#include <iostream>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/binary_search.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "cuda_uint128.h"
#include "CUDASieve/cudasieve.hpp"
#include "P2.cuh"
#include "cudapix.hpp"
#include "general/device_functions.cuh"

uint64_t maxRange = 1ul << 33;
const uint16_t threadsPerBlock = 256;

uint128_t P2(uint128_t x, uint64_t y)
{
    uint128_t total = 0;

    uint64_t top = x/y;
    uint64_t sqrt_x = uint128_t::sqrt(x);

    uint64_t gap = 0;
    bool run = 1;

    PrimeArray hi, lo;
    lo.top = sqrt_x;
    hi.bottom = sqrt_x;
    hi.top = sqrt_x + maxRange;

    ResetCounter counter;

    do{
      if(hi.top >= top) {hi.top = top; run = 0;}
      lo.bottom = x/hi.top;
      lo.bottom += lo.bottom & 1ull; // make sure bounds of sieving intervals are even

      // Get primes P1 such that y < p < sqrt(x)
      lo.d_primes = CudaSieve::getDevicePrimes(lo.bottom, lo.top, lo.len, 0);
      // Get primes P2 such that x/y > p > sqrt(x)
      hi.d_primes = CudaSieve::getDevicePrimes(hi.bottom, hi.top, hi.len, 0);


      divXbyY(x, lo.d_primes, lo.len);

      // this will return pi(x/p) - pi(hi.bottom) for all P1
      thrust::upper_bound(thrust::device, hi.d_primes, hi.d_primes+hi.len, lo.d_primes, lo.d_primes+lo.len, lo.d_primes);
      //
      total += thrust::reduce(thrust::device, lo.d_primes, lo.d_primes+lo.len); // sum of pi(x/p) - pi(hi.bottom)
      total += ((lo.len+1)*lo.len/2 + lo.len*(gap)); // sum(pi(lo.top) - pi(p)) + sum(pi(hi.bottom) - pi(lo.top))

      gap += lo.len + hi.len; // change ranges for next sieving interval
      hi.bottom = hi.top;
      hi.top += maxRange;
      lo.top = lo.bottom;

      cudaFree(hi.d_primes);
      cudaFree(lo.d_primes);

      std::cout << "\t" << (float)100*(hi.bottom - sqrt_x)/(top - sqrt_x) << "% complete         \r";
      std::cout << "\t\t\t\t\tTotal: " << total << "\r";
      std::cout << std::flush;

      counter.increment();
    }while(run);
    std::cout << std::endl;
    return total;
}


// this is the faster P2 implementation and does all of the critical operations on
// the GPU, however it is limited to 64 bit X due to CUDA's lack of native
// support for extended precision integers
uint64_t P2(uint64_t x, uint64_t y)
{
  uint64_t total = 0; // z is a temp for conversion to uint64_t

  uint64_t sqrt_x = sqrt(x);
  uint64_t top = x/y;
  uint64_t * d_sums, gap = 0;
  bool run = 1;
  x_Over_y op(x);

  PrimeArray hi, lo;
  lo.top = sqrt_x;
  hi.bottom = sqrt_x;
  hi.top = sqrt_x + maxRange;

  ResetCounter counter;

  do{
    if(hi.top >= top) {hi.top = top; run = 0;}

    lo.bottom = x/hi.top;
    lo.bottom += lo.bottom & 1ull; // make sure bounds of sieving intervals are even

    // Get primes P1 such that y < p < sqrt(x)
    lo.d_primes = CudaSieve::getDevicePrimes(lo.bottom, lo.top, lo.len, 0);

    // Get primes P2 such that x/y > p > sqrt(x)
    hi.d_primes = CudaSieve::getDevicePrimes(hi.bottom, hi.top, hi.len, 0);

    // divXbyY<<<lo.len/THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(x, lo.d_primes, lo.len);
    thrust::transform(thrust::device, lo.d_primes, lo.d_primes + lo.len, lo.d_primes, op);

    cudaMalloc(&d_sums, lo.len*sizeof(uint64_t));

    // this will return pi(x/p) - pi(hi.bottom) for all P1
    thrust::upper_bound(thrust::device, hi.d_primes, hi.d_primes+hi.len, lo.d_primes, lo.d_primes+lo.len, d_sums);

    total += thrust::reduce(thrust::device, d_sums, d_sums+lo.len); // sum of pi(x/p) - pi(hi.bottom)
    total += (lo.len+1)*lo.len/2 + lo.len*(gap); // sum(pi(lo.top) - pi(p)) + sum(pi(hi.bottom) - pi(lo.top))

    gap += lo.len + hi.len; // change ranges for next sieving interval
    hi.bottom = hi.top;
    hi.top += maxRange;
    lo.top = lo.bottom;

    cudaFree(hi.d_primes);
    cudaFree(lo.d_primes);
    cudaFree(d_sums);

    std::cout << "\t" << (float)100*(hi.bottom - sqrt_x)/(top - sqrt_x) << "% complete         \r";
    std::cout << "\t\t\t\t\tTotal: " << total << "\r";
    std::cout << std::flush;

    counter.increment();
  }while(run);
  std::cout << std::endl;
  return total;
}

void divXbyY(uint128_t x, uint64_t * y, size_t len)
{
  global::divXbyY<<<len/threadsPerBlock + 1, threadsPerBlock>>>(x, y, len);
}
