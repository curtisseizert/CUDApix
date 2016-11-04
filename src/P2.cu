#include <iostream>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/binary_search.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "uint128_t.cuh"
#include "CUDASieve/cudasieve.hpp"
#include "P2.cuh"
#include "cudapix.hpp"
#include "general/device_functions.cuh"

uint64_t maxRange = 1ul << 33;
const uint16_t threadsPerBlock = 256;



#ifndef _UINT128_T_CUDA_H

#include <omp.h>
#include <gmpxx.h>
#include <future>

// this is a less efficient P2 implementation compared to the one below, but
// it has the advantage of not being limited to a 64 bit x, which it accomplishes
// by using the gmp library and performing the division operation x/p on the CPU
// rather than the GPU as below

mpz_class P2(mpz_class x, mpz_class y)
{
  mpz_class z, total = 0; // z is a temp for conversion to uint64_t

  z = sqrt(x);
  uint64_t sqrt_x = z.get_ui();

  z = x/y;
  uint64_t top = z.get_ui();

  uint64_t * d_sums, gap = 0;
  bool run = 1;

  PrimeArray hi, lo;
  lo.top = sqrt_x;
  hi.bottom = sqrt_x;
  hi.top = sqrt_x + maxRange;

  ResetCounter counter;

  do{
    if(hi.top >= top) {hi.top = top; run = 0;}

    z = x/hi.top;
    lo.bottom = z.get_ui();
    lo.bottom += lo.bottom & 1ull; // make sure bounds of sieving intervals are even

    // Get primes P1 such that y < p < sqrt(x)
    lo.h_primes = CudaSieve::getHostPrimes(lo.bottom, lo.top, lo.len, 0);

    // Get primes P2 such that x/y > p > sqrt(x)
    std::future<uint64_t *> fut = std::async(std::launch::async, CudaSieve::getDevicePrimes, hi.bottom, hi.top, std::ref(hi.len), 0);

    #pragma omp parallel for // calculate x/y for primes y < p < sqrt(x)
    for(uint32_t i = 0; i < lo.len; i++){
      mpz_class n;
      n = x/lo.h_primes[i];
      lo.h_primes[i] = n.get_ui();
    }

    cudaMalloc(&d_sums, lo.len*sizeof(uint64_t));
    cudaMalloc(&lo.d_primes, lo.len*sizeof(uint64_t));

    cudaMemcpy(lo.d_primes, lo.h_primes, lo.len*sizeof(uint64_t), cudaMemcpyHostToDevice);

    hi.d_primes = fut.get();

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
    cudaFreeHost(lo.h_primes);

    std::cout << "\t" << (float)100*(hi.bottom - sqrt_x)/(top - sqrt_x) << "% complete         \r";
    std::cout << "\t\t\t\t\tTotal: " << total << "\r";
    std::cout << std::flush;

    counter.increment();

  }while(run);
  std::cout << std::endl;
  return total;
}
#endif // #ifndef _UINT128_T_CUDA

uint128_t P2(uint128_t x, uint64_t y)
{
    uint128_t total = 0;

    uint64_t top = x/y;
    uint64_t sqrt_x = uint128_t::sqrt(x);

    uint64_t * d_sums, gap = 0;
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

      cudaMalloc(&d_sums, lo.len*sizeof(uint64_t));

      // this will return pi(x/p) - pi(hi.bottom) for all P1
      thrust::upper_bound(thrust::device, hi.d_primes, hi.d_primes+hi.len, lo.d_primes, lo.d_primes+lo.len, d_sums);

      total += thrust::reduce(thrust::device, d_sums, d_sums+lo.len); // sum of pi(x/p) - pi(hi.bottom)
      total += ((lo.len+1)*lo.len/2 + lo.len*(gap)); // sum(pi(lo.top) - pi(p)) + sum(pi(hi.bottom) - pi(lo.top))

      gap += lo.len + hi.len; // change ranges for next sieving interval
      hi.bottom = hi.top;
      hi.top += maxRange;
      lo.top = lo.bottom;

      if(counter.isReset()){
        size_t free, tot;
        cudaMemGetInfo(&free, &tot);
        if(2*free > tot + (1ull << 30)) maxRange *= 2;
      }

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
