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
#include "Gourdon/gourdonvariant.hpp"

uint64_t GourdonVariant64::B()
{
  uint64_t total = 0;

  uint64_t * d_sums, gap = pi_sqrtx;
  bool run = 1;
  x_Over_y op(x);

  PrimeArray hi, lo;
  lo.top = sqrtx;
  hi.bottom = sqrtx;
  hi.top = sqrtx + maxRange_;

  ResetCounter counter;

  do{
    if(hi.top >= z) {hi.top = z; run = 0;}

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
    total += lo.len*(gap); // sum(pi(lo.top) - pi(p)) + sum(pi(hi.bottom) - pi(lo.top))

    gap += hi.len; // change ranges for next sieving interval
    hi.bottom = hi.top;
    hi.top += maxRange_;
    lo.top = lo.bottom;

    cudaFree(hi.d_primes);
    cudaFree(lo.d_primes);
    cudaFree(d_sums);

    std::cout << "\t" << (float)100*(hi.bottom - sqrtx)/(z - sqrtx) << "% complete         \r";
    std::cout << "\t\t\t\t\tTotal: " << total << "\r";
    std::cout << std::flush;

    counter.increment();
  }while(run);
  std::cout << std::endl;
  return total;
}
