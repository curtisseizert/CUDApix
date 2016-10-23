/*
  U calculates Sum(pi(y) - pi(x/p^2)) p such that sqrt(x/y) < p < cbrt(x)
 */

#include <iostream>
#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "P2.cuh"
#include "trivial.cuh"
#include "CUDASieve/cudasieve.hpp"

#define THREADS_PER_BLOCK 256

uint64_t S1_trivial(uint64_t x, uint64_t y)
{
  uint64_t lower_bound = std::sqrt(x/y);
  uint64_t upper_bound = std::cbrt(x);

  PrimeArray lo, hi;

  lo.bottom = lower_bound;
  lo.top = upper_bound;

  hi.bottom = upper_bound;
  hi.top = y;

  lo.d_primes = CudaSieve::getDevicePrimes(lo.bottom, lo.top, lo.len, 0);
  hi.d_primes = CudaSieve::getDevicePrimes(hi.bottom, hi.top, hi.len, 0);

  uint32_t blocks = 1 + lo.len/THREADS_PER_BLOCK;

  x_over_psquared<<<blocks, THREADS_PER_BLOCK>>>(lo.d_primes, x, lo.len);

  thrust::upper_bound(thrust::device, hi.d_primes, hi.d_primes + hi.len, lo.d_primes, lo.d_primes + lo.len, lo.d_primes);

  x_minus_array<<<blocks, THREADS_PER_BLOCK>>>(lo.d_primes, (uint64_t) hi.len, lo.len);

  uint64_t u = thrust::reduce(thrust::device, lo.d_primes, lo.d_primes + lo.len);
  u += hi.len*(hi.len - 1)/2;

  cudaFree(lo.d_primes);
  cudaFree(hi.d_primes);

  return u;
}

__global__ void x_over_psquared(uint64_t * p, uint64_t x, size_t len)
{
  uint32_t tidx = threadIdx.x + blockDim.x*blockIdx.x;

  if(tidx < len) p[tidx] = x / (p[tidx] * p[tidx]);
}

__global__ void x_minus_array(uint64_t * a, uint64_t x, size_t len)
{
  uint32_t tidx = threadIdx.x + blockDim.x*blockIdx.x;

  if(tidx < len) a[tidx] = x - a[tidx];
}
