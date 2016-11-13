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

#include "cudapix.hpp"
#include "trivial.cuh"
#include "CUDASieve/cudasieve.hpp"
#include "general/device_functions.cuh"
#include "uint128_t.cuh"

const uint16_t threadsPerBlock = 256;


__global__ void x_over_psquared(uint64_t * p, uint64_t x, size_t len)
{
  uint32_t tidx = threadIdx.x + blockDim.x*blockIdx.x;

  if(tidx < len) p[tidx] = x / (p[tidx] * p[tidx]);
}


uint64_t S2_trivial(uint64_t x, uint64_t y)
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

  uint32_t blocks = 1 + lo.len/threadsPerBlock;

  x_over_psquared<<<blocks, threadsPerBlock>>>(lo.d_primes, (uint64_t)x, lo.len);

  thrust::upper_bound(thrust::device, hi.d_primes, hi.d_primes + hi.len, lo.d_primes, lo.d_primes + lo.len, lo.d_primes);

  x_minus_array(lo.d_primes, (uint64_t) hi.len, lo.len);

  uint64_t u = thrust::reduce(thrust::device, lo.d_primes, lo.d_primes + lo.len);
  u += hi.len*(hi.len - 1)/2;

  cudaFree(lo.d_primes);
  cudaFree(hi.d_primes);

  return u;
}

uint128_t S2_trivial(uint128_t x, uint64_t y) // due to fitting p^2 (p_max = cbrt(x)) in a uint64_t
                                              // this imposes a limit of 2^96 for x
{
  uint64_t lower_bound = uint128_t::sqrt(x/y);
  uint64_t upper_bound = uint128_t::sqrt(x);

  std::cout << upper_bound << std::endl;

  upper_bound = pow(upper_bound, (double)(2.0/3.0));

  std::cout << upper_bound << std::endl;

  PrimeArray lo(lower_bound, upper_bound);
  PrimeArray hi(upper_bound, y);

  lo.d_primes = CudaSieve::getDevicePrimes(lo.bottom, lo.top, lo.len, 0);
  hi.d_primes = CudaSieve::getDevicePrimes(hi.bottom, hi.top, hi.len, 0);

  xOverPSquared(lo.d_primes, x, lo.len);

  thrust::upper_bound(thrust::device, hi.d_primes, hi.d_primes + hi.len, lo.d_primes, lo.d_primes + lo.len, lo.d_primes);

  x_minus_array(lo.d_primes, (uint64_t) hi.len, lo.len);

  uint128_t u = hi.len * (hi.len - 1)/2;
  u += thrust::reduce(thrust::device, lo.d_primes, lo.d_primes + lo.len);

  cudaFree(lo.d_primes);
  cudaFree(hi.d_primes);

  return u;
}

inline void xOverPSquared(uint64_t * p, uint128_t x, size_t len)
{
  global::xOverPSquared<<<len/threadsPerBlock + 1, threadsPerBlock>>>(p, x, len);
}

inline void x_minus_array(uint64_t * p, uint64_t x, size_t len)
{
  uint32_t blocks = 1 + len/threadsPerBlock;
  global::x_minus_array<<<blocks, threadsPerBlock>>>(p, (uint64_t) x, len);
}
