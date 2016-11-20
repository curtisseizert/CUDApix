// C.cu
//
// A simple implemenation of the (modified) sum "C" in Xavier Gourdon's variant of the
// Deleglise-Rivat prime counting algorithm.  While the implemenation is
// efficient, it requires that all values of pi(x) from 0 to sqrt(x) are available
// to the kernel, which becomes very expensive at higher ranges.
//
// Copywrite (c) 2016 Curtis Seizert <cseizert@gmail.com>

#include <iostream>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <CUDASieve/cudasieve.hpp>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <CUDASieve/host.hpp>
#include <math_functions.h>
#include <cuda_profiler_api.h>

#include "general/tools.hpp"
#include "general/device_functions.cuh"
#include "Gourdon/C.cuh"
#include "Gourdon/gourdonvariant.hpp"
#include "cudapix.hpp"
#include "pitable.cuh"

uint64_t GourdonVariant64::C()
{
  uint64_t sum = 0, blocks = 0;
  uint64_t sqrty = std::sqrt(y);
  uint64_t pi_sqrty = CudaSieve::countPrimes(sqrty);
  PrimeArray pq(sqrty, y);

  uint64_t num_p = pi_qrtx - pi_sqrty;

  pq.d_primes = CudaSieve::getDevicePrimes(pq.bottom, pq.top, pq.len, 0);

  uint64_t maxblocks = 1 + ((pi_y - pi_qrtx)/threadsPerBlock);

  uint64_t * d_sums;
  cudaMalloc(&d_sums, maxblocks * sizeof(uint64_t));
  global::zero<<<maxblocks, threadsPerBlock>>>(d_sums, maxblocks);
  cudaDeviceSynchronize();
  KernelTime timer;
  timer.start();

  PiTable * pi_table = new PiTable(sqrtx);
  uint32_t * d_pitable = pi_table->getCurrent();

  blocks = 1 + ((pq.len)/threadsPerBlock);
  C_nonseg<<<blocks, threadsPerBlock>>>(x, pq.d_primes, d_sums, pi_sqrty, num_p, pq.len, d_pitable, pi_qrtx - pi_sqrty + 1);
  cudaDeviceSynchronize();

  sum = thrust::reduce(thrust::device, d_sums, d_sums + maxblocks);

  timer.stop();
  timer.displayTime();
  timer.~KernelTime();

  cudaFree(d_sums);
  cudaFree(pq.d_primes);
  cudaDeviceReset();
  return sum;
}

__global__
void C_nonseg( uint64_t x, uint64_t * pq, uint64_t * sums,
                 uint64_t pi_sqrty, uint32_t num_p, uint64_t maxQidx,
                 uint32_t * d_pitable, uint64_t first_q_offset)
{
  uint64_t q = 0, quot = 0;
  __shared__ uint64_t s_pi_quot[numThreads];
  s_pi_quot[threadIdx.x] = 0;

  uint32_t qidx = first_q_offset + (threadIdx.x + blockIdx.x * blockDim.x);
  if(qidx < maxQidx)
    q = pq[qidx];

  for(uint32_t i = 0; i < num_p; i++){
    uint64_t p = pq[i];
    quot = checkAndDiv_C(q, p, x);

    if(quot != 0){
      s_pi_quot[threadIdx.x] += calculatePiChi_C(quot, d_pitable);
      s_pi_quot[threadIdx.x] += 2 - (i + pi_sqrty);
    }
  }
  q = 0;
  // q is repurposed here to act as the register holding the reduction result
  __syncthreads();
  q = thrust::reduce(thrust::device, s_pi_quot, s_pi_quot + numThreads);
  if(threadIdx.x == 0)
    sums[blockIdx.x] += q;
}

__device__
uint64_t calculatePiChi_C(uint64_t quot, uint32_t * d_pitable)
{
  uint64_t r = d_pitable[(quot + 1)/2];

  return (uint64_t)r;
}


__device__
inline uint64_t checkAndDiv_C(uint64_t q, uint64_t p, uint64_t x)
{
  uint64_t quot = x / (p * q);
  quot = (q > 1 + x / (p * p * p) ? quot : 0);

  return quot;
}
