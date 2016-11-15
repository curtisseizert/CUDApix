// A1.cu
//
// A simple implemenation of the sum "A" in Xavier Gourdon's variant of the
// Deleglise-Rivat prime counting algorithm.  While the implemenation is
// efficient, it requires that all values of pi(x) from 0 to sqrt(x) are available
// to the kernel, which becomes very expensive at higher ranges.  A solution to
// this problem is segmentation (predictably), and an implemenation of this
// algorithm that employs segmentation can be found in A2.cu.
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
#include "Gourdon/A.cuh"
#include "Gourdon/gourdonvariant.hpp"
#include "cudapix.hpp"
#include "pitable.cuh"

uint64_t GourdonVariant64::A()
{
  // sum(x^1/4 < p <= x^1/3, sum(p < q <= sqrt(x/p), chi(x/(p*q)) * pi(x/(p * q))))
  // where chi(n) = 1 if n >= y and 2 if n < y

  uint64_t sum = 0, blocks = 0;
  PrimeArray pq(qrtx, sqrt(x / qrtx));

  uint64_t num_p = pi_cbrtx - pi_qrtx;

  pq.d_primes = CudaSieve::getDevicePrimes(pq.bottom, pq.top, pq.len, 0);

  uint64_t maxblocks = 1 + (pq.len/threadsPerBlock);

  uint64_t * d_sums;
  cudaMalloc(&d_sums, maxblocks * sizeof(uint64_t));
  global::zero<<<maxblocks, threadsPerBlock>>>(d_sums, maxblocks);
  cudaDeviceSynchronize();
  KernelTime timer;
  timer.start();

  PiTable * pi_table = new PiTable(sqrtx);
  uint32_t * d_pitable = pi_table->getCurrent();

  blocks = 1 + ((pq.len)/threadsPerBlock);
  A_lo_p_reg<<<blocks, threadsPerBlock>>>(x, pq.d_primes, d_sums, y, num_p, pq.len, d_pitable);
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
void A_lo_p_reg( uint64_t x, uint64_t * pq, uint64_t * sums,
                 uint64_t y, uint32_t num_p, uint64_t max_num_q,
                 uint32_t * d_pitable)
{
  uint64_t q = 0, quot = 0;
  __shared__ uint64_t s_pi_quot[numThreads];
  s_pi_quot[threadIdx.x] = 0;

  uint32_t qidx = 1 + (threadIdx.x + blockIdx.x * blockDim.x);
  if(qidx < max_num_q)
    q = pq[qidx];

  for(uint32_t i = 0; i < qidx; i++){
    uint64_t p = pq[i];
    quot = checkAndDiv(q, p, x);

    if(quot != 0)
      s_pi_quot[threadIdx.x] += calculatePiChi(quot, y, d_pitable);
    else break;
  }
  q = 0;
  // q is repurposed here to act as the register holding the reduction result
  __syncthreads();
  q = thrust::reduce(thrust::device, s_pi_quot, s_pi_quot + numThreads);
  if(threadIdx.x == 0)
    sums[blockIdx.x] += q;
}

__device__
uint64_t calculatePiChi(uint64_t quot, uint64_t y, uint32_t * d_pitable)
{
  uint64_t r = d_pitable[(quot + 1)/2];
  if(quot < y)
    r <<= 1;
  return r;
}


__device__
inline uint64_t checkAndDiv(uint64_t q, uint64_t p, uint64_t x)
{
  uint64_t quot = (q <= __dsqrt_rd(x/p) && q != 0) ? x/(p * q) : 0;
  quot = (p < q ? quot : 0);

  return quot;
}
