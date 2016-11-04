#include <iostream>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <uint128_t.cuh>
#include <CUDASieve/cudasieve.hpp>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include "general/device_functions.cuh"
#include "Gourdon/A.cuh"
#include "Gourdon/gourdonvariant.hpp"
#include "cudapix.hpp"

__constant__ const uint16_t numThreads = 256;
const uint16_t threadsPerBlock = 256;
__constant__ const uint16_t numQ = 1024;
const uint16_t QperBlock = 1024;

__constant__ uint64_t p[4096];

uint64_t GourdonVariant64::A()
{
  // sum(x^1/4 < p <= x^1/3, sum(p < q <= sqrt(x/p), chi(x/(p*q)) * pi(x/(p * q))))
  // where chi(n) = 1 if n >= y and 2 if n < y

  uint64_t sum = 0, pSoFar = 0, blocks = 0;
  PrimeArray pi(0, sqrtx);
  PrimeArray pq(qrtx, sqrt(x / qrtx));

  uint64_t num_p = pi_cbrtx - pi_qrtx;

  pi.d_primes = CudaSieve::getDevicePrimes(pi.bottom, pi.top, pi.len, 0);
  pq.d_primes = CudaSieve::getDevicePrimes(pq.bottom, pq.top, pq.len, 0);

  blocks = 1 + (pq.len/QperBlock);

  uint64_t * d_sums = (uint64_t *)malloc(blocks * sizeof(uint64_t));
  global::zero<<<blocks, threadsPerBlock>>>(d_sums, pq.len/QperBlock);

  while(num_p > pSoFar){
    uint16_t pPerBlock = min(num_p, (uint64_t)4096);
    cudaMemcpyToSymbol(p, pq.d_primes + pSoFar, pPerBlock * sizeof(uint64_t), cudaMemcpyDefault);
    pSoFar += pPerBlock;

    blocks = 1 + ((pq.len - pSoFar)/QperBlock);
    A_lo_p<<<blocks, threadsPerBlock>>>(x, pq.d_primes + pSoFar, pi.d_primes, d_sums, pi_y, pPerBlock, pq.len);
    cudaDeviceSynchronize();
  }

  sum = thrust::reduce(thrust::device, d_sums, d_sums + pq.len/QperBlock);
  return sum;
}

__global__ void A_lo_p(uint64_t x, uint64_t * q, uint64_t * pi, uint64_t * sums,
                       uint32_t pi_y, uint32_t num_p, uint64_t max_num_q)
{
  __shared__ uint64_t s_q[numQ];
  __shared__ uint64_t s_quot[numQ];

  uint64_t sum = 0;

  fill_s_q(q, s_q, max_num_q);

  for(uint32_t iter = 0; iter < num_p; iter++){
    calculateQuotient(x, s_q, s_quot, iter);
    __syncthreads();
    if(s_q[0] == 0 && s_q[numQ - 1] == 0) break;

    thrust::upper_bound(thrust::device, pi, pi + numQ, s_quot, s_quot + numQ, s_quot);

    chiXPQ(pi_y, s_quot);
    __syncthreads();

    sum += thrust::reduce(thrust::device, s_quot, s_quot + numQ);
  }
  sums[blockIdx.x] = sum;
}

__device__ void fill_s_q(uint64_t * q, uint64_t * s_q, uint64_t max_num_q)
{
  uint32_t qidx = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
  uint32_t sqidx = 4 * threadIdx.x;

  #pragma unroll 4
  for(uint16_t j = 0; j < numQ/numThreads; j++)
    if(qidx < max_num_q)
      s_q[sqidx + j] = q[qidx + j];
    else
    s_q[sqidx + j] = 0;
}

__device__ void calculateQuotient(uint64_t x, uint64_t * s_q, uint64_t * s_quot, uint32_t iter)
{
  uint32_t sqidx = 4 * threadIdx.x;

  #pragma unroll 4
  for(uint16_t j = 0; j < numQ/numThreads; j++){
    if((s_q[sqidx + j] > p[iter]) && (s_q[sqidx + j] <= x / (p[iter] * p[iter]))){
      s_quot[sqidx + j] = x / (p[iter] * s_q[sqidx + j]);
    }else{
      s_quot[sqidx + j] = 0;
      s_q[sqidx + j] = 0;
    }
  }
}

__device__ void chiXPQ(uint32_t pi_y, uint64_t * s_quot)
{
  uint32_t sqidx = 4 * threadIdx.x;

  #pragma unroll 4
  for(uint16_t j = 0; j < numQ/numThreads; j++){
    if(s_quot[sqidx + j] < pi_y)
      s_quot[sqidx + j] *= 2;
  }
}
