#include <iostream>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <uint128_t.cuh>
#include <CUDASieve/cudasieve.hpp>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <CUDASieve/host.hpp>
#include <math_functions.h>

#include "general/tools.hpp"
#include "general/device_functions.cuh"
#include "Gourdon/A.cuh"
#include "Gourdon/gourdonvariant.hpp"
#include "cudapix.hpp"
#include "pitable.cuh"

__constant__ const uint16_t numThreads = 256;
const uint16_t threadsPerBlock = 256;
__constant__ const uint16_t numQ = 256;
const uint16_t QperBlock = 256;

__constant__ uint64_t p[4096];

uint64_t GourdonVariant64::A()
{
  // sum(x^1/numQ / numThreads < p <= x^1/3, sum(p < q <= sqrt(x/p), chi(x/(p*q)) * pi(x/(p * q))))
  // where chi(n) = 1 if n >= y and 2 if n < y

  uint64_t sum = 0, pSoFar = 0, blocks = 0;
  PrimeArray pi(0, sqrtx);
  PrimeArray pq(qrtx, sqrt(x / qrtx));

  uint64_t num_p = pi_cbrtx - pi_qrtx;

  pi.d_primes = CudaSieve::getDevicePrimes(pi.bottom, pi.top, pi.len, 0);
  pq.d_primes = CudaSieve::getDevicePrimes(pq.bottom, pq.top, pq.len, 0);

  uint64_t maxblocks = 1 + (pq.len/QperBlock);

  uint64_t * d_sums;
  cudaMalloc(&d_sums, maxblocks * sizeof(uint64_t));
  global::zero<<<maxblocks, threadsPerBlock>>>(d_sums, maxblocks);
  cudaDeviceSynchronize();
  KernelTime timer;
  timer.start();

  uint32_t * d_pitable = get_d_piTable(sqrtx);

  while(num_p > pSoFar){
    uint16_t pPerBlock = min(num_p, (uint64_t)4096);
    cudaMemcpyToSymbol(p, pq.d_primes +pSoFar, pPerBlock * sizeof(uint64_t), 0, cudaMemcpyDefault);

    blocks = 1 + ((pq.len - pSoFar)/QperBlock);
    A_lo_p_reg<<<blocks, threadsPerBlock>>>(x, pq.d_primes + pSoFar, pi.d_primes, d_sums, pi_y, pPerBlock, pq.len, d_pitable);
    cudaDeviceSynchronize();
    // std::cout << blocks << std::endl;
    pSoFar += pPerBlock;
  }

  sum = thrust::reduce(thrust::device, d_sums, d_sums + maxblocks);

  timer.stop();
  timer.displayTime();
  timer.~KernelTime();

  cudaFree(d_sums);
  cudaFree(d_pitable);
  cudaFree(pi.d_primes);
  cudaFree(pq.d_primes);

  return sum;
}

uint64_t GourdonVariant64::checkA()
{
  PrimeArray pi(0, sqrtx);
  PrimeArray pq(qrtx, sqrt(x / qrtx));

  uint64_t sum = 0, num_p = pi_cbrtx - pi_qrtx;

  pi.h_primes = CudaSieve::getHostPrimes(pi.bottom, pi.top, pi.len, 0);
  pq.h_primes = CudaSieve::getHostPrimes(pq.bottom, pq.top, pq.len, 0);
  CudaSieve * sieve = new CudaSieve((uint64_t)0, sqrtx, 0);

  for(uint32_t i = 1; i < num_p; i++){
    uint32_t p = pq.h_primes[i];
    std::cout << "\t" << p << "\t(" << i << "/" << num_p << ")\r";
    std::cout << std::flush;
    for(uint32_t j = i + 1; j <= pq.len; j++){
      uint32_t q = pq.h_primes[j];
      if(q > sqrt(x / p)) break;
      uint64_t n = x / (p * q);
      uint64_t pi_n = sieve->countPrimesSegment(0, n);
      if(pi_n < pi_y) pi_n *= 2;
      // std::cout << pi_n << std::endl;
      sum += pi_n;
    }
  }
  std::cout << std::endl;
  return sum;
}

__global__ void A_lo_p(uint64_t x, uint64_t * q, uint64_t * pi, uint64_t * sums,
                       uint32_t pi_y, uint32_t num_p, uint64_t max_num_q,
                       uint32_t * d_pitable)
{
  __shared__ uint64_t s_q[numQ];
  __shared__ uint64_t s_quot[numQ];

  uint64_t sum = 0;

  fill_s_q(q, s_q, max_num_q);


  for(uint32_t iter = 0; iter < num_p; iter++){
    calculateQuotient(x, s_q, s_quot, iter);
    __syncthreads();

    getPiOfQuot(s_quot, d_pitable);

    chiXPQ(pi_y, s_quot);
    __syncthreads();

    sum = thrust::reduce(thrust::device, s_quot, s_quot + numQ);
    __syncthreads();

    if(threadIdx.x == 0)
      sums[blockIdx.x] += sum;
  }
}

__global__ void A_lo_p_reg(uint64_t x, uint64_t * q, uint64_t * pi, uint64_t * sums,
                           uint32_t pi_y, uint32_t num_p, uint64_t max_num_q,
                           uint32_t * d_pitable)
{
  uint64_t qt, quot, sum;
  __shared__ uint64_t s_pi_quot[numThreads];
  s_pi_quot[threadIdx.x] = 0;

  uint32_t qidx = numQ / numThreads * (threadIdx.x + blockIdx.x * blockDim.x);

  qt = q[qidx];

  for(uint32_t iter = 1; iter < num_p; iter++){
    calculateQuotient(x, qt, quot, iter);
    __syncthreads();

    getPiOfQuot(quot, pi_y, d_pitable, s_pi_quot);

    __syncthreads();
  }

  sum = thrust::reduce(thrust::device, s_pi_quot, s_pi_quot + numThreads);
  if(threadIdx.x == 0)
    sums[blockIdx.x] += sum;
}

__device__ void fill_s_q(uint64_t * q, uint64_t * s_q, uint64_t max_num_q)
{
  uint32_t qidx = numQ / numThreads * (threadIdx.x + blockIdx.x * blockDim.x);
  uint32_t sqidx = numQ / numThreads * threadIdx.x;

  #pragma unroll numQ / numThreads
  for(uint16_t j = 0; j < numQ/numThreads; j++)
    if(qidx < max_num_q)
      s_q[sqidx + j] = q[qidx + j];
    else
    s_q[sqidx + j] = 0;
}

__device__ void calculateQuotient(uint64_t x, uint64_t * s_q, uint64_t * s_quot, uint32_t iter)
{
  uint32_t sqidx = numQ / numThreads * threadIdx.x;

  #pragma unroll numQ / numThreads
  for(uint16_t j = 0; j < numQ/numThreads; j++){
    if((s_q[sqidx + j] > p[iter]) && (s_q[sqidx + j] <= sqrtf(x / p[iter]))){ // this may have to be changed to double precision
      s_quot[sqidx + j] = x / (p[iter] * s_q[sqidx + j]);
    }else{
      s_quot[sqidx  + j] = 0;
      s_q[sqidx + j] = 0;
    }
  }
}

__device__ void calculateQuotient(uint64_t x, uint64_t qt, uint64_t & quot, uint32_t iter)
{
  if((qt > p[iter]) && (qt <= sqrtf(x / p[iter]))){
    quot = x / (p[iter] * qt);
  }else{
    quot = 0;
    qt = 0;
  }
}

__device__ void chiXPQ(uint32_t pi_y, uint64_t * s_quot)
{
  uint32_t sqidx = numQ / numThreads * threadIdx.x;

  #pragma unroll numQ / numThreads
  for(uint16_t j = 0; j < numQ/numThreads; j++){
    if(s_quot[sqidx + j] < pi_y)
      s_quot[sqidx + j] *= 2;
  }
}

__device__ void chiXPQ_reg(uint32_t pi_y, uint64_t * s_pi_quot)
{
  if(s_pi_quot[threadIdx.x] < pi_y)
    s_pi_quot[threadIdx.x] *= 2;
}

__device__ void getPiOfQuot(uint64_t * s_quot, uint32_t * d_pitable)
{
  uint32_t sqidx = numQ / numThreads * threadIdx.x;

  #pragma unroll numQ / numThreads
  for(uint16_t j = 0; j < numQ/numThreads; j++){
    s_quot[sqidx + j] = d_pitable[(1 + s_quot[sqidx + j])/2 - 1];
  }
}

// This version of this function incorporates chi(x/(p * q)) unlike the above
// which uses a separate function to evaluate chi.
__device__ void getPiOfQuot(uint64_t quot, uint32_t pi_y, uint32_t * d_pitable,
                            uint64_t * s_pi_quot)
{
  uint64_t r = d_pitable[(1 + quot)/2 - 1];
  s_pi_quot[threadIdx.x] += r;
  if(r < pi_y)
    s_pi_quot[threadIdx.x] += r;
  // if(quot != 0) printf("%llu\n", s_pi_quot[threadIdx.x]);
}
