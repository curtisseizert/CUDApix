// omega3.cu
//
// 128-bit
//
// A segmented implemenation of the sum "C" in Xavier Gourdon's variant of the
// Deleglise-Rivat prime counting algorithm with modified bounds to simplify
// implementation with a range of y values.
//
// Copywrite (c) 2016 Curtis Seizert <cseizert@gmail.com>

#include <iostream>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_uint128.h>
#include <cuda_uint128_primitives.cuh>
#include <CUDASieve/cudasieve.hpp>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <CUDASieve/host.hpp>
#include <math_functions.h>
#include <cuda_profiler_api.h>

#include "general/tools.hpp"
#include "general/device_functions.cuh"
#include "Deleglise-Rivat/omega3.cuh"
#include "Deleglise-Rivat/deleglise-rivat.hpp"
#include "cudapix.hpp"
#include "pitable.cuh"

uint128_t deleglise_rivat128::omega3()
{
  cudaStream_t stream[3];
  uint128_t sum = 0;
  PrimeArray pq(0, y);
  uint64_t sqrty = _isqrt(y);
  uint64_t pi_sqrty = CudaSieve::countPrimes(0, sqrty);

  uint64_t num_p = pi_qrtx - pi_sqrty;

  pq.d_primes = CudaSieve::getDevicePrimes(pq.bottom, pq.top, pq.len, 0);

  uint64_t maxblocks = 1 + ((pq.len - pi_qrtx)/threadsPerBlock);

  uint64_t * d_sums;
  uint64_t * d_lastQ, * d_nextQ;
  cudaMalloc(&d_sums, maxblocks * sizeof(uint64_t));
  cudaMalloc(&d_lastQ, pi_qrtx * sizeof(uint64_t));
  cudaMalloc(&d_nextQ, pi_qrtx * sizeof(uint64_t));
  cudaMallocHost(&pq.h_primes, pq.len * sizeof(uint64_t));
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);
  cudaStreamCreate(&stream[2]);

  cudaMemcpy(pq.h_primes, pq.d_primes, pq.len*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  Omega3_lower_bound<<<num_p/threadsPerBlock + 1, threadsPerBlock, 0, stream[1]>>>
    (x, d_lastQ, pq.d_primes, pi_sqrty, pi_qrtx);
  global::zero<<<maxblocks, threadsPerBlock, 0, stream[2]>>>(d_sums, maxblocks);
  cudaDeviceSynchronize();
  thrust::upper_bound(thrust::device, pq.d_primes, pq.d_primes + pq.len, d_lastQ, d_lastQ + pi_qrtx, d_lastQ);

  cudaDeviceSynchronize();
  KernelTime timer;
  timer.start();

  PiTable pi_table(sqrtx, 0);
  pi_table.set_pi_base(pi_sqrtx);
  cudaFuncSetCacheConfig(Omega3_kernel, cudaFuncCachePreferL1);

  // find all (p,q) pairs such that x/(p * q) >= x^(3/8)
  while(pi_table.get_base() > 0){
  // for(uint32_t i = 0; i < 3; i++){ // for testing
    // nextQ is copied from lastQ each iteration rather than switching pointers
    // to form the basis of a compare and swap operation in the kernel that evaluates
    // whether a given value has changed
    cudaMemcpyAsync(d_nextQ, d_lastQ, pi_qrtx * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream[1]);
    uint64_t pMin = std::max((uint64_t)pow(sqrtx, 0.3333333), _isqrt(pi_table.getNextBaseDown()));
    uint64_t pCorner = std::max(sqrty, _isqrt(pi_table.get_base()));
    uint64_t qMax = std::min(y, (x / (pMin * pMin * pMin)));
    pMin = std::max(sqrty + 1, pMin);
    uint32_t pMinIdx = lowerBound(pq.h_primes, pi_sqrty, pi_qrtx, pMin);
    uint32_t pCornerIdx = upperBound(pq.h_primes, pi_sqrty, pi_qrtx, pCorner);
    uint64_t qMaxIdx = upperBound(pq.h_primes, 0, pq.len, qMax);

    uint64_t blocks = (qMaxIdx - pi_qrtx) / threadsPerBlock + 1;

    // get this iteration's pi table and bounds
    uint64_t pi_max = pi_table.get_pi_base();
    uint32_t * d_piTable = pi_table.getNextDown();
    std::cout << "Base: " << pi_table.get_base() << "\t\tBlocks: " << blocks << std::endl;

    // launch kernel
    cudaDeviceSynchronize();
    cudaProfilerStart();
    Omega3_kernel<<<blocks, threadsPerBlock, 0, stream[0]>>>
      (x, pq.d_primes, d_piTable, (pi_table.get_pi_base() & (~1ull)), pi_table.get_base(),
      pMinIdx, pCornerIdx, pi_qrtx, d_sums, d_nextQ, d_lastQ, (uint64_t)pq.len);
    cudaProfilerStop();
    cudaDeviceSynchronize();
    // dispDeviceArray(d_lastQ + pMinIdx, pi_qrtx - pMinIdx);

  }

  sum = cuda128::reduce64to128(d_sums, maxblocks);
  // std::cout << "Omega 3:\t" << sum << std::endl;

  timer.stop();
  timer.displayTime();
  timer.~KernelTime();
  pi_table.~PiTable();

  cudaFree(d_sums);
  cudaFree(pq.d_primes);
  cudaFreeHost(pq.h_primes);

  cudaDeviceReset();

  return sum;
}

__global__
void Omega3_kernel( uint128_t x, uint64_t * pq, uint32_t * d_pitable, uint64_t pi_0,
                    uint64_t base, uint32_t pMinIdx, uint32_t pCornerIdx, uint32_t pMaxIdx,
                    uint64_t * sums, uint64_t * nextQ, uint64_t * lastQ, uint64_t maxQidx)
{
  uint64_t sum = 0;
  uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ uint64_t s_pi[numThreads];
  s_pi[threadIdx.x] = 0;
  __shared__ uint64_t s_lastQ[numThreads];

  for(uint64_t j = pMinIdx; j < pMaxIdx - 1; j += numThreads){
    s_lastQ[threadIdx.x] = (uint64_t)-1;
    __syncthreads();
    uint64_t i = j;
    for(; i < min((uint32_t)pCornerIdx, (uint32_t)j + numThreads); i++){

      uint64_t qidx = nextQ[i] + tidx;
      // if(tidx == 0) printf("%llu\t%llu\n", qidx, i);
      if(qidx >= maxQidx){
        atomicCAS((unsigned long long *)&s_lastQ[i - j], (unsigned long long)-1, (unsigned long long)maxQidx);
      } else {
        uint64_t q = pq[qidx];
        uint64_t p = pq[i];

        // calculate x/(p * q) and store value in q
        q = div128to64(x, (p * q));

        // check to make sure quotient is > pi_0, and coordinate this block's value
        // of lastQ if not
        q = checkRange(q, base, s_lastQ[i - j], qidx);

        // calculate pi(x/(p * q)) * chi(x/(p * q)) if q is in range
        if(q != 0){
          s_pi[threadIdx.x] += lookupPi(q, d_pitable, pi_0, base) + 2;
          s_pi[threadIdx.x] -= i;
        }
      }
    }

    for(; i < min((uint32_t)pMaxIdx, (uint32_t)j + numThreads); i++){

      uint64_t qidx = nextQ[i] + tidx;
      // if(tidx == 0) printf("%llu\t%llu\n", qidx, i);
      if(qidx >= maxQidx){
        atomicCAS((unsigned long long *)&s_lastQ[i - j], (unsigned long long)-1, (unsigned long long)maxQidx);
      } else {
        uint64_t q = pq[qidx];
        uint64_t p = pq[i];

        // calculate x/(p * q) and store value in q
        q = div128to64(x, (p * q));

        // check to make sure quotient is > pi_0, and coordinate this block's value
        // of lastQ if not
        q = checkRange(q, base, s_lastQ[i - j], qidx);

        // calculate pi(x/(p * q)) * chi(x/(p * q)) if q is in range
        if(q != 0){
          s_pi[threadIdx.x] += lookupPi(q, d_pitable, pi_0, base) + 2;
          s_pi[threadIdx.x] -= i;
        }else{
          // break;
          q = 1;
        }
      }
    } // repeat for all p values in range
    __syncthreads();

    // get global minimum value of lastQ
    minLastQ(j, s_lastQ, nextQ, lastQ);
  }
  __syncthreads();
  sum = thrust::reduce(thrust::device, s_pi, s_pi + numThreads);
  if(threadIdx.x == 0)
    sums[blockIdx.x] += sum;
}

__device__
inline uint64_t checkRange(uint64_t q, uint64_t base, uint64_t & s_lastQ, uint64_t qidx)
{
  if(q + 1 < base){
    atomicMin((unsigned long long *)&s_lastQ, (unsigned long long)qidx);
    q = 0;
  }
  return q;
}

__device__
inline uint64_t lookupPi(uint64_t q, uint32_t * d_pitable,
                                uint64_t pi_0, uint64_t base)
{
  uint64_t r;
  uint32_t *ptr = &d_pitable[(q + 1 - (base & ~1ull))/2];
  asm("ld.global.u32.ca   %0, [%1];\n\t"
       : "=l" (r)
       : "l" (ptr));
  r += pi_0;
  return r;
}

__device__
inline void minLastQ(uint32_t j, uint64_t * s_lastQ, uint64_t * nextQ, uint64_t * lastQ)
{
  uint32_t i = j + threadIdx.x;
  if(s_lastQ[threadIdx.x] != (uint64_t)-1){
    atomicCAS((unsigned long long *)&lastQ[i], (unsigned long long)nextQ[i], (unsigned long long)s_lastQ[threadIdx.x]);
    atomicMin((unsigned long long *)&lastQ[i], (unsigned long long)s_lastQ[threadIdx.x]);
  }
}

__global__
void Omega3_lower_bound(uint128_t x, uint64_t * nextQ, uint64_t * pq,
                        uint64_t pMinIdx, uint64_t pMaxIdx)
{
  uint64_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  uint64_t pidx = pMinIdx + tidx;
  uint64_t p = pq[pidx];

  if(pidx <= pMaxIdx)
    nextQ[pidx] = div128to64(x, p*p*p)+1; // this is going to overflow at ~x = 2^85
}
