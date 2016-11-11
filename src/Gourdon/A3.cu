// A2.cu
//
// A segmented implemenation of the sum "A" in Xavier Gourdon's variant of the
// Deleglise-Rivat prime counting algorithm with an upper bound less constrained
// by the memory requirements of holding pi(x) values from 0 to sqrt(x) as in
// A1.cu
//
// Copywrite (c) 2016 Curtis Seizert <cseizert@gmail.com>

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
#include <cuda_profiler_api.h>

#include "general/tools.hpp"
#include "general/device_functions.cuh"
#include "Gourdon/A.cuh"
#include "Gourdon/gourdonvariant.hpp"
#include "cudapix.hpp"
#include "pitable.cuh"


uint64_t GourdonVariant64::A_vert()
{
// this is a segmented variant of the above, where each iteration evaluates
// all of the pi(x/(p * q)) values that fall within the range of the pi table.
// The basic idea is that each time the maximum value of q is exceeded (as defined
// by the minimum of x/(p*q) imposed by the minimum value of our pi table), the
// id of this element of the array will be saved in a different array, which will
// serve as the first q used for each p in the next iteration (with the next smaller
// pi table)
  cudaStream_t stream[3];
  uint64_t sum = 0;
  PrimeArray pq(qrtx, sqrt(x / qrtx));

  uint64_t num_p = pi_cbrtx - pi_qrtx;

  pq.d_primes = CudaSieve::getDevicePrimes(pq.bottom, pq.top, pq.len, 0);

  uint32_t maxblocks = 2 + (num_p/threadsPerBlock);

  uint64_t * d_sums;
  uint32_t * d_lastQ, * d_nextQ;
  cudaMalloc(&d_sums, maxblocks * sizeof(uint64_t));
  cudaMalloc(&d_lastQ, num_p * sizeof(uint64_t));
  cudaMalloc(&d_nextQ, num_p * sizeof(uint64_t));
  cudaMallocHost(&pq.h_primes, pq.len * sizeof(uint64_t));
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);
  cudaStreamCreate(&stream[2]);

  cudaMemcpy(pq.h_primes, pq.d_primes, pq.len*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  global::setXPlusB<<<num_p/threadsPerBlock + 1, threadsPerBlock, 0, stream[1]>>>(d_lastQ, num_p+1, (uint32_t)1);
  global::zero<<<maxblocks, threadsPerBlock, 0, stream[2]>>>(d_sums, maxblocks);

  cudaDeviceSynchronize();
  KernelTime timer;
  timer.start();

  PiTable pi_table(sqrtx, pq.top);
  pi_table.set_pi_base(pi_sqrtx);

  // we need to get the maximum number of q values in a given iteration in order
  // to calculate the number of blocks needed.  abs((n/p)-(m/p)) is greatest at lowest
  // values of p, which obviously occur at the defined lower bound of p (p > x^(1/4)).
  // This is the value of p where we will define delta q so we don't miss any q's
  // by underestimating the number of blocks we need
  uint64_t pMax = sqrt(x/pi_table.getNextBaseDown());
  uint32_t pMaxIdx = (uint32_t)upperBound(pq.h_primes, 0, num_p, pMax);
  uint64_t pMin = 0;
  uint32_t pMinIdx = 0;
  uint64_t qMax = (sqrtx * qrtx) / pi_table.getNextBaseDown();
  uint32_t qMaxIdx = (uint32_t)upperBound(pq.h_primes, 0, pq.len, qMax);
  uint64_t qMinIdx = 0;

  // find all (p,q) pairs such that x/(p * q) >= x^(3/8)
  while(pi_table.get_base() > pq.top){
    // nextQ is copied from lastQ each iteration rather than switching pointers
    // to form the basis of a compare and swap operation in the kernel that evaluates
    // whether a given value has changed
    cudaMemcpyAsync(d_nextQ, d_lastQ, num_p * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream[1]);

    // get this iteration's pi table and bounds
    uint64_t pi_max = pi_table.get_pi_base();
    uint32_t * d_piTable = pi_table.getNextDown();

    // calculate number of blocks to span maximum range of q values (defined at
    // p = x^(1/4))
    // uint32_t blocks_a = (pMaxIdx)/threadsPerBlock + 1 - blockCutoff_;
    uint32_t blocks_b = blockCutoff_ * (1 + (qMaxIdx - qMinIdx) / threadsPerBlock);
    std::cout << "Blocks A : " << blocks_a << "\t Blocks B : " << blocks_b << std::endl;

    // launch kernel
    cudaDeviceSynchronize();
    cudaProfilerStart();
    A_large_loPQ_vert_a<<<blocks_a, threadsPerBlock, 0, stream[0]>>>
      (x, y, pq.d_primes, d_piTable, pi_table.get_pi_base(), pi_table.get_base(),
      pMaxIdx, d_sums, d_nextQ, d_lastQ, (uint32_t)pq.len);
    // A_large_loPQ_vert_b<<<blocks_b, threadsPerBlock, 0, stream[1]>>>
    //   (x, y, pq.d_primes, d_piTable, pi_table.get_pi_base(), pi_table.get_base(),
    //   pMaxIdx, d_sums, d_nextQ, d_lastQ, (uint32_t)pq.len);
    cudaProfilerStop();
    // calculate the minimum and maximum p values and indices for next iteration
    pMax = sqrt(x/pi_table.getNextBaseDown());
    pMaxIdx = upperBound(pq.h_primes, 0, num_p, pMax);
    cudaDeviceSynchronize();

  }

  // the maximum number of q values does not fall on p = x^(1/4) in the clustered
  // easy leaves but rather where the base of the pi_table meets the curve defined
  // by (x/p)^(1/2).  To find the maximum number of q values, we thus need to find the
  // maximum value of p_a such that b = [base of the pi table] > (x / p_a)^(1/2).
  // Thus, p < (x/b^2)^(1/2) <= p_(a+1).  To be safe, we will use the interval
  // b_0 < q <= b_1 for this rather than b_0 < q <= (x/p)^(1/2) since the former
  // upper bound will be marginally larger.
  pi_table.set_bottom(cbrtx);

  pMax = sqrt(x/pi_table.getNextBaseDown());
  pMaxIdx = (uint32_t)upperBound(pq.h_primes, 0, num_p, pMax);
  pMin = x / (pi_table.get_base() * pi_table.get_base());
  pMinIdx = upperBound(pq.h_primes, 0, num_p, pMin);


  // sum = thrust::reduce(thrust::device, d_sums, d_sums + maxblocks);
  // std::cout << "Sparse Leaves:\t\t" << sum << std::endl;


  // find all (p,q) pairs such that x^(1/3) <= x/(p*q) < x^(3/8).  The maximum
  // q for each p at this range is (x/p)^(1/2).  In this loop, we need to keep
  // track of the minimum value of p necessary to find the remaining p,q pairs
  // as well as the minimum value of q for each p within range.

  while(pi_table.getNextBaseDown() > cbrtx){
    cudaMemcpyAsync(d_nextQ, d_lastQ, num_p * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream[1]);

    // get this iteration's pi table and bounds
    uint64_t pi_max = pi_table.get_pi_base();
    uint32_t * d_piTable = pi_table.getNextDown();

    // calculate number of blocks to span maximum range of q values (defined at
    // p = x^(1/4))
    uint32_t blocks = (pMaxIdx - pMinIdx) / threadsPerBlock + 1;
    std::cout << "Blocks : " << blocks << std::endl;

    // launch kernel
    cudaDeviceSynchronize();
    cudaProfilerStart();
    A_large_hiPQ_vert<<<blocks, threadsPerBlock>>>
      (x, y, pq.d_primes, d_piTable, pi_table.get_pi_base(), pi_table.get_base(),
      pMinIdx, pMaxIdx, d_sums, d_nextQ, d_lastQ, (uint32_t)pq.len);
    cudaProfilerStop();
    // calculate the minimum and maximum p values and indices for next iteration

    pMax = sqrt(x/pi_table.getNextBaseDown());
    pMaxIdx = (uint32_t)upperBound(pq.h_primes, 0, num_p, pMax);
    pMin = x / (pi_table.getNextBaseDown() * pi_table.getNextBaseDown());
    pMinIdx = upperBound(pq.h_primes, 0, num_p, pMin);
    cudaDeviceSynchronize();
  }

  while(pi_table.get_base() > cbrtx){
    cudaMemcpyAsync(d_nextQ, d_lastQ, num_p * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream[1]);

    // get this iteration's pi table and bounds
    uint64_t pi_max = pi_table.get_pi_base();
    uint32_t * d_piTable = pi_table.getNextDown();

    // calculate number of blocks to span maximum range of q values (defined at
    // p = x^(1/4))
    uint32_t blocks = (pMaxIdx - pMinIdx )/ threadsPerBlock + 1;
    std::cout << "Blocks : " << blocks << std::endl;

    // launch kernel
    cudaDeviceSynchronize();
    cudaProfilerStart();
    A_large_lastPQ_vert<<<blocks, threadsPerBlock>>>
      (x, y, pq.d_primes, d_piTable, pi_table.get_pi_base(), pi_table.get_base(),
       pMinIdx, pMaxIdx, d_sums, d_nextQ, (uint32_t)pq.len);
    cudaProfilerStop();

    cudaDeviceSynchronize();
  }


  sum = thrust::reduce(thrust::device, d_sums, d_sums + maxblocks);
  std::cout << "Clustered Leaves:\t" << sum << std::endl;

  timer.stop();
  timer.displayTime();
  timer.~KernelTime();
  pi_table.~PiTable();

  cudaFree(d_sums);
  cudaFree(pq.d_primes);
  cudaFreeHost(pq.h_primes);

  cudaDeviceReset();

  return 0;//sum + sum2;
}

__global__
void A_large_loPQ_vert_a(uint64_t x, uint64_t y, uint64_t * pq, uint32_t * d_pitable,
                        uint64_t pi_0, uint64_t base, uint32_t pMaxIdx, uint64_t * sums,
                        uint32_t * nextQ, uint32_t * lastQ, uint32_t maxQidx)
{
  uint32_t tidx = threadIdx.x + blockDim.x * (blockIdx.x + blockCutoff);
  __shared__ uint64_t s_pi_chi[numThreads];
  s_pi_chi[threadIdx.x] = 0;

  uint64_t p = pq[tidx];
  uint32_t qidx = nextQ[tidx];
  uint64_t q = pq[qidx];
  uint64_t maxQ = x / (p * base);

  while(q <= maxQ && qidx < maxQidx){
    // calculate x/(p * q) and store value in q
    q = x / (p * q);

    // calculate pi(x/(p * q)) * chi(x/(p * q)) if q is in range
    s_pi_chi[threadIdx.x] += calculatePiChi(q, y, d_pitable, pi_0, base);

    qidx++;
    q = pq[qidx];
  } // repeat for all q values in range

  lastQ[tidx] = qidx;
  __syncthreads();

  // repurpose p as the sum for this block
  p = thrust::reduce(thrust::device, s_pi_chi, s_pi_chi + numThreads);
  if(threadIdx.x == 0)
    sums[blockIdx.x] += p;
}

__global__
void A_large_loPQ_vert_b(uint64_t x, uint64_t y, uint64_t * pq, uint32_t * d_pitable,
                        uint64_t pi_0, uint64_t base, uint32_t pMaxIdx, uint64_t * sums,
                        uint32_t * nextQ, uint32_t * lastQ, uint32_t maxQidx)
{
  uint32_t htidx = threadIdx.x + blockDim.x * (blockIdx.x % blockCutoff);
  uint32_t vtidx = threadIdx.x + blockDim.x * (blockIdx.x / blockCutoff);
  __shared__ uint64_t s_pi_chi[numThreads];
  s_pi_chi[threadIdx.x] = 0;

  uint64_t p = pq[htidx];
  uint32_t qidx = nextQ[htidx] + vtidx;
  uint64_t q, maxQ;
  if(qidx < maxQidx)
    q = pq[qidx];
  else
    goto EndVB;
  maxQ = x / (p * base);

  for(uint16_t count = 0; count < threadsPerBlock; count ++){
    if(q > maxQ || qidx >= maxQidx){
      lastQ[htidx] = qidx;
    }
    // calculate x/(p * q) and store value in q
    q = x / (p * q);

    // calculate pi(x/(p * q)) * chi(x/(p * q)) if q is in range
    s_pi_chi[threadIdx.x] += calculatePiChi(q, y, d_pitable, pi_0, base);

    qidx++;
    q = pq[qidx];
  } // repeat for all q values in range

EndVB:
  __syncthreads();

  // repurpose p as the sum for this block
  p = thrust::reduce(thrust::device, s_pi_chi, s_pi_chi + numThreads);
  if(threadIdx.x == 0)
    sums[blockIdx.x] += p;
}

__global__
void A_large_hiPQ_vert( uint64_t x, uint64_t y, uint64_t * pq, uint32_t * d_pitable,
                        uint64_t pi_0, uint64_t base, uint32_t pMinIdx, uint32_t pMaxIdx,
                        uint64_t * sums, uint32_t * nextQ, uint32_t * lastQ, uint32_t maxQidx)
{
  uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ uint64_t s_pi_chi[numThreads];
  s_pi_chi[threadIdx.x] = 0;

  uint64_t p = pq[tidx + pMinIdx];
  uint32_t qidx = nextQ[tidx + pMinIdx];
  uint64_t q = pq[qidx];
  uint64_t maxQ = min(x / (p * base), (uint64_t)__dsqrt_rd(x/p));

  while(q <= maxQ && qidx < maxQidx){
    // calculate x/(p * q) and store value in q
    q = x / (p * q);

    // calculate pi(x/(p * q)) * chi(x/(p * q)) if q is in range
    s_pi_chi[threadIdx.x] += calculatePiChi(q, y, d_pitable, pi_0, base);

    qidx++;
    q = pq[qidx];
  } // repeat for all q values in range

  lastQ[tidx] = qidx;
  __syncthreads();

  // repurpose p as the sum for this block
  p = thrust::reduce(thrust::device, s_pi_chi, s_pi_chi + numThreads);
  if(threadIdx.x == 0)
    sums[blockIdx.x] += p;
}


__global__
void A_large_lastPQ_vert(uint64_t x, uint64_t y, uint64_t * pq, uint32_t * d_pitable,
                    uint64_t pi_0, uint64_t base, uint32_t pMinIdx, uint32_t pMaxIdx,
                    uint64_t * sums, uint32_t * nextQ, uint32_t maxQidx)
{
  uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ uint64_t s_pi_chi[numThreads];
  s_pi_chi[threadIdx.x] = 0;
  uint64_t p, qidx, q, maxQ;

  if(tidx + pMinIdx > pMaxIdx) goto End;
  p = pq[tidx + pMinIdx];
  qidx = nextQ[tidx + pMinIdx];
  if(qidx < maxQidx)
    q = pq[qidx];
  maxQ = __dsqrt_rd(x/p);

  while(q <= maxQ && qidx < maxQidx){
    // calculate x/(p * q) and store value in q
    q = x / (p * q);

    // calculate pi(x/(p * q)) * chi(x/(p * q)) if q is in range
    s_pi_chi[threadIdx.x] += calculatePiChi(q, y, d_pitable, pi_0, base);

    qidx++;
    q = pq[qidx];
  } // repeat for all q values in range
End:
  __syncthreads();

  // repurpose p as the sum for this block
  p = thrust::reduce(thrust::device, s_pi_chi, s_pi_chi + numThreads);
  if(threadIdx.x == 0)
    sums[blockIdx.x] += p;
}
/*
__device__
uint64_t checkRange(uint64_t  q, uint64_t base, uint32_t & s_lastQ, uint32_t qidx)
{
  if(q < base){
    atomicMin(&s_lastQ, qidx);
    q = 0;
  }
  return q;
}

__device__
uint64_t checkRangeHi(uint64_t q, uint64_t base, uint32_t & s_lastQ, uint32_t qidx)
{
  if(q < base && q != 0){
    atomicMin(&s_lastQ, qidx);
    return 0;
  }
  else return q;
}
*/
__device__
inline uint64_t calculatePiChi(uint64_t q, uint64_t y, uint32_t * d_pitable,
                        uint64_t pi_0, uint64_t base)
{
  uint64_t r = d_pitable[(q + 1 - base)/2] + pi_0;
  if(q < y)
    r <<= 1;
  return r;
}
/*
__device__
void minLastQ(uint32_t j, uint32_t * s_lastQ, uint32_t * nextQ, uint32_t * lastQ)
{
  uint32_t i = j + threadIdx.x;
  if(s_lastQ[threadIdx.x] != ~0){
    atomicCAS(&lastQ[i], nextQ[i], s_lastQ[threadIdx.x]);
    atomicMin(&lastQ[i], s_lastQ[threadIdx.x]);
  }
}

__device__
inline uint64_t checkAndDiv(uint64_t q, uint64_t p, uint64_t x)
{
  uint64_t quot = (q <= __dsqrt_rd(x/p) && q != 0) ? x/(p * q) : 0;
  quot = (p < q ? quot : 0);

  return quot;
}
*/
