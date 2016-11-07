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
  PrimeArray pq(qrtx, sqrt(x / qrtx));

  uint64_t num_p = pi_cbrtx - pi_qrtx;

  pq.d_primes = CudaSieve::getDevicePrimes(pq.bottom, pq.top, pq.len, 0);

  uint64_t maxblocks = 1 + (pq.len/QperBlock);

  uint64_t * d_sums;
  cudaMalloc(&d_sums, maxblocks * sizeof(uint64_t));
  global::zero<<<maxblocks, threadsPerBlock>>>(d_sums, maxblocks);
  cudaDeviceSynchronize();
  KernelTime timer;
  timer.start();

  // uint32_t * d_pitable = get_d_piTable(sqrtx);

  PiTable * pi_table = new PiTable(sqrtx);
  uint32_t * d_pitable = pi_table->getCurrent();

  while(num_p > pSoFar){
    uint16_t pPerBlock = min(num_p, (uint64_t)4096);
    cudaMemcpyToSymbol(p, pq.d_primes +pSoFar, pPerBlock * sizeof(uint64_t), 0, cudaMemcpyDefault);

    blocks = 1 + ((pq.len - pSoFar)/QperBlock);
    A_lo_p_reg<<<blocks, threadsPerBlock>>>(x, pq.d_primes + pSoFar, d_sums, y, pPerBlock, pq.len, d_pitable);
    cudaDeviceSynchronize();
    std::cout << blocks << " " << maxblocks << std::endl;
    pSoFar += pPerBlock;
  }

  sum = thrust::reduce(thrust::device, d_sums, d_sums + maxblocks);

  timer.stop();
  timer.displayTime();
  timer.~KernelTime();

  cudaFree(d_sums);
  cudaFree(pq.d_primes);

  return sum;
}

uint64_t GourdonVariant64::A_large()
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

  uint64_t maxblocks = 1 + (pq.len/QperBlock);

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
  global::setXPlusB<<<num_p/threadsPerBlock + 1, threadsPerBlock, 0, stream[1]>>>(d_lastQ, num_p, (uint32_t)1);
  global::setXPlusB<<<num_p/threadsPerBlock + 1, threadsPerBlock, 0, stream[1]>>>(d_nextQ, num_p, (uint32_t)1);
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
  uint64_t qMax = (sqrtx * qrtx) / pi_table.getNextBaseDown();
  uint32_t qMaxIdx = (uint32_t)upperBound(pq.h_primes, 0, pq.len, qMax);
  uint64_t qMinIdx = 0;

  std::cout << "Sparse Leaves:\n" << std::endl;
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
    uint32_t blocks = (qMaxIdx - qMinIdx)/threadsPerBlock + 2;
    std::cout << pi_table.get_base() << " " << pi_table.get_bottom() << " " << blocks << " " << qMaxIdx << " " << qMinIdx << std::endl;

    // launch kernel
    cudaDeviceSynchronize();
    A_large_loPQ<<<blocks, threadsPerBlock>>>
      (x, y, pq.d_primes, d_piTable, pi_table.get_pi_base(), pi_table.get_base(),
      pMaxIdx, d_sums, d_nextQ, d_lastQ, (uint32_t)pq.len);

    // calculate the minimum and maximum p values and indices for next iteration
    qMinIdx = qMaxIdx;
    pMax = sqrt(x/pi_table.getNextBaseDown());
    pMaxIdx = upperBound(pq.h_primes, 0, num_p, pMax);
    qMax = (sqrtx * qrtx) / pi_table.getNextBaseDown();
    qMaxIdx = upperBound(pq.h_primes, 0, pq.len, qMax);
    cudaDeviceSynchronize();

    // ** see if passing p array by parameter is faster/slower/same
    // ** see if registers can be saved by copying x and y to symbol
    // ** next we do an iteration along the p axis keeping track of the minimum
    //    q value for each p that did not fall in the range of the pi table and
    //    using that as the first q for that p in the next pi table-wise
    //    iteration
  }

  // the maximum number of q values does not fall on p = x^(1/4) in the clustered
  // easy leaves but rather where the base of the pi_table meets the curve defined
  // by (x/p)^(1/2).  To find the maximum number of q values, we thus need to find the
  // maximum value of p_a such that b = [base of the pi table] > (x / p_a)^(1/2).
  // Thus, p < (x/b^2)^(1/2) <= p_(a+1).  To be safe, we will use the interval
  // b_0 < q <= b_1 for this rather than b_0 < q <= (x/p)^(1/2) since the former
  // upper bound will be marginally larger.
  pi_table.set_bottom(pq.bottom);
  uint64_t pMinNext = x / (pi_table.getNextBaseDown() * pi_table.getNextBaseDown());
  uint64_t pMinNextIdx = upperBound(pq.h_primes, (uint64_t)0, num_p, pMax);
  qMax = sqrt(x/pq.h_primes[pMinNextIdx]);
  qMaxIdx = upperBound(pq.h_primes, (uint64_t)0, pq.len, qMax);
  uint64_t qMin = max(x / (pq.h_primes[pMinNextIdx]*pi_table.get_base()), pq.h_primes[pMinNextIdx]);
  qMinIdx = upperBound(pq.h_primes, (uint64_t)0, pq.len, qMin);
  uint64_t pMinIdx = 0;

  std::cout << "Clustered Leaves:\n" << std::endl;

  // find all (p,q) pairs such that x^(1/3) <= x/(p*q) < x^(3/8).  The maximum
  // q for each p at this range is (x/p)^(1/2).  In this loop, we need to keep
  // track of the minimum value of p necessary to find the remaining p,q pairs
  // as well as the minimum value of q for each p within range.
  while(pi_table.get_base() > pq.bottom){
    cudaMemcpyAsync(d_nextQ, d_lastQ, num_p * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream[1]);

    // get this iteration's pi table and bounds
    uint64_t pi_max = pi_table.get_pi_base();
    uint32_t * d_piTable = pi_table.getNextDown();

    // calculate number of blocks to span maximum range of q values (defined at
    // p = x^(1/4))
    uint32_t blocks = (qMaxIdx - qMinIdx)/threadsPerBlock + 2;
    std::cout << pi_table.get_base() << " " << pi_table.get_bottom() << " " << blocks << " " << qMaxIdx << " " << qMinIdx << std::endl;

    // launch kernel
    cudaDeviceSynchronize();
    A_large_hiPQ<<<blocks, threadsPerBlock>>>
      (x, y, pq.d_primes, d_piTable, pi_table.get_pi_base(), pi_table.get_base(),
       pMinIdx, pMaxIdx, d_sums, d_nextQ, d_lastQ, (uint32_t)pq.len);

    // calculate the minimum and maximum p values and indices for next iteration
    pMinIdx = pMinNextIdx;
    pMinNext = x / (pi_table.getNextBaseDown() * pi_table.getNextBaseDown());
    pMinNextIdx = upperBound(pq.h_primes, (uint64_t)0, num_p, pMax);
    qMax = sqrt(x/pq.h_primes[pMinNextIdx]);
    qMaxIdx = upperBound(pq.h_primes, (uint64_t)0, pq.len, qMax);
    qMin = max(x / (pq.h_primes[pMinNextIdx]*pi_table.get_base()), pq.h_primes[pMinNextIdx]);
    qMinIdx = upperBound(pq.h_primes, (uint64_t)0, pq.len, qMin);
    cudaDeviceSynchronize();
  }


  // while(pi_table.get_base() > cbrtx){
  //
  // }

  sum = thrust::reduce(thrust::device, d_sums, d_sums + maxblocks);

  timer.stop();
  timer.displayTime();
  timer.~KernelTime();
  pi_table.~PiTable();

  cudaFree(d_sums);
  cudaFree(pq.d_primes);
  cudaFreeHost(pq.h_primes);

  return sum;
}

__global__
void A_large_loPQ(uint64_t x, uint64_t y, uint64_t * pq, uint32_t * d_pitable,
                  uint64_t pi_0, uint64_t base, uint32_t pMaxIdx, uint64_t * sums,
                  uint32_t * nextQ, uint32_t * lastQ, uint32_t maxQidx)
{
  uint64_t sum = 0;
  __shared__ uint64_t s_pi_chi[numThreads];
  s_pi_chi[threadIdx.x] = 0;

  __shared__ uint32_t s_lastQ[1];


  for(uint32_t i = 0; i <= pMaxIdx; i++){
    if(threadIdx.x == 0)s_lastQ[0] = (uint32_t)-1;
    uint32_t qidx = nextQ[i] + threadIdx.x + blockDim.x * blockIdx.x;
    if(qidx >= maxQidx) break;
    uint64_t q = pq[qidx];
    uint64_t p = pq[i];

    // calculate x/(p * q) and store value in q
    q = x / (p * q);

    // check to make sure quotient is > pi_0, and coordinate this block's value
    // of lastQ if not
    checkRange(q, base, s_lastQ, qidx);

    // calculate pi(x/(p * q)) * chi(x/(p * q)) if q is in range
    if(q != 0)
      s_pi_chi[threadIdx.x] += calculatePiChi(q, y, d_pitable, pi_0, base);
    __syncthreads();

    // get global minimum value of lastQ
    minLastQ(i, s_lastQ, nextQ, lastQ);
  // if(threadIdx.x == 0 && blockIdx.x == 0) printf("%u\t%u\t%u\n", nextQ[i], lastQ[i], i);
  } // repeat for all p values in range

  __syncthreads();
  sum = thrust::reduce(thrust::device, s_pi_chi, s_pi_chi + numThreads);
  if(threadIdx.x == 0)
    sums[blockIdx.x] += sum;
}

__global__
void A_large_hiPQ(uint64_t x, uint64_t y, uint64_t * pq, uint32_t * d_pitable,
                  uint64_t pi_0, uint64_t base, uint32_t pMinIdx, uint32_t pMaxIdx,
                  uint64_t * sums, uint32_t * nextQ, uint32_t * lastQ, uint32_t maxQidx)
{
  uint64_t sum = 0;
  __shared__ uint64_t s_pi_chi[numThreads];
  s_pi_chi[threadIdx.x] = 0;

  __shared__ uint32_t s_lastQ[1];


  for(uint32_t i = 0; i <= pMaxIdx; i++){
    if(threadIdx.x == 0)s_lastQ[0] = (uint32_t)-1;
    uint32_t qidx = nextQ[i] + threadIdx.x + blockDim.x * blockIdx.x;
    if(qidx >= maxQidx) break;
    uint64_t q = pq[qidx];
    uint64_t p = pq[i];
    // calculate x/(p * q) and store value in q
    q = x / (p * q);

    // check to make sure quotient is > pi_0, and coordinate this block's value
    // of lastQ if not
    q = checkRangeHi(q, base, s_lastQ, qidx, p, x);

    // calculate pi(x/(p * q)) * chi(x/(p * q)) if q is in range
    if(q != 0)
      s_pi_chi[threadIdx.x] += calculatePiChi(q, y, d_pitable, pi_0, base);
    __syncthreads();

    // get global minimum value of lastQ
    minLastQ(i, s_lastQ, nextQ, lastQ);
  // if(threadIdx.x == 0 && blockIdx.x == 0) printf("%u\t%u\t%u\n", nextQ[i], lastQ[i], i);
  } // repeat for all p values in range

  __syncthreads();
  sum = thrust::reduce(thrust::device, s_pi_chi, s_pi_chi + numThreads);
  if(threadIdx.x == 0)
    sums[blockIdx.x] += sum;
}

__device__
void checkRange(uint64_t & q, uint64_t base, uint32_t * s_lastQ, uint32_t qidx)
{
  if(q < base){
    atomicMin(&s_lastQ[0], qidx);
    q = 0;
  }
}

__device__
uint64_t checkRangeHi(uint64_t q, uint64_t base, uint32_t * s_lastQ, uint32_t qidx, uint64_t p, uint64_t x)
{
  if(q < sqrtf(x/p))
    return 0;
  if(q < base){
    atomicMin(&s_lastQ[0], qidx);
    return 0;
  }
  else return q;
}

__device__
uint64_t calculatePiChi(uint64_t q, uint64_t y, uint32_t * d_pitable,
                        uint64_t pi_0, uint64_t base)
{
  if(q < base) printf("\t\t\t\t%llu  %llu\n", q, base);
  uint64_t r = d_pitable[(q + 1 - base)/2] + pi_0;
  if(q < y)
    r <<= 1;
  return r;
}

__device__
void minLastQ(uint32_t i, uint32_t * s_lastQ, uint32_t * nextQ, uint32_t * lastQ)
{
  if(threadIdx.x == 0 && s_lastQ[0] != ~0){
    atomicCAS(&lastQ[i], nextQ[i], s_lastQ[0]);
    atomicMin(&lastQ[i], s_lastQ[0]);
  }
  __syncthreads();
}

uint64_t GourdonVariant64::checkA()
{
  PrimeArray pi(0, sqrtx);
  PrimeArray pq(qrtx, sqrt(x / qrtx));

  uint64_t sum = 0, num_p = pi_cbrtx - pi_qrtx;

  pi.h_primes = CudaSieve::getHostPrimes(pi.bottom, pi.top, pi.len, 0);
  pq.h_primes = CudaSieve::getHostPrimes(pq.bottom, pq.top, pq.len, 0);
  CudaSieve * sieve = new CudaSieve((uint64_t)0, sqrtx, 0);

  for(uint32_t i = 0; i < num_p; i++){
    uint32_t p = pq.h_primes[i];
    std::cout << "\t" << p << "\t(" << i << "/" << num_p << ")\r";
    std::cout << std::flush;
    for(uint32_t j = i + 1; j <= pq.len; j++){
      uint32_t q = pq.h_primes[j];
      if(q > sqrt(x / p)) break;
      uint64_t n = x / (p * q);
      uint64_t pi_n = sieve->countPrimesSegment(0, n);
      if(n < y) pi_n *= 2;
      // std::cout << pi_n << std::endl;
      sum += pi_n;
    }
  }
  std::cout << std::endl;
  return sum;
}

__global__ void A_lo_p_reg(uint64_t x, uint64_t * q, uint64_t * sums,
                           uint64_t y, uint32_t num_p, uint64_t max_num_q,
                           uint32_t * d_pitable)
{
  uint64_t qt = 0, quot = 0, sum = 0;
  __shared__ uint64_t s_pi_quot[numThreads];
  s_pi_quot[threadIdx.x] = 0;

  uint32_t qidx = numQ / numThreads * (threadIdx.x + blockIdx.x * blockDim.x);
  if(qidx < max_num_q)
    qt = q[qidx];

  for(uint32_t iter = 0; iter < num_p; iter++){
    calculateQuotient(x, qt, quot, iter);
    if(qt == 0) break;
    __syncthreads();

    getPiOfQuot(quot, y, d_pitable, s_pi_quot);

    __syncthreads();
  }

  __syncthreads();
  sum = thrust::reduce(thrust::device, s_pi_quot, s_pi_quot + numThreads);
  if(threadIdx.x == 0)
    sums[blockIdx.x] += sum;
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

__device__ void getPiOfQuot(uint64_t quot, uint64_t y, uint32_t * d_pitable,
                            uint64_t * s_pi_quot)
{
  uint64_t r = d_pitable[(quot + 1)/2];
  if(quot < y)
    r <<= 1;
  s_pi_quot[threadIdx.x] += r;

  // if(quot != 0) printf("%llu\t\t%llu\n", quot, r);
}
