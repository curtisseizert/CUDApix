#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "CUDASieve/cudasieve.hpp"
#include "CUDASieve/launch.cuh"
#include "sieve/phisieve_device.cuh"
#include "sieve/phisieve_host.cuh"

Phisieve::Phisieve(uint32_t maxPrime)
{
  this->maxPrime_ = maxPrime;

  init();
}

Phisieve::Phisieve(uint32_t maxPrime, uint32_t blockSize)
{
  this->maxPrime_ = maxPrime;
  this->blockSize_ = blockSize;

  init();
}

Phisieve::~Phisieve()
{
  cudaFree(d_primeList_);
  cudaFreeHost(h_primeList_);
  cudaFree(d_sieve_);
  cudaFree(d_count_);
}

inline void Phisieve::init()
{
  d_primeList_ = PrimeList::getSievingPrimes(maxPrime_, primeListLength_, 1);
  cudaMallocHost(&h_primeList_, primeListLength_ * sizeof(uint32_t));
  cudaMemcpy(h_primeList_, d_primeList_, primeListLength_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  cudaMalloc(&d_sieve_, (size_t)blockSize_/8);
  cudaMalloc(&d_count_, blockSize_ * sizeof(uint32_t));

  threads_ = 256;
  blocks_ = blockSize_/(32 * threads_);
}

void Phisieve::firstSieve(uint16_t c)
{
  a_current_ = c;
  phiglobal::sieveCountInit<<<blocks_, threads_>>>(d_sieve_, d_count_, bstart_, c);
  cudaDeviceSynchronize();

  thrust::exclusive_scan(thrust::device, d_count_, d_count_ + blockSize_, d_count_);
  cudaDeviceSynchronize();
}

void Phisieve::markNext()
{
  if(a_current_ < 20){
    phiglobal::markSmallPrimes<<<blocks_, threads_>>>(d_sieve_, bstart_, a_current_);
  }else{
    uint32_t p = h_primeList_[a_current_-12];
    uint32_t sieveBits = 1u << 14;
    uint32_t blocks = blockSize_ / (sieveBits * threads_);
    phiglobal::markMedPrimes<<<blocks, threads_>>>(d_sieve_, p, bstart_, sieveBits);
    std::cout << "\t" << p << std::endl;
  }
  cudaDeviceSynchronize();
  a_current_++;
}

void Phisieve::updateCount()
{
  phiglobal::updateCount<<<blocks_, threads_>>>(d_sieve_, d_count_);
  cudaDeviceSynchronize();

  thrust::exclusive_scan(thrust::device, d_count_, d_count_ + blockSize_, d_count_);
  cudaDeviceSynchronize();
}

uint32_t * Phisieve::getCountHost()
{
  uint32_t * h_count = (uint32_t *)malloc(blockSize_ * sizeof(uint32_t));

  cudaMemcpy(h_count, d_count_, blockSize_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  return h_count;
}
