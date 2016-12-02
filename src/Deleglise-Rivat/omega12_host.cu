#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <CUDASieve/cudasieve.hpp>
#include <CUDASieve/host.hpp>
#include <CUDASieve/primelist.cuh>

#include "general/tools.hpp"
#include "general/device_functions.cuh"
#include "Deleglise-Rivat/omega12.cuh"
#include "sieve/lpf_mu.cuh"
#include "S3.cuh"

#include <cuda_profiler_api.h>

const uint16_t h_threadsPerBlock = 256;

Omega12Host::Omega12Host(uint128_t x, uint64_t y, uint16_t c)
{
  maxPrime_ = _isqrt(div128to128(x,y));
  makeData(x, y, c);
  allocate();

  global::zero<<<1 + h_cdata.elPerBlock/512, 512, 0, stream[0]>>>(data->d_totals, h_cdata.elPerBlock);

  zero();
}

Omega12Host::Omega12Host(uint128_t x, uint64_t y, uint16_t c, uint64_t maxPrime)
{
  maxPrime_ = maxPrime;
  makeData(x, y, c);
  allocate();

  global::zero<<<1 + h_cdata.elPerBlock/512, 512, 0, stream[0]>>>(data->d_totals, h_cdata.elPerBlock);

  zero();
}

Omega12Host::~Omega12Host()
{
  deallocate();
}

void Omega12Host::makeData(uint128_t x, uint64_t y, uint16_t c)
{
  cudaMallocManaged((void **)&data, sizeof(omega12data_128));
  cudaDeviceSynchronize();
  h_cdata.x = x;
  h_cdata.y = y;
  h_cdata.sqrty = _isqrt(y);
  h_cdata.z = div128to64(x,y);
  h_cdata.c = c;
  h_cdata.sieveWords = sieveWords_;
  h_cdata.bstart = 0;
  h_cdata.mstart = 1;
  h_cdata.blocks = std::min(1 + (uint64_t)(h_cdata.z/(64 * h_cdata.sieveWords)), 512ul); // must be <= 1024

  h_cdata.maxPrime = maxPrime_;
  data->d_primeList = CudaSieve::getDevicePrimes32(0, h_cdata.maxPrime, h_cdata.primeListLength, 0);
  data->d_bitsieve = CudaSieve::genDeviceBitSieve(0, y, 0);
  h_cdata.elPerBlock = h_cdata.primeListLength - 2;

  transferConstants();

  data->d_mu = gen_d_mu((uint64_t)0, (uint64_t)y);
  data->d_lpf = gen_d_lpf((uint64_t)0, (uint64_t)y);
}

void Omega12Host::setupNextIter()
{
  // if(h_cdata.sieveWords < 4096) h_cdata.sieveWords += 1024;
  h_cdata.bstart += h_cdata.blocks * (64 * h_cdata.sieveWords);
  h_cdata.blocks = std::min(1 + (uint64_t)(h_cdata.z - h_cdata.bstart)/(64 * h_cdata.sieveWords), 512ul);

  transferConstants();
  zero();
}

void Omega12Host::allocate()
{
  for(uint16_t i = 0; i < numStreams_; i++)
    cudaStreamCreate(&stream[i]);
  cudaMalloc(&data->d_sums, h_cdata.elPerBlock * h_cdata.blocks * sizeof(uint64_t));
  cudaMalloc(&data->d_partialsums, h_cdata.blocks * sizeof(int64_t));
  cudaMalloc(&data->d_num, h_cdata.elPerBlock * h_cdata.blocks * sizeof(int16_t));
  cudaMalloc(&data->d_totals, h_cdata.elPerBlock * sizeof(uint64_t));
  cudaMalloc(&data->d_totalsNext, h_cdata.elPerBlock * sizeof(uint64_t));
}

void Omega12Host::zero()
{
  global::zero<<<1 + h_cdata.elPerBlock/512, 512, 0, stream[1]>>>(data->d_totalsNext, h_cdata.elPerBlock);
  global::zero<<<h_cdata.elPerBlock, h_cdata.blocks, 0, stream[2]>>>(data->d_sums, h_cdata.elPerBlock * h_cdata.blocks);
  global::zero<<<1 + h_cdata.blocks/512, 512, 0, stream[3]>>>(data->d_partialsums, h_cdata.blocks);
  global::zero<<<h_cdata.elPerBlock, h_cdata.blocks, 0, stream[4]>>>(data->d_num, h_cdata.elPerBlock * h_cdata.blocks);
}

void Omega12Host::deallocate()
{
  cudaFree(data->d_primeList);
  cudaFree(data->d_mu);
  cudaFree(data->d_lpf);
  cudaFree(data->d_sums);
  cudaFree(data->d_partialsums);
  cudaFree(data->d_num);
  cudaFree(data->d_totals);
  cudaFree(data->d_totalsNext);
}

uint128_t	 Omega12Host::launchIter()
{
  uint128_t omega12_res = 0;

  cudaProfilerStart();
  Omega12Global::omega12_ctl<<<h_cdata.blocks, h_threadsPerBlock, h_cdata.sieveWords*sizeof(uint32_t)>>>(data);//, bstart, h_cdata.sieveWords);
  cudaProfilerStop();
  cudaDeviceSynchronize();

  Omega12Global::scanVectorized<<<h_cdata.elPerBlock, h_cdata.blocks, h_cdata.blocks*sizeof(int64_t)>>>(data->d_sums, data->d_totalsNext);
  cudaDeviceSynchronize();

  Omega12Global::addMultiply<<<h_cdata.elPerBlock, h_cdata.blocks>>>(data->d_sums, data->d_totals, data->d_num);
  cudaDeviceSynchronize();

  Omega12Global::addArrays<<<h_cdata.elPerBlock, h_cdata.blocks, 0, stream[1]>>>(data->d_totals, data->d_totalsNext, h_cdata.blocks);

  omega12_res = thrust::reduce(thrust::device, data->d_partialsums, data->d_partialsums + h_cdata.blocks);

  omega12_res += thrust::reduce(thrust::device, data->d_sums, data->d_sums + (h_cdata.blocks * h_cdata.elPerBlock));
  cudaDeviceSynchronize();

  // std::cout << "\t" <<100 * h_cdata.bstart/h_cdata.z << "% complete\r";
  // std::cout << std::flush;
  //

  return omega12_res;
}

uint128_t Omega12Host::omega12(uint128_t x, uint64_t y, uint16_t c)
{
  uint128_t sum = 0;

  Omega12Host * s2h = new Omega12Host(x, y, c);
  KernelTime timer;
  timer.start();

  sum += s2h->launchIter();

  s2h->setupNextIter();

  while(s2h->h_cdata.bstart < s2h->h_cdata.z){
    sum += s2h->launchIter();
    s2h->setupNextIter();
  }
  timer.stop();
  timer.displayTime();
  return sum;
}
