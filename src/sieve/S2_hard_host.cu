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
#include "sieve/S2_hard_host.cuh"
#include "sieve/S2_hard_device.cuh"
#include "sieve/lpf_mu.cuh"
#include "S3.cuh"

#include <cuda_profiler_api.h>

const uint16_t h_threadsPerBlock = 256;

S2hardHost::S2hardHost(uint64_t x, uint64_t y, uint16_t c)
{
  makeData(x, y, c);
  allocate();

  global::zero<<<1 + h_cdata.elPerBlock/512, 512, 0, stream[0]>>>(data->d_totals, h_cdata.elPerBlock);

  zero();
}

S2hardHost::~S2hardHost()
{
  deallocate();
}

void S2hardHost::makeData(uint64_t x, uint64_t y, uint16_t c)
{
  cudaMallocManaged((void **)&data, sizeof(S2data_64));
  cudaDeviceSynchronize();
  h_cdata.x = x;
  h_cdata.y = y;
  h_cdata.z = x/y;
  h_cdata.c = c;
  h_cdata.sieveWords = sieveWords_;
  h_cdata.bstart = 0;
  h_cdata.mstart = 1;
  h_cdata.blocks = std::min(1 + (uint32_t)(h_cdata.z/(64 * h_cdata.sieveWords)), 512u); // must be <= 1024

  h_cdata.maxPrime = sqrt(h_cdata.z);
  data->d_primeList = CudaSieve::getDevicePrimes32(0, h_cdata.maxPrime, h_cdata.primeListLength, 0);
  data->d_bitsieve = CudaSieve::genDeviceBitSieve(0, y, 0);
  h_cdata.elPerBlock = h_cdata.primeListLength - 2;

  transferConstants();

  data->d_mu = gen_d_mu((uint32_t)0, (uint32_t)y);
  data->d_lpf = gen_d_lpf((uint32_t)0, (uint32_t)y);
}

void S2hardHost::setupNextIter()
{
  // if(h_cdata.sieveWords < 4096) h_cdata.sieveWords += 1024;
  h_cdata.bstart += h_cdata.blocks * (64 * h_cdata.sieveWords);
  h_cdata.blocks = std::min(1 + (uint32_t)(h_cdata.z - h_cdata.bstart)/(64 * h_cdata.sieveWords), 512u);

  transferConstants();
  zero();
}

void S2hardHost::allocate()
{
  for(uint16_t i = 0; i < numStreams_; i++)
    cudaStreamCreate(&stream[i]);
  cudaMalloc(&data->d_sums, h_cdata.elPerBlock * h_cdata.blocks * sizeof(uint64_t));
  cudaMalloc(&data->d_partialsums, h_cdata.blocks * sizeof(int64_t));
  cudaMalloc(&data->d_num, h_cdata.elPerBlock * h_cdata.blocks * sizeof(int16_t));
  cudaMalloc(&data->d_totals, h_cdata.elPerBlock * sizeof(uint64_t));
  cudaMalloc(&data->d_totalsNext, h_cdata.elPerBlock * sizeof(uint64_t));
}

void S2hardHost::zero()
{
  global::zero<<<1 + h_cdata.elPerBlock/512, 512, 0, stream[1]>>>(data->d_totalsNext, h_cdata.elPerBlock);
  global::zero<<<h_cdata.elPerBlock, h_cdata.blocks, 0, stream[2]>>>(data->d_sums, h_cdata.elPerBlock * h_cdata.blocks);
  global::zero<<<1 + h_cdata.blocks/512, 512, 0, stream[3]>>>(data->d_partialsums, h_cdata.blocks);
  global::zero<<<h_cdata.elPerBlock, h_cdata.blocks, 0, stream[4]>>>(data->d_num, h_cdata.elPerBlock * h_cdata.blocks);
}

void S2hardHost::deallocate()
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

int64_t S2hardHost::launchIter()
{
  int64_t s2_hard = 0;

  cudaProfilerStart();
  S2glob::S2ctl<<<h_cdata.blocks, h_threadsPerBlock, h_cdata.sieveWords*sizeof(uint32_t)>>>(data);//, bstart, h_cdata.sieveWords);
  cudaProfilerStop();
  cudaDeviceSynchronize();

  // dispDevicePartialSums(data->d_sums, h_cdata.blocks*h_cdata.elPerBlock, h_cdata.blocks);
  // dispDeviceArray(data->d_totals, 20u);

  S2glob::scanVectorized<<<h_cdata.elPerBlock, h_cdata.blocks, h_cdata.blocks*sizeof(int64_t)>>>(data->d_sums, data->d_totalsNext);
  cudaDeviceSynchronize();

  S2glob::addMultiply<<<h_cdata.elPerBlock, h_cdata.blocks>>>(data->d_sums, data->d_totals, data->d_num);
  cudaDeviceSynchronize();

  S2glob::addArrays<<<h_cdata.elPerBlock, h_cdata.blocks, 0, stream[1]>>>(data->d_totals, data->d_totalsNext, h_cdata.blocks);

  dispDevicePartialSums(data->d_num, (h_cdata.blocks * h_cdata.elPerBlock), h_cdata.blocks);
  // dispDevicePartialSums(data->d_sums, h_cdata.blocks*h_cdata.elPerBlock, h_cdata.blocks);

  s2_hard = thrust::reduce(thrust::device, data->d_partialsums, data->d_partialsums + h_cdata.blocks);
  std::cout << s2_hard << std::endl;

  s2_hard += thrust::reduce(thrust::device, data->d_sums, data->d_sums + (h_cdata.blocks * h_cdata.elPerBlock));
  cudaDeviceSynchronize();

  // std::cout << "\t" <<100 * h_cdata.bstart/h_cdata.z << "% complete\r";
  // std::cout << std::flush;
  //

  return s2_hard;
}

uint64_t S2hardHost::S2hard(uint64_t x, uint64_t y, uint16_t c)
{
  uint64_t sum = 0;

  S2hardHost * s2h = new S2hardHost(x, y, c);
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
