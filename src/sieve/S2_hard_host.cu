#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <CUDASieve/cudasieve.hpp>
#include <CUDASieve/host.hpp>

#include "general/tools.hpp"
#include "general/device_functions.cuh"
#include "sieve/S2_hard_host.cuh"
#include "sieve/S2_hard_device.cuh"
#include "sieve/lpf_mu.cuh"
#include "S3.cuh"

#include <cuda_profiler_api.h>

const uint16_t h_cutoff = 12;
const uint16_t h_threadsPerBlock = 256;
const uint32_t blockSpan = 1024 * 64;

S2hardHost::S2hardHost(uint64_t x, uint64_t y, uint16_t c)
{
  makeData(x, y, c);
  allocate();
}

S2hardHost::~S2hardHost()
{
  deallocate();
}

void S2hardHost::makeData(uint64_t x, uint64_t y, uint16_t c)
{
  cudaMallocManaged((void **)&data, sizeof(S2data_64));
  cudaDeviceSynchronize();
  data->x = x;
  data->y = y;
  data->c = c;
  data->bstart = 0;
  data->mstart = 1;
  data->blocks = std::min(1 + (uint32_t)(x/y)/blockSpan, 512u); // must be <= 1024

  uint32_t maxPrime = sqrt(y);
  data->d_primeList = PrimeList::getSievingPrimes(maxPrime, data->primeListLength, 1);
  data->elPerBlock = data->primeListLength + h_cutoff;

  data->d_mu = gen_d_mu((uint32_t)0, (uint32_t)y);
  data->d_lpf = gen_d_lpf((uint32_t)0, (uint32_t)y);
}

void S2hardHost::allocate()
{
  cudaMalloc(&data->d_sums, data->elPerBlock * data->blocks * sizeof(uint64_t));
  cudaMalloc(&data->d_partialsums, data->blocks * sizeof(int64_t));
  cudaMalloc(&data->d_num, data->elPerBlock * data->blocks * sizeof(int16_t));
  cudaMalloc(&data->d_totals, data->elPerBlock * sizeof(uint64_t));
  cudaMalloc(&data->d_totalsNext, data->elPerBlock * sizeof(uint64_t));

  cudaMemset(data->d_sums, data->elPerBlock * data->blocks * sizeof(int64_t), 0); // todo: implement streams here
  cudaMemset(data->d_partialsums, data->blocks * sizeof(int64_t), 0);
  gen::zero<<<data->elPerBlock, data->blocks>>>(data->d_num, data->elPerBlock * data->blocks);
  gen::zero<<<1 + data->elPerBlock/512, 512>>>(data->d_totals, data->elPerBlock);
  gen::zero<<<1 + data->elPerBlock/512, 512>>>(data->d_totalsNext, data->elPerBlock);

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
  KernelTime timer;
  int64_t s2_hard = 0;
  timer.start();
  cudaProfilerStart();
  S2glob::S2ctl<<<data->blocks, h_threadsPerBlock>>>(data);
  cudaProfilerStop();
  cudaDeviceSynchronize();
  timer.stop();
  timer.displayTime();

  // dispDevicePartialSums(data->d_sums, data->blocks*data->elPerBlock, data->blocks);
  // dispDeviceArray(data->d_sums, data->blocks*data->elPerBlock);

  S2glob::scanVectorized<<<data->elPerBlock, data->blocks, data->blocks*sizeof(int64_t)>>>(data->d_sums, data->d_totalsNext);
  cudaDeviceSynchronize();

  // dispDeviceArray(data->d_totalsNext, data->elPerBlock);

  S2glob::addMultiply<<<data->elPerBlock, data->blocks>>>(data->d_sums, data->d_totals, data->d_num);
  cudaDeviceSynchronize();

  uint64_t * dummy = data->d_totals;
  data->d_totals = data->d_totalsNext;
  data->d_totalsNext = dummy;

  dispDevicePartialSums(data->d_num, (data->blocks * data->elPerBlock), data->blocks);

  s2_hard = thrust::reduce(thrust::device, data->d_partialsums, data->d_partialsums + data->blocks);

  std::cout << s2_hard << std::endl;

  s2_hard += thrust::reduce(thrust::device, data->d_sums, data->d_sums + (data->blocks * data->elPerBlock));

  return s2_hard;
}

uint64_t S2hardHost::S2hard(uint64_t x, uint64_t y, uint16_t c)
{
  uint64_t sum = 0;

  S2hardHost * s2h = new S2hardHost(x, y, c);

  sum += s2h->launchIter();

  // int32_t * d_test;
  // cudaMalloc(&d_test, 1024);
  //
  // gen::set<<<1, 256>>>(d_test, 256u, 1);
  //
  // testRed<<<1, 256>>>(d_test);
  // cudaDeviceSynchronize();
  //
  // dispDeviceArray(d_test, 256u);

  return sum;
}
