#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <CUDASieve/cudasieve.hpp>

#include "pitable.cuh"

const uint16_t threadsPerBlock = 256;

uint32_t * get_d_piTable(uint32_t hi)
{
  size_t len;
  uint32_t * d_primes = CudaSieve::getDevicePrimes32(0, hi, len, 0);

  uint32_t * d_pitable;
  cudaMalloc(&d_pitable, hi/2 * sizeof(uint32_t));
  cudaDeviceSynchronize();

  transposePrimes<<<1 + len/threadsPerBlock, threadsPerBlock>>>(d_primes, d_pitable, hi, len);

  cudaDeviceSynchronize();
  cudaFree(d_primes);

  return d_pitable;
}

PiTable::PiTable(uint64_t range)
{
  this->range = range;
  allocate();
}

PiTable::PiTable(uint64_t base, uint64_t bottom, uint64_t range)
{
  this->bottom = bottom;
  this->base = base;
  if(range == 0) this->range = std::min(getMaxRange(), base - bottom);
  else           this->range = range;

  allocate();
}

PiTable::~PiTable()
{
  // if(d_pitable != NULL) cudaFree(d_pitable);
}

uint64_t PiTable::getMaxRange()
{
  cudaMemGetInfo(&free_mem, &tot_mem);
  uint64_t r = free_mem/2.5;

  return r;
}

uint64_t PiTable::setMaxRange()
{
  range = getMaxRange();
  reallocate();
  return range;
}

void PiTable::allocate()
{
  if(d_pitable == NULL) cudaMalloc(&d_pitable, range/2 * sizeof(uint32_t));
  allocatedRange = range;
  cudaMemsetAsync(d_pitable, 0, range/2*sizeof(uint32_t));
}

void PiTable::reallocate()
{
  if(d_pitable != NULL) cudaFree(d_pitable);
  cudaMalloc(&d_pitable, range/2 * sizeof(uint32_t));
  allocatedRange = range;
  cudaMemsetAsync(d_pitable, 0, range/2*sizeof(uint32_t));
}

uint32_t * PiTable::getNextUp()
{
  if(!isPiCurrent) calc_pi_base();

  base += range;
  pi_base += len;

  uint64_t * d_primes = CudaSieve::getDevicePrimes(base, base + range, len, 0);

  cudaDeviceSynchronize();
  transposePrimes<<<1 + len/threadsPerBlock, threadsPerBlock>>>(d_primes, d_pitable, base, range, len, pi_base);
  cudaDeviceSynchronize();
  cudaFree(d_primes);
  isTableCurrent = 1;

  return d_pitable;
}

uint32_t * PiTable::getNextDown()
{
  if(!isPiCurrent) pi_base = get_pi_base();

  if((int64_t)bottom < (int64_t)base - (int64_t)range)
    base -= range;
  else {
    range = base - bottom;
    base = bottom;
  }

  uint64_t * d_primes = CudaSieve::getDevicePrimes(base, base + range, len, 0);

  pi_base -= len;
  // std::cout << base << " " << len << std::endl;

  cudaDeviceSynchronize();
  transposePrimes<<<1 + len/threadsPerBlock, threadsPerBlock>>>(d_primes, d_pitable, base, range, len, pi_base);
  cudaDeviceSynchronize();
  cudaFree(d_primes);
  isTableCurrent = 1;

  return d_pitable;
}

uint32_t * PiTable::getCurrent()
{
  if(!isPiCurrent) pi_base = get_pi_base();
  if(isTableCurrent) return d_pitable;

  uint64_t * d_primes = CudaSieve::getDevicePrimes(base, base + range, len, 0);

  cudaDeviceSynchronize();
  transposePrimes<<<1 + len/threadsPerBlock, threadsPerBlock>>>(d_primes, d_pitable, base, range, len, pi_base);
  cudaDeviceSynchronize();
  cudaFree(d_primes);
  isTableCurrent = 1;

  return d_pitable;
}

uint64_t PiTable::get_pi_base()
{
  if(base == 0)          return 0;
  if(!isPiCurrent)     calc_pi_base();

  return pi_base;
}

void PiTable::calc_pi_base()
{
  if(base != 0) pi_base = CudaSieve::countPrimes(0, base, 0);
  else        pi_base = 0;
  isPiCurrent = 1;
}

void PiTable::set_base(uint64_t base)
{
  this->base = base;
  isPiCurrent = 0;
  isTableCurrent = 0;
}

void PiTable::set_pi_base(uint64_t pi_base)
{
  this->pi_base = pi_base;
  isPiCurrent = 1;
}

void PiTable::set_range(uint64_t range)
{
  this->range = range;
  std::cout << this->range << std::endl;
  isTableCurrent = 0;
}

__global__
void transposePrimes(uint32_t * d_primes, uint32_t * d_pitable,
                     uint32_t top, size_t len)
{
  uint32_t tidx = threadIdx.x + blockIdx.x*blockDim.x;

  if(tidx < len){

    uint32_t lo, hi;
    if(tidx < len - 1){
      lo = (d_primes[tidx])/2 + 1;
      hi = (d_primes[tidx + 1])/2 + 1;
    }else{
      lo = (d_primes[tidx])/2 + 1;
      hi = top/2;
    }

    while(lo < hi){
      d_pitable[lo] = tidx + 1;
      lo++;
    }
  }
}

__global__
void transposePrimes(uint64_t * d_primes, uint32_t * d_pitable, uint64_t bottom,
                     uint64_t range, size_t len, uint64_t pi_base)
{
  uint32_t tidx = threadIdx.x + blockIdx.x*blockDim.x;

  if(tidx < len){

    uint64_t lo, hi;
    if(tidx < len - 1){
      lo = (d_primes[tidx] - bottom)/2 + 1;
      hi = (d_primes[tidx + 1] - bottom)/2 + 1;
    }else{
      lo = (d_primes[tidx] - bottom)/2 + 1;
      hi = (4 + range)/2;
    }

    while(lo < hi){
      d_pitable[lo] = tidx + 1;
      lo++;
    }
  }
}
