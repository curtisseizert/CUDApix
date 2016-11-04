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

__global__ void transposePrimes(uint32_t * d_primes, uint32_t * d_pitable, uint32_t top, size_t len)
{
  uint32_t tidx = threadIdx.x + blockIdx.x*blockDim.x;

  if(tidx < len){

    uint32_t lo, hi;
    if(tidx < len - 1){
      lo = d_primes[tidx]/2;
      hi = d_primes[tidx + 1]/2;
    }else{
      lo = d_primes[tidx]/2;
      hi = top/2;
    }

    while(lo < hi){
      d_pitable[lo] = tidx + 1;
      lo++;
    }
  }
}
