#include <stdint.h>
#include <cuda.h>

#include "general/device_functions.cuh"

template<typename T, typename U>
__global__ void gen::zero(T * array, U max)
{
  U tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if(tidx< max) array[tidx] = 0;
}

template __global__ void gen::zero<uint64_t, uint32_t>(uint64_t *, uint32_t);

template<typename T, typename U>
__global__ void gen::set(T * array, U max, T set)
{
  U tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if(tidx< max) array[tidx] = set;
}

template __global__ void gen::set<int32_t, uint32_t>(int32_t *, uint32_t, int32_t);
template __global__ void gen::set<int64_t, uint32_t>(int64_t *, uint32_t, int64_t);

__global__ void gen::multiply(uint32_t * a, int8_t * b, int32_t * c, uint32_t numElements)
{
  uint64_t tidx = threadIdx.x + blockIdx.x*blockDim.x;
  if(tidx < numElements)
    c[tidx] = a[tidx] * b[tidx];
}
