#include <stdint.h>
#include <cuda.h>

#include "general/device_functions.cuh"

template<typename T, typename U>
__global__ void global::zero(T * array, U max)
{
  U tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if(tidx< max) array[tidx] = 0;
}
template __global__ void global::zero<int16_t, uint32_t>(int16_t *, uint32_t);
template __global__ void global::zero<int64_t, uint32_t>(int64_t *, uint32_t);
template __global__ void global::zero<uint64_t, uint32_t>(uint64_t *, uint32_t);
template __global__ void global::zero<uint64_t, uint64_t>(uint64_t *, uint64_t);

template<typename T, typename U>
__global__ void global::set(T * array, U max, T set)
{
  U tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if(tidx< max) array[tidx] = set;
}

template __global__ void global::set<int32_t, uint32_t>(int32_t *, uint32_t, int32_t);
template __global__ void global::set<int64_t, uint32_t>(int64_t *, uint32_t, int64_t);

__global__ void global::multiply(uint32_t * a, int8_t * b, int32_t * c, uint32_t numElements)
{
  uint64_t tidx = threadIdx.x + blockIdx.x*blockDim.x;
  if(tidx < numElements)
    c[tidx] = a[tidx] * b[tidx];
}


///  For sigma4:
///  array[i] = x / (array[i] * y);
///  which represents the equation n = x (p_i * y)
///
__global__ void global::xOverPY(uint64_t * p, uint128_t x, uint64_t y, size_t len)
{
  uint32_t tidx = threadIdx.x + blockDim.x*blockIdx.x;

  if(tidx < len) p[tidx] = x / (p[tidx] * y);
}

__global__ void global::xOverPY(uint64_t * p, uint64_t x, uint64_t y, size_t len)
{
  uint32_t tidx = threadIdx.x + blockDim.x*blockIdx.x;

  if(tidx < len) p[tidx] = x / (p[tidx] * y);
}


///  For sigma5:
///  array[i] = x / (array[i] * array[i]);
///  which represents the equation n = x / (p_i * p_i)
///
__global__ void global::xOverPSquared(uint64_t * p, uint128_t x, size_t len)
{
  uint32_t tidx = threadIdx.x + blockDim.x*blockIdx.x;

  if(tidx < len) p[tidx] = x / (p[tidx] * p[tidx]);
}

__global__ void global::xOverPSquared(uint64_t * p, uint64_t x, size_t len)
{
  uint32_t tidx = threadIdx.x + blockDim.x*blockIdx.x;

  if(tidx < len) p[tidx] = x / (p[tidx] * p[tidx]);
}


///  For sigma6:
///  array[i] = sqrt(x) / sqrt(p);
///  which represents the equation n = x^(1/2) / p_i^(1/2)
///  note that sqrt(x) is precomputed, so don't call this function x.
///
__global__ void global::sqrtxOverSqrtp(uint64_t * p, uint64_t sqrtx, size_t len)
{
  uint32_t tidx = threadIdx.x + blockDim.x*blockIdx.x;

  if(tidx < len) p[tidx] = sqrtx / __double2ull_rd(__dsqrt_rd(p[tidx]));
}

///  For sigma6:
///  array[i] = array[i]^2;
///  which represents the expression pi(n)^2
///
__global__ void global::squareEach(uint64_t * pi, size_t len)
{
  uint32_t tidx = threadIdx.x + blockDim.x*blockIdx.x;

  if(tidx < len) pi[tidx] = pi[tidx] * pi[tidx];
}

///  For trivial leaves:
///

__global__ void global::x_minus_array(uint64_t * a, uint64_t x, size_t len)
{
  uint32_t tidx = threadIdx.x + blockDim.x*blockIdx.x;

  if(tidx < len) a[tidx] = x - a[tidx];
}

/// For P2(x, y):
///
///

__global__ void global::divXbyY(uint128_t x, uint64_t * y, size_t len)
{
  uint64_t tidx = threadIdx.x + blockIdx.x*blockDim.x;

  if(tidx < len) y[tidx] = x/y[tidx];
}
