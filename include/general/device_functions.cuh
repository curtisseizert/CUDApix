#include <stdint.h>
#include <cuda.h>
#include <cuda_uint128.h>

#ifndef _DEVICE_FUNCTIONS_CUDAPIX
#define _DEVICE_FUNCTIONS_CUDAPIX

namespace global
{
  template<typename T, typename U>
  __global__ void zero(T * array, U max);

  template<typename T, typename U>
  __global__ void set(T * array, U max, T set)
  {
    U tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if(tidx< max) array[tidx] = set;
  }

  template <typename T, typename U, typename V>
  __global__ void addToArray(T * a, U len, V c)
  {
    uint64_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if(tidx < len)
      a[tidx] += c;
  }

// a[x] = x + b
  template<typename T, typename U>
  __global__ void setXPlusB(T * array, U max, T b);

  __global__ void multiply(uint32_t * a, int8_t * b, int32_t * c, uint32_t numElements);
  __global__ void xOverPY(uint64_t * p, uint128_t x, uint64_t y, size_t len);
  __global__ void xOverPY(uint64_t * p, uint64_t x, uint64_t y, size_t len);
  __global__ void xOverPSquared(uint64_t * p, uint128_t x, size_t len);
  __global__ void xOverPSquared(uint64_t * p, uint64_t x, size_t len);
  __global__ void sqrtxOverSqrtp(uint64_t * p, uint64_t sqrtx, size_t len);
  __global__ void squareEach(uint64_t * pi, size_t len);
  __global__ void x_minus_array(uint64_t * a, uint64_t x, size_t len);
  __global__ void divXbyY(uint128_t x, uint64_t * y, size_t len);
  __global__ void addToArray(uint64_t * a, size_t len, uint64_t c);
}

#endif
