#include <stdint.h>
#include <cuda.h>
#include <uint128_t.cuh>

namespace global
{
  template<typename T, typename U>
  __global__ void zero(T * array, U max);

  template<typename T, typename U>
  __global__ void set(T * array, U max, T set);

  __global__ void multiply(uint32_t * a, int8_t * b, int32_t * c, uint32_t numElements);
  __global__ void xOverPY(uint64_t * p, uint128_t x, uint64_t y, size_t len);
  __global__ void xOverPY(uint64_t * p, uint64_t x, uint64_t y, size_t len);
  __global__ void xOverPSquared(uint64_t * p, uint128_t x, size_t len);
  __global__ void xOverPSquared(uint64_t * p, uint64_t x, size_t len);
  __global__ void sqrtxOverSqrtp(uint64_t * p, uint64_t sqrtx, size_t len);
  __global__ void squareEach(uint64_t * pi, size_t len);
  __global__ void x_minus_array(uint64_t * a, uint64_t x, size_t len);
  __global__ void divXbyY(uint128_t x, uint64_t * y, size_t len);
}
