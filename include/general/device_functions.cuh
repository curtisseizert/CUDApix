#include <stdint.h>
#include <cuda.h>

namespace gen
{
  template<typename T, typename U>
  __global__ void zero(T * array, U max);

  template<typename T, typename U>
  __global__ void set(T * array, U max, T set);

  __global__ void multiply(uint32_t * a, int8_t * b, int32_t * c, uint32_t numElements);
}
