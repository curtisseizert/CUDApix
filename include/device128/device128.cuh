#include <stdint.h>
#include "uint128_t.cuh"

namespace device{
__global__ void divXbyY(uint128_t x, uint64_t * y, size_t len);
__global__ void x_over_psquared(uint64_t * p, uint128_t x, size_t len);
}

namespace launch{
  void divXbyY(uint128_t x, uint64_t * y, size_t len);
  void xOverPSquared(uint64_t * p, uint128_t x, size_t len);
}
