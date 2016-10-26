#include <cuda.h>
#include <cuda_runtime.h>

#include <uint128_t.cu>

#include "device128/device128.cuh"


const uint16_t threadsPerBlock = 256;

void launch::divXbyY(uint128_t x, uint64_t * y, size_t len)
{
  device::divXbyY<<<len/threadsPerBlock + 1, threadsPerBlock>>>(x, y, len);
}

void launch::xOverPSquared(uint64_t * p, uint128_t x, size_t len)
{
  device::x_over_psquared<<<len/threadsPerBlock + 1, threadsPerBlock>>>(p, x, len);
}

__global__ void device::divXbyY(uint128_t x, uint64_t * y, size_t len)
{
  uint64_t tidx = threadIdx.x + blockIdx.x*blockDim.x;
  if(tidx < len) y[tidx] = x/y[tidx];
}

__global__ void device::x_over_psquared(uint64_t * p, uint128_t x, size_t len)
{
  uint32_t tidx = threadIdx.x + blockDim.x*blockIdx.x;

  if(tidx < len) p[tidx] = x / (p[tidx] * p[tidx]);
}
