#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <uint128_t.cuh>

uint64_t Ordinary(uint64_t x, uint64_t y, uint16_t c);
uint128_t Ordinary(uint128_t x, uint64_t y, uint16_t c);

__global__ void ordinaryKernel(int8_t * d_mu, int64_t * d_quot, uint32_t * d_lpf, uint32_t * d_phi, uint64_t x, uint64_t y, uint16_t c);
__global__ void ordinaryKernel(int8_t * d_mu, int64_t * d_quot, uint32_t * d_lpf, uint32_t * d_phi, uint128_t x, uint64_t y, uint16_t c);
