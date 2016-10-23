#include <stdint.h>

uint64_t S3(uint64_t x, uint64_t y, uint32_t c);

__global__ void S3_phi(uint64_t x, uint32_t p, uint64_t y, int64_t * d_sums, uint32_t * d_lpf, int8_t * d_mu, uint32_t * d_phi);
__global__ void zero(int64_t * array, uint32_t max);
