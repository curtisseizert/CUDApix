#include <stdint.h>

namespace phidev{
__device__ void sieveCountInit(uint32_t * d_sieve, uint32_t * d_count, uint64_t bstart, uint16_t c);
__device__ void markSmallPrimes(uint32_t * d_sieve, uint64_t bstart, uint16_t c);
__device__ void markMedPrimes(uint32_t * d_sieve, uint32_t p, uint64_t bstart, uint32_t sieveBits);

}

namespace phiglobal{
  __global__ void sieveCountInit(uint32_t * d_sieve, uint32_t * d_count, uint64_t bstart, uint16_t c);
  __global__ void markSmallPrimes(uint32_t * d_sieve, uint64_t bstart, uint16_t c);
  __global__ void markMedPrimes(uint32_t * d_sieve, uint32_t p, uint64_t bstart, uint32_t sieveBits);
  __global__ void updateCount(uint32_t * d_sieve, uint32_t * d_count);
}
