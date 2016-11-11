#include <stdint.h>
#include <cuda.h>

__constant__ const uint16_t numThreads = 256;
const uint16_t threadsPerBlock = 256;

__global__ void S2_a(uint64_t x, uint64_t * pq, uint64_t * sums,
                           uint64_t y, uint32_t num_p, uint64_t max_num_q,
                           uint32_t * d_pitable);

__device__ inline uint64_t checkAndDiv(uint64_t q, uint64_t p, uint64_t x);
__device__ inline uint64_t calculatePiChi(uint64_t q, uint64_t y, uint32_t * d_pitable);
