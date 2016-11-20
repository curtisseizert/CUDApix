#include <cuda.h>
#include <stdint.h>

__constant__ const uint16_t numThreads = 256;
const uint16_t threadsPerBlock = 256;

__global__
void C_nonseg( uint64_t x, uint64_t * pq, uint64_t * sums,
                 uint64_t pi_sqrty, uint32_t num_p, uint64_t max_num_q,
                 uint32_t * d_pitable, uint64_t first_q_offset);

__device__
inline uint64_t checkAndDiv_C(uint64_t q, uint64_t p, uint64_t x);

__device__
uint64_t calculatePiChi_C(uint64_t quot, uint32_t * d_pitable);
