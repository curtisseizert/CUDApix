#include <stdint.h>
#include <cuda.h>


__global__ void A_lo_p_reg(uint64_t x, uint64_t * q, uint64_t * sums,
                           uint64_t y, uint32_t num_p, uint64_t max_num_q,
                           uint32_t * d_pitable);

__device__ void calculateQuotient(uint64_t x, uint64_t qt, uint64_t & quot, uint32_t iter);
__device__ void getPiOfQuot(uint64_t quot, uint64_t y, uint32_t * d_pitable, uint64_t * s_pi_quot);

__global__
void A_large_loPQ(uint64_t x, uint64_t y, uint64_t * pq, uint32_t * d_pitable,
                  uint64_t pi_0, uint64_t base, uint32_t pMaxIdx, uint64_t * sums,
                  uint32_t * nextQ, uint32_t * lastQ, uint32_t maxQidx);

__device__ void checkRange(uint64_t & q, uint64_t base, uint32_t * s_lastQ, uint32_t qidx);
__device__ uint64_t checkRangeHi(uint64_t q, uint64_t base, uint32_t * s_lastQ, uint32_t qidx, uint64_t p, uint64_t y);
__device__ uint64_t calculatePiChi(uint64_t q, uint64_t y, uint32_t * d_pitable, uint64_t pi_0, uint64_t base);
__device__ void minLastQ(uint32_t i, uint32_t * s_lastQ, uint32_t * nextQ, uint32_t * lastQ);

__global__
void A_large_loPQ(uint64_t x, uint64_t y, uint64_t * pq, uint32_t * d_pitable,
                  uint64_t pi_0, uint64_t base, uint32_t pMaxIdx, uint64_t * sums,
                  uint32_t * nextQ, uint32_t * lastQ, uint32_t maxQidx);

__global__
void A_large_hiPQ(uint64_t x, uint64_t y, uint64_t * pq, uint32_t * d_pitable,
                  uint64_t pi_0, uint64_t base, uint32_t pMinIdx, uint32_t pMaxIdx,
                  uint64_t * sums, uint32_t * nextQ, uint32_t * lastQ, uint32_t maxQidx);
