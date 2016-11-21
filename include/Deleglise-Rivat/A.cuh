#include <stdint.h>
#include <uint128_t.cuh>
///
/// 128-bit implementation
///

__constant__ const uint16_t numThreads = 256;
const uint16_t threadsPerBlock = 256;

__global__
void A_large_loPQ(uint128_t x, uint64_t y, uint64_t * pq, uint32_t * d_pitable,
                  uint64_t pi_0, uint64_t base, uint32_t pMaxIdx, uint64_t * sums,
                  uint64_t * nextQ, uint64_t * lastQ, uint64_t maxQidx);

__global__
void A_large_hiPQ_vert(uint128_t x, uint64_t y, uint64_t * pq, uint32_t * d_pitable,
                        uint64_t pi_0, uint64_t base, uint32_t pMinIdx, uint32_t pMaxIdx,
                        uint64_t * sums, uint64_t * nextQ, uint64_t * lastQ, uint64_t maxQidx);

__device__
uint64_t checkRange(uint64_t  q, uint64_t base, uint64_t & s_lastQ, uint64_t qidx);

__device__
uint64_t calculatePiChi(uint64_t q, uint64_t y, uint32_t * d_pitable,
                        uint64_t pi_0, uint64_t base);
__device__
void minLastQ(uint32_t j, uint64_t * s_lastQ, uint64_t * nextQ, uint64_t * lastQ);
