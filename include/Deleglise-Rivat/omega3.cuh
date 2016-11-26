#include <stdint.h>
#include <cuda_uint128.h>

__constant__ const uint16_t numThreads = 256;
const uint16_t threadsPerBlock = 256;

__global__
void Omega3_kernel( uint128_t x, uint64_t * pq, uint32_t * d_pitable, uint64_t pi_0,
                    uint64_t base, uint32_t pMinIdx, uint32_t pCornerIdx, uint32_t pMaxIdx,
                    uint64_t * sums, uint64_t * nextQ, uint64_t * lastQ, uint64_t maxQidx);

__device__
inline uint64_t checkRange(uint64_t q, uint64_t base, uint64_t & s_lastQ, uint64_t qidx);

__device__
inline uint64_t lookupPi(uint64_t q, uint32_t * d_pitable,
                                uint64_t pi_0, uint64_t base);
__device__
inline void minLastQ(uint32_t j, uint64_t * s_lastQ, uint64_t * nextQ, uint64_t * lastQ);
__global__
void Omega3_lower_bound(uint128_t x, uint64_t * nextQ, uint64_t * pq,
                        uint64_t p0Idx, uint64_t pMaxIdx);
