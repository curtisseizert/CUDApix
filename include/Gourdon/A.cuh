#include <stdint.h>
#include <cuda.h>

__global__ void A_lo_p(uint64_t x, uint64_t * q, uint64_t * pi, uint64_t * sums,
                       uint32_t pi_y, uint32_t num_p, uint64_t max_num_q,
                       uint32_t * d_pitable);

__device__ void fill_s_q(uint64_t * q, uint64_t * s_q, uint64_t max_num_q);
__device__ void calculateQuotient(uint64_t x, uint64_t * s_q, uint64_t * s_quot, uint32_t iter);
__device__ void chiXPQ(uint32_t pi_y, uint64_t * s_quot);
__device__ void getPiOfQuot(uint64_t * s_quot, uint32_t * d_pitable);

__global__ void A_lo_p_reg(uint64_t x, uint64_t * q, uint64_t * pi, uint64_t * sums,
                           uint32_t pi_y, uint32_t num_p, uint64_t max_num_q,
                           uint32_t * d_pitable);

__device__ void calculateQuotient(uint64_t x, uint64_t qt, uint64_t & quot, uint32_t iter);
__device__ void getPiOfQuot(uint64_t quot, uint32_t pi_y, uint32_t * d_pitable, uint64_t * s_pi_quot);
