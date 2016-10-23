#include <stdint.h>

template <typename T>
T S0(T x, T y);

__global__ void S0kernel(int8_t * d_mu, int64_t * d_quot, uint64_t x, uint64_t y);
