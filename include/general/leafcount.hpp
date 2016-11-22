#include <stdint.h>
#include <cuda_uint128.h>

namespace leafcount{

uint64_t gourdon_A(uint64_t x);
uint128_t gourdon_A(uint128_t x);
uint64_t gourdon_C_simple(uint64_t x, uint64_t y);
uint64_t omega1(uint64_t x, uint64_t y, uint16_t c);
uint64_t omega2(uint64_t x, uint64_t y);
uint64_t omega3(uint64_t x, uint64_t y);

} // namespace leafcount
