#include <stdint.h>
#include "cudapix.hpp"

#pragma once

int64_t V(uint64_t x, uint64_t y);
int64_t V_a(uint64_t x, PrimeArray & pi, PrimeArray & p, PrimeArray & q);
int64_t V_b(uint64_t x, PrimeArray & pi, PrimeArray & p, PrimeArray & q);


namespace device
{
__global__ void XoverPQ(uint64_t x, uint64_t * q, uint64_t * quot, size_t len);
__global__ void XoverPSquared(uint64_t x, uint64_t * p, uint64_t * quot, size_t len);
} // namespace device
