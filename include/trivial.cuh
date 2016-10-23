#include <stdint.h>

#pragma once

uint64_t S1_trivial(uint64_t x, uint64_t y);

__global__ void x_over_psquared(uint64_t * p, uint64_t x, size_t len);
__global__ void x_minus_array(uint64_t * a, uint64_t x, size_t len);
