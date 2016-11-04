#include <stdint.h>
#include <uint128_t.cuh>
#include <general/device_functions.cuh>

#pragma once

#ifndef _UINT128_T_CUDA_H

uint64_t S1_trivial(uint64_t x, uint64_t y);

#else

uint128_t S1_trivial(uint128_t x, uint64_t y);
inline void xOverPSquared(uint64_t * p, uint128_t x, size_t len);
inline void x_minus_array(uint64_t * p, uint64_t x, size_t len);


#endif
