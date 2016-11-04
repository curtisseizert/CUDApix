#include <stdint.h>

#ifndef _PITABLE
#define _PITABLE

uint32_t * get_d_piTable(uint32_t hi);

__global__ void transposePrimes(uint32_t * d_primes, uint32_t * d_pitable, uint32_t top, size_t len);

#endif
