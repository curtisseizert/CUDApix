/*
P2.cuh - header file for computing second partial sieve function

Curtis Seizert <cseizert@gmail.com>

*/

#include <gmpxx.h>
#include "uint128_t.cuh"

#ifndef _P2
#define _P2

#define THREADS_PER_BLOCK 256

uint64_t P2(uint64_t x, uint64_t y);
uint128_t P2(uint128_t x, uint64_t y);

// mpz_class P2(mpz_class x, mpz_class y);

void divXbyY(uint128_t x, uint64_t * y, size_t len);

#endif
