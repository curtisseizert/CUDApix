/*
P2.cuh - header file for computing second partial sieve function

Curtis Seizert <cseizert@gmail.com>

*/

#include <gmpxx.h>

#ifndef _P2
#define _P2

#define THREADS_PER_BLOCK 256
typedef unsigned __int128 uint128_t;

uint64_t P2(uint64_t x, uint64_t y);
uint128_t P2(uint128_t x, uint128_t y);
mpz_class P2(mpz_class x, mpz_class y);


struct PrimeArray{
  uint64_t * h_primes, * d_primes, bottom, top, pi_bottom, pi_top, diff_pi;
  size_t len;
};

class ResetCounter{
private:
  uint16_t counter = 0;
public:
  void increment();
};

class x_Over_y{
private:
  uint64_t x;
public:
  __device__ __host__ x_Over_y(uint64_t x): x(x){}
  __device__ __host__ uint64_t operator() (uint64_t y) { return x / y;}
};

__global__ void divXbyY(uint64_t x, uint64_t * y, size_t len);

#endif
