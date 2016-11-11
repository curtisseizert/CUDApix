#include <stdint.h>
#include <iostream>
#include <math.h>
#include <CUDASieve/cudasieve.hpp>

#include "Deleglise-Rivat/deleglise-rivat.hpp"
#include "P2.cuh"
#include "ordinary.cuh"

deleglise_rivat64::deleglise_rivat64(uint64_t x, uint64_t y, uint16_t c)
{
  this->x = x;
  this->y = y;
  this->c = c;
  calculateBounds();
  calculatePiValues();
}

void deleglise_rivat64::calculateBounds()
{
  z = x / y;
  sqrtx = sqrt(x);
  cbrtx = cbrt(x);
  qrtx = sqrt(sqrtx);
  sqrtz = sqrt(z);
}

void deleglise_rivat64::calculatePiValues()
{
  pi_qrtx = CudaSieve::countPrimes(0ull, qrtx);
  pi_cbrtx = pi_qrtx + CudaSieve::countPrimes(qrtx, cbrtx);
  pi_y = pi_cbrtx + CudaSieve::countPrimes(cbrtx, y);
  pi_sqrtx = pi_y + CudaSieve::countPrimes(y, sqrtx);
  pi_sqrtz = CudaSieve::countPrimes(0ull, sqrtz);
}

uint64_t deleglise_rivat64::S1()
{
  return (uint64_t)sigma1();
}

uint64_t deleglise_rivat64::pi_deleglise_rivat(uint64_t x, uint64_t y, uint16_t c)
{
  deleglise_rivat64 pi_dr(x, y, c);

  uint64_t pi_y = pi_dr.pi_y;
  uint64_t p2, s0, s1, s2;//, s3;

  p2 = P2(x, y);

  s0 = ordinary(x, y, c);

  s1 = pi_dr.S1();

  s2 = pi_dr.S2();




  uint64_t pi = s0 + s1 + s2 + pi_y - 1;

  return pi;
}
