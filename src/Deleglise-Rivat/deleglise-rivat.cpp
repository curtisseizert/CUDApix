#include <stdint.h>
#include <iostream>
#include <math.h>
#include <CUDASieve/cudasieve.hpp>
#include <cuda_uint128.h>

#include "Deleglise-Rivat/deleglise-rivat.hpp"
#include "Deleglise-Rivat/omega12.cuh"
#include "P2.cuh"
#include "trivial.cuh"
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
  return (uint64_t)this->sigma1();
}

uint64_t deleglise_rivat64::S0()
{
  return Ordinary(x, y, c);
}

uint64_t deleglise_rivat64::pi_deleglise_rivat(uint64_t x, uint64_t y, uint16_t c)
{
  deleglise_rivat64 pi_dr(x, y, c);

  uint64_t pi_y = pi_dr.pi_y;
  uint64_t p2, ordinary, trivial, easy, hard;

  p2 = P2(x, y);

  ordinary = pi_dr.S0();

  trivial = S2_trivial(x, y);

  std::cout << pi_dr.S1() << std::endl;

  easy = pi_dr.S2();

  hard = pi_dr.S3();

  std::cout << "pi(x) = phi(x, a) - 1 + a - P2(x, a), where a = pi(y)" << std::endl;
  std::cout << "P2\t\t:\t" << p2 << std::endl;
  std::cout << "Ordinary Leaves\t:\t" << ordinary << std::endl;
  std::cout << "Trivial Leaves \t:\t" << trivial << std::endl;
  std::cout << "Easy Leaves    \t:\t" << easy << std::endl;
  std::cout << "Hard Leaves\t:\t" << hard << std::endl;
  std::cout << std::endl;

  uint64_t pi = ordinary + trivial + easy + hard + pi_y - 1 - p2;

  return pi;
}

///
/// Deleglise-Rivat 128-bit implemenation
///

deleglise_rivat128::deleglise_rivat128(uint128_t x, uint64_t y, uint16_t c)
{
  this->x = x;
  this->y = y;
  this->c = c;

  calculateBounds();
  calculatePiValues();
}

void deleglise_rivat128::calculateBounds()
{
  z = x / y;
  sqrtx = _isqrt(x);
  cbrtx = _icbrt(x);
  qrtx = _iqrt(x);
  sqrtz = _isqrt(z);
}

void deleglise_rivat128::calculatePiValues()
{
  pi_qrtx = CudaSieve::countPrimes(0ull, qrtx);
  pi_cbrtx = pi_qrtx + CudaSieve::countPrimes(qrtx, cbrtx);
  pi_y = pi_cbrtx + CudaSieve::countPrimes(cbrtx, y);
  pi_sqrtx = pi_y + CudaSieve::countPrimes(y, sqrtx);
  pi_sqrtz = CudaSieve::countPrimes(0ull, sqrtz);
}

uint128_t deleglise_rivat128::omega12()
{
	return Omega12Host::omega12(x, y, c);
}

uint128_t deleglise_rivat128::S0()
{
  return Ordinary(x, y, c);
}

uint128_t deleglise_rivat128::pi_deleglise_rivat(uint128_t x, uint64_t y, uint16_t c)
{
  deleglise_rivat128 pi_dr(x, y, c);

  uint64_t pi_y = pi_dr.pi_y;
  uint128_t p2, a, w3, pi, s0, s1, s2;

  // p2 = P2(x, y);

  std::cout <<  pi_dr.S0() << std::endl;

  // a = pi_dr.A();
  // w3 = pi_dr.omega3();
  // s1 = pi_dr.sigma1();
  //
  // s2 = a - pi_dr.sigma2() + pi_dr.sigma3() + pi_dr.sigma4() + pi_dr.sigma5() - pi_dr.sigma6();

  std::cout << "S0 = " << s0 <<  " S1 = " << s1 << " S2 = " << s2 << " omega3 = " << w3 << std::endl;
  // std::cout << pi_dr.omega12() << std::endl;
  // std::cout << pi_dr.A_cpu() << std::endl;
  // std::cout << "Sigma = " << pi_dr.sigma() << std::endl;
  //
  // std::cout << pi_dr.sigma1()<< std::endl;
  // std::cout << pi_dr.sigma2()<< std::endl;
  // std::cout << pi_dr.sigma3()<< std::endl;
  // std::cout << pi_dr.sigma4()<< std::endl;
  // std::cout << pi_dr.sigma5()<< std::endl;
  // std::cout << pi_dr.sigma6()<< std::endl;


  std::cout << "pi(x) = phi(x, a) - 1 + a - P2(x, a), where a = pi(y)" << std::endl;
  pi = s1 + s2 + w3 - 1 + pi_y - p2;
  std::cout << std::endl;


  return pi;
}
