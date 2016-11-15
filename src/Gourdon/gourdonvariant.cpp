#include <stdint.h>
#include <CUDASieve/cudasieve.hpp>
#include <uint128_t.cuh>

#include "sieve/S2_hard_host.cuh"
#include "Gourdon/gourdonvariant.hpp"

GourdonVariant64::GourdonVariant64(uint64_t x, uint64_t y, uint16_t c)
{
  this->x = x;
  this->y = y;
  this->c = c;
  calculateBounds();
  calculatePiValues();
}

void GourdonVariant64::calculateBounds()
{
  z = x / y;
  sqrtx = sqrt(x);
  cbrtx = cbrt(x);
  qrtx = sqrt(sqrtx);
  sqrtz = sqrt(z);
}

void GourdonVariant64::calculatePiValues()
{
  pi_qrtx = CudaSieve::countPrimes(0ull, qrtx);
  pi_cbrtx = pi_qrtx + CudaSieve::countPrimes(qrtx, cbrtx);
  pi_y = pi_cbrtx + CudaSieve::countPrimes(cbrtx, y);
  pi_sqrtx = pi_y + CudaSieve::countPrimes(y, sqrtx);
  pi_sqrtz = CudaSieve::countPrimes(0ull, sqrtz);
}

uint64_t GourdonVariant64::piGourdon(uint64_t x, uint64_t y, uint16_t c)
{
  GourdonVariant64 * pi_gourdon = new GourdonVariant64(x, y, c);

  uint64_t a, b, s, w, p0;

  std::cout << "starting a" << std::endl;
  //
  // a = pi_gourdon->A_large();
  // std::cout << "A = " << a << "\n" << std::endl;

  // a = pi_gourdon->A();
  // std::cout << "A (gpu) = " << a << "\n" << std::endl;

  a = pi_gourdon->A_cpu();
  std::cout << "A (cpu) = " << a << "\n" << std::endl;

  // b = pi_gourdon->B();
  // std::cout << "B = " << b << "\n" << std::endl;
  //
  //
  // w = S2hardHost::S2hard(x, y, c);
  // std::cout << "Omega = " << w << "\n" << std::endl;
  //
  // s = pi_gourdon->sigma();
  // std::cout << "Sigma = " << s << "\n" << std::endl;
  //
  // p0 = pi_gourdon->phi_0();
  // std::cout << "phi_0 = " << p0 << "\n" << std::endl;


  return a;//- b + p0 + w + s;

}
