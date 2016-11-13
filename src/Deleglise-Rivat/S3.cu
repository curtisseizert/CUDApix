#include <stdint.h>
#include <iostream>
#include "Deleglise-Rivat/deleglise-rivat.hpp"
#include "sieve/S2_hard_host.cuh"

uint64_t deleglise_rivat64::S3()
{
  return S2hardHost::S2hard(x, y, c, qrtx);
}
