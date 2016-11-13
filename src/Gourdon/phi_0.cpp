#include <stdint.h>

#include "ordinary.cuh"
#include "Gourdon/gourdonvariant.hpp"

uint64_t GourdonVariant64::phi_0()
{
  return Ordinary(x, y, c);
}
