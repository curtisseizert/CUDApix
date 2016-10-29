#include <stdint.h>

uint64_t S0(uint64_t x, uint64_t y, uint16_t c);
uint64_t h_S0(uint64_t x, uint64_t y, uint16_t c);


__global__ void S0kernel(int8_t * d_mu, int64_t * d_quot, uint32_t * d_lpf, uint32_t * d_phi, uint64_t x, uint64_t y, uint16_t c);

class PhiRec{
private:
  uint64_t * h_primes, sum = 0;
  void phi(uint64_t, uint64_t, int8_t);
public:
  PhiRec(uint64_t);
  uint64_t phiRec(uint64_t, uint64_t);
};
