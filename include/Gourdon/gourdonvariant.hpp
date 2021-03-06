#include <stdint.h>
#include <cuda_uint128.h>

#ifndef _GOURDON_64
#define _GOURDON_64

class GourdonVariant64{
private:
  uint64_t maxRange_ = 1ull << 33;

  uint16_t c;
  uint64_t x, y, z, sqrtx, cbrtx, qrtx, sqrtz;
  uint64_t pi_y, pi_cbrtx, pi_sqrtz, pi_qrtx, pi_sqrtx;

  int64_t sigma0();
  int64_t sigma1();
  int64_t sigma2();
  int64_t sigma3();
  int64_t sigma4();
  int64_t sigma5();
  int64_t sigma6();

  int64_t sigma0_cpu();
  int64_t sigma1_cpu();
  int64_t sigma2_cpu();
  int64_t sigma3_cpu();
  int64_t sigma4_cpu();
  int64_t sigma5_cpu();
  int64_t sigma6_cpu();
public:

  static uint64_t piGourdon(uint64_t x, uint64_t y, uint16_t c);

  GourdonVariant64(uint64_t x, uint64_t y, uint16_t c);

  void calculateBounds();
  void calculatePiValues();

  uint64_t pi();

  uint64_t A();
  uint64_t A_large();
  uint64_t checkA();
  uint64_t A_cpu();
  uint64_t A2_cpu();

  uint64_t B();

  uint64_t C();
  uint64_t C_cpu();

  int64_t sigma();
  int64_t sigma_cpu();

  uint64_t phi_0();
};

inline void xOverPY(uint64_t * p, uint128_t x, uint64_t y, size_t len);
inline void xOverPY(uint64_t * p, uint64_t x, uint64_t y, size_t len);
inline void xOverPSquared(uint64_t * p, uint128_t x, size_t len);
inline void xOverPSquared(uint64_t * p, uint64_t x, size_t len);
inline void sqrtxOverSqrtp(uint64_t * p, uint64_t sqrtx, size_t len);
inline void squareEach(uint64_t * pi, size_t len);
inline void addToArray(uint64_t * pi, size_t len, uint64_t k);

#endif
