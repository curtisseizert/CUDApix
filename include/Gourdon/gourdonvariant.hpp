#include <stdint.h>
#include <uint128_t.cuh>


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

  uint64_t B();

  int64_t sigma();

  uint64_t phi_0();
};

inline void xOverPY(uint64_t * p, uint128_t x, uint64_t y, size_t len);
inline void xOverPY(uint64_t * p, uint64_t x, uint64_t y, size_t len);
inline void xOverPSquared(uint64_t * p, uint128_t x, size_t len);
inline void xOverPSquared(uint64_t * p, uint64_t x, size_t len);
inline void sqrtxOverSqrtp(uint64_t * p, uint64_t sqrtx, size_t len);
inline void squareEach(uint64_t * pi, size_t len);
inline void addToArray(uint64_t * pi, size_t len, uint64_t k);
