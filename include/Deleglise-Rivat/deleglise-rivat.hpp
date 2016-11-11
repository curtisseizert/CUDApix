#include <stdint.h>
#include <uint128_t.cuh>

class deleglise_rivat64{
private:
  uint16_t c;
  uint64_t x, y, z, sqrtx, cbrtx, qrtx, sqrtz;
  uint64_t pi_y, pi_cbrtx, pi_sqrtz, pi_qrtx, pi_sqrtx;

  int64_t sigma1();
  int64_t sigma2();
  int64_t sigma3();
  int64_t sigma4();
  int64_t sigma5();
  int64_t sigma6();

  void calculateBounds();
  void calculatePiValues();

  uint64_t S1();
  uint64_t S2();
public:
  static uint64_t pi_deleglise_rivat(uint64_t x, uint64_t y, uint16_t c);

  deleglise_rivat64(uint64_t x, uint64_t y, uint16_t c);


};

inline void xOverPY(uint64_t * p, uint128_t x, uint64_t y, size_t len);
inline void xOverPY(uint64_t * p, uint64_t x, uint64_t y, size_t len);
inline void xOverPSquared(uint64_t * p, uint128_t x, size_t len);
inline void xOverPSquared(uint64_t * p, uint64_t x, size_t len);
inline void sqrtxOverSqrtp(uint64_t * p, uint64_t sqrtx, size_t len);
inline void squareEach(uint64_t * pi, size_t len);
