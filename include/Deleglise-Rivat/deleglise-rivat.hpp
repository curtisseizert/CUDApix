#include <stdint.h>
#include <cuda_uint128.h>

#ifndef _DELEGLISE_RIVAT_128
#define _DELEGLISE_RIVAT_128

class deleglise_rivat64{
private:
  uint16_t c;
  uint64_t x, y, z, sqrtx, cbrtx, qrtx, sqrtz;
  uint64_t pi_y, pi_cbrtx, pi_sqrtz, pi_qrtx, pi_sqrtx;

  int64_t sigma1() const;
  int64_t sigma2() const;
  int64_t sigma3() const;
  int64_t sigma4() const;
  int64_t sigma5() const;
  int64_t sigma6() const;

  void calculateBounds();
  void calculatePiValues();

  uint64_t S0();
  uint64_t S1();
  uint64_t S2();
  uint64_t S3();
public:
  static uint64_t pi_deleglise_rivat(uint64_t x, uint64_t y, uint16_t c);

  deleglise_rivat64(uint64_t x, uint64_t y, uint16_t c);


};

class deleglise_rivat128{
private:
  uint16_t c;
  uint128_t x;
  uint64_t y, z, sqrtx, cbrtx, qrtx, sqrtz;
  uint64_t pi_y, pi_cbrtx, pi_sqrtz, pi_qrtx, pi_sqrtx;

  uint128_t sigma() const;
  uint128_t sigma1() const;
  uint128_t sigma2() const;
  uint128_t sigma3() const;
  uint128_t sigma4() const;
  uint128_t sigma5() const;
  uint128_t sigma6() const;

  uint128_t A();
  uint128_t A_cpu();

  uint128_t omega12();
  uint128_t omega3();

  void calculateBounds();
  void calculatePiValues();

  deleglise_rivat128(uint128_t x, uint64_t y, uint16_t c);

public:
  static uint128_t pi_deleglise_rivat(uint128_t x, uint64_t y, uint16_t c);
};

inline void xOverPY(uint64_t * p, uint128_t x, uint64_t y, size_t len);
inline void xOverPY(uint64_t * p, uint64_t x, uint64_t y, size_t len);
inline void xOverPSquared(uint64_t * p, uint128_t x, size_t len);
inline void xOverPSquared(uint64_t * p, uint64_t x, size_t len);
inline void sqrtxOverSqrtp(uint64_t * p, uint64_t sqrtx, size_t len);
inline void squareEach(uint64_t * pi, size_t len);

#endif
