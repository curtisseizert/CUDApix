#include <stdint.h>
#include <uint128_t.cuh>
#include <cuda.h>

inline void xOverPY(uint64_t * p, uint128_t x, uint64_t y, size_t len);
inline void xOverPY(uint64_t * p, uint64_t x, uint64_t y, size_t len);
inline void xOverPSquared(uint64_t * p, uint128_t x, size_t len);
inline void xOverPSquared(uint64_t * p, uint64_t x, size_t len);
inline void sqrtxOverSqrtp(uint64_t * p, uint64_t sqrtx, size_t len);
inline void squareEach(uint64_t * pi, size_t len);
