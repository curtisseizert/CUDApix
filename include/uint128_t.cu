#include <stdint.h>
#include <cuda.h>
#include <math.h>
#ifdef __CUDA_ARCH__
#include <math_functions.h>
#endif

// this dill likely have to exist only as a header in order to allod
// the compiler to do its thing effectively

#ifndef _UINT128_T
#define _UINT128_T

class uint128_t{
private:
  uint64_t lo = 0, hi = 0; // d == most significant bits
public:
  __host__ __device__ uint128_t(){};

  template<typename T>
  __host__ __device__ uint128_t(const T & a){this->lo = a;}


// operator overloading
  template <typename T>
  __host__ __device__ uint128_t & operator=(const T n){this->lo = n; return * this;}

  template <typename T>
  __host__ __device__ friend uint128_t operator+(uint128_t a, const T & b){return add128(a, b);}

  template <typename T>
  __host__ __device__ uint128_t operator+=(const T & b){return add128(*this, b);}

  template <typename T>
  __host__ __device__ uint128_t operator-=(const T & b){return sub128(*this, b);}

  __host__ __device__ uint128_t & operator=(const uint128_t & n);

  __host__ __device__ friend uint64_t operator/(uint128_t x, const uint64_t & v){return div128(x, v);}
  __host__ __device__ friend bool operator<(uint128_t a, uint128_t b){return isLessThan(a, b);}
  __host__ __device__ friend bool operator>(uint128_t a, uint128_t b){return isGreaterThan(a, b);}
  __host__ __device__ friend bool operator<=(uint128_t a, uint128_t b){return isLessThanOrEqual(a, b);}
  __host__ __device__ friend bool operator>=(uint128_t a, uint128_t b){return isGreaterThanOrEqual(a, b);}
  __host__ __device__ friend bool operator==(uint128_t a, uint128_t b){return isEqualTo(a, b);}
  __host__ __device__ friend bool operator!=(uint128_t a, uint128_t b){return isNotEqualTo(a, b);}
  __host__ __device__ uint128_t friend operator-(uint128_t a, uint128_t b){return sub128(a, b);}

// comparisons
  __host__ __device__ static inline bool isLessThan(uint128_t a, uint128_t b);
  __host__ __device__ static inline bool isLessThanOrEqual(uint128_t a, uint128_t b);
  __host__ __device__ static inline bool isGreaterThan(uint128_t a, uint128_t b);
  __host__ __device__ static inline bool isGreaterThanOrEqual(uint128_t a, uint128_t b);
  __host__ __device__ static inline bool isEqualTo(uint128_t a, uint128_t b);
  __host__ __device__ static inline bool isNotEqualTo(uint128_t a, uint128_t b);

// arithmetic
  __host__ __device__ static inline uint128_t add128(uint128_t x, uint128_t y);
  __host__ __device__ static inline uint128_t add128(uint128_t x, uint64_t y);
  __host__ __device__ static inline uint128_t mul128(uint64_t x, uint64_t y);
  __host__ __device__ static inline uint64_t div128(uint128_t x, uint64_t v, uint64_t * r = NULL); // x / v
  __host__ __device__ static inline uint128_t sub128(uint128_t x, uint128_t y); // x - y
  __host__ __device__ uint64_t static inline sqrt(uint128_t & x);

  __host__ int32_t static inline clzll(uint64_t a);


}; // class uint128_t

__host__ __device__ bool uint128_t::isEqualTo(uint128_t a, uint128_t b)
{
  if(a.lo == b.lo && a.hi == b.hi) return 1;
  else return 0;
}

__host__ __device__ bool uint128_t::isNotEqualTo(uint128_t a, uint128_t b)
{
  if(a.lo != b.lo || a.hi != b.hi) return 1;
  else return 0;
}

__host__ __device__ bool uint128_t::isGreaterThan(uint128_t a, uint128_t b)
{
  if(a.hi < b.hi) return 0;
  if(a.hi > b.hi) return 1;
  if(a.lo <= b.lo) return 0;
  else return 1;
}

__host__ __device__ bool uint128_t::isLessThan(uint128_t a, uint128_t b)
{
  if(a.hi < b.hi) return 1;
  if(a.hi > b.hi) return 0;
  if(a.lo < b.lo) return 1;
  else return 0;
}

__host__ __device__ bool uint128_t::isGreaterThanOrEqual(uint128_t a, uint128_t b)
{
  if(a.hi < b.hi) return 0;
  if(a.hi > b.hi) return 1;
  if(a.lo < b.lo) return 0;
  else return 1;
}

__host__ __device__ bool uint128_t::isLessThanOrEqual(uint128_t a, uint128_t b)
{
  if(a.hi < b.hi) return 1;
  if(a.hi > b.hi) return 0;
  if(a.lo <= b.lo) return 1;
  else return 0;
}

__host__ __device__ uint128_t & uint128_t::operator=(const uint128_t & n)
{
  lo = n.lo;
  hi = n.hi;
  return * this;
}

__host__ __device__ uint64_t uint128_t::div128(uint128_t x, uint64_t v, uint64_t * r)
{
  const uint64_t b = 1ull << 32;
  uint64_t  un1, un0,
            vn1, vn0,
            q1, q0,
            un64, un21, un10,
            rhat;
  int s;

  if(x.hi >= v){
    if( r != NULL) *r = (uint64_t) -1;
    return  (uint64_t) -1;
  }

#ifdef __CUDA_ARCH__
  s = __clzll(v);
#else
  s = clzll(v);
#endif

  if(s > 0){
    v = v << s;
    un64 = (x.hi << s) | ((x.lo >> (64 - s)) & (-s >> 31));
    un10 = x.lo << s;
  }else{
    un64 = x.lo | x.hi;
    un10 = x.lo;
  }

  vn1 = v >> 32;
  vn0 = v & 0xffffffff;

  un1 = un10 >> 32;
  un0 = un10 & 0xffffffff;

  q1 = un64/vn1;
  rhat = un64 - q1*vn1;

again1:
  if (q1 >= b || q1*vn0 > b*rhat + un1){
    q1 -= 1;
    rhat = rhat + vn1;
    if(rhat < b) goto again1;
   }

   un21 = un64*b + un1 - q1*v;

   q0 = un21/vn1;
   rhat = un21 - q0*vn1;
again2:
  if(q0 >= b || q0 * vn0 > b*rhat + un0){
    q0 = q0 - 1;
    rhat = rhat + vn1;
    if(rhat < b) goto again2;
  }

  if(r != NULL) *r = (un21*b + un0 - q0*v) >> s;
  return q1*b + q0;
}


__host__ __device__ uint128_t uint128_t::add128(uint128_t x, uint64_t y)
{
  uint128_t res;
#ifdef __CUDA_ARCH__
  asm(  "add.cc.u64    %0 %2 %4\n\t"
        "addc.u64      %1 %3 0\n\t"
        : "=l" (res.lo) "=l" (res.hi)
        : "l" (x.lo) "l" (x.hi)
          "l" (y));
#else
  asm(  "add    %3, %0\n\t"
        "adc    %5, %1\n\t"
        : "=r" (res.lo) "=r" (res.hi)
        : "%0" (x.lo) "%1" (x.hi)
          "r" (y) "r" (0ull)
        :  "cc");
#endif
  return res;
}

__host__ __device__ uint128_t uint128_t::add128(uint128_t x, uint128_t y)
{
   uint128_t res;
 #ifdef __CUDA_ARCH__
   asm( "add.cc.u64    %0 %2 %4;\n\t"
        "addc.u64      %1 %3 %5\n\t"
        : "=l" (res.lo) "=l" (res.hi)
        : "l" (x.lo), "l" (x.hi),
          "l" (y.lo), "l" (y.hi));
#else
    asm("add    %3, %0\n\t"
        "adc    %5, %1\n\t"
        : "=r" (res.lo) "=r" (res.hi)
        : "%0" (x.lo) "%1" (x.hi)
          "r" (y.lo) "r" (y.hi)
        :  "cc");
#endif
    return res;
}

__host__ __device__ uint128_t uint128_t::mul128(uint64_t x, uint64_t y)
{
  uint128_t res;
#ifdef __CUDA_ARCH__
  asm(  "mul.lo.u64    %0 %2 %3\n\t"
        "mul.hi.u64    %1 %2 %3\n\t"
        : "=l" (res.lo) "=l" (res.hi)
        : "l" (x)
          "l" (y));
#else
  asm ("mulq %3\n\t"
   : "=a" (res.lo), "=d" (res.hi)
   : "%0" (x), "rm" (y));
#endif
  return res;
}

__host__ inline int32_t uint128_t::clzll(uint64_t x)
{
  uint64_t res1;
  int32_t res;
  asm("lzcnt %1, %0" : "=l" (res1) : "l" (x));
  res = res1;
  return res;
}

__host__ __device__ uint128_t uint128_t::sub128(uint128_t x, uint128_t y)
{
  uint128_t res;

  res.lo = x.lo - y.lo;
  res.hi = x.hi - y.hi;
  if(x.lo < y.lo) res.hi--;

  return res;
}

__host__ __device__ uint64_t uint128_t::sqrt(uint128_t & x)
{
  int32_t i = 64;

#ifdef __CUDA_ARCH__
  i -= __clzll(x.hi)/2;
#else
  i -= clzll(x.hi)/2;
#endif
  uint128_t cmp;
  uint64_t res = 1ull << i, err;
  do{
    cmp = mul128(res,res);
    if(cmp > x){
      cmp -= x;
      err = cmp/(2 * res + err);
      res -= err;
    }
    if(cmp < x){
      cmp = x - cmp;
      err = cmp/(2 * res + err);
      res += err;
    }
  }while(cmp != x);
  return res;
}

#endif
