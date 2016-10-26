#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

#include "S3.cuh"
#include "phi.cuh"
#include "S0.cuh"
#include "uint128_t.cuh"
#include "sieve/lpf_mu.cuh"
#include "P2.cuh"
#include "trivial.cuh"
#include "V.cuh"

int main()
{
    uint128_t x, /*p2,*/ s1_t;
    uint64_t y = 16499370765;

    x = 1;
    x <<= 80;

    std::cout << x << std::endl;

    // p2 = P2(x, y);
    //
    // std::cout << p2 << std::endl;

    s1_t = S1_trivial(x, y);

    std::cout << s1_t << std::endl;

    return 0;
}
