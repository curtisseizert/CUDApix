#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include <algorithm>

#include "li.cuh"
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
    uint64_t x = pow(10, 10);
    // uint64_t y = 10000;
    // uint64_t s3;

    // s3 = S3(x, y, 6);
    //
    // std::cout << (int64_t)s3 << std::endl;
    //
    double lix, xd = (double) x;

    lix = li(x);

    std::cout << lix << std::endl;

    return 0;
}
