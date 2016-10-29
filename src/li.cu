#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_functions.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "li.cuh"

__global__ void nint_li(double x, double * d_li, double precision)
{
  uint64_t totThreads = blockDim.x * gridDim.x;
  uint64_t tidx = threadIdx.x + blockIdx.x*blockDim.x;
  double du = (x - 2) / totThreads;

  d_li[tidx] = 0;

  for(double ddu = 2 + tidx * du; ddu < 2 + (tidx + 1) * du; ddu += precision){
    d_li[tidx] += (1.0 / ((double) log(ddu + precision))) - (1.0 / ((double) log(ddu)));
  }
}

double li(double x)
{
  double sum, * d_li, precision = 1;
  uint32_t blocks = 1u << 20, threads = 1u << 8;
  cudaMalloc(&d_li, blocks*threads*sizeof(double));

  nint_li<<<blocks, threads>>>(x, d_li, precision);

  sum = 1.045163 + thrust::reduce(thrust::device, d_li, d_li + blocks*threads);
  return sum;
}
