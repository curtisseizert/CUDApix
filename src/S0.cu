#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "S0.cuh"
#include "sieve/lpf.cuh"

int8_t mu[3] = {1, -1, -1};

template <typename T>
T S0(T x, T y)
{
  uint16_t threads = 256;
  T sum = 0;
  int64_t * d_quot = NULL, * h_quot = NULL, h_sum = 0;
  int8_t * d_mu;
  uint32_t arraySize = y + (4 * threads) - y % (4 * threads);

  d_mu = get_d_mu((uint64_t)0, y);

  cudaMalloc(&d_quot, arraySize * sizeof(int64_t));
  cudaMallocHost(&h_quot, arraySize * sizeof(int64_t));

  S0kernel<<<arraySize/1024, threads>>>(d_mu, d_quot, x, y);

  for(uint8_t i = 0; i < 3; i++) sum += mu[i] * x / (i + 1);

  cudaDeviceSynchronize();
  sum += thrust::reduce(thrust::device, d_quot + 4, d_quot + y);

  cudaMemcpy(h_quot, d_quot, arraySize * sizeof(int64_t), cudaMemcpyDeviceToHost);

  for(uint32_t i = 0; i < y; i++){
    std::cout << (long long int) h_quot[i] << std::endl;
    h_sum += h_quot[i];
  }

  cudaFree(d_mu);
  cudaFree(d_quot);

  std::cout << h_sum << std::endl;

  return sum;
}

template uint64_t S0<uint64_t>(uint64_t x, uint64_t y);

__global__ void S0kernel(int8_t * d_mu, int64_t * d_quot, uint64_t x, uint64_t y)
{
  int64_t x_int = x, tidx = threadIdx.x + blockIdx.x * blockDim.x;

  d_quot[4 * tidx] = d_mu[2 * tidx] * x_int / (4 * tidx + 1);
  d_quot[4 * tidx + 1] = -d_mu[tidx] * x_int / (4 * tidx + 2);
  d_quot[4 * tidx + 2] = d_mu[2 * tidx + 1] * x_int / (4 * tidx + 3);
  d_quot[4 * tidx + 3] = 0;
}
