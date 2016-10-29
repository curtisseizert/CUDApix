#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <CUDASieve/cudasieve.hpp>

#include "phi.cuh"
#include "S0.cuh"
#include "sieve/lpf_mu.cuh"

__constant__ uint16_t d_small[8] = {2, 3, 5, 7, 11, 13, 17, 19};
__constant__ uint32_t d_wheel[8] = {2, 6, 30, 210, 2310, 30030, 510510, 9699690};
__constant__ uint32_t d_totient[8] = {1, 2, 8, 48, 480, 5760, 92160, 1658880};

const uint16_t h_small[8] = {2, 3, 5, 7, 11, 13, 17, 19};
const uint32_t h_wheel[8] = {2, 6, 30, 210, 2310, 30030, 510510, 9699690};
const uint32_t h_totient[8] = {1, 2, 8, 48, 480, 5760, 92160, 1658880};

uint64_t S0(uint64_t x, uint64_t y, uint16_t c)
{
  uint16_t threads = 256;
  uint64_t sum = 0;
  int64_t * d_quot = NULL, * h_quot = NULL, h_sum = 0;
  int8_t * d_mu;
  uint32_t arraySize = 1+ y/2;
  uint32_t * d_phi, * d_lpf;

  d_mu = gen_d_mu((uint64_t)0, y);
  d_lpf = gen_d_lpf(0u, (uint32_t)y);

  Phi phi(h_wheel[c], h_small[c]);
  d_phi = phi.generateRange(h_wheel[c], (uint32_t) c);

  cudaMalloc(&d_quot, arraySize * sizeof(int64_t));
  cudaMallocHost(&h_quot, arraySize * sizeof(int64_t));
  cudaMemset(d_quot, arraySize * sizeof(int64_t), 0);

  S0kernel<<<1 + arraySize/threads, threads>>>(d_mu, d_quot, d_lpf, d_phi, x, y, c);

  cudaDeviceSynchronize();
  sum += thrust::reduce(thrust::device, d_quot, d_quot + (y / 2));

  cudaMemcpy(h_quot, d_quot, arraySize * sizeof(int64_t), cudaMemcpyDeviceToHost);

  for(uint32_t i = 0; i < y/2; i++){
    //  std::cout << 2 * i + 1 << "\t" << h_quot[i] << "\t" << std::endl;
     h_sum += h_quot[i];
   }

  cudaFree(d_mu);
  cudaFree(d_quot);
  cudaFree(d_phi);
  cudaFree(d_lpf);

  std::cout << h_sum << std::endl;

  return sum;
}


uint64_t h_S0(uint64_t x, uint64_t y, uint16_t c)
{
  int64_t h_sum = 0;
  int8_t * h_mu;
  uint32_t * h_lpf;
  c--;

  h_mu = gen_h_mu((uint64_t)0, y);
  h_lpf = gen_h_lpf(0u, (uint32_t)y);

  Phi phi(h_wheel[c], 19);

  for(uint32_t n = 1; n < y; n += 2){
    if(h_lpf[n/2] > h_small[c]){
      int64_t phi_0 = h_totient[c] * ((x/n) / h_wheel[c]);
      phi_0 += phi.phi((x/n) % h_wheel[c], c);
      phi_0 *= h_mu[n/2];
      // int64_t phi_0 = h_mu[n/2] * phi.phiRec(x/n, c);
      std::cout << n << "\t" << (x/n) << "\t" << phi_0 << "\t" << std::endl;
      h_sum += phi_0;
    }
  }

  std::cout << h_sum << std::endl;

  return (uint64_t)h_sum;
}

PhiRec::PhiRec(uint64_t max_p_a)
{
  size_t len;
  h_primes = CudaSieve::getHostPrimes(0, std::max(150u,(unsigned int)max_p_a), len, 0);
}

uint64_t PhiRec::phiRec(uint64_t x, uint64_t a)
{
  uint64_t temp;
  phi(x, a, 1);
  temp = sum;
  sum = 0;
  return temp;
}

void PhiRec::phi(uint64_t x, uint64_t a, int8_t sign)
{
  while(x >= h_primes[a]){
    if(a == 0) break;
    a--;
    phi(x/h_primes[a], a, -sign);
  }
  if(a == 0) sum += sign * x;
  else sum += sign;
}

__global__ void S0kernel(int8_t * d_mu, int64_t * d_quot, uint32_t * d_lpf, uint32_t * d_phi, uint64_t x, uint64_t y, uint16_t c)
{
  uint64_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
  uint64_t n = 2 * tidx + 1;
  if(n <= y && n >= d_small[c]){
    uint64_t m = x / n;
    d_quot[tidx] = m / d_wheel[c];
    d_quot[tidx] *= d_totient[c];
    int64_t phidx = (m % d_wheel[c]) / 2;
    int64_t phi = d_phi[phidx];
    d_quot[tidx] += phi;
    d_quot[tidx] *= d_mu[tidx];
    if(d_lpf[tidx] <= d_small[c]) d_quot[tidx] = 0;
  }
}
