#include <stdint.h>
#include <cuda.h>

#pragma once

__global__ void lpf_kernel(uint32_t * d_primeList, uint32_t * d_lpf, uint32_t primeListLength, uint16_t sieveWords, uint32_t bottom);
__global__ void lpf_kernel(uint32_t * d_primeList, uint64_t * d_lpf, uint32_t primeListLength, uint16_t sieveWords, uint64_t bottom);

__global__ void mu_kernel(uint32_t * d_primeList, int8_t * d_mu, uint32_t primeListLength, uint16_t sieveWords, uint32_t bottom);


template <typename T>
T* gen_d_lpf(T bottom, T top);

template <typename T>
int8_t * gen_d_mu(T bottom, T top);
