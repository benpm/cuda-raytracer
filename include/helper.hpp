#pragma once

#include <iostream>
#include <cuda.h>
#include <glm/glm.hpp>
#include <curand_kernel.h>

//CUDA error checker macro
#define catchErr(val) checkCUDA( (val), #val, __FILE__, __LINE__ )
//Random vector
#define RANDVEC3(R) glm::vec3(curand_uniform((R)),curand_uniform((R)),curand_uniform((R)))

//Print CUDA error
void checkCUDA(cudaError_t result, char const *const func, 
    const char *const file, int const line);
//Random vector in unit sphere
__device__ glm::vec3 randVecUnitSphere(curandState *randState);
//Reflect ray
__device__ glm::vec3 reflect(const glm::vec3 &a, const glm::vec3 &b);