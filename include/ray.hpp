#pragma once

#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

class Ray {
public:
    glm::vec3 a, b;

    __host__ __device__ Ray(const glm::vec3& a, const glm::vec3& b);
    __host__ __device__ glm::vec3 pointAtTime(float t) const;
};