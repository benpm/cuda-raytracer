#pragma once

#include <vector>
#include <volume.hpp>
#include <curand_kernel.h>

class Scene {
public:
    Volume** volumes;
    size_t capacity;

    __host__ Scene(size_t capacity);
    __device__ glm::vec3 colorAt(const Ray& ray, curandState *randState) const;
};