#pragma once

#include <vector>
#include <volume.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class Scene {
public:
    Volume** volumes;
    size_t capacity;

    __host__ __device__ Scene(size_t capacity);
    __device__ glm::vec3 colorAt(const Ray& ray) const;
};