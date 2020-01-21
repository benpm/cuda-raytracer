#pragma once

#include <vector>
#include <volume.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class Scene {
public:
    __host__ __device__ Scene();
    __device__ glm::vec3 colorAt(const Ray& ray, const Sphere* volume) const;
};