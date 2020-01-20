#pragma once

#include <ray.hpp>

class Scene {
public:


    __host__ __device__ Scene();
    __host__ __device__ glm::vec3 colorAt(const Ray& ray) const;
};