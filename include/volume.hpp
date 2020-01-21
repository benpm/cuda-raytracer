#pragma once

#include <ray.hpp>

class Volume {
public:
    __device__ virtual float intersect(const Ray& ray) const = 0;
};

class Sphere : public Volume {
public:
    glm::vec3 pos;
    float radius;

    __device__ Sphere(const glm::vec3& pos, float radius);
    __device__ virtual float intersect(const Ray& ray) const;
};