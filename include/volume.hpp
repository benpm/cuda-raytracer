#pragma once

#include <ray.hpp>

struct Hit {
    float t;
    glm::vec3 point;
    glm::vec3 normal;
};

class Volume {
public:
    __device__ virtual bool intersect(const Ray& ray, Hit& hit) const = 0;
};

class Sphere : public Volume {
public:
    glm::vec3 pos;
    float radius;

    __device__ Sphere(const glm::vec3& pos, float radius);
    __device__ virtual bool intersect(const Ray& ray, Hit& hit) const;
};