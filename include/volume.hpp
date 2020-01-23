#pragma once

#include <ray.hpp>

struct Hit {
    float t = 0;
    glm::vec3 point;
    glm::vec3 normal;
};

class Volume {
public:
    __device__ virtual bool intersect(const Ray& ray, float minT, float maxT, Hit& hit) const = 0;
};

class Sphere : public Volume {
public:
    glm::vec3 pos;
    float radius;

    __device__ Sphere(const glm::vec3& pos, float radius);
    __device__ virtual bool intersect(const Ray& ray, float minT, float maxT, Hit& hit) const;
};

class Plane : public Volume {
public:
    float height;

    __device__ Plane(float height);
    __device__ virtual bool intersect(const Ray& ray, float minT, float maxT, Hit& hit) const;
};