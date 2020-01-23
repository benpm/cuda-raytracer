#pragma once

#include <material.hpp>

class Material;

struct Hit {
    float t = 0;
    glm::vec3 point;
    glm::vec3 normal;
};

class Volume {
public:
    __device__ virtual Material* getMat() const = 0;
    __device__ virtual bool intersect(const Ray& ray, float minT, float maxT, Hit& hit) const = 0;
};

class Sphere : public Volume {
public:
    Material* mat;
    glm::vec3 pos;
    float radius;

    __device__ Sphere(Material* mat, const glm::vec3& pos, float radius);
    __device__ virtual Material* getMat() const;
    __device__ virtual bool intersect(const Ray& ray, float minT, float maxT, Hit& hit) const;
};

class Plane : public Volume {
public:
    Material* mat;
    float height;

    __device__ Plane(Material* mat, float height);
    __device__ virtual Material* getMat() const;
    __device__ virtual bool intersect(const Ray& ray, float minT, float maxT, Hit& hit) const;
};