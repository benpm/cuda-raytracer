#pragma once

#include <helper.hpp>
#include <ray.hpp>
#include <volume.hpp>

struct Hit;

class Material {
public:
    __device__ virtual float hit(const Hit& hit, Ray& ray, float energy, curandState* randState) = 0;
};

class MatteMaterial : public Material {
public:
    glm::vec3 color;

    __device__ MatteMaterial(glm::vec3 color);
    __device__ virtual float hit(const Hit& hit, Ray& ray, float energy, curandState* randState);
};

class MetallicMaterial : public Material {
public:
    glm::vec3 color;
    float roughness;

    __device__ MetallicMaterial(glm::vec3 color, float roughness);
    __device__ virtual float hit(const Hit& hit, Ray& ray, float energy, curandState* randState);
};