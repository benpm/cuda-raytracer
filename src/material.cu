#include <material.hpp>

__device__ MatteMaterial::MatteMaterial(glm::vec3 color) : color(color) {
}

__device__ float MatteMaterial::hit(const Hit& hit, Ray& ray, float energy, curandState* randState) {
    ray.b = hit.normal + randVecUnitSphere(randState);
    ray.a = hit.point;
    return energy * 0.5f;
}


__device__ MetallicMaterial::MetallicMaterial(glm::vec3 color, float roughness) : color(color), roughness(roughness) {
}

__device__ float MetallicMaterial::hit(const Hit& hit, Ray& ray, float energy, curandState* randState) {
    const glm::vec3 reflected = reflect(glm::normalize(ray.b), hit.normal);
    ray.b = reflected + randVecUnitSphere(randState) * roughness;
    ray.a = hit.point;
    return energy * 0.95f;
}