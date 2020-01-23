#include <material.hpp>

__device__ MatteMaterial::MatteMaterial(glm::vec3 color) : color(color) {
}

__device__ float MatteMaterial::hit(const Hit& hit, Ray& ray, float energy, curandState* randState) {
    ray.b = hit.normal + randVecUnitSphere(randState);
    ray.a = hit.point;
    return energy * 0.5f;
}