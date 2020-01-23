#include <material.hpp>

/* Matte Material */

__device__ MatteMaterial::MatteMaterial(glm::vec3 color) : color(color) {
}

__device__ void MatteMaterial::hit(const Hit& hit, Ray& ray, glm::vec3& energy, curandState* randState) {
    ray.b = hit.normal + randVecUnitSphere(randState);
    ray.a = hit.point;
    energy *= color * 0.5f * color;
}


/* Metallic Material */

__device__ MetallicMaterial::MetallicMaterial(glm::vec3 color, float roughness) : color(color), roughness(roughness) {
}

__device__ void MetallicMaterial::hit(const Hit& hit, Ray& ray, glm::vec3& energy, curandState* randState) {
    const glm::vec3 reflected = reflect(glm::normalize(ray.b), hit.normal);
    ray.b = reflected + randVecUnitSphere(randState) * roughness;
    ray.a = hit.point;
    energy *= color * 0.95f * color;
}


/* Dielectric Material */

// __device__ refract(const glm::vec3& )

__device__ DielectricMaterial::DielectricMaterial(
    glm::vec3 color, float roughness, float refractiveIndex)
    : color(color), roughness(roughness), refractiveIndex(refractiveIndex) {
}

__device__ void DielectricMaterial::hit(const Hit& hit, Ray& ray, glm::vec3& energy, curandState* randState) {
    const glm::vec3 reflected = reflect(glm::normalize(ray.b), hit.normal);
    ray.b = reflected + randVecUnitSphere(randState) * roughness;
    ray.a = hit.point;
    energy *= color * 0.95f;
}


__device__ TestMaterial::TestMaterial() {
}

__device__ void TestMaterial::hit(const Hit& hit, Ray& ray, glm::vec3& energy, curandState* randState) {
    ray.b = ray.b + RANDVEC3(randState) * 0.05f;
    ray.a = hit.point;
    energy *= 0.95f;
}