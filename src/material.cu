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

__device__ float schlick(float cosine, float refractiveIndex) {
    float r0 = (1 - refractiveIndex) / (1 + refractiveIndex);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__device__ bool refract(const glm::vec3& v, const glm::vec3& n, float nint, glm::vec3& refracted) {
    glm::vec3 uv = glm::normalize(v);
    float dt = glm::dot(uv, n);
    float discriminant = 1.0f - nint * nint * (1.0f - dt * dt);
    if (discriminant > 0) {
        refracted = nint * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    } else {
        return false;
    }
}

__device__ DielectricMaterial::DielectricMaterial(
    glm::vec3 color, float roughness, float refractiveIndex)
    : color(color), roughness(roughness), refractiveIndex(refractiveIndex) {
}

__device__ void DielectricMaterial::hit(const Hit& hit, Ray& ray, glm::vec3& energy, curandState* randState) {
    glm::vec3 outwardNorm;
    glm::vec3 reflected = reflect(ray.b, hit.normal);
    float nint;
    energy *= color;
    glm::vec3 refracted;
    float reflectProb;
    float cosine;

    if (glm::dot(ray.b, hit.normal) > 0) {
        outwardNorm = -hit.normal;
        nint = refractiveIndex;
        cosine = refractiveIndex * glm::dot(ray.b, hit.normal) / glm::length(ray.b);
    }
    else {
        outwardNorm = hit.normal;
        nint = 1.0f / refractiveIndex;
        cosine = -glm::dot(ray.b, hit.normal) / glm::length(ray.b);
    }

    if (refract(ray.b, outwardNorm, nint, refracted)) {
        reflectProb = schlick(cosine, refractiveIndex);
    }
    else {
        reflectProb = 1.0f;
    }

    if (curand_uniform(randState) < reflectProb) {
        ray.a = hit.point;
        ray.b = reflected;
    } else {
        ray.a = hit.point;
        ray.b = refracted;
    }
}


__device__ TestMaterial::TestMaterial() {
}

__device__ void TestMaterial::hit(const Hit& hit, Ray& ray, glm::vec3& energy, curandState* randState) {
    ray.b = ray.b + RANDVEC3(randState) * 0.05f;
    ray.a = hit.point;
    energy *= 0.95f;
}