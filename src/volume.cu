#include <volume.hpp>

Sphere::Sphere(const glm::vec3& pos, float radius) : radius(radius), pos(pos) {
}

__device__ float Sphere::intersect(const Ray& ray) const {
    const glm::vec3 oc = ray.a - this->pos;
    const float a = glm::dot(ray.b, ray.b);
    const float b = 2.0f * glm::dot(oc, ray.b);
    const float c = glm::dot(oc, oc) - radius * radius;
    return (b * b) - (4 * a * c);
}