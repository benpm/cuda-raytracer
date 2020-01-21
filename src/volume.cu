#include <volume.hpp>

__device__ Sphere::Sphere(const glm::vec3& pos, float radius) : radius(radius), pos(pos) {
}

__device__ bool Sphere::intersect(const Ray& ray, Hit& hit) const {
    const glm::vec3 oc = ray.a - this->pos;
    const float a = glm::dot(ray.b, ray.b);
    const float b = 2.0f * glm::dot(oc, ray.b);
    const float c = glm::dot(oc, oc) - radius * radius;
    const float discriminant = (b * b) - (4 * a * c);
    if (discriminant < 0) {
        return false;
    } else {
        const float t = (-b - sqrt(discriminant)) / (2.0f * a);
        hit.t = t;
        hit.point = ray.pointAtTime(t);
        hit.normal = glm::normalize(hit.point - pos);
        return true;
    }
}