#include <volume.hpp>

__device__ Sphere::Sphere(const glm::vec3& pos, float radius) : radius(radius), pos(pos) {
}

__device__ bool Sphere::intersect(const Ray& ray, Hit& hit) const {
    const glm::vec3 oc = ray.a - pos;
    const float a = glm::dot(ray.b, ray.b);
    const float b = glm::dot(oc, ray.b);
    const float c = glm::dot(oc, oc) - radius * radius;
    const float discriminant = (b * b) - (a * c);
    if (discriminant > 0) {
        float t = (-b - sqrt(discriminant)) / a;
        if (t > 0) {
            hit.t = t;
            hit.point = ray.pointAtTime(t);
            hit.normal = glm::normalize((hit.point - pos) / radius);
            return true;
        }
        t = (-b + sqrt(discriminant)) / a;
        if (t > 0) {
            hit.t = t;
            hit.point = ray.pointAtTime(t);
            hit.normal = glm::normalize((hit.point - pos) / radius);
            return true;
        }
    }
    return false;
}

__device__ Plane::Plane(float height) : height(height) {
}

__device__ bool Plane::intersect(const Ray& ray, Hit& hit) const {
    hit.t = (height - ray.a.y) / ray.b.y;
    if (hit.t > 0) {
        hit.t = -ray.a.y / ray.b.y;
        hit.point = ray.pointAtTime(hit.t);
        hit.normal = glm::vec3(0, 1, 0);
        return true;
    }

    return false;
}