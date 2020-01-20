#include <ray.hpp>

Ray::Ray(const glm::vec3& a, const glm::vec3& b) : a(a), b(b) {
}

glm::vec3 Ray::pointAtTime(float t) const {
    return a + t * b;
}