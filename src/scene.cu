#include <scene.hpp>

Scene::Scene() {}

glm::vec3 Scene::colorAt(const Ray& ray) const {
    const glm::vec3 dir = glm::normalize(ray.b);
    const float t = 0.5 * (dir.y + 1.0);
    return (1.0f - t) * glm::vec3(1, 1, 1) + t * glm::vec3(0.5, 0.7, 1);
}
