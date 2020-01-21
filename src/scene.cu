#include <scene.hpp>

Scene::Scene() {}

__device__ glm::vec3 Scene::colorAt(const Ray& ray, Volume** volumes, size_t nvolumes) const {
    Hit hit;
    for (size_t i = 0; i < nvolumes; ++i) {
        if (volumes[i]->intersect(ray, hit)) {
            return 0.5f * (hit.normal + glm::vec3(1, 1, 1));
        }
    }
    const glm::vec3 dir = glm::normalize(ray.b);
    const float t = 0.5 * (dir.y + 1.0);
    return (1.0f - t) * glm::vec3(1, 1, 1) + t * glm::vec3(0.5, 0.7, 1);
}
