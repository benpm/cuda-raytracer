#include <scene.hpp>

Scene::Scene(size_t capacity) : capacity(capacity) {
    catchErr(cudaMalloc((void**)&this->volumes, capacity * sizeof(Volume *)));
}

__device__ glm::vec3 Scene::colorAt(const Ray& ray) const {
    Hit hit;
    for (size_t i = 0; i < capacity; ++i) {
        if (volumes[i]->intersect(ray, hit)) {
            return 0.5f * (hit.normal + glm::vec3(1, 1, 1));
        }
    }
    if (ray.b.y < 0) {
        return glm::vec3(0.5, 0.5, 0.5);
    }
    const glm::vec3 dir = glm::normalize(ray.b);
    const float t = 0.5 * (dir.y + 1.0);
    return (1.0f - t) * glm::vec3(1, 1, 1) + t * glm::vec3(0.5, 0.7, 1);
}
