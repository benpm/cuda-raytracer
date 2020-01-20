#include <camera.hpp>

Camera::Camera(const glm::vec3& origin, float width, float height) :
    origin(origin), size(width, height) {
}

Ray Camera::ray(const glm::vec2& uv) const {
    return Ray(origin, glm::vec3(uv * size - size / 2.0f, -1.0f));
}