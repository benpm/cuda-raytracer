#pragma once

#include <ray.hpp>

class Camera {
public:
    glm::vec3 origin;
    glm::vec2 size;
    const uint samplesPerPixel = 1024;

    __host__ __device__ Camera(const glm::vec3& origin, float width, float height);
    __host__ __device__ Ray ray(const glm::vec2& uv) const;
};