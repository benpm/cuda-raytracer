#pragma once

#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

class Renderer {
private:
    float* framebuffer;
public:
    const uint width;
    const uint height;
    const uint framebufferLen;

    Renderer(const uint width, const uint height);
    ~Renderer();
    void render(float* dest);
};