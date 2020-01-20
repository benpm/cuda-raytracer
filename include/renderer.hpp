#pragma once

#include <cuda.h>
#define GLM_FORCE_CUDA
#include <scene.hpp>
#include <camera.hpp>

class Renderer {
private:
    float* framebuffer;
public:
    const uint width;
    const uint height;
    const uint framebufferLen;

    Scene scene;
    Camera camera;

    Renderer(const uint width, const uint height);
    ~Renderer();
    void render(float* dest);
};