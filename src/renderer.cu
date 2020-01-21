#include <iostream>
#include <renderer.hpp>
#include <time.h>





__global__ void _render(float* fb, uint width, uint height, 
    const Scene* scene, const Camera* cam, const Sphere* volume) {
    const uint i = threadIdx.x + blockIdx.x * blockDim.x;
    const uint j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= width || j >= height) return;

    const uint pixel = (j * width + i) * 3;
    const glm::vec2 uv(float(i) / float(width), float(j) / float(height));
    const Ray ray = cam->ray(uv);
    const glm::vec3 color = scene->colorAt(ray, volume);
    // float t = volume->intersect(ray);
    // const glm::vec3 color(t, t, t);
    fb[pixel + 0] = color.x;
    fb[pixel + 1] = color.y;
    fb[pixel + 2] = color.z;
}

Renderer::Renderer(const uint width, const uint height) :
    width(width), height(height), framebufferLen(width * height * sizeof(float) * 3),
    camera(glm::vec3(0, 0, 0), float(width) / float(height), 1.0f) {
    catchErr(cudaMallocManaged((void **)&framebuffer, framebufferLen));
}

Renderer::~Renderer() {
    catchErr(cudaFree(framebuffer));
}

void Renderer::render(float* dest) {
    printf("Rendering %ux%u image...\n", width, height);
    const uint blockSize = 16;

    //Timing clock
    clock_t start, stop;
    start = clock();

    //Copy memory
    Camera* _camera;
    catchErr(cudaMalloc((void**)&_camera, sizeof(Camera)));
    catchErr(cudaMemcpy(_camera, &this->camera, sizeof(Camera), cudaMemcpyHostToDevice));
    Scene* _scene;
    catchErr(cudaMalloc((void**)&_scene, sizeof(Scene)));
    catchErr(cudaMemcpy(_scene, &this->scene, sizeof(Scene), cudaMemcpyHostToDevice));
    Sphere* _volume;
    Sphere sphere(glm::vec3(0, 0, -4), 1);
    catchErr(cudaMalloc((void**)&_volume, sizeof(Sphere)));
    catchErr(cudaMemcpy(_volume, &sphere, sizeof(Sphere), cudaMemcpyHostToDevice));

    //Render to framebuffer
    dim3 blocks(width / blockSize + 1, height / blockSize + 1);
    dim3 threads(blockSize, blockSize);
    _render<<<blocks, threads>>>(framebuffer, width, height, _scene, _camera, _volume);

    //Catch errors, print time
    catchErr(cudaGetLastError());
    catchErr(cudaDeviceSynchronize());
    stop = clock();
    double seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Finished in %lf seconds\n", seconds);

    //Copy out
    catchErr(cudaMemcpy(
        (void *)dest, (void *)framebuffer, 
        framebufferLen, cudaMemcpyDeviceToHost));
    
    //Free memory
    catchErr(cudaFree(_camera));
    catchErr(cudaFree(_scene));
}

