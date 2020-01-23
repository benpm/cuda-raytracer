#include <iostream>
#include <curand_kernel.h>
#include <renderer.hpp>
#include <time.h>



__global__ void _construct(Scene* scene) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        scene->volumes[0] = new Sphere(glm::vec3(-2, 0, -4), 0.5);
        scene->volumes[1] = new Sphere(glm::vec3(0, 0, -8), 0.75);
        scene->volumes[2] = new Sphere(glm::vec3(2, 0, -4), 1);
        scene->volumes[3] = new Plane(-1);
    }
}

__global__ void _render_init(uint width, uint height, curandState *randState) {
    const uint i = threadIdx.x + blockIdx.x * blockDim.x;
    const uint j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= width || j >= height) return;

    const uint pixel = j * width + i;

    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1234, pixel, 0, &randState[pixel]);
}

__global__ void _render(float* fb, uint width, uint height, 
    const Scene* scene, const Camera* cam, curandState *randState) {
    const uint i = threadIdx.x + blockIdx.x * blockDim.x;
    const uint j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= width || j >= height) return;

    const uint pixel = (j * width + i) * 3;
    curandState localRandState = randState[pixel];
    glm::vec3 color(0, 0, 0);
    for (uint s = 0; s < cam->samplesPerPixel; ++s) {
        const glm::vec2 uv(
            (float(i) + curand_uniform(&localRandState)) / float(width), 
            (float(j) + curand_uniform(&localRandState)) / float(height));
        const Ray ray = cam->ray(uv);
        color += scene->colorAt(ray, &localRandState);
    }
    color /= float(cam->samplesPerPixel);
    fb[pixel + 0] = sqrt(color.x);
    fb[pixel + 1] = sqrt(color.y);
    fb[pixel + 2] = sqrt(color.z);
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

    //CUDA Random state
    curandState* randState;
    catchErr(cudaMalloc((void**)&randState, width * height * sizeof(curandState)));
    
    //Construct scene
    Scene scene(4);
    Scene* _scene;
    catchErr(cudaMalloc((void**)&_scene, sizeof(Scene)));
    catchErr(cudaMemcpy(_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice));
    puts("Constructing scene..");
    _construct<<<1, 1>>>(_scene);
    catchErr(cudaGetLastError());
    catchErr(cudaDeviceSynchronize());

    //Copy memory
    Camera* _camera;
    catchErr(cudaMalloc((void**)&_camera, sizeof(Camera)));
    catchErr(cudaMemcpy(_camera, &this->camera, sizeof(Camera), cudaMemcpyHostToDevice));

    //Render to framebuffer
    dim3 blocks(width / blockSize + 1, height / blockSize + 1);
    dim3 threads(blockSize, blockSize);

    puts("Initializing render...");
    _render_init<<<blocks, threads>>>(width, height, randState);
    catchErr(cudaGetLastError());
    catchErr(cudaDeviceSynchronize());

    puts("Rendering...");
    _render<<<blocks, threads>>>(framebuffer, width, height, _scene, _camera, randState);
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

