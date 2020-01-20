#include <iostream>
#include <renderer.hpp>
#include <time.h>

//CUDA error checker macro
#define catchErr(val) checkCUDA( (val), #val, __FILE__, __LINE__ )
//Print CUDA error
void checkCUDA(cudaError_t result, char const *const func, const char *const file, int const line);



__global__ void _render(float* fb, uint width, uint height) {
    const uint i = threadIdx.x + blockIdx.x * blockDim.x;
    const uint j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= width || j >= height) return;

    const uint pixel = (j * width + i) * 3;
    const float u = float(i) / float(width);
    const float v = float(j) / float(height);
    fb[pixel + 0] = 1.0;
    fb[pixel + 1] = u;
    fb[pixel + 2] = v;
}

Renderer::Renderer(const uint width, const uint height) :
    width(width), height(height), framebufferLen(width * height * sizeof(float) * 3) {
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

    //Render to framebuffer
    dim3 blocks(width / blockSize + 1, height / blockSize + 1);
    dim3 threads(blockSize, blockSize);
    _render<<<blocks, threads>>>(framebuffer, width, height);

    //Catch errors, print time
    catchErr(cudaGetLastError());
    catchErr(cudaDeviceSynchronize());
    stop = clock();
    double seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Finished in %lf seconds\n", seconds);

    catchErr(cudaMemcpy(
        (void *)dest, (void *)framebuffer, 
        framebufferLen, cudaMemcpyDeviceToHost));
}

void checkCUDA(cudaError_t result, char const *const func, 
    const char *const file, int const line) {
    if (result) {
        std::cerr << "! CUDA ERROR: " << cudaGetErrorName(result) << std::endl;
        std::cerr << "\t" << cudaGetErrorString(result) << std::endl;
        std::cerr << "\tat " << file << ":" << line << " " << func << std::endl;
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}