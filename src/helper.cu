#include <helper.hpp>

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

__device__ glm::vec3 randVecUnitSphere(curandState *randState) {
    glm::vec3 point;
    do {
        point = 2.0f * RANDVEC3(randState) - glm::vec3(1, 1, 1);
    } while (glm::dot(point, point) >= 1.0f);
    return point;
}

__device__ glm::vec3 reflect(const glm::vec3 &a, const glm::vec3 &b) {
    return a - 2.0f * glm::dot(a, b) * b;
}