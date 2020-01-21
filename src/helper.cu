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