# CUDA Raytracer
A simple raytracer using CUDA and OpenGL. Raytracing is performed by a CUDA kernel,
and drawing is done with OpenGL.

![Screenshot](screenshot.png)

## Features
- Draws result to window

## Dependencies
- CUDA 10
- OpenGL 4
- glm
- GLFW 3
- glew
- CMake

## Building and Running
*Tested on Ubuntu 19.10*

1. Install dependencies: `nvidia-cuda-toolkit libglm-dev libglfw3-dev libglew-dev cmake`
2. `mkdir build && cd build`
3. `cmake ..`
4. `make`
5. `./main`
