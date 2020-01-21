cmake_minimum_required(VERSION 3.0.0)
project("cuda-raytracing" LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
set(INCLUDE "${CMAKE_SOURCE_DIR}/include")
set(SOURCE "${CMAKE_SOURCE_DIR}/src")

set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA)

include_directories(${INCLUDE})
add_executable(main
    "${SOURCE}/main.cu"
    "${SOURCE}/renderer.cu"
    "${SOURCE}/scene.cu"
    "${SOURCE}/ray.cu"
    "${SOURCE}/camera.cu"
    "${SOURCE}/volume.cu"
    "${SOURCE}/helper.cu"
)

# Use C++11 standards
target_compile_features(main PUBLIC cxx_std_11)

# CUDA settings
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# OpenGL Preference
set(OpenGL_GL_PREFERENCE LEGACY)

# glfw3
find_package(glfw3 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})
link_libraries(${GLFW_LIBRARY_DIRS})

# opengl
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

# glew
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

# glm
find_package(glm REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

# link libraries
target_link_libraries(
    main
    glfw
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${EXTRA_LIBS}
)