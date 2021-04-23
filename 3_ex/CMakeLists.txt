cmake_minimum_required(VERSION 3.19)
project(3_ex CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(3_ex main.cu)

set_target_properties(
        3_ex
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)