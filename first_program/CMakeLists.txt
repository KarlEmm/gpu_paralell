cmake_minimum_required(VERSION 3.19)
project(first_program CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(first_program main.cu)

set_target_properties(
        first_program
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)