cmake_minimum_required(VERSION 3.19)
project(mapping_data CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(mapping_data main.cu)

set_target_properties(
        mapping_data
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)