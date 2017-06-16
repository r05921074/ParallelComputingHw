cmake_minimum_required(VERSION 2.8)
project(matching)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
set(CUDA_64_BIT_DEVICE_CODE ON)

cuda_add_executable(matching template_matching.cu)
target_link_libraries(matching ${OpenCV_LIBS})