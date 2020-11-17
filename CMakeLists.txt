cmake_minimum_required(VERSION 3.18)

project(objCDchecker CUDA CXX )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)



option(PRINT_CDTRI_INDEX "print cd triangle index" OFF)

configure_file(
        "${PROJECT_SOURCE_DIR}/config/config.h.in"
        "${PROJECT_SOURCE_DIR}/include/config.h"
)

find_package(CUDA REQUIRED)
find_package(GLUT REQUIRED)
find_package( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

aux_source_directory(./src SRC)

add_executable(objCDchecker ${SRC} )

target_include_directories(objCDchecker PRIVATE include common /opt/cuda/include)

target_link_libraries(objCDchecker glut GL m GLU OpenMP::OpenMP_CXX)

set_target_properties(objCDchecker
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75)