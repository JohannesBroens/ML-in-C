# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(MLinC LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 99)

# Conditional compilation for CUDA
option(USE_CUDA "Build with CUDA support" ON)

# Add subdirectories
add_subdirectory(models/mlp)

# Add the executable
add_executable(main main.c data_loader.c)

# Link the mlp library and math library
target_link_libraries(main PRIVATE mlp m)

# Include directories
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(USE_CUDA)
    message(STATUS "Building with CUDA support")
    target_compile_definitions(main PRIVATE USE_CUDA)
else()
    message(STATUS "Building without CUDA support")
    # Find OpenMP
    find_package(OpenMP)
    if(OPENMP_FOUND)
        target_link_libraries(main PUBLIC OpenMP::OpenMP_C)
    endif()
endif()
