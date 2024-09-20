cmake_minimum_required(VERSION 3.10)

# Set the project name (optional)
project(mlp LANGUAGES C CUDA)

# Conditional compilation for CUDA
option(USE_CUDA "Build with CUDA support" ON)

if(USE_CUDA)
    message(STATUS "Building MLP with CUDA support")
    add_library(mlp SHARED mlp.cu)
    target_compile_definitions(mlp PRIVATE USE_CUDA)
    set_target_properties(mlp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # Set CUDA architectures to your GPU's compute capability
    set_property(TARGET mlp PROPERTY CUDA_ARCHITECTURES 61)
    # Set the library paths for linking
    link_directories(/usr/lib/x86_64-linux-gnu)
    # Suppress incompatible library warnings
    #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xlinker=-ignore_incompatible")
else()
    message(STATUS "Building MLP without CUDA support")
    add_library(mlp SHARED mlp_cpu.c)
    # Find OpenMP
    find_package(OpenMP)
    if(OPENMP_FOUND)
        target_link_libraries(mlp PUBLIC OpenMP::OpenMP_C)
    endif()
    # Link the math library
    target_link_libraries(mlp PUBLIC m)
endif()

# Include directories
target_include_directories(mlp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
