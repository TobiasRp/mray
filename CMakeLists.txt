cmake_minimum_required(VERSION 3.10)

project(mray VERSION 1.0 LANGUAGES CXX)

set(mray_DESCRIPTION "Image-based Visualization of Large Volumetric Data Using Moments")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 14)

option(COMPILER_WARNINGS "Enable ALL compiler warnings" ON)
if (COMPILER_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

option(SANITIZE_MEMORY "Enable compiler sanitizer options to detect memory leaks etc. Only for development! (Can cause issues with CUDA and older Ubuntu versions)" OFF)
option(SANITIZE_THREADS "Enable compiler sanitizer options to detect race conditions. Only for development! (Can cause issues with CUDA and older Ubuntu versions)" OFF)

if (SANITIZE_MEMORY AND SANITIZE_THREADS)
    message(FATAL_ERROR "Memory and thread sanitizers can not be enabled at the same time!")
endif()

if (SANITIZE_MEMORY)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -fsanitize=undefined -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=leak -fsanitize=undefined -fsanitize=address")
endif(SANITIZE_MEMORY)

if (SANITIZE_THREADS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
endif(SANITIZE_THREADS)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -D_GLIBCXX_PARALLEL")
endif()

include(cmake/cuda.cmake)

option (NVTX_TRACING "Enable the NVTX extension" OFF)
if (ENABLE_CUDA AND NVTX_TRACING)
    find_package(NVTX)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_NVTX=${NVTX_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NVTX=${NVTX_VERSION}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_NVTX=${NVTX_VERSION}")
endif()

if (SANITIZE_MEMORY AND ENABLE_CUDA)
    message(FATAL_ERROR "Memory sanitizer and CUDA don't work well together!")
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
    message(STATUS "OpenMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(LZ4 REQUIRED)
include_directories(${LZ4_INCLUDE_DIRS})

find_package(NetCDF REQUIRED)
include_directories(${NetCDF_INCLUDE_DIRS})

find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

set(LINK_LIBRARIES)

if (ENABLE_CUDA)
    file(GLOB_RECURSE SOURCE ${PROJECT_SOURCE_DIR}/src/*)
    list(REMOVE_ITEM SOURCE ${PROJECT_SOURCE_DIR}/src/main.cpp)

    cuda_add_library(mray_LIB SHARED ${SOURCE})
    cuda_add_executable(mray ${PROJECT_SOURCE_DIR}/src/main.cpp)
else()
    file(GLOB_RECURSE SOURCE ${PROJECT_SOURCE_DIR}/src/*.cpp)
    list(REMOVE_ITEM SOURCE ${PROJECT_SOURCE_DIR}/src/main.cpp)

    add_library(mray_LIB SHARED ${SOURCE})
    add_executable(mray ${PROJECT_SOURCE_DIR}/src/main.cpp)
endif()

list(APPEND LINK_LIBRARIES mray_LIB ${netCDF_LIBRARIES}
        ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES} ${LZ4_LIBRARIES} ${NVTX_LIBRARIES})

target_link_libraries(mray ${LINK_LIBRARIES})

option(BUILD_TESTS "Build the tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif()
