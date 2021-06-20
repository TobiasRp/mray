find_package(CUDA)

option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" FALSE)
option(CUDA_ENABLE_CUPTI_INSTRUMENTATION "enable CUPTI instrumentation" FALSE)

set(CUDA_64_BIT_DEVICE_CODE ON)

set(CUDA_NVCC_FLAGS "-use_fast_math;-ftz=true;-fmad=true;-prec-div=false;-prec-sqrt=false;-std=c++14")

if(${CUDA_FOUND})
if(${CMAKE_VERSION} VERSION_GREATER "3.7.0")

    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
    LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

else()
    option(CUDA_BUILD_CC30 "Build with compute capability 3.0 support" FALSE)
    option(CUDA_BUILD_CC50 "Build with compute capability 5.0 support" TRUE)
    option(CUDA_BUILD_CC60 "Build with compute capability 6.0 support" FALSE)

    message("Cannot autodetect CUDA hardware on cmake version older than 3.7, Please specify your CUDA compute capability! (CUDA_BUILD_CC)")
    if(CUDA_BUILD_CC30)
    	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_30,code=sm_30;")
    endif()
    if(CUDA_BUILD_CC50)
    	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_50,code=sm_50;")
    endif()
    if(CUDA_BUILD_CC60)
    	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode=arch=compute_60,code=sm_60;")
    endif()
endif()
endif()

if(CUDA_BUILD_INFO)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-keep;--ptxas-options=-v;-lineinfo")
endif()

option(ENABLE_CUDA "Enable CUDA" ${CUDA_FOUND})

if(ENABLE_CUDA)
    add_definitions(-DCUDA_SUPPORT)
endif()
