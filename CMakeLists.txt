cmake_minimum_required(VERSION 2.6)			

project(FAST)						

set(CPACK_GENERATOR "ZIP")				
include(CPack)						

find_package(OpenCV REQUIRED)				
find_package(CUDA)					
if (NOT CUDA_FOUND)					
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include/ ${CUDA_INCLUDE_DIRS})
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_75,code=sm_75")
set(cuda_sources
  src/fast_kernel.cu
  src/fast_kernel.h
  src/orb.cpp
  src/orb.h
  src/structures.h
  src/utils.h
)  

set(sources
  src/main.cpp
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

#SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(FAST ${cuda_sources} ${sources} OPTIONS)


set_target_properties(FAST PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(FAST ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
