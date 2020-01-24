cmake_minimum_required(VERSION 3.0)
project(sparse-scene-flow)

# Build Options
option(SHOW_PATH "Show path of libs and headers" ON)
set(OpenCV_DIR "/home/osep/local/share/OpenCV" CACHE PATH "Path to local OpenCV e.g. USER/local/share/OpenCV")

# Configuration
# Compiler Flags
set(CMAKE_CXX_COMPILER_FLAGS "${CMAKE_CXX_COMPILER_FLAGS} -Wall") # Enables all compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")  # Generate position independent code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")  # Enable c++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

find_package(Boost 1.58.0 REQUIRED COMPONENTS program_options filesystem)
find_package(OpenCV 3.2.0 REQUIRED COMPONENTS core highgui imgproc calib3d)
find_package(pybind11 REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include ${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} )

# Include own files
file(GLOB PROJ_SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
include_directories(${CMAKE_SOURCE_DIR}/src)

file(GLOB SFLOW_SRC_FILES ${CMAKE_SOURCE_DIR}/src/sparseflow/*.cpp)
set(SFLOW_SRC_DIR ${CMAKE_SOURCE_DIR}/src/sparseflow)
include_directories(${SFLOW_SRC_DIR})

# Libraries
add_library (viso2
        src/libviso2/matrix.cpp
        src/libviso2/viso.cpp
        src/libviso2/viso_mono.cpp
        src/libviso2/viso_stereo.cpp
        src/libviso2/matcher.cpp
        src/libviso2/reconstruction.cpp
        src/libviso2/triangle.cpp
        src/libviso2/filter.cpp

        include/libviso2/viso.h
        include/libviso2/viso_mono.h
        include/libviso2/viso_stereo.h
        include/libviso2/matcher.h
        include/libviso2/reconstruction.h
        include/libviso2/triangle.h
        include/libviso2/filter.h
        )

add_library (kitti
        src/libkitti/kitti.cpp
        src/libkitti/kitti.h
        )

target_link_libraries(viso2)
target_link_libraries(kitti)

add_subdirectory(apps)

# Debug infos
if(SHOW_PATH)
    message(STATUS "Boost_LIBRARIES: " ${Boost_LIBRARIES})
    message(STATUS ${OpenCV_LIBS})
    message(STATUS "Project Source Dirs: " ${PROJECT_SOURCE_DIR})
    message(STATUS "OpenCV Include Dirs: " ${OpenCV_INCLUDE_DIRS})
    message(STATUS "Eigen Include Dirs: " ${EIGEN3_INCLUDE_DIR})
endif()
