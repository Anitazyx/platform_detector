cmake_minimum_required(VERSION 3.5)
# project name
PROJECT(test)

# find opencv in system 
set(OpenCV_DIR /home/yangye/tensorrt_apps/TarTensorrt/opencv-3.4.9/build)
find_package(OpenCV REQUIRED)
# find opencv with fixed path
# find_package(OpenCV PATHS /home/yangye/tensorrt_apps/TarTensorrt/opencv-3.4.9 NO_DEFAULT_PATH)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test platform_detector.cpp)
# 链接OpenCV库
target_link_libraries(test ${OpenCV_LIBS})
