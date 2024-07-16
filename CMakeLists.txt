cmake_minimum_required(VERSION 3.10)
# 设置项目名称和版本
project(t5_engine VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE DEBUG)
if(WIN32)
    # 如果是 Windows
    message("当前操作系统为 Windows")
    set(SENTENCEPIECE_DIR D:/3rdparty/sentencepiece-0.2.0-win64)
    link_directories(${SENTENCEPIECE_DIR}/lib)
    include_directories(${SENTENCEPIECE_DIR}/include)
    SET(ONNXRUNTIME_DIR D:/3rdparty/onnxruntime-win-x64-1.16.1)
    link_directories(${ONNXRUNTIME_DIR}/lib)
    include_directories(${ONNXRUNTIME_DIR}/include)

elseif(UNIX AND NOT APPLE)
    # 如果是 Linux
    message("当前操作系统为 Linux")
    # set(CUDA_DIR /usr/local/cuda)
    # link_directories(${CUDA_DIR}/lib64)
    # include_directories(${CUDA_DIR}/include)    
    # set(TENSORRT_DIR /usr/local/tensorrt)
    # link_directories(${TENSORRT_DIR}/lib)
    # include_directories(${TENSORRT_DIR}/include)
    SET(ONNXRUNTIME_DIR /workspace/onnxruntime-linux-x64-1.14.0)
    link_directories(${ONNXRUNTIME_DIR}/lib)
    include_directories(${ONNXRUNTIME_DIR}/include)

    set(SENTENCEPIECE_DIR /workspace/sentencepiece-0.2.0-Linux)
    link_directories(${SENTENCEPIECE_DIR}/lib)
    include_directories(${SENTENCEPIECE_DIR}/include)
else()
    message(FATAL_ERROR "不支持的操作系统类型")
endif()
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")


file(GLOB FRONT_SOURCES
    main.cpp 
    *.cpp
    *.h
)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog_include)
add_executable(t5_engine ${FRONT_SOURCES})
# target_link_libraries(t5_engine PRIVATE cudart nvinfer) # 添加cuda和tensorrt库
# target_link_libraries(t5_engine PRIVATE cudart) # 添加cuda和tensorrt库
# target_link_libraries(t5_engine PRIVATE nvinfer) # 添加cuda和tensorrt库
# target_link_libraries(t5_engine PRIVATE D:/3rdparty/sentencepiece-0.2.0-win64/lib/libsentencepiece.a) # 添加cuda和tensorrt库
target_link_libraries(t5_engine PRIVATE onnxruntime libsentencepiece.a) # 添加cuda和tensorrt库
message("cmake done!")


