cmake_minimum_required(VERSION 3.22.1)
project(lsfg-android LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(THIRD_PARTY "${CMAKE_CURRENT_LIST_DIR}/../../../../third_party" ABSOLUTE)
set(ZEROFG_PREBUILT_SHADERS ON)
set(ZEROFG_VULKAN_HEADERS_DIR "${CMAKE_SYSROOT}/usr/include")
add_subdirectory("${THIRD_PARTY}/zerofg" zerofg)
add_library(volk STATIC "${THIRD_PARTY}/volk/volk.c")
target_include_directories(volk PUBLIC "${THIRD_PARTY}/volk")
target_compile_definitions(volk PUBLIC VK_USE_PLATFORM_ANDROID_KHR=1)

find_library(log-lib log)
find_library(neuralnetworks-lib neuralnetworks)

add_library(lsfg-android SHARED
    lsfg_jni.cpp
    android_vk_session.cpp
    ahb_image_bridge.cpp
    gpu_postprocess.cpp
    lsfg_render_loop.cpp
    nnapi_npu.cpp
    nnapi_postprocess.cpp
    cpu_postprocess.cpp
    crash_reporter.cpp
)

target_link_libraries(lsfg-android
    PRIVATE
        zerofg::zerofg
        volk
        ${log-lib}
        ${neuralnetworks-lib}
        android
)
