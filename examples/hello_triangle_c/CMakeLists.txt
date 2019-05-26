cmake_minimum_required(VERSION 3.11b)

project(hello_triangle)

set(TARGET_NAME hello_triangle)

set(BACKEND_VULKAN "vulkan")
set(BACKEND_METAL "metal")
set(BACKEND_DX12 "dx12")
set(AVAILABLE_BACKENDS ${BACKEND_VULKAN} ${BACKEND_METAL} ${BACKEND_DX12})

if(NOT DEFINED BACKEND)
    message(FATAL_ERROR "Backend has not been set, please do so by using the -DBACKEND= argument. \
    Available backends: ${AVAILABLE_BACKENDS}")
endif()

add_executable(hello_triangle main.c)

if(MSVC)
    add_definitions(-DWGPU_TARGET=WGPU_TARGET_WINDOWS)
    target_compile_options(${TARGET_NAME} PRIVATE /W4)
    set(GLFW_LIBRARY glfw3)
    set(OS_LIBRARIES "userenv" "ws2_32" "Dwmapi" "dbghelp")    
    if("${BACKEND}" STREQUAL "${BACKEND_DX12}")
        list(APPEND OS_LIBRARIES "d3dcompiler" "D3D12" "DXGI")
    endif()
elseif(APPLE)
    add_definitions(-DWGPU_TARGET=WGPU_TARGET_MACOS)
    set(OS_LIBRARIES "-framework Cocoa" "-framework CoreVideo" "-framework IOKit" "-framework QuartzCore")
    target_compile_options(${TARGET_NAME} PRIVATE -x objective-c)
    set(GLFW_LIBRARY glfw)
else(MSVC)
    add_definitions(-DWGPU_TARGET=WGPU_TARGET_LINUX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -pedantic)
    set(GLFW_LIBRARY glfw)
endif(MSVC)

find_package(glfw3)

find_library(WGPU_LIBRARY wgpu_native
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug"
)

target_link_libraries(${TARGET_NAME} ${GLFW_LIBRARY} ${WGPU_LIBRARY} ${OS_LIBRARIES})
