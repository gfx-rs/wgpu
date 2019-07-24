cmake_minimum_required(VERSION 3.11b)

set(BACKEND_VULKAN "vulkan")
set(BACKEND_METAL "metal")
set(BACKEND_DX11 "dx11")
set(BACKEND_DX12 "dx12")
set(AVAILABLE_BACKENDS 
    ${BACKEND_VULKAN}
    ${BACKEND_METAL}
    ${BACKEND_DX11}
    ${BACKEND_DX12})

if(NOT DEFINED BACKEND OR NOT "${BACKEND}" IN_LIST AVAILABLE_BACKENDS)
    message(FATAL_ERROR "BACKEND invalid or undefined, available backends: ${AVAILABLE_BACKENDS}")
endif()

project(compute)

set(TARGET_NAME compute)

add_executable(compute main.c)

if(MSVC)
    add_definitions(-DWGPU_TARGET=WGPU_TARGET_WINDOWS)
    target_compile_options(${TARGET_NAME} PRIVATE /W4)
    set(OS_LIBRARIES "userenv" "ws2_32" "Dwmapi" "dbghelp")    
    if("${BACKEND}" STREQUAL "${BACKEND_DX11}")
        list(APPEND OS_LIBRARIES "d3dcompiler" "D3D11" "DXGI")
    elseif("${BACKEND}" STREQUAL "${BACKEND_DX12}")
        list(APPEND OS_LIBRARIES "d3dcompiler" "D3D12" "DXGI")
    endif()
elseif(APPLE)
    add_definitions(-DWGPU_TARGET=WGPU_TARGET_MACOS)
    set(OS_LIBRARIES "-framework Cocoa" "-framework CoreVideo" "-framework IOKit" "-framework QuartzCore")
    target_compile_options(${TARGET_NAME} PRIVATE -x objective-c)
else(MSVC)
    add_definitions(-DWGPU_TARGET=WGPU_TARGET_LINUX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -pedantic)
endif(MSVC)

find_package(glfw3 3.3 REQUIRED
    HINTS "$ENV{GLFW3_INSTALL_DIR}"
)

find_library(WGPU_LIBRARY wgpu_native
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug"
)

target_include_directories(${TARGET_NAME} PUBLIC $ENV{GLFW3_INCLUDE_DIR})

target_link_libraries(${TARGET_NAME} glfw ${WGPU_LIBRARY} ${OS_LIBRARIES})