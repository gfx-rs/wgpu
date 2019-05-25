cmake_minimum_required(VERSION 3.11b)

project(hello_triangle)

set(TARGET_NAME hello_triangle)

add_executable(hello_triangle main.c)

if(MSVC)
    add_definitions(-DWGPU_TARGET=WGPU_TARGET_WINDOWS)
    target_compile_options(${TARGET_NAME} PRIVATE /W4)
    set(GLFW_LIBRARY glfw3)
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

find_package(glfw3 REQUIRED)

find_library(WGPU_LIBRARY wgpu_native
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug"
)

target_link_libraries(${TARGET_NAME} ${GLFW_LIBRARY} ${WGPU_LIBRARY} ${OS_LIBRARIES})
