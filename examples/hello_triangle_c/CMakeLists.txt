cmake_minimum_required(VERSION 3.11b)

project(hello_triangle)

set(TARGET_NAME hello_triangle)

add_executable(hello_triangle main.c)

if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE /W4)
else(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -pedantic)
endif(MSVC)

if(APPLE)
    add_compile_definitions(WGPU_TARGET WGPU_TARGET_MACOS)
    set(OS_LIBRARIES "-framework Cocoa" "-framework CoreVideo" "-framework IOKit" "-framework QuartzCore")
    target_compile_options(${TARGET_NAME} PRIVATE -x objective-c)
endif(APPLE)

find_package(glfw3)

find_library(WGPU_LIBRARY wgpu_native
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug"
)

target_link_libraries(${TARGET_NAME} glfw3 ${WGPU_LIBRARY} ${OS_LIBRARIES})
