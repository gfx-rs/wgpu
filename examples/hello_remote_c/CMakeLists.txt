cmake_minimum_required(VERSION 3.11b)

project(hello_remote)

set(TARGET_NAME hello_remote)

add_executable(hello_remote main.c)

find_package(glfw3)

find_library(WGPU_LIBRARY wgpu_remote
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug"
)

target_link_libraries(${TARGET_NAME} ${WGPU_LIBRARY})
