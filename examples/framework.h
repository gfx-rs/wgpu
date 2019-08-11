#ifndef WGPU_H
#define WGPU_H
#include "wgpu.h"
#endif

WGPUU32Array read_file(const char *name);

void read_buffer_map(
    WGPUBufferMapAsyncStatus status, 
    const uint8_t *data, 
    uint8_t *userdata);
