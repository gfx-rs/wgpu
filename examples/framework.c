#ifndef WGPU_H
#define WGPU_H
#include "wgpu.h"
#endif

#include <stdio.h>
#include <stdlib.h>

WGPUU32Array read_file(const char *name) {
    FILE *file = fopen(name, "rb");
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    unsigned char *bytes = malloc(length);
    fseek(file, 0, SEEK_SET);
    fread(bytes, 1, length, file);
    fclose(file);
    return (WGPUU32Array){
        .bytes = (uint32_t*) bytes,
        .length = length,
    };
}

void read_buffer_map(
    WGPUBufferMapAsyncStatus status,
    const uint8_t *data,
    uint8_t *userdata) {
    (void)userdata;
    if (status == WGPUBufferMapAsyncStatus_Success) {
        uint32_t *times = (uint32_t *) data;
        printf("Times: [%d, %d, %d, %d]",
            times[0],
            times[1],
            times[2],
            times[3]);
    }
}
