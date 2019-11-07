#ifndef WGPU_H
#define WGPU_H
#include "wgpu.h"
#endif

#include <stdio.h>
#include <stdlib.h>

WGPUU32Array read_file(const char *name) {
    FILE *file = fopen(name, "rb");
    if (!file) {
        printf("Unable to open %s\n", name);
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    unsigned char *bytes = malloc(length);
    fseek(file, 0, SEEK_SET);
    fread(bytes, 1, length, file);
    fclose(file);
    return (WGPUU32Array){
        .bytes = (uint32_t*) bytes,
        .length = length / 4,
    };
}
