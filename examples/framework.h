/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef WGPU_H
#define WGPU_H
#include "wgpu.h"
#endif

WGPUU32Array read_file(const char *name);

void read_buffer_map(
    WGPUBufferMapAsyncStatus status,
    const uint8_t *data,
    uint8_t *userdata);
