/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef WGPU_H
#define WGPU_H
#include "wgpu.h"
#endif

#include "framework.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BIND_ENTRIES_LENGTH (1)
#define BIND_GROUP_LAYOUTS_LENGTH (1)

void request_adapter_callback(WGPUAdapterId received, void *userdata) {
    *(WGPUAdapterId*)userdata = received;
}

void read_buffer_map(
    WGPUBufferMapAsyncStatus status,
    const uint8_t *data,
    uint8_t *userdata) {
    (void)userdata;
    if (status == WGPUBufferMapAsyncStatus_Success) {
        uint32_t *times = (uint32_t *) data;
        printf("Times: [%d, %d, %d, %d]\n",
            times[0],
            times[1],
            times[2],
            times[3]);
    }
}

int main(
    int argc,
    char *argv[]) {

    if (argc != 5) {
        printf("You must pass 4 positive integers!\n");
        return 0;
    }

    uint32_t numbers[] = {
        strtoul(argv[1], NULL, 0),
        strtoul(argv[2], NULL, 0),
        strtoul(argv[3], NULL, 0),
        strtoul(argv[4], NULL, 0),
    };

    uint32_t size = sizeof(numbers);

    uint32_t numbers_length = size / sizeof(uint32_t);

    WGPUAdapterId adapter = { 0 };
    wgpu_request_adapter_async(
        NULL,
        2 | 4 | 8,
        request_adapter_callback,
        (void *) &adapter
    );

    WGPUDeviceId device = wgpu_adapter_request_device(adapter, NULL);

    uint8_t *staging_memory;

    WGPUBufferId buffer = wgpu_device_create_buffer_mapped(device,
            &(WGPUBufferDescriptor){
                .label = "buffer",
                .size = size,
				.usage = WGPUBufferUsage_STORAGE | WGPUBufferUsage_MAP_READ},
            &staging_memory);

	memcpy((uint32_t *) staging_memory, numbers, size);

	wgpu_buffer_unmap(buffer);

    WGPUBindGroupLayoutId bind_group_layout =
        wgpu_device_create_bind_group_layout(device,
            &(WGPUBindGroupLayoutDescriptor){
                .label = "bind group layout",
                .entries = &(WGPUBindGroupLayoutEntry){
					.binding = 0,
                    .visibility = WGPUShaderStage_COMPUTE,
                    .ty = WGPUBindingType_StorageBuffer},
                .entries_length = BIND_ENTRIES_LENGTH});

	WGPUBindingResource resource = {
		.tag = WGPUBindingResource_Buffer,
        .buffer = {(WGPUBufferBinding){
            .buffer = buffer,
			.size = size,
			.offset = 0}}};

    WGPUBindGroupId bind_group = wgpu_device_create_bind_group(device,
            &(WGPUBindGroupDescriptor){
                .label = "bind group",
                .layout = bind_group_layout,
                .entries = &(WGPUBindGroupEntry){
					.binding = 0,
					.resource = resource},
                .entries_length = BIND_ENTRIES_LENGTH});

	WGPUBindGroupLayoutId bind_group_layouts[BIND_GROUP_LAYOUTS_LENGTH] = {
        bind_group_layout};

    WGPUPipelineLayoutId pipeline_layout =
            wgpu_device_create_pipeline_layout(device,
                &(WGPUPipelineLayoutDescriptor){
                    .bind_group_layouts = bind_group_layouts,
                    .bind_group_layouts_length = BIND_GROUP_LAYOUTS_LENGTH});

    WGPUShaderModuleId shader_module = wgpu_device_create_shader_module(device,
        &(WGPUShaderModuleDescriptor){
            .code = read_file("./../../data/collatz.comp.spv")});

    WGPUComputePipelineId compute_pipeline =
        wgpu_device_create_compute_pipeline(device,
            &(WGPUComputePipelineDescriptor){
				.layout = pipeline_layout,
                .compute_stage = (WGPUProgrammableStageDescriptor){
                    .module = shader_module,
					.entry_point = "main"
                }});

    WGPUCommandEncoderId encoder = wgpu_device_create_command_encoder(
        device, &(WGPUCommandEncoderDescriptor){
            .label = "command encoder",
        });

    WGPUComputePassId command_pass =
        wgpu_command_encoder_begin_compute_pass(encoder, NULL);
    wgpu_compute_pass_set_pipeline(command_pass, compute_pipeline);

    wgpu_compute_pass_set_bind_group(command_pass, 0, bind_group, NULL, 0);
    wgpu_compute_pass_dispatch(command_pass, numbers_length, 1, 1);
    wgpu_compute_pass_end_pass(command_pass);

    WGPUQueueId queue = wgpu_device_get_default_queue(device);

    WGPUCommandBufferId command_buffer = wgpu_command_encoder_finish(encoder, NULL);

    wgpu_queue_submit(queue, &command_buffer, 1);

    wgpu_buffer_map_read_async(buffer, 0, size, read_buffer_map, NULL);

    wgpu_device_poll(device, true);

    return 0;
}
