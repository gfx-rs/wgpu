#ifndef WGPU_H
#define WGPU_H
#include "wgpu.h"
#endif

#include "framework.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BINDINGS_LENGTH (1)
#define BIND_GROUP_LAYOUTS_LENGTH (1)

int main(
    int argc,
    char *argv[]) {

    if (argc != 5) {
        printf("You must pass 4 positive integers!");
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

    WGPUAdapterId adapter = wgpu_request_adapter(NULL);
    WGPUDeviceId device = wgpu_adapter_request_device(adapter, NULL);

	uint8_t *staging_memory;

    WGPUBufferId staging_buffer = wgpu_device_create_buffer_mapped(device,
            &(WGPUBufferDescriptor){
                .size = size,
				.usage = WGPUBufferUsage_MAP_READ},
            &staging_memory);

	memcpy((uint32_t *) staging_memory, numbers, size);

	wgpu_buffer_unmap(staging_buffer);

    WGPUBufferId storage_buffer = wgpu_device_create_buffer(device,
        &(WGPUBufferDescriptor){
			.size = size,
            .usage = WGPUBufferUsage_STORAGE});

    WGPUBindGroupLayoutId bind_group_layout =
        wgpu_device_create_bind_group_layout(device,
            &(WGPUBindGroupLayoutDescriptor){
                .bindings = &(WGPUBindGroupLayoutBinding){
					.binding = 0,
                    .visibility = WGPUShaderStage_COMPUTE,
                    .ty = WGPUBindingType_StorageBuffer},
                .bindings_length = BINDINGS_LENGTH});

	WGPUBindingResource resource = {
		.tag = WGPUBindingResource_Buffer,
        .buffer = (WGPUBufferBinding){
            .buffer = storage_buffer,
			.size = size,
			.offset = 0}};

    WGPUBindGroupId bind_group = wgpu_device_create_bind_group(device,
            &(WGPUBindGroupDescriptor){.layout = bind_group_layout,
                .bindings = &(WGPUBindGroupBinding){
					.binding = 0,
					.resource = resource},
                .bindings_length = BINDINGS_LENGTH});

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
            .todo = 0
        });

    wgpu_command_encoder_copy_buffer_to_buffer(
        encoder, staging_buffer, 0, storage_buffer, 0, size);

    WGPUComputePassId command_pass =
        wgpu_command_encoder_begin_compute_pass(encoder, NULL);
    wgpu_compute_pass_set_pipeline(command_pass, compute_pipeline);

    wgpu_compute_pass_set_bind_group(command_pass, 0, bind_group, NULL, 0);
    wgpu_compute_pass_dispatch(command_pass, numbers_length, 1, 1);
    wgpu_compute_pass_end_pass(command_pass);

    wgpu_command_encoder_copy_buffer_to_buffer(
        encoder, storage_buffer, 0, staging_buffer, 0, size);

    WGPUQueueId queue = wgpu_device_get_queue(device);

    WGPUCommandBufferId command_buffer = wgpu_command_encoder_finish(encoder, NULL);

    wgpu_queue_submit(queue, &command_buffer, 1);

    wgpu_buffer_map_read_async(staging_buffer, 0, size, read_buffer_map, NULL);

    wgpu_device_poll(device, true);

    return 0;
}
