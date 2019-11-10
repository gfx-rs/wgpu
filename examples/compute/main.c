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
#define INPUT_LENGTH (4)

typedef enum {
    ApplicationStatus_Initial,
    ApplicationStatus_WaitingForEvent,
    ApplicationStatus_ReceivedAdapter,
    ApplicationStatus_QuitRequested,
} ApplicationStatus;

typedef struct {
    ApplicationStatus status;
    WGPUEventLoopId event_loop;
    uint32_t *input;
    uint32_t input_length;
    uint32_t input_size;
    WGPUAdapterId adapter;
} Application;

void received_adapter(WGPUAdapterId const *adapter, void *userdata) {
    Application *app = (Application *)userdata;
    app->adapter = *adapter;
    app->status = ApplicationStatus_ReceivedAdapter;
}

void get_adapter(Application *app) {
    app->status = ApplicationStatus_WaitingForEvent;
    wgpu_request_adapter_async(
        NULL, app->event_loop, received_adapter, (void *)app);
}

void print_compute_result(
    WGPUBufferMapAsyncStatus status, const uint8_t *data, uint8_t *userdata) {
    Application *app = (Application *)userdata;
    if (status != WGPUBufferMapAsyncStatus_Success) {
        printf("Failed to map buffer");
        exit(1);
    }

    uint32_t *result = (uint32_t *)data;
    printf("Times: [%d, %d, %d, %d]\n", result[0], result[1], result[2],
        result[3]);
    app->status = ApplicationStatus_QuitRequested;
}

void dispatch_compute(Application *app) {
    app->status = ApplicationStatus_WaitingForEvent;
    WGPUDeviceId device = wgpu_adapter_request_device(app->adapter, NULL);

    uint8_t *staging_memory;

    WGPUBufferId buffer = wgpu_device_create_buffer_mapped(device,
        &(WGPUBufferDescriptor){.size = app->input_size,
            .usage = WGPUBufferUsage_STORAGE | WGPUBufferUsage_MAP_READ},
        &staging_memory);

    memcpy((uint32_t *)staging_memory, app->input, app->input_size);

    wgpu_buffer_unmap(buffer);

    WGPUBindGroupLayoutId bind_group_layout =
        wgpu_device_create_bind_group_layout(device,
            &(WGPUBindGroupLayoutDescriptor){
                .bindings = &(WGPUBindGroupLayoutBinding){.binding = 0,
                    .visibility = WGPUShaderStage_COMPUTE,
                    .ty = WGPUBindingType_StorageBuffer},
                .bindings_length = BINDINGS_LENGTH});

    WGPUBindingResource resource = {.tag = WGPUBindingResource_Buffer,
        .buffer = {(WGPUBufferBinding){
            .buffer = buffer, .size = app->input_size, .offset = 0}}};

    WGPUBindGroupId bind_group = wgpu_device_create_bind_group(device,
        &(WGPUBindGroupDescriptor){.layout = bind_group_layout,
            .bindings =
                &(WGPUBindGroupBinding){.binding = 0, .resource = resource},
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
            &(WGPUComputePipelineDescriptor){.layout = pipeline_layout,
                .compute_stage = (WGPUProgrammableStageDescriptor){
                    .module = shader_module, .entry_point = "main"}});

    WGPUCommandEncoderId encoder = wgpu_device_create_command_encoder(
        device, &(WGPUCommandEncoderDescriptor){.todo = 0});

    WGPUComputePassId command_pass =
        wgpu_command_encoder_begin_compute_pass(encoder, NULL);
    wgpu_compute_pass_set_pipeline(command_pass, compute_pipeline);
    wgpu_compute_pass_set_bind_group(command_pass, 0, bind_group, NULL, 0);
    wgpu_compute_pass_dispatch(command_pass, app->input_length, 1, 1);
    wgpu_compute_pass_end_pass(command_pass);

    WGPUQueueId queue = wgpu_device_get_queue(device);

    WGPUCommandBufferId command_buffer =
        wgpu_command_encoder_finish(encoder, NULL);

    wgpu_queue_submit(queue, &command_buffer, 1, app->event_loop);
    wgpu_buffer_map_read_async(buffer, 0, app->input_size, print_compute_result,
        (uint8_t *)app, app->event_loop);
}

void quit(Application *app) {
    wgpu_destroy_event_loop(app->event_loop);
    exit(0);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("You must pass 4 positive integers!\n");
        return 1;
    }

    Application app = {.status = ApplicationStatus_Initial,
        .event_loop = wgpu_create_event_loop(),
        .input =
            (uint32_t[4]){
                strtoul(argv[1], NULL, 0),
                strtoul(argv[2], NULL, 0),
                strtoul(argv[3], NULL, 0),
                strtoul(argv[4], NULL, 0),
            },
        .input_length = INPUT_LENGTH,
        .input_size = INPUT_LENGTH * sizeof(uint32_t)};

    while (true) {
        switch (app.status) {
        case ApplicationStatus_Initial:
            get_adapter(&app);
            break;
        case ApplicationStatus_ReceivedAdapter:
            dispatch_compute(&app);
            break;
        case ApplicationStatus_QuitRequested:
            quit(&app);
            break;
        case ApplicationStatus_WaitingForEvent:
            break;
        }

        wgpu_process_events(app.event_loop);
    }
}
