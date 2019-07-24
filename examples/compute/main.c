#include "./../../ffi/wgpu.h"
#include <stdio.h>
#include <stdlib.h>

#define BINDINGS_LENGTH (1)
#define BIND_GROUP_LAYOUTS_LENGTH (1)

WGPUByteArray read_file(const char *name) {
    FILE *file = fopen(name, "rb");
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    unsigned char *bytes = malloc(length);
    fseek(file, 0, SEEK_SET);
    fread(bytes, 1, length, file);
    fclose(file);
    return (WGPUByteArray){
        .bytes = bytes,
        .length = length,
    };
}

void read_buffer_map(
    WGPUBufferMapAsyncStatus status, 
    const uint8_t *data, 
    uint8_t *userdata) {
    if (status == WGPUBufferMapAsyncStatus_Success) {
        uint32_t *times = (uint32_t *) data;
        printf("Times: [%d, %d, %d, %d]",
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

    WGPUInstanceId instance = wgpu_create_instance();

    WGPUAdapterId adapter = wgpu_instance_get_adapter(instance,
        &(WGPUAdapterDescriptor){
            .power_preference = WGPUPowerPreference_LowPower,
        });

    WGPUDeviceId device = wgpu_adapter_request_device(adapter,
        &(WGPUDeviceDescriptor){
            .extensions = {
				.anisotropic_filtering = false,
            },
        });

	uint8_t *staging_memory;

    WGPUBufferId staging_buffer = wgpu_device_create_buffer_mapped(device,
        &(WGPUBufferDescriptor){
			.size = size,
            .usage = WGPUBufferUsage_MAP_READ | WGPUBufferUsage_TRANSFER_DST |
                WGPUBufferUsage_TRANSFER_SRC},
            &staging_memory);

	memcpy((uint32_t *) staging_memory, numbers, size);
	
	wgpu_buffer_unmap(staging_buffer);	

    WGPUBufferId storage_buffer = wgpu_device_create_buffer(device,
        &(WGPUBufferDescriptor){
			.size = size,
            .usage = WGPUBufferUsage_STORAGE | WGPUBufferUsage_TRANSFER_DST |
                WGPUBufferUsage_TRANSFER_SRC});

    WGPUBindGroupLayoutId bind_group_layout =
        wgpu_device_create_bind_group_layout(device,
            &(WGPUBindGroupLayoutDescriptor){
                .bindings = &(WGPUBindGroupLayoutBinding){
					.binding = 0,
                    .visibility = WGPUShaderStage_COMPUTE,
                    .ty = WGPUBindingType_StorageBuffer
                },
				.bindings_length = BINDINGS_LENGTH
            }
        );

	WGPUBindingResource resource = {
        .tag = WGPUBindingResource_Buffer,
        .buffer = (WGPUBufferBinding){
			.buffer = storage_buffer, 
			.size = size, 
			.offset = 0
		}
	};

    WGPUBindGroupId bind_group = wgpu_device_create_bind_group(device,
        &(WGPUBindGroupDescriptor){
            .layout = bind_group_layout,
            .bindings =
                &(WGPUBindGroupBinding){
                    .binding = 0, 
                    .resource = resource
            },
            .bindings_length = BINDINGS_LENGTH
        }
    );

	WGPUBindGroupLayoutId bind_group_layouts[BIND_GROUP_LAYOUTS_LENGTH] = {
        bind_group_layout
    };

    WGPUPipelineLayoutId pipeline_layout =
        wgpu_device_create_pipeline_layout(device,
            &(WGPUPipelineLayoutDescriptor){
                .bind_group_layouts = bind_group_layouts,
                .bind_group_layouts_length = BIND_GROUP_LAYOUTS_LENGTH
		});

    WGPUShaderModuleId shader_module = wgpu_device_create_shader_module(device,
        &(WGPUShaderModuleDescriptor){
            .code = read_file("./../../data/collatz.comp.spv")
        }
    );

    WGPUComputePipelineId compute_pipeline =
        wgpu_device_create_compute_pipeline(device,
            &(WGPUComputePipelineDescriptor){
				.layout = pipeline_layout,
                .compute_stage = (WGPUPipelineStageDescriptor){
                    .module = shader_module,
					.entry_point = "main"
                }
            }
        );

    WGPUCommandEncoderId encoder = wgpu_device_create_command_encoder(
        device, &(WGPUCommandEncoderDescriptor){
            .todo = 0
        }
    );

    wgpu_command_buffer_copy_buffer_to_buffer(
        encoder, staging_buffer, 0, storage_buffer, 0, size);

    WGPUComputePassId command_pass =
        wgpu_command_encoder_begin_compute_pass(encoder);
    wgpu_compute_pass_set_pipeline(command_pass, compute_pipeline);

    wgpu_compute_pass_set_bind_group(command_pass, 0, bind_group, NULL, 0);
    wgpu_compute_pass_dispatch(command_pass, numbers_length, 1, 1);
    wgpu_compute_pass_end_pass(command_pass);

    wgpu_command_buffer_copy_buffer_to_buffer(
        encoder, storage_buffer, 0, staging_buffer, 0, size);

    WGPUQueueId queue = wgpu_device_get_queue(device);

    WGPUCommandBufferId command_buffer = wgpu_command_encoder_finish(encoder);

    wgpu_queue_submit(queue, &command_buffer, 1);

    wgpu_buffer_map_read_async(staging_buffer, 0, size, read_buffer_map, NULL);

    wgpu_device_poll(device, true);

    return 0;
}