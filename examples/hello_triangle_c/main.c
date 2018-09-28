#include <stdio.h>
#include "./../../wgpu-bindings/wgpu.h"

WGPUByteArray read_file(const char *name)
{
    FILE *file = fopen(name, "rb");
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    unsigned char *bytes = malloc(length);
    fseek(file, 0, SEEK_SET);
    fread(bytes, 1, length, file);
    fclose(file);
    WGPUByteArray ret = {
        .bytes = bytes,
        .length = length,
    };
    return ret;
}

int main()
{
    WGPUInstanceId instance = wgpu_create_instance();
    WGPUAdapterDescriptor adapter_desc = {
        .power_preference = WGPUPowerPreference_LowPower,
    };
    WGPUAdapterId adapter = wgpu_instance_get_adapter(instance, adapter_desc);
    WGPUDeviceDescriptor device_desc = {
        .extensions = {
            .anisotropic_filtering = false,
        },
    };
    WGPUDeviceId device = wgpu_adapter_create_device(adapter, device_desc);
    WGPUShaderModuleDescriptor vs_desc = {
        .code = read_file("./../data/hello_triangle.vert.spv"),
    };
    WGPUShaderModuleId _vs = wgpu_device_create_shader_module(device, vs_desc);
    WGPUShaderModuleDescriptor fs_desc = {
        .code = read_file("./../data/hello_triangle.frag.spv"),
    };
    WGPUShaderModuleId _fs = wgpu_device_create_shader_module(device, fs_desc);

    WGPUCommandBufferDescriptor cmd_buf_desc = {
    };
    WGPUCommandBufferId cmd_buf = wgpu_device_create_command_buffer(device, cmd_buf_desc);
    WGPUQueueId queue = wgpu_device_get_queue(device);
    wgpu_queue_submit(queue, &cmd_buf, 1);
    return 0;
}
