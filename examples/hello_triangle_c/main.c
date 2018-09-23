#include <stdio.h>
#include "./../../wgpu-bindings/wgpu.h"

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
    /*WGPUShaderModuleDescriptor vs_desc = {
        .code = "",
    };
    WGPUShaderModuleId _vs = wgpu_device_create_shader_module(device, vs_desc);
    */
    return 0;
}
