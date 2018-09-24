#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

typedef enum {
  WGPUPowerPreference_Default = 0,
  WGPUPowerPreference_LowPower = 1,
  WGPUPowerPreference_HighPerformance = 2,
} WGPUPowerPreference;

typedef uint32_t WGPUId;

typedef WGPUId WGPUDeviceId;

typedef WGPUId WGPUAdapterId;

typedef struct {
  bool anisotropic_filtering;
} WGPUExtensions;

typedef struct {
  WGPUExtensions extensions;
} WGPUDeviceDescriptor;

typedef WGPUId WGPUComputePassId;

typedef WGPUId WGPURenderPassId;

typedef WGPUId WGPUCommandBufferId;

typedef WGPUId WGPUInstanceId;

typedef WGPUId WGPUShaderModuleId;

typedef struct {
  const uint8_t *bytes;
  uintptr_t length;
} WGPUByteArray;

typedef struct {
  WGPUByteArray code;
} WGPUShaderModuleDescriptor;

typedef struct {
  WGPUPowerPreference power_preference;
} WGPUAdapterDescriptor;

WGPUDeviceId wgpu_adapter_create_device(WGPUAdapterId adapter_id, WGPUDeviceDescriptor desc);

WGPUComputePassId wgpu_command_buffer_begin_compute_pass(void);

WGPURenderPassId wgpu_command_buffer_begin_render_pass(WGPUCommandBufferId command_buffer);

WGPUInstanceId wgpu_create_instance(void);

WGPUShaderModuleId wgpu_device_create_shader_module(WGPUDeviceId device_id,
                                                    WGPUShaderModuleDescriptor desc);

WGPUAdapterId wgpu_instance_get_adapter(WGPUInstanceId instance_id, WGPUAdapterDescriptor desc);
