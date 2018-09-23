#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

typedef enum {
  Default = 0,
  LowPower = 1,
  HighPerformance = 2,
} PowerPreference;

typedef struct ShaderModuleDescriptor ShaderModuleDescriptor;

typedef uint32_t Id;

typedef Id DeviceId;

typedef Id AdapterId;

typedef struct {
  bool anisotropic_filtering;
} Extensions;

typedef struct {
  Extensions extensions;
} DeviceDescriptor;

typedef Id ComputePassId;

typedef Id RenderPassId;

typedef Id CommandBufferId;

typedef Id InstanceId;

typedef Id ShaderModuleId;

typedef struct {
  PowerPreference power_preference;
} AdapterDescriptor;

DeviceId adapter_create_device(AdapterId adapter_id, DeviceDescriptor desc);

ComputePassId command_buffer_begin_compute_pass(void);

RenderPassId command_buffer_begin_render_pass(CommandBufferId command_buffer);

InstanceId create_instance(void);

ShaderModuleId device_create_shader_module(DeviceId device_id, ShaderModuleDescriptor desc);

AdapterId instance_get_adapter(InstanceId instance_id, AdapterDescriptor desc);
