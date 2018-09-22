#include <cstdint>
#include <cstdlib>

enum class PowerPreference {
  Default = 0,
  LowPower = 1,
  HighPerformance = 2,
};

struct ShaderModuleDescriptor;

using Id = uint32_t;

using DeviceId = Id;

using AdapterId = Id;

struct Extensions {
  bool anisotropic_filtering;
};

struct DeviceDescriptor {
  Extensions extensions;
};

using ComputePassId = Id;

using RenderPassId = Id;

using CommandBufferId = Id;

using InstanceId = Id;

using ShaderModuleId = Id;

struct AdapterDescriptor {
  PowerPreference power_preference;
};

extern "C" {

DeviceId adapter_create_device(AdapterId adapter_id, DeviceDescriptor desc);

ComputePassId command_buffer_begin_compute_pass();

RenderPassId command_buffer_begin_render_pass(CommandBufferId command_buffer);

InstanceId create_instance();

ShaderModuleId device_create_shader_module(DeviceId device_id, ShaderModuleDescriptor desc);

AdapterId instance_get_adapter(InstanceId instance_id, AdapterDescriptor desc);

} // extern "C"
