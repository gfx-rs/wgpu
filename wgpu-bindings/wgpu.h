#ifdef WGPU_REMOTE
    typedef uint32_t WGPUId;
#else
    typedef void *WGPUId;
#endif

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#define WGPUShaderStageFlags_COMPUTE 4

#define WGPUShaderStageFlags_FRAGMENT 2

#define WGPUShaderStageFlags_NONE 0

#define WGPUShaderStageFlags_VERTEX 1

typedef enum {
  WGPUBindingType_UniformBuffer = 0,
  WGPUBindingType_Sampler = 1,
  WGPUBindingType_SampledTexture = 2,
  WGPUBindingType_StorageBuffer = 3,
} WGPUBindingType;

typedef enum {
  WGPUPowerPreference_Default = 0,
  WGPUPowerPreference_LowPower = 1,
  WGPUPowerPreference_HighPerformance = 2,
} WGPUPowerPreference;

typedef enum {
  WGPUPrimitiveTopology_PointList = 0,
  WGPUPrimitiveTopology_LineList = 1,
  WGPUPrimitiveTopology_LineStrip = 2,
  WGPUPrimitiveTopology_TriangleList = 3,
  WGPUPrimitiveTopology_TriangleStrip = 4,
} WGPUPrimitiveTopology;

typedef enum {
  WGPUShaderStage_Vertex = 0,
  WGPUShaderStage_Fragment = 1,
  WGPUShaderStage_Compute = 2,
} WGPUShaderStage;

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

typedef struct {

} WGPUCommandBufferDescriptor;
typedef WGPUId WGPUBindGroupLayoutId;

typedef uint32_t WGPUShaderStageFlags;

typedef struct {
  uint32_t binding;
  WGPUShaderStageFlags visibility;
  WGPUBindingType ty;
} WGPUBindGroupLayoutBinding;

typedef struct {
  const WGPUBindGroupLayoutBinding *bindings;
  uintptr_t bindings_length;
} WGPUBindGroupLayoutDescriptor;

typedef WGPUId WGPUPipelineLayoutId;

typedef struct {
  const WGPUBindGroupLayoutId *bind_group_layouts;
  uintptr_t bind_group_layouts_length;
} WGPUPipelineLayoutDescriptor;

typedef WGPUId WGPURenderPipelineId;

typedef WGPUId WGPUShaderModuleId;

typedef struct {
  WGPUShaderModuleId module;
  WGPUShaderStage stage;
  const char *entry_point;
} WGPUPipelineStageDescriptor;

typedef WGPUId WGPUBlendStateId;

typedef WGPUId WGPUDepthStencilStateId;

typedef WGPUId WGPUAttachmentStateId;

typedef struct {
  WGPUPipelineLayoutId layout;
  const WGPUPipelineStageDescriptor *stages;
  uintptr_t stages_length;
  WGPUPrimitiveTopology primitive_topology;
  const WGPUBlendStateId *blend_state;
  uintptr_t blend_state_length;
  WGPUDepthStencilStateId depth_stencil_state;
  WGPUAttachmentStateId attachment_state;
} WGPURenderPipelineDescriptor;

typedef struct {
  const uint8_t *bytes;
  uintptr_t length;
} WGPUByteArray;

typedef struct {
  WGPUByteArray code;
} WGPUShaderModuleDescriptor;

typedef WGPUId WGPUQueueId;

typedef struct {
  WGPUPowerPreference power_preference;
} WGPUAdapterDescriptor;

WGPUDeviceId wgpu_adapter_create_device(WGPUAdapterId adapter_id, WGPUDeviceDescriptor _desc);

WGPUComputePassId wgpu_command_buffer_begin_compute_pass(void);

WGPURenderPassId wgpu_command_buffer_begin_render_pass(WGPUCommandBufferId _command_buffer);

WGPUInstanceId wgpu_create_instance(void);

WGPUCommandBufferId wgpu_device_create_command_buffer(WGPUDeviceId device_id,
                                                      WGPUCommandBufferDescriptor desc);
WGPUBindGroupLayoutId wgpu_device_create_bind_group_layout(WGPUDeviceId device_id,
                                                           WGPUBindGroupLayoutDescriptor desc);

WGPUPipelineLayoutId wgpu_device_create_pipeline_layout(WGPUDeviceId device_id,
                                                        WGPUPipelineLayoutDescriptor desc);

WGPURenderPipelineId wgpu_device_create_render_pipeline(WGPUDeviceId device_id,
                                                        WGPURenderPipelineDescriptor desc);

WGPUShaderModuleId wgpu_device_create_shader_module(WGPUDeviceId device_id,
                                                    WGPUShaderModuleDescriptor desc);

WGPUQueueId wgpu_device_get_queue(WGPUDeviceId device_id);

WGPUAdapterId wgpu_instance_get_adapter(WGPUInstanceId instance_id, WGPUAdapterDescriptor desc);

void wgpu_queue_submit(WGPUQueueId queue_id,
                       const WGPUCommandBufferId *command_buffer_ptr,
                       uintptr_t command_buffer_count);
