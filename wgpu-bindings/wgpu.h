#ifdef WGPU_REMOTE
    typedef uint32_t WGPUId;
#else
    typedef void *WGPUId;
#endif

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

typedef enum {
  WGPUBindingType_UniformBuffer = 0,
  WGPUBindingType_Sampler = 1,
  WGPUBindingType_SampledTexture = 2,
  WGPUBindingType_StorageBuffer = 3,
} WGPUBindingType;

typedef enum {
  WGPUBlendFactor_Zero = 0,
  WGPUBlendFactor_One = 1,
  WGPUBlendFactor_SrcColor = 2,
  WGPUBlendFactor_OneMinusSrcColor = 3,
  WGPUBlendFactor_SrcAlpha = 4,
  WGPUBlendFactor_OneMinusSrcAlpha = 5,
  WGPUBlendFactor_DstColor = 6,
  WGPUBlendFactor_OneMinusDstColor = 7,
  WGPUBlendFactor_DstAlpha = 8,
  WGPUBlendFactor_OneMinusDstAlpha = 9,
  WGPUBlendFactor_SrcAlphaSaturated = 10,
  WGPUBlendFactor_BlendColor = 11,
  WGPUBlendFactor_OneMinusBlendColor = 12,
} WGPUBlendFactor;

typedef enum {
  WGPUBlendOperation_Add = 0,
  WGPUBlendOperation_Subtract = 1,
  WGPUBlendOperation_ReverseSubtract = 2,
  WGPUBlendOperation_Min = 3,
  WGPUBlendOperation_Max = 4,
} WGPUBlendOperation;

typedef enum {
  WGPUCompareFunction_Never = 0,
  WGPUCompareFunction_Less = 1,
  WGPUCompareFunction_Equal = 2,
  WGPUCompareFunction_LessEqual = 3,
  WGPUCompareFunction_Greater = 4,
  WGPUCompareFunction_NotEqual = 5,
  WGPUCompareFunction_GreaterEqual = 6,
  WGPUCompareFunction_Always = 7,
} WGPUCompareFunction;

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

typedef enum {
  WGPUStencilOperation_Keep = 0,
  WGPUStencilOperation_Zero = 1,
  WGPUStencilOperation_Replace = 2,
  WGPUStencilOperation_Invert = 3,
  WGPUStencilOperation_IncrementClamp = 4,
  WGPUStencilOperation_DecrementClamp = 5,
  WGPUStencilOperation_IncrementWrap = 6,
  WGPUStencilOperation_DecrementWrap = 7,
} WGPUStencilOperation;

typedef enum {
  WGPUTextureDimension_D1,
  WGPUTextureDimension_D2,
  WGPUTextureDimension_D3,
} WGPUTextureDimension;

typedef enum {
  WGPUTextureFormat_R8g8b8a8Unorm = 0,
  WGPUTextureFormat_R8g8b8a8Uint = 1,
  WGPUTextureFormat_B8g8r8a8Unorm = 2,
  WGPUTextureFormat_D32FloatS8Uint = 3,
} WGPUTextureFormat;

typedef enum {
  WGPUTextureViewDimension_D1,
  WGPUTextureViewDimension_D2,
  WGPUTextureViewDimension_D2Array,
  WGPUTextureViewDimension_Cube,
  WGPUTextureViewDimension_CubeArray,
  WGPUTextureViewDimension_D3,
} WGPUTextureViewDimension;

typedef struct WGPURenderPassDescriptor_TextureViewId WGPURenderPassDescriptor_TextureViewId;

typedef struct WGPUTextureUsageFlags WGPUTextureUsageFlags;

typedef WGPUId WGPUDeviceId;

typedef WGPUId WGPUAdapterId;

typedef struct {
  bool anisotropic_filtering;
} WGPUExtensions;

typedef struct {
  WGPUExtensions extensions;
} WGPUDeviceDescriptor;

typedef WGPUId WGPUComputePassId;

typedef WGPUId WGPUCommandBufferId;

typedef WGPUId WGPURenderPassId;

typedef WGPUId WGPUInstanceId;

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

typedef WGPUId WGPUBlendStateId;

typedef struct {
  WGPUBlendFactor src_factor;
  WGPUBlendFactor dst_factor;
  WGPUBlendOperation operation;
} WGPUBlendDescriptor;

typedef uint32_t WGPUColorWriteFlags;

typedef struct {
  bool blend_enabled;
  WGPUBlendDescriptor alpha;
  WGPUBlendDescriptor color;
  WGPUColorWriteFlags write_mask;
} WGPUBlendStateDescriptor;

typedef struct {

} WGPUCommandBufferDescriptor;

typedef WGPUId WGPUDepthStencilStateId;

typedef struct {
  WGPUCompareFunction compare;
  WGPUStencilOperation stencil_fail_op;
  WGPUStencilOperation depth_fail_op;
  WGPUStencilOperation pass_op;
} WGPUStencilStateFaceDescriptor;

typedef struct {
  bool depth_write_enabled;
  WGPUCompareFunction depth_compare;
  WGPUStencilStateFaceDescriptor front;
  WGPUStencilStateFaceDescriptor back;
  uint32_t stencil_read_mask;
  uint32_t stencil_write_mask;
} WGPUDepthStencilStateDescriptor;

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

typedef struct {
  WGPUTextureFormat format;
  uint32_t samples;
} WGPUAttachment;

typedef struct {
  const WGPUAttachment *color_attachments;
  uintptr_t color_attachments_length;
  const WGPUAttachment *depth_stencil_attachment;
} WGPUAttachmentsState;

typedef struct {
  WGPUPipelineLayoutId layout;
  const WGPUPipelineStageDescriptor *stages;
  uintptr_t stages_length;
  WGPUPrimitiveTopology primitive_topology;
  WGPUAttachmentsState attachments_state;
  const WGPUBlendStateId *blend_states;
  uintptr_t blend_states_length;
  WGPUDepthStencilStateId depth_stencil_state;
} WGPURenderPipelineDescriptor;

typedef struct {
  const uint8_t *bytes;
  uintptr_t length;
} WGPUByteArray;

typedef struct {
  WGPUByteArray code;
} WGPUShaderModuleDescriptor;

typedef WGPUId WGPUTextureId;

typedef struct {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
} WGPUExtent3d;

typedef struct {
  WGPUExtent3d size;
  uint32_t array_size;
  WGPUTextureDimension dimension;
  WGPUTextureFormat format;
  WGPUTextureUsageFlags usage;
} WGPUTextureDescriptor;

typedef WGPUId WGPUQueueId;

typedef struct {
  WGPUPowerPreference power_preference;
} WGPUAdapterDescriptor;

typedef WGPUId WGPUTextureViewId;

typedef uint32_t WGPUTextureAspectFlags;

typedef struct {
  WGPUTextureFormat format;
  WGPUTextureViewDimension dimension;
  WGPUTextureAspectFlags aspect;
  uint32_t base_mip_level;
  uint32_t level_count;
  uint32_t base_array_layer;
  uint32_t array_count;
} WGPUTextureViewDescriptor;

#define WGPUBufferUsageFlags_INDEX (WGPUBufferUsageFlags){ .bits = 16 }

#define WGPUBufferUsageFlags_MAP_READ (WGPUBufferUsageFlags){ .bits = 1 }

#define WGPUBufferUsageFlags_MAP_WRITE (WGPUBufferUsageFlags){ .bits = 2 }

#define WGPUBufferUsageFlags_NONE (WGPUBufferUsageFlags){ .bits = 0 }

#define WGPUBufferUsageFlags_STORAGE (WGPUBufferUsageFlags){ .bits = 128 }

#define WGPUBufferUsageFlags_TRANSFER_DST (WGPUBufferUsageFlags){ .bits = 8 }

#define WGPUBufferUsageFlags_TRANSFER_SRC (WGPUBufferUsageFlags){ .bits = 4 }

#define WGPUBufferUsageFlags_UNIFORM (WGPUBufferUsageFlags){ .bits = 64 }

#define WGPUBufferUsageFlags_VERTEX (WGPUBufferUsageFlags){ .bits = 32 }

#define WGPUColorWriteFlags_ALL (WGPUColorWriteFlags){ .bits = 15 }

#define WGPUColorWriteFlags_ALPHA (WGPUColorWriteFlags){ .bits = 8 }

#define WGPUColorWriteFlags_BLUE (WGPUColorWriteFlags){ .bits = 4 }

#define WGPUColorWriteFlags_COLOR (WGPUColorWriteFlags){ .bits = 7 }

#define WGPUColorWriteFlags_GREEN (WGPUColorWriteFlags){ .bits = 2 }

#define WGPUColorWriteFlags_RED (WGPUColorWriteFlags){ .bits = 1 }

#define WGPUColor_BLACK (WGPUColor){ .r = 0, .g = 0, .b = 0, .a = 1 }

#define WGPUColor_BLUE (WGPUColor){ .r = 0, .g = 0, .b = 1, .a = 1 }

#define WGPUColor_GREEN (WGPUColor){ .r = 0, .g = 1, .b = 0, .a = 1 }

#define WGPUColor_RED (WGPUColor){ .r = 1, .g = 0, .b = 0, .a = 1 }

#define WGPUColor_TRANSPARENT (WGPUColor){ .r = 0, .g = 0, .b = 0, .a = 0 }

#define WGPUColor_WHITE (WGPUColor){ .r = 1, .g = 1, .b = 1, .a = 1 }

#define WGPUShaderStageFlags_COMPUTE (WGPUShaderStageFlags){ .bits = 4 }

#define WGPUShaderStageFlags_FRAGMENT (WGPUShaderStageFlags){ .bits = 2 }

#define WGPUShaderStageFlags_VERTEX (WGPUShaderStageFlags){ .bits = 1 }

#define WGPUTextureAspectFlags_COLOR (WGPUTextureAspectFlags){ .bits = 1 }

#define WGPUTextureAspectFlags_DEPTH (WGPUTextureAspectFlags){ .bits = 2 }

#define WGPUTextureAspectFlags_STENCIL (WGPUTextureAspectFlags){ .bits = 4 }

#define WGPUTextureUsageFlags_NONE (WGPUTextureUsageFlags){ .bits = 0 }

#define WGPUTextureUsageFlags_OUTPUT_ATTACHMENT (WGPUTextureUsageFlags){ .bits = 16 }

#define WGPUTextureUsageFlags_PRESENT (WGPUTextureUsageFlags){ .bits = 32 }

#define WGPUTextureUsageFlags_SAMPLED (WGPUTextureUsageFlags){ .bits = 4 }

#define WGPUTextureUsageFlags_STORAGE (WGPUTextureUsageFlags){ .bits = 8 }

#define WGPUTextureUsageFlags_TRANSFER_DST (WGPUTextureUsageFlags){ .bits = 2 }

#define WGPUTextureUsageFlags_TRANSFER_SRC (WGPUTextureUsageFlags){ .bits = 1 }

#define WGPUTrackPermit_EXTEND (WGPUTrackPermit){ .bits = 1 }

#define WGPUTrackPermit_REPLACE (WGPUTrackPermit){ .bits = 2 }

WGPUDeviceId wgpu_adapter_create_device(WGPUAdapterId adapter_id,
                                        const WGPUDeviceDescriptor *_desc);

WGPUComputePassId wgpu_command_buffer_begin_compute_pass(WGPUCommandBufferId command_buffer_id);

WGPURenderPassId wgpu_command_buffer_begin_render_pass(WGPUCommandBufferId command_buffer_id,
                                                       WGPURenderPassDescriptor_TextureViewId desc);

WGPUCommandBufferId wgpu_compute_pass_end_pass(WGPUComputePassId pass_id);

WGPUInstanceId wgpu_create_instance(void);

WGPUBindGroupLayoutId wgpu_device_create_bind_group_layout(WGPUDeviceId device_id,
                                                           const WGPUBindGroupLayoutDescriptor *desc);

WGPUBlendStateId wgpu_device_create_blend_state(WGPUDeviceId _device_id,
                                                const WGPUBlendStateDescriptor *desc);

WGPUCommandBufferId wgpu_device_create_command_buffer(WGPUDeviceId device_id,
                                                      const WGPUCommandBufferDescriptor *_desc);

WGPUDepthStencilStateId wgpu_device_create_depth_stencil_state(WGPUDeviceId _device_id,
                                                               const WGPUDepthStencilStateDescriptor *desc);

WGPUPipelineLayoutId wgpu_device_create_pipeline_layout(WGPUDeviceId device_id,
                                                        const WGPUPipelineLayoutDescriptor *desc);

WGPURenderPipelineId wgpu_device_create_render_pipeline(WGPUDeviceId device_id,
                                                        const WGPURenderPipelineDescriptor *desc);

WGPUShaderModuleId wgpu_device_create_shader_module(WGPUDeviceId device_id,
                                                    const WGPUShaderModuleDescriptor *desc);

WGPUTextureId wgpu_device_create_texture(WGPUDeviceId device_id, const WGPUTextureDescriptor *desc);

WGPUQueueId wgpu_device_get_queue(WGPUDeviceId device_id);

WGPUAdapterId wgpu_instance_get_adapter(WGPUInstanceId instance_id,
                                        const WGPUAdapterDescriptor *desc);

void wgpu_queue_submit(WGPUQueueId queue_id,
                       const WGPUCommandBufferId *command_buffer_ptr,
                       uintptr_t command_buffer_count);

WGPUCommandBufferId wgpu_render_pass_end_pass(WGPURenderPassId pass_id);

WGPUTextureViewId wgpu_texture_create_default_texture_view(WGPUTextureId texture_id);

WGPUTextureViewId wgpu_texture_create_texture_view(WGPUTextureId texture_id,
                                                   const WGPUTextureViewDescriptor *desc);

void wgpu_texture_destroy(WGPUDeviceId texture_id);

void wgpu_texture_view_destroy(WGPUTextureViewId _texture_view_id);
