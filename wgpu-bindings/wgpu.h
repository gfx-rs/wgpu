#ifdef WGPU_REMOTE
    typedef uint32_t WGPUId;
#else
    typedef void *WGPUId;
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define WGPUBITS_PER_BYTE 8

typedef enum {
  WGPUAddressMode_ClampToEdge = 0,
  WGPUAddressMode_Repeat = 1,
  WGPUAddressMode_MirrorRepeat = 2,
  WGPUAddressMode_ClampToBorderColor = 3,
} WGPUAddressMode;

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
  WGPUBorderColor_TransparentBlack = 0,
  WGPUBorderColor_OpaqueBlack = 1,
  WGPUBorderColor_OpaqueWhite = 2,
} WGPUBorderColor;

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
  WGPUFilterMode_Nearest = 0,
  WGPUFilterMode_Linear = 1,
} WGPUFilterMode;

typedef enum {
  WGPUIndexFormat_Uint16 = 0,
  WGPUIndexFormat_Uint32 = 1,
} WGPUIndexFormat;

typedef enum {
  WGPUInputStepMode_Vertex = 0,
  WGPUInputStepMode_Instance = 1,
} WGPUInputStepMode;

typedef enum {
  WGPULoadOp_Clear = 0,
  WGPULoadOp_Load = 1,
} WGPULoadOp;

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
  WGPUStoreOp_Store = 0,
} WGPUStoreOp;

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

typedef enum {
  WGPUVertexFormat_FloatR32G32B32A32 = 0,
  WGPUVertexFormat_FloatR32G32B32 = 1,
  WGPUVertexFormat_FloatR32G32 = 2,
  WGPUVertexFormat_FloatR32 = 3,
} WGPUVertexFormat;

typedef WGPUId WGPUDeviceId;

typedef WGPUId WGPUAdapterId;

typedef struct {
  bool anisotropic_filtering;
} WGPUExtensions;

typedef struct {
  WGPUExtensions extensions;
} WGPUDeviceDescriptor;

typedef WGPUId WGPUBufferId;

typedef WGPUId WGPUComputePassId;

typedef WGPUId WGPUCommandBufferId;

typedef WGPUId WGPURenderPassId;

typedef WGPUId WGPUTextureViewId;

typedef struct {
  float r;
  float g;
  float b;
  float a;
} WGPUColor;

typedef struct {
  WGPUTextureViewId attachment;
  WGPULoadOp load_op;
  WGPUStoreOp store_op;
  WGPUColor clear_color;
} WGPURenderPassColorAttachmentDescriptor_TextureViewId;

typedef struct {
  WGPUTextureViewId attachment;
  WGPULoadOp depth_load_op;
  WGPUStoreOp depth_store_op;
  float clear_depth;
  WGPULoadOp stencil_load_op;
  WGPUStoreOp stencil_store_op;
  uint32_t clear_stencil;
} WGPURenderPassDepthStencilAttachmentDescriptor_TextureViewId;

typedef struct {
  const WGPURenderPassColorAttachmentDescriptor_TextureViewId *color_attachments;
  uintptr_t color_attachments_length;
  const WGPURenderPassDepthStencilAttachmentDescriptor_TextureViewId *depth_stencil_attachment;
} WGPURenderPassDescriptor;

typedef struct {
  WGPUBufferId buffer;
  uint32_t offset;
  uint32_t row_pitch;
  uint32_t image_height;
} WGPUBufferCopyView;

typedef WGPUId WGPUTextureId;

typedef struct {
  float x;
  float y;
  float z;
} WGPUOrigin3d;

typedef struct {
  WGPUTextureId texture;
  uint32_t level;
  uint32_t slice;
  WGPUOrigin3d origin;
} WGPUTextureCopyView;

typedef struct {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
} WGPUExtent3d;

typedef WGPUId WGPUBindGroupId;

typedef WGPUId WGPUComputePipelineId;

typedef WGPUId WGPUInstanceId;

typedef WGPUId WGPUBindGroupLayoutId;

typedef struct {
  WGPUBufferId buffer;
  uint32_t offset;
  uint32_t size;
} WGPUBufferBinding;

typedef WGPUId WGPUSamplerId;

typedef enum {
  WGPUBindingResource_Buffer,
  WGPUBindingResource_Sampler,
  WGPUBindingResource_TextureView,
} WGPUBindingResource_Tag;

typedef struct {
  WGPUBufferBinding _0;
} WGPUBindingResource_WGPUBuffer_Body;

typedef struct {
  WGPUSamplerId _0;
} WGPUBindingResource_WGPUSampler_Body;

typedef struct {
  WGPUTextureViewId _0;
} WGPUBindingResource_WGPUTextureView_Body;

typedef struct {
  WGPUBindingResource_Tag tag;
  union {
    WGPUBindingResource_WGPUBuffer_Body buffer;
    WGPUBindingResource_WGPUSampler_Body sampler;
    WGPUBindingResource_WGPUTextureView_Body texture_view;
  };
} WGPUBindingResource;

typedef struct {
  uint32_t binding;
  WGPUBindingResource resource;
} WGPUBinding;

typedef struct {
  WGPUBindGroupLayoutId layout;
  const WGPUBinding *bindings;
  uintptr_t bindings_length;
} WGPUBindGroupDescriptor;

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

typedef uint32_t WGPUBufferUsageFlags;

typedef struct {
  uint32_t size;
  WGPUBufferUsageFlags usage;
} WGPUBufferDescriptor;

typedef struct {
  uint32_t todo;
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

typedef uint32_t WGPUShaderAttributeIndex;

typedef struct {
  uint32_t offset;
  WGPUVertexFormat format;
  WGPUShaderAttributeIndex attribute_index;
} WGPUVertexAttributeDescriptor;

typedef struct {
  uint32_t stride;
  WGPUInputStepMode step_mode;
  const WGPUVertexAttributeDescriptor *attributes;
  uintptr_t attributes_count;
} WGPUVertexBufferDescriptor;

typedef struct {
  WGPUIndexFormat index_format;
  const WGPUVertexBufferDescriptor *vertex_buffers;
  uintptr_t vertex_buffers_count;
} WGPUVertexBufferStateDescriptor;

typedef struct {
  WGPUPipelineLayoutId layout;
  const WGPUPipelineStageDescriptor *stages;
  uintptr_t stages_length;
  WGPUPrimitiveTopology primitive_topology;
  WGPUAttachmentsState attachments_state;
  const WGPUBlendStateId *blend_states;
  uintptr_t blend_states_length;
  WGPUDepthStencilStateId depth_stencil_state;
  WGPUVertexBufferStateDescriptor vertex_buffer_state;
} WGPURenderPipelineDescriptor;

typedef struct {
  WGPUAddressMode r_address_mode;
  WGPUAddressMode s_address_mode;
  WGPUAddressMode t_address_mode;
  WGPUFilterMode mag_filter;
  WGPUFilterMode min_filter;
  WGPUFilterMode mipmap_filter;
  float lod_min_clamp;
  float lod_max_clamp;
  uint32_t max_anisotropy;
  WGPUCompareFunction compare_function;
  WGPUBorderColor border_color;
} WGPUSamplerDescriptor;

typedef struct {
  const uint8_t *bytes;
  uintptr_t length;
} WGPUByteArray;

typedef struct {
  WGPUByteArray code;
} WGPUShaderModuleDescriptor;

typedef WGPUId WGPUSwapChainId;

typedef WGPUId WGPUSurfaceId;

typedef uint32_t WGPUTextureUsageFlags;

typedef struct {
  WGPUTextureUsageFlags usage;
  WGPUTextureFormat format;
  uint32_t width;
  uint32_t height;
} WGPUSwapChainDescriptor;

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

typedef struct {
  WGPUTextureId texture_id;
  WGPUTextureViewId view_id;
} WGPUSwapChainOutput;

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

#define WGPUBufferUsageFlags_INDEX 16

#define WGPUBufferUsageFlags_MAP_READ 1

#define WGPUBufferUsageFlags_MAP_WRITE 2

#define WGPUBufferUsageFlags_NONE 0

#define WGPUBufferUsageFlags_STORAGE 128

#define WGPUBufferUsageFlags_TRANSFER_DST 8

#define WGPUBufferUsageFlags_TRANSFER_SRC 4

#define WGPUBufferUsageFlags_UNIFORM 64

#define WGPUBufferUsageFlags_VERTEX 32

#define WGPUColorWriteFlags_ALL 15

#define WGPUColorWriteFlags_ALPHA 8

#define WGPUColorWriteFlags_BLUE 4

#define WGPUColorWriteFlags_COLOR 7

#define WGPUColorWriteFlags_GREEN 2

#define WGPUColorWriteFlags_RED 1

#define WGPUColor_BLACK (WGPUColor){ .r = 0, .g = 0, .b = 0, .a = 1 }

#define WGPUColor_BLUE (WGPUColor){ .r = 0, .g = 0, .b = 1, .a = 1 }

#define WGPUColor_GREEN (WGPUColor){ .r = 0, .g = 1, .b = 0, .a = 1 }

#define WGPUColor_RED (WGPUColor){ .r = 1, .g = 0, .b = 0, .a = 1 }

#define WGPUColor_TRANSPARENT (WGPUColor){ .r = 0, .g = 0, .b = 0, .a = 0 }

#define WGPUColor_WHITE (WGPUColor){ .r = 1, .g = 1, .b = 1, .a = 1 }

#define WGPUShaderStageFlags_COMPUTE 4

#define WGPUShaderStageFlags_FRAGMENT 2

#define WGPUShaderStageFlags_VERTEX 1

#define WGPUTextureAspectFlags_COLOR 1

#define WGPUTextureAspectFlags_DEPTH 2

#define WGPUTextureAspectFlags_STENCIL 4

#define WGPUTextureUsageFlags_NONE 0

#define WGPUTextureUsageFlags_OUTPUT_ATTACHMENT 16

#define WGPUTextureUsageFlags_SAMPLED 4

#define WGPUTextureUsageFlags_STORAGE 8

#define WGPUTextureUsageFlags_TRANSFER_DST 2

#define WGPUTextureUsageFlags_TRANSFER_SRC 1

#define WGPUTextureUsageFlags_UNINITIALIZED 65535

#define WGPUTrackPermit_EXTEND (WGPUTrackPermit){ .bits = 1 }

#define WGPUTrackPermit_REPLACE (WGPUTrackPermit){ .bits = 2 }

WGPUDeviceId wgpu_adapter_create_device(WGPUAdapterId adapter_id,
                                        const WGPUDeviceDescriptor *_desc);

void wgpu_buffer_destroy(WGPUBufferId buffer_id);

void wgpu_buffer_set_sub_data(WGPUBufferId buffer_id,
                              uint32_t start,
                              uint32_t count,
                              const uint8_t *data);

WGPUComputePassId wgpu_command_buffer_begin_compute_pass(WGPUCommandBufferId command_buffer_id);

WGPURenderPassId wgpu_command_buffer_begin_render_pass(WGPUCommandBufferId command_buffer_id,
                                                       WGPURenderPassDescriptor desc);

void wgpu_command_buffer_copy_buffer_to_buffer(WGPUCommandBufferId command_buffer_id,
                                               WGPUBufferId src,
                                               uint32_t src_offset,
                                               WGPUBufferId dst,
                                               uint32_t dst_offset,
                                               uint32_t size);

void wgpu_command_buffer_copy_buffer_to_texture(WGPUCommandBufferId command_buffer_id,
                                                const WGPUBufferCopyView *source,
                                                const WGPUTextureCopyView *destination,
                                                WGPUExtent3d copy_size);

void wgpu_command_buffer_copy_texture_to_buffer(WGPUCommandBufferId command_buffer_id,
                                                const WGPUTextureCopyView *source,
                                                const WGPUBufferCopyView *destination,
                                                WGPUExtent3d copy_size);

void wgpu_command_buffer_copy_texture_to_texture(WGPUCommandBufferId command_buffer_id,
                                                 const WGPUTextureCopyView *source,
                                                 const WGPUTextureCopyView *destination,
                                                 WGPUExtent3d copy_size);

void wgpu_compute_pass_dispatch(WGPUComputePassId pass_id, uint32_t x, uint32_t y, uint32_t z);

WGPUCommandBufferId wgpu_compute_pass_end_pass(WGPUComputePassId pass_id);

void wgpu_compute_pass_set_bind_group(WGPUComputePassId pass_id,
                                      uint32_t index,
                                      WGPUBindGroupId bind_group_id);

void wgpu_compute_pass_set_pipeline(WGPUComputePassId pass_id, WGPUComputePipelineId pipeline_id);

WGPUInstanceId wgpu_create_instance(void);

WGPUBindGroupId wgpu_device_create_bind_group(WGPUDeviceId device_id,
                                              const WGPUBindGroupDescriptor *desc);

WGPUBindGroupLayoutId wgpu_device_create_bind_group_layout(WGPUDeviceId device_id,
                                                           const WGPUBindGroupLayoutDescriptor *desc);

WGPUBlendStateId wgpu_device_create_blend_state(WGPUDeviceId _device_id,
                                                const WGPUBlendStateDescriptor *desc);

WGPUBufferId wgpu_device_create_buffer(WGPUDeviceId device_id, const WGPUBufferDescriptor *desc);

WGPUCommandBufferId wgpu_device_create_command_buffer(WGPUDeviceId device_id,
                                                      const WGPUCommandBufferDescriptor *_desc);

WGPUDepthStencilStateId wgpu_device_create_depth_stencil_state(WGPUDeviceId _device_id,
                                                               const WGPUDepthStencilStateDescriptor *desc);

WGPUPipelineLayoutId wgpu_device_create_pipeline_layout(WGPUDeviceId device_id,
                                                        const WGPUPipelineLayoutDescriptor *desc);

WGPURenderPipelineId wgpu_device_create_render_pipeline(WGPUDeviceId device_id,
                                                        const WGPURenderPipelineDescriptor *desc);

WGPUSamplerId wgpu_device_create_sampler(WGPUDeviceId device_id, const WGPUSamplerDescriptor *desc);

WGPUShaderModuleId wgpu_device_create_shader_module(WGPUDeviceId device_id,
                                                    const WGPUShaderModuleDescriptor *desc);

WGPUSwapChainId wgpu_device_create_swap_chain(WGPUDeviceId device_id,
                                              WGPUSurfaceId surface_id,
                                              const WGPUSwapChainDescriptor *desc);

WGPUTextureId wgpu_device_create_texture(WGPUDeviceId device_id, const WGPUTextureDescriptor *desc);

WGPUQueueId wgpu_device_get_queue(WGPUDeviceId device_id);

WGPUSurfaceId wgpu_instance_create_surface_from_macos_layer(WGPUInstanceId instance_id,
                                                            void *layer);

WGPUSurfaceId wgpu_instance_create_surface_from_windows_hwnd(WGPUInstanceId instance_id,
                                                             void *hinstance,
                                                             void *hwnd);

WGPUSurfaceId wgpu_instance_create_surface_from_xlib(WGPUInstanceId instance_id,
                                                     const void **display,
                                                     uint64_t window);

WGPUAdapterId wgpu_instance_get_adapter(WGPUInstanceId instance_id,
                                        const WGPUAdapterDescriptor *desc);

void wgpu_queue_submit(WGPUQueueId queue_id,
                       const WGPUCommandBufferId *command_buffer_ptr,
                       uintptr_t command_buffer_count);

void wgpu_render_pass_draw(WGPURenderPassId pass_id,
                           uint32_t vertex_count,
                           uint32_t instance_count,
                           uint32_t first_vertex,
                           uint32_t first_instance);

void wgpu_render_pass_draw_indexed(WGPURenderPassId pass_id,
                                   uint32_t index_count,
                                   uint32_t instance_count,
                                   uint32_t first_index,
                                   int32_t base_vertex,
                                   uint32_t first_instance);

WGPUCommandBufferId wgpu_render_pass_end_pass(WGPURenderPassId pass_id);

void wgpu_render_pass_set_bind_group(WGPURenderPassId pass_id,
                                     uint32_t index,
                                     WGPUBindGroupId bind_group_id);

void wgpu_render_pass_set_index_buffer(WGPURenderPassId pass_id,
                                       WGPUBufferId buffer_id,
                                       uint32_t offset);

void wgpu_render_pass_set_pipeline(WGPURenderPassId pass_id, WGPURenderPipelineId pipeline_id);

void wgpu_render_pass_set_vertex_buffers(WGPURenderPassId pass_id,
                                         const WGPUBufferId *buffer_ptr,
                                         const uint32_t *offset_ptr,
                                         uintptr_t count);

WGPUSwapChainOutput wgpu_swap_chain_get_next_texture(WGPUSwapChainId swap_chain_id);

void wgpu_swap_chain_present(WGPUSwapChainId swap_chain_id);

WGPUTextureViewId wgpu_texture_create_default_texture_view(WGPUTextureId texture_id);

WGPUTextureViewId wgpu_texture_create_texture_view(WGPUTextureId texture_id,
                                                   const WGPUTextureViewDescriptor *desc);

void wgpu_texture_destroy(WGPUTextureId texture_id);

void wgpu_texture_view_destroy(WGPUTextureViewId _texture_view_id);
