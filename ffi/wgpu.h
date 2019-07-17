#define WGPU_LOCAL

/* Generated with cbindgen:0.9.0 */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define WGPUMAX_BIND_GROUPS 4

#define WGPUMAX_COLOR_TARGETS 4

#define WGPUMAX_MIP_LEVELS 16

#define WGPUMAX_VERTEX_BUFFERS 8

typedef enum {
  WGPUAddressMode_ClampToEdge = 0,
  WGPUAddressMode_Repeat = 1,
  WGPUAddressMode_MirrorRepeat = 2,
} WGPUAddressMode;

typedef enum {
  WGPUBindingType_UniformBuffer = 0,
  WGPUBindingType_Sampler = 1,
  WGPUBindingType_SampledTexture = 2,
  WGPUBindingType_StorageBuffer = 3,
  WGPUBindingType_UniformBufferDynamic = 4,
  WGPUBindingType_StorageBufferDynamic = 5,
  WGPUBindingType_StorageTexture = 10,
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
  WGPUBufferMapAsyncStatus_Success,
  WGPUBufferMapAsyncStatus_Error,
  WGPUBufferMapAsyncStatus_Unknown,
  WGPUBufferMapAsyncStatus_ContextLost,
} WGPUBufferMapAsyncStatus;

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
  WGPUCullMode_None = 0,
  WGPUCullMode_Front = 1,
  WGPUCullMode_Back = 2,
} WGPUCullMode;

typedef enum {
  WGPUFilterMode_Nearest = 0,
  WGPUFilterMode_Linear = 1,
} WGPUFilterMode;

typedef enum {
  WGPUFrontFace_Ccw = 0,
  WGPUFrontFace_Cw = 1,
} WGPUFrontFace;

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
  WGPUPresentMode_NoVsync = 0,
  WGPUPresentMode_Vsync = 1,
} WGPUPresentMode;

typedef enum {
  WGPUPrimitiveTopology_PointList = 0,
  WGPUPrimitiveTopology_LineList = 1,
  WGPUPrimitiveTopology_LineStrip = 2,
  WGPUPrimitiveTopology_TriangleList = 3,
  WGPUPrimitiveTopology_TriangleStrip = 4,
} WGPUPrimitiveTopology;

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
  WGPUTextureFormat_R8Unorm = 0,
  WGPUTextureFormat_R8UnormSrgb = 1,
  WGPUTextureFormat_R8Snorm = 2,
  WGPUTextureFormat_R8Uint = 3,
  WGPUTextureFormat_R8Sint = 4,
  WGPUTextureFormat_R16Unorm = 5,
  WGPUTextureFormat_R16Snorm = 6,
  WGPUTextureFormat_R16Uint = 7,
  WGPUTextureFormat_R16Sint = 8,
  WGPUTextureFormat_R16Float = 9,
  WGPUTextureFormat_Rg8Unorm = 10,
  WGPUTextureFormat_Rg8UnormSrgb = 11,
  WGPUTextureFormat_Rg8Snorm = 12,
  WGPUTextureFormat_Rg8Uint = 13,
  WGPUTextureFormat_Rg8Sint = 14,
  WGPUTextureFormat_B5g6r5Unorm = 15,
  WGPUTextureFormat_R32Uint = 16,
  WGPUTextureFormat_R32Sint = 17,
  WGPUTextureFormat_R32Float = 18,
  WGPUTextureFormat_Rg16Unorm = 19,
  WGPUTextureFormat_Rg16Snorm = 20,
  WGPUTextureFormat_Rg16Uint = 21,
  WGPUTextureFormat_Rg16Sint = 22,
  WGPUTextureFormat_Rg16Float = 23,
  WGPUTextureFormat_Rgba8Unorm = 24,
  WGPUTextureFormat_Rgba8UnormSrgb = 25,
  WGPUTextureFormat_Rgba8Snorm = 26,
  WGPUTextureFormat_Rgba8Uint = 27,
  WGPUTextureFormat_Rgba8Sint = 28,
  WGPUTextureFormat_Bgra8Unorm = 29,
  WGPUTextureFormat_Bgra8UnormSrgb = 30,
  WGPUTextureFormat_Rgb10a2Unorm = 31,
  WGPUTextureFormat_Rg11b10Float = 32,
  WGPUTextureFormat_Rg32Uint = 33,
  WGPUTextureFormat_Rg32Sint = 34,
  WGPUTextureFormat_Rg32Float = 35,
  WGPUTextureFormat_Rgba16Unorm = 36,
  WGPUTextureFormat_Rgba16Snorm = 37,
  WGPUTextureFormat_Rgba16Uint = 38,
  WGPUTextureFormat_Rgba16Sint = 39,
  WGPUTextureFormat_Rgba16Float = 40,
  WGPUTextureFormat_Rgba32Uint = 41,
  WGPUTextureFormat_Rgba32Sint = 42,
  WGPUTextureFormat_Rgba32Float = 43,
  WGPUTextureFormat_D16Unorm = 44,
  WGPUTextureFormat_D32Float = 45,
  WGPUTextureFormat_D24UnormS8Uint = 46,
  WGPUTextureFormat_D32FloatS8Uint = 47,
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
  WGPUVertexFormat_Uchar2 = 1,
  WGPUVertexFormat_Uchar4 = 3,
  WGPUVertexFormat_Char2 = 5,
  WGPUVertexFormat_Char4 = 7,
  WGPUVertexFormat_Uchar2Norm = 9,
  WGPUVertexFormat_Uchar4Norm = 11,
  WGPUVertexFormat_Char2Norm = 14,
  WGPUVertexFormat_Char4Norm = 16,
  WGPUVertexFormat_Ushort2 = 18,
  WGPUVertexFormat_Ushort4 = 20,
  WGPUVertexFormat_Short2 = 22,
  WGPUVertexFormat_Short4 = 24,
  WGPUVertexFormat_Ushort2Norm = 26,
  WGPUVertexFormat_Ushort4Norm = 28,
  WGPUVertexFormat_Short2Norm = 30,
  WGPUVertexFormat_Short4Norm = 32,
  WGPUVertexFormat_Half2 = 34,
  WGPUVertexFormat_Half4 = 36,
  WGPUVertexFormat_Float = 37,
  WGPUVertexFormat_Float2 = 38,
  WGPUVertexFormat_Float3 = 39,
  WGPUVertexFormat_Float4 = 40,
  WGPUVertexFormat_Uint = 41,
  WGPUVertexFormat_Uint2 = 42,
  WGPUVertexFormat_Uint3 = 43,
  WGPUVertexFormat_Uint4 = 44,
  WGPUVertexFormat_Int = 45,
  WGPUVertexFormat_Int2 = 46,
  WGPUVertexFormat_Int3 = 47,
  WGPUVertexFormat_Int4 = 48,
} WGPUVertexFormat;

typedef uint32_t WGPUIndex;

typedef uint32_t WGPUEpoch;

typedef struct {
  WGPUIndex _0;
  WGPUEpoch _1;
} WGPUId;

typedef WGPUId WGPUDeviceId;

typedef WGPUId WGPUAdapterId;

typedef struct {
  bool anisotropic_filtering;
} WGPUExtensions;

typedef struct {
  uint32_t max_bind_groups;
} WGPULimits;

typedef struct {
  WGPUExtensions extensions;
  WGPULimits limits;
} WGPUDeviceDescriptor;

typedef WGPUId WGPUBindGroupId;

typedef WGPUId WGPUBufferId;

typedef uint64_t WGPUBufferAddress;

typedef void (*WGPUBufferMapReadCallback)(WGPUBufferMapAsyncStatus status, const uint8_t *data, uint8_t *userdata);

typedef void (*WGPUBufferMapWriteCallback)(WGPUBufferMapAsyncStatus status, uint8_t *data, uint8_t *userdata);

typedef WGPUId WGPUCommandBufferId;

typedef struct {
  WGPUBufferId buffer;
  WGPUBufferAddress offset;
  uint32_t row_pitch;
  uint32_t image_height;
} WGPUBufferCopyView;

typedef WGPUId WGPUTextureId;

typedef struct {
  float x;
  float y;
  float z;
} WGPUOrigin3d;
#define WGPUOrigin3d_ZERO (WGPUOrigin3d){ .x = 0, .y = 0, .z = 0 }

typedef struct {
  WGPUTextureId texture;
  uint32_t mip_level;
  uint32_t array_layer;
  WGPUOrigin3d origin;
} WGPUTextureCopyView;

typedef struct {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
} WGPUExtent3d;

typedef WGPUId WGPUComputePassId;

typedef WGPUCommandBufferId WGPUCommandEncoderId;

typedef WGPUId WGPURenderPassId;

typedef WGPUId WGPUTextureViewId;

typedef struct {
  float r;
  float g;
  float b;
  float a;
} WGPUColor;
#define WGPUColor_TRANSPARENT (WGPUColor){ .r = 0, .g = 0, .b = 0, .a = 0 }
#define WGPUColor_BLACK (WGPUColor){ .r = 0, .g = 0, .b = 0, .a = 1 }
#define WGPUColor_WHITE (WGPUColor){ .r = 1, .g = 1, .b = 1, .a = 1 }
#define WGPUColor_RED (WGPUColor){ .r = 1, .g = 0, .b = 0, .a = 1 }
#define WGPUColor_GREEN (WGPUColor){ .r = 0, .g = 1, .b = 0, .a = 1 }
#define WGPUColor_BLUE (WGPUColor){ .r = 0, .g = 0, .b = 1, .a = 1 }

typedef struct {
  WGPUTextureViewId attachment;
  const WGPUTextureViewId *resolve_target;
  WGPULoadOp load_op;
  WGPUStoreOp store_op;
  WGPUColor clear_color;
} WGPURenderPassColorAttachmentDescriptor;

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
  const WGPURenderPassColorAttachmentDescriptor *color_attachments;
  uintptr_t color_attachments_length;
  const WGPURenderPassDepthStencilAttachmentDescriptor_TextureViewId *depth_stencil_attachment;
} WGPURenderPassDescriptor;

typedef const char *WGPURawString;

typedef WGPUId WGPUComputePipelineId;

typedef WGPUId WGPUInstanceId;

typedef WGPUId WGPUBindGroupLayoutId;

typedef struct {
  WGPUBufferId buffer;
  WGPUBufferAddress offset;
  WGPUBufferAddress size;
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
} WGPUBindGroupBinding;

typedef struct {
  WGPUBindGroupLayoutId layout;
  const WGPUBindGroupBinding *bindings;
  uintptr_t bindings_length;
} WGPUBindGroupDescriptor;

typedef uint32_t WGPUShaderStage;
#define WGPUShaderStage_NONE 0
#define WGPUShaderStage_VERTEX 1
#define WGPUShaderStage_FRAGMENT 2
#define WGPUShaderStage_COMPUTE 4

typedef struct {
  uint32_t binding;
  WGPUShaderStage visibility;
  WGPUBindingType ty;
} WGPUBindGroupLayoutBinding;

typedef struct {
  const WGPUBindGroupLayoutBinding *bindings;
  uintptr_t bindings_length;
} WGPUBindGroupLayoutDescriptor;

typedef uint32_t WGPUBufferUsage;
#define WGPUBufferUsage_MAP_READ 1
#define WGPUBufferUsage_MAP_WRITE 2
#define WGPUBufferUsage_TRANSFER_SRC 4
#define WGPUBufferUsage_TRANSFER_DST 8
#define WGPUBufferUsage_INDEX 16
#define WGPUBufferUsage_VERTEX 32
#define WGPUBufferUsage_UNIFORM 64
#define WGPUBufferUsage_STORAGE 128
#define WGPUBufferUsage_NONE 0
#define WGPUBufferUsage_WRITE_ALL 2 + 8 + 128

typedef struct {
  WGPUBufferAddress size;
  WGPUBufferUsage usage;
} WGPUBufferDescriptor;

typedef struct {
  uint32_t todo;
} WGPUCommandEncoderDescriptor;

typedef WGPUId WGPUPipelineLayoutId;

typedef WGPUId WGPUShaderModuleId;

typedef struct {
  WGPUShaderModuleId module;
  WGPURawString entry_point;
} WGPUPipelineStageDescriptor;

typedef struct {
  WGPUPipelineLayoutId layout;
  WGPUPipelineStageDescriptor compute_stage;
} WGPUComputePipelineDescriptor;

typedef struct {
  const WGPUBindGroupLayoutId *bind_group_layouts;
  uintptr_t bind_group_layouts_length;
} WGPUPipelineLayoutDescriptor;

typedef WGPUId WGPURenderPipelineId;

typedef struct {
  WGPUFrontFace front_face;
  WGPUCullMode cull_mode;
  int32_t depth_bias;
  float depth_bias_slope_scale;
  float depth_bias_clamp;
} WGPURasterizationStateDescriptor;

typedef struct {
  WGPUBlendFactor src_factor;
  WGPUBlendFactor dst_factor;
  WGPUBlendOperation operation;
} WGPUBlendDescriptor;

typedef uint32_t WGPUColorWrite;
#define WGPUColorWrite_RED 1
#define WGPUColorWrite_GREEN 2
#define WGPUColorWrite_BLUE 4
#define WGPUColorWrite_ALPHA 8
#define WGPUColorWrite_COLOR 7
#define WGPUColorWrite_ALL 15

typedef struct {
  WGPUTextureFormat format;
  WGPUBlendDescriptor alpha_blend;
  WGPUBlendDescriptor color_blend;
  WGPUColorWrite write_mask;
} WGPUColorStateDescriptor;

typedef struct {
  WGPUCompareFunction compare;
  WGPUStencilOperation fail_op;
  WGPUStencilOperation depth_fail_op;
  WGPUStencilOperation pass_op;
} WGPUStencilStateFaceDescriptor;

typedef struct {
  WGPUTextureFormat format;
  bool depth_write_enabled;
  WGPUCompareFunction depth_compare;
  WGPUStencilStateFaceDescriptor stencil_front;
  WGPUStencilStateFaceDescriptor stencil_back;
  uint32_t stencil_read_mask;
  uint32_t stencil_write_mask;
} WGPUDepthStencilStateDescriptor;

typedef uint32_t WGPUShaderLocation;

typedef struct {
  WGPUBufferAddress offset;
  WGPUVertexFormat format;
  WGPUShaderLocation shader_location;
} WGPUVertexAttributeDescriptor;

typedef struct {
  WGPUBufferAddress stride;
  WGPUInputStepMode step_mode;
  const WGPUVertexAttributeDescriptor *attributes;
  uintptr_t attributes_length;
} WGPUVertexBufferDescriptor;

typedef struct {
  WGPUIndexFormat index_format;
  const WGPUVertexBufferDescriptor *vertex_buffers;
  uintptr_t vertex_buffers_length;
} WGPUVertexInputDescriptor;

typedef struct {
  WGPUPipelineLayoutId layout;
  WGPUPipelineStageDescriptor vertex_stage;
  const WGPUPipelineStageDescriptor *fragment_stage;
  WGPUPrimitiveTopology primitive_topology;
  WGPURasterizationStateDescriptor rasterization_state;
  const WGPUColorStateDescriptor *color_states;
  uintptr_t color_states_length;
  const WGPUDepthStencilStateDescriptor *depth_stencil_state;
  WGPUVertexInputDescriptor vertex_input;
  uint32_t sample_count;
} WGPURenderPipelineDescriptor;

typedef struct {
  WGPUAddressMode address_mode_u;
  WGPUAddressMode address_mode_v;
  WGPUAddressMode address_mode_w;
  WGPUFilterMode mag_filter;
  WGPUFilterMode min_filter;
  WGPUFilterMode mipmap_filter;
  float lod_min_clamp;
  float lod_max_clamp;
  WGPUCompareFunction compare_function;
} WGPUSamplerDescriptor;

typedef struct {
  const uint8_t *bytes;
  uintptr_t length;
} WGPUByteArray;

typedef struct {
  WGPUByteArray code;
} WGPUShaderModuleDescriptor;

typedef WGPUId WGPUSurfaceId;

typedef WGPUSurfaceId WGPUSwapChainId;

typedef uint32_t WGPUTextureUsage;
#define WGPUTextureUsage_TRANSFER_SRC 1
#define WGPUTextureUsage_TRANSFER_DST 2
#define WGPUTextureUsage_SAMPLED 4
#define WGPUTextureUsage_STORAGE 8
#define WGPUTextureUsage_OUTPUT_ATTACHMENT 16
#define WGPUTextureUsage_NONE 0
#define WGPUTextureUsage_WRITE_ALL 2 + 8 + 16
#define WGPUTextureUsage_UNINITIALIZED 65535

typedef struct {
  WGPUTextureUsage usage;
  WGPUTextureFormat format;
  uint32_t width;
  uint32_t height;
  WGPUPresentMode present_mode;
} WGPUSwapChainDescriptor;

typedef struct {
  WGPUExtent3d size;
  uint32_t array_layer_count;
  uint32_t mip_level_count;
  uint32_t sample_count;
  WGPUTextureDimension dimension;
  WGPUTextureFormat format;
  WGPUTextureUsage usage;
} WGPUTextureDescriptor;

typedef WGPUDeviceId WGPUQueueId;

typedef struct {
  WGPUPowerPreference power_preference;
} WGPUAdapterDescriptor;

typedef struct {
  WGPUTextureId texture_id;
  WGPUTextureViewId view_id;
} WGPUSwapChainOutput;

typedef uint32_t WGPUTextureAspectFlags;
#define WGPUTextureAspectFlags_COLOR 1
#define WGPUTextureAspectFlags_DEPTH 2
#define WGPUTextureAspectFlags_STENCIL 4

typedef struct {
  WGPUTextureFormat format;
  WGPUTextureViewDimension dimension;
  WGPUTextureAspectFlags aspect;
  uint32_t base_mip_level;
  uint32_t level_count;
  uint32_t base_array_layer;
  uint32_t array_count;
} WGPUTextureViewDescriptor;

#if defined(WGPU_LOCAL)
WGPUDeviceId wgpu_adapter_request_device(WGPUAdapterId adapter_id,
                                         const WGPUDeviceDescriptor *desc);
#endif

void wgpu_bind_group_destroy(WGPUBindGroupId bind_group_id);

void wgpu_buffer_destroy(WGPUBufferId buffer_id);

void wgpu_buffer_map_read_async(WGPUBufferId buffer_id,
                                WGPUBufferAddress start,
                                WGPUBufferAddress size,
                                WGPUBufferMapReadCallback callback,
                                uint8_t *userdata);

void wgpu_buffer_map_write_async(WGPUBufferId buffer_id,
                                 WGPUBufferAddress start,
                                 WGPUBufferAddress size,
                                 WGPUBufferMapWriteCallback callback,
                                 uint8_t *userdata);

void wgpu_buffer_unmap(WGPUBufferId buffer_id);

void wgpu_command_buffer_copy_buffer_to_buffer(WGPUCommandBufferId command_buffer_id,
                                               WGPUBufferId src,
                                               WGPUBufferAddress src_offset,
                                               WGPUBufferId dst,
                                               WGPUBufferAddress dst_offset,
                                               WGPUBufferAddress size);

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

#if defined(WGPU_LOCAL)
WGPUComputePassId wgpu_command_encoder_begin_compute_pass(WGPUCommandEncoderId command_encoder_id);
#endif

#if defined(WGPU_LOCAL)
WGPURenderPassId wgpu_command_encoder_begin_render_pass(WGPUCommandEncoderId command_encoder_id,
                                                        const WGPURenderPassDescriptor *desc);
#endif

WGPUCommandBufferId wgpu_command_encoder_finish(WGPUCommandEncoderId command_encoder_id);

void wgpu_compute_pass_dispatch(WGPUComputePassId pass_id, uint32_t x, uint32_t y, uint32_t z);

WGPUCommandBufferId wgpu_compute_pass_end_pass(WGPUComputePassId pass_id);

void wgpu_compute_pass_insert_debug_marker(WGPUComputePassId _pass_id, WGPURawString _label);

void wgpu_compute_pass_pop_debug_group(WGPUComputePassId _pass_id);

void wgpu_compute_pass_push_debug_group(WGPUComputePassId _pass_id, WGPURawString _label);

void wgpu_compute_pass_set_bind_group(WGPUComputePassId pass_id,
                                      uint32_t index,
                                      WGPUBindGroupId bind_group_id,
                                      const WGPUBufferAddress *offsets,
                                      uintptr_t offsets_length);

void wgpu_compute_pass_set_pipeline(WGPUComputePassId pass_id, WGPUComputePipelineId pipeline_id);

#if defined(WGPU_LOCAL)
WGPUInstanceId wgpu_create_instance(void);
#endif

#if defined(WGPU_LOCAL)
WGPUBindGroupId wgpu_device_create_bind_group(WGPUDeviceId device_id,
                                              const WGPUBindGroupDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
WGPUBindGroupLayoutId wgpu_device_create_bind_group_layout(WGPUDeviceId device_id,
                                                           const WGPUBindGroupLayoutDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
WGPUBufferId wgpu_device_create_buffer(WGPUDeviceId device_id, const WGPUBufferDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
WGPUBufferId wgpu_device_create_buffer_mapped(WGPUDeviceId device_id,
                                              const WGPUBufferDescriptor *desc,
                                              uint8_t **mapped_ptr_out);
#endif

#if defined(WGPU_LOCAL)
WGPUCommandEncoderId wgpu_device_create_command_encoder(WGPUDeviceId device_id,
                                                        const WGPUCommandEncoderDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
WGPUComputePipelineId wgpu_device_create_compute_pipeline(WGPUDeviceId device_id,
                                                          const WGPUComputePipelineDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
WGPUPipelineLayoutId wgpu_device_create_pipeline_layout(WGPUDeviceId device_id,
                                                        const WGPUPipelineLayoutDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
WGPURenderPipelineId wgpu_device_create_render_pipeline(WGPUDeviceId device_id,
                                                        const WGPURenderPipelineDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
WGPUSamplerId wgpu_device_create_sampler(WGPUDeviceId device_id, const WGPUSamplerDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
WGPUShaderModuleId wgpu_device_create_shader_module(WGPUDeviceId device_id,
                                                    const WGPUShaderModuleDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
WGPUSwapChainId wgpu_device_create_swap_chain(WGPUDeviceId device_id,
                                              WGPUSurfaceId surface_id,
                                              const WGPUSwapChainDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
WGPUTextureId wgpu_device_create_texture(WGPUDeviceId device_id, const WGPUTextureDescriptor *desc);
#endif

void wgpu_device_destroy(WGPUDeviceId device_id);

WGPUQueueId wgpu_device_get_queue(WGPUDeviceId device_id);

void wgpu_device_poll(WGPUDeviceId device_id, bool force_wait);

#if defined(WGPU_LOCAL)
WGPUSurfaceId wgpu_instance_create_surface_from_macos_layer(WGPUInstanceId instance_id,
                                                            void *layer);
#endif

#if defined(WGPU_LOCAL)
WGPUSurfaceId wgpu_instance_create_surface_from_windows_hwnd(WGPUInstanceId instance_id,
                                                             void *hinstance,
                                                             void *hwnd);
#endif

#if defined(WGPU_LOCAL)
WGPUSurfaceId wgpu_instance_create_surface_from_xlib(WGPUInstanceId instance_id,
                                                     const void **display,
                                                     uint64_t window);
#endif

#if defined(WGPU_LOCAL)
WGPUAdapterId wgpu_instance_get_adapter(WGPUInstanceId instance_id,
                                        const WGPUAdapterDescriptor *desc);
#endif

void wgpu_queue_submit(WGPUQueueId queue_id,
                       const WGPUCommandBufferId *command_buffers,
                       uintptr_t command_buffers_length);

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

void wgpu_render_pass_insert_debug_marker(WGPURenderPassId _pass_id, WGPURawString _label);

void wgpu_render_pass_pop_debug_group(WGPURenderPassId _pass_id);

void wgpu_render_pass_push_debug_group(WGPURenderPassId _pass_id, WGPURawString _label);

void wgpu_render_pass_set_bind_group(WGPURenderPassId pass_id,
                                     uint32_t index,
                                     WGPUBindGroupId bind_group_id,
                                     const WGPUBufferAddress *offsets,
                                     uintptr_t offsets_length);

void wgpu_render_pass_set_blend_color(WGPURenderPassId pass_id, const WGPUColor *color);

void wgpu_render_pass_set_index_buffer(WGPURenderPassId pass_id,
                                       WGPUBufferId buffer_id,
                                       WGPUBufferAddress offset);

void wgpu_render_pass_set_pipeline(WGPURenderPassId pass_id, WGPURenderPipelineId pipeline_id);

void wgpu_render_pass_set_scissor_rect(WGPURenderPassId pass_id,
                                       uint32_t x,
                                       uint32_t y,
                                       uint32_t w,
                                       uint32_t h);

void wgpu_render_pass_set_stencil_reference(WGPURenderPassId pass_id, uint32_t value);

void wgpu_render_pass_set_vertex_buffers(WGPURenderPassId pass_id,
                                         const WGPUBufferId *buffers,
                                         const WGPUBufferAddress *offsets,
                                         uintptr_t length);

void wgpu_render_pass_set_viewport(WGPURenderPassId pass_id,
                                   float x,
                                   float y,
                                   float w,
                                   float h,
                                   float min_depth,
                                   float max_depth);

WGPUSwapChainOutput wgpu_swap_chain_get_next_texture(WGPUSwapChainId swap_chain_id);

void wgpu_swap_chain_present(WGPUSwapChainId swap_chain_id);

#if defined(WGPU_LOCAL)
WGPUTextureViewId wgpu_texture_create_default_view(WGPUTextureId texture_id);
#endif

#if defined(WGPU_LOCAL)
WGPUTextureViewId wgpu_texture_create_view(WGPUTextureId texture_id,
                                           const WGPUTextureViewDescriptor *desc);
#endif

void wgpu_texture_destroy(WGPUTextureId texture_id);

void wgpu_texture_view_destroy(WGPUTextureViewId texture_view_id);
