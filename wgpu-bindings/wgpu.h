#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define WGPUBITS_PER_BYTE 8

#define WGPUMAX_BIND_GROUPS 4

#define WGPUMAX_COLOR_TARGETS 4

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
  WGPUBindingType_UniformBufferDynamic = 4,
  WGPUBindingType_StorageBufferDynamic = 5,
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
  WGPUVertexFormat_Uchar = 0,
  WGPUVertexFormat_Uchar2 = 1,
  WGPUVertexFormat_Uchar3 = 2,
  WGPUVertexFormat_Uchar4 = 3,
  WGPUVertexFormat_Char = 4,
  WGPUVertexFormat_Char2 = 5,
  WGPUVertexFormat_Char3 = 6,
  WGPUVertexFormat_Char4 = 7,
  WGPUVertexFormat_UcharNorm = 8,
  WGPUVertexFormat_Uchar2Norm = 9,
  WGPUVertexFormat_Uchar3Norm = 10,
  WGPUVertexFormat_Uchar4Norm = 11,
  WGPUVertexFormat_Uchar4NormBgra = 12,
  WGPUVertexFormat_CharNorm = 13,
  WGPUVertexFormat_Char2Norm = 14,
  WGPUVertexFormat_Char3Norm = 15,
  WGPUVertexFormat_Char4Norm = 16,
  WGPUVertexFormat_Ushort = 17,
  WGPUVertexFormat_Ushort2 = 18,
  WGPUVertexFormat_Ushort3 = 19,
  WGPUVertexFormat_Ushort4 = 20,
  WGPUVertexFormat_Short = 21,
  WGPUVertexFormat_Short2 = 22,
  WGPUVertexFormat_Short3 = 23,
  WGPUVertexFormat_Short4 = 24,
  WGPUVertexFormat_UshortNorm = 25,
  WGPUVertexFormat_Ushort2Norm = 26,
  WGPUVertexFormat_Ushort3Norm = 27,
  WGPUVertexFormat_Ushort4Norm = 28,
  WGPUVertexFormat_ShortNorm = 29,
  WGPUVertexFormat_Short2Norm = 30,
  WGPUVertexFormat_Short3Norm = 31,
  WGPUVertexFormat_Short4Norm = 32,
  WGPUVertexFormat_Half = 33,
  WGPUVertexFormat_Half2 = 34,
  WGPUVertexFormat_Half3 = 35,
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

typedef struct WGPUBufferMapAsyncStatus WGPUBufferMapAsyncStatus;

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
  WGPUExtensions extensions;
} WGPUDeviceDescriptor;

typedef WGPUId WGPUBindGroupId;

typedef WGPUId WGPUBufferId;

typedef void (*WGPUBufferMapReadCallback)(WGPUBufferMapAsyncStatus status, const uint8_t *data, uint8_t *userdata);

typedef void (*WGPUBufferMapWriteCallback)(WGPUBufferMapAsyncStatus status, uint8_t *data, uint8_t *userdata);

typedef WGPUId WGPUCommandBufferId;

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

typedef uint32_t WGPUBufferUsageFlags;

typedef struct {
  uint32_t size;
  WGPUBufferUsageFlags usage;
} WGPUBufferDescriptor;

typedef struct {
  uint32_t todo;
} WGPUCommandEncoderDescriptor;

typedef WGPUId WGPUPipelineLayoutId;

typedef WGPUId WGPUShaderModuleId;

typedef struct {
  WGPUShaderModuleId module;
  const char *entry_point;
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

typedef uint32_t WGPUColorWriteFlags;

typedef struct {
  WGPUTextureFormat format;
  WGPUBlendDescriptor alpha;
  WGPUBlendDescriptor color;
  WGPUColorWriteFlags write_mask;
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
  WGPUPipelineStageDescriptor vertex_stage;
  WGPUPipelineStageDescriptor fragment_stage;
  WGPUPrimitiveTopology primitive_topology;
  WGPURasterizationStateDescriptor rasterization_state;
  const WGPUColorStateDescriptor *color_states;
  uintptr_t color_states_length;
  const WGPUDepthStencilStateDescriptor *depth_stencil_state;
  WGPUVertexBufferStateDescriptor vertex_buffer_state;
  uint32_t sample_count;
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

typedef WGPUId WGPUSurfaceId;

typedef WGPUSurfaceId WGPUSwapChainId;

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

typedef WGPUDeviceId WGPUQueueId;

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

#define WGPUPipelineFlags_BLEND_COLOR 1

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

WGPUDeviceId wgpu_adapter_create_device(WGPUAdapterId adapter_id, const WGPUDeviceDescriptor *desc);

void wgpu_bind_group_destroy(WGPUBindGroupId bind_group_id);

void wgpu_buffer_destroy(WGPUBufferId buffer_id);

void wgpu_buffer_map_read_async(WGPUBufferId buffer_id,
                                uint32_t start,
                                uint32_t size,
                                WGPUBufferMapReadCallback callback,
                                uint8_t *userdata);

void wgpu_buffer_map_write_async(WGPUBufferId buffer_id,
                                 uint32_t start,
                                 uint32_t size,
                                 WGPUBufferMapWriteCallback callback,
                                 uint8_t *userdata);

void wgpu_buffer_set_sub_data(WGPUBufferId buffer_id,
                              uint32_t start,
                              uint32_t count,
                              const uint8_t *data);

void wgpu_buffer_unmap(WGPUBufferId buffer_id);

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

WGPUComputePassId wgpu_command_encoder_begin_compute_pass(WGPUCommandEncoderId command_encoder_id);

WGPURenderPassId wgpu_command_encoder_begin_render_pass(WGPUCommandEncoderId command_encoder_id,
                                                        WGPURenderPassDescriptor desc);

WGPUCommandBufferId wgpu_command_encoder_finish(WGPUCommandEncoderId command_encoder_id);

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

WGPUBufferId wgpu_device_create_buffer(WGPUDeviceId device_id, const WGPUBufferDescriptor *desc);

WGPUBufferId wgpu_device_create_buffer_mapped(WGPUDeviceId device_id,
                                              const WGPUBufferDescriptor *desc,
                                              uint8_t **mapped_ptr_out);

WGPUCommandEncoderId wgpu_device_create_command_encoder(WGPUDeviceId device_id,
                                                        const WGPUCommandEncoderDescriptor *desc);

WGPUComputePipelineId wgpu_device_create_compute_pipeline(WGPUDeviceId device_id,
                                                          const WGPUComputePipelineDescriptor *desc);

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

void wgpu_device_destroy(WGPUDeviceId device_id);

WGPUQueueId wgpu_device_get_queue(WGPUDeviceId device_id);

void wgpu_device_poll(WGPUDeviceId device_id, bool force_wait);

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

void wgpu_render_pass_set_blend_color(WGPURenderPassId pass_id, const WGPUColor *color);

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

WGPUTextureViewId wgpu_texture_create_default_view(WGPUTextureId texture_id);

WGPUTextureViewId wgpu_texture_create_view(WGPUTextureId texture_id,
                                           const WGPUTextureViewDescriptor *desc);

void wgpu_texture_destroy(WGPUTextureId texture_id);

void wgpu_texture_view_destroy(WGPUTextureViewId texture_view_id);
