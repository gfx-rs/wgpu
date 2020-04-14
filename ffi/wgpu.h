/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/* Generated with cbindgen:0.14.0 */

/* DO NOT MODIFY THIS MANUALLY! This file was generated using cbindgen.
 * To generate this file:
 *   1. Get the latest cbindgen using `cargo install --force cbindgen`
 *      a. Alternatively, you can clone `https://github.com/eqrion/cbindgen` and use a tagged release
 *   2. Run `rustup run nightly cbindgen toolkit/library/rust/ --lockfile Cargo.lock --crate wgpu-remote -o dom/webgpu/ffi/wgpu_ffi_generated.h`
 */

typedef unsigned long long WGPUNonZeroU64;
typedef unsigned long long WGPUOption_AdapterId;
typedef unsigned long long WGPUOption_SurfaceId;
typedef unsigned long long WGPUOption_TextureViewId;


#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Bound uniform/storage buffer offsets must be aligned to this number.
 */
#define WGPUBIND_BUFFER_ALIGNMENT 256

#define WGPUDEFAULT_BIND_GROUPS 4

#define WGPUDESIRED_NUM_FRAMES 3

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
  WGPUBindingType_StorageBuffer = 1,
  WGPUBindingType_ReadonlyStorageBuffer = 2,
  WGPUBindingType_Sampler = 3,
  WGPUBindingType_ComparisonSampler = 4,
  WGPUBindingType_SampledTexture = 5,
  WGPUBindingType_ReadonlyStorageTexture = 6,
  WGPUBindingType_WriteonlyStorageTexture = 7,
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
  WGPUCompareFunction_Undefined = 0,
  WGPUCompareFunction_Never = 1,
  WGPUCompareFunction_Less = 2,
  WGPUCompareFunction_Equal = 3,
  WGPUCompareFunction_LessEqual = 4,
  WGPUCompareFunction_Greater = 5,
  WGPUCompareFunction_NotEqual = 6,
  WGPUCompareFunction_GreaterEqual = 7,
  WGPUCompareFunction_Always = 8,
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
  /**
   * The presentation engine does **not** wait for a vertical blanking period and
   * the request is presented immediately. This is a low-latency presentation mode,
   * but visible tearing may be observed. Will fallback to `Fifo` if unavailable on the
   * selected  platform and backend. Not optimal for mobile.
   */
  WGPUPresentMode_Immediate = 0,
  /**
   * The presentation engine waits for the next vertical blanking period to update
   * the current image, but frames may be submitted without delay. This is a low-latency
   * presentation mode and visible tearing will **not** be observed. Will fallback to `Fifo`
   * if unavailable on the selected platform and backend. Not optimal for mobile.
   */
  WGPUPresentMode_Mailbox = 1,
  /**
   * The presentation engine waits for the next vertical blanking period to update
   * the current image. The framerate will be capped at the display refresh rate,
   * corresponding to the `VSync`. Tearing cannot be observed. Optimal for mobile.
   */
  WGPUPresentMode_Fifo = 2,
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
  WGPUStoreOp_Clear = 0,
  WGPUStoreOp_Store = 1,
} WGPUStoreOp;

typedef enum {
  WGPUTextureAspect_All,
  WGPUTextureAspect_StencilOnly,
  WGPUTextureAspect_DepthOnly,
} WGPUTextureAspect;

typedef enum {
  WGPUTextureComponentType_Float,
  WGPUTextureComponentType_Sint,
  WGPUTextureComponentType_Uint,
} WGPUTextureComponentType;

typedef enum {
  WGPUTextureDimension_D1,
  WGPUTextureDimension_D2,
  WGPUTextureDimension_D3,
} WGPUTextureDimension;

typedef enum {
  WGPUTextureFormat_R8Unorm = 0,
  WGPUTextureFormat_R8Snorm = 1,
  WGPUTextureFormat_R8Uint = 2,
  WGPUTextureFormat_R8Sint = 3,
  WGPUTextureFormat_R16Uint = 4,
  WGPUTextureFormat_R16Sint = 5,
  WGPUTextureFormat_R16Float = 6,
  WGPUTextureFormat_Rg8Unorm = 7,
  WGPUTextureFormat_Rg8Snorm = 8,
  WGPUTextureFormat_Rg8Uint = 9,
  WGPUTextureFormat_Rg8Sint = 10,
  WGPUTextureFormat_R32Uint = 11,
  WGPUTextureFormat_R32Sint = 12,
  WGPUTextureFormat_R32Float = 13,
  WGPUTextureFormat_Rg16Uint = 14,
  WGPUTextureFormat_Rg16Sint = 15,
  WGPUTextureFormat_Rg16Float = 16,
  WGPUTextureFormat_Rgba8Unorm = 17,
  WGPUTextureFormat_Rgba8UnormSrgb = 18,
  WGPUTextureFormat_Rgba8Snorm = 19,
  WGPUTextureFormat_Rgba8Uint = 20,
  WGPUTextureFormat_Rgba8Sint = 21,
  WGPUTextureFormat_Bgra8Unorm = 22,
  WGPUTextureFormat_Bgra8UnormSrgb = 23,
  WGPUTextureFormat_Rgb10a2Unorm = 24,
  WGPUTextureFormat_Rg11b10Float = 25,
  WGPUTextureFormat_Rg32Uint = 26,
  WGPUTextureFormat_Rg32Sint = 27,
  WGPUTextureFormat_Rg32Float = 28,
  WGPUTextureFormat_Rgba16Uint = 29,
  WGPUTextureFormat_Rgba16Sint = 30,
  WGPUTextureFormat_Rgba16Float = 31,
  WGPUTextureFormat_Rgba32Uint = 32,
  WGPUTextureFormat_Rgba32Sint = 33,
  WGPUTextureFormat_Rgba32Float = 34,
  WGPUTextureFormat_Depth32Float = 35,
  WGPUTextureFormat_Depth24Plus = 36,
  WGPUTextureFormat_Depth24PlusStencil8 = 37,
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

typedef WGPUNonZeroU64 WGPUId_Adapter_Dummy;

typedef WGPUId_Adapter_Dummy WGPUAdapterId;

typedef WGPUNonZeroU64 WGPUId_Device_Dummy;

typedef WGPUId_Device_Dummy WGPUDeviceId;

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

typedef WGPUNonZeroU64 WGPUId_BindGroup_Dummy;

typedef WGPUId_BindGroup_Dummy WGPUBindGroupId;

typedef WGPUNonZeroU64 WGPUId_Buffer_Dummy;

typedef WGPUId_Buffer_Dummy WGPUBufferId;

typedef uint64_t WGPUBufferAddress;

typedef void (*WGPUBufferMapReadCallback)(WGPUBufferMapAsyncStatus status, const uint8_t *data, uint8_t *userdata);

typedef void (*WGPUBufferMapWriteCallback)(WGPUBufferMapAsyncStatus status, uint8_t *data, uint8_t *userdata);

typedef WGPUNonZeroU64 WGPUId_CommandBuffer_Dummy;

typedef WGPUId_CommandBuffer_Dummy WGPUCommandBufferId;

typedef WGPUCommandBufferId WGPUCommandEncoderId;

typedef struct {
  uint8_t *data;
  uint8_t *base;
  uintptr_t capacity;
  WGPUCommandEncoderId parent;
} WGPURawPass;

typedef struct {
  uint32_t todo;
} WGPUComputePassDescriptor;

typedef WGPUNonZeroU64 WGPUId_TextureView_Dummy;

typedef WGPUId_TextureView_Dummy WGPUTextureViewId;

typedef struct {
  double r;
  double g;
  double b;
  double a;
} WGPUColor;
#define WGPUColor_TRANSPARENT (WGPUColor){ .r = 0.0, .g = 0.0, .b = 0.0, .a = 0.0 }
#define WGPUColor_BLACK (WGPUColor){ .r = 0.0, .g = 0.0, .b = 0.0, .a = 1.0 }
#define WGPUColor_WHITE (WGPUColor){ .r = 1.0, .g = 1.0, .b = 1.0, .a = 1.0 }
#define WGPUColor_RED (WGPUColor){ .r = 1.0, .g = 0.0, .b = 0.0, .a = 1.0 }
#define WGPUColor_GREEN (WGPUColor){ .r = 0.0, .g = 1.0, .b = 0.0, .a = 1.0 }
#define WGPUColor_BLUE (WGPUColor){ .r = 0.0, .g = 0.0, .b = 1.0, .a = 1.0 }

typedef struct {
  WGPUTextureViewId attachment;
  WGPUOption_TextureViewId resolve_target;
  WGPULoadOp load_op;
  WGPUStoreOp store_op;
  WGPUColor clear_color;
} WGPURenderPassColorAttachmentDescriptorBase_TextureViewId;

typedef WGPURenderPassColorAttachmentDescriptorBase_TextureViewId WGPURenderPassColorAttachmentDescriptor;

typedef struct {
  WGPUTextureViewId attachment;
  WGPULoadOp depth_load_op;
  WGPUStoreOp depth_store_op;
  float clear_depth;
  WGPULoadOp stencil_load_op;
  WGPUStoreOp stencil_store_op;
  uint32_t clear_stencil;
} WGPURenderPassDepthStencilAttachmentDescriptorBase_TextureViewId;

typedef WGPURenderPassDepthStencilAttachmentDescriptorBase_TextureViewId WGPURenderPassDepthStencilAttachmentDescriptor;

typedef struct {
  const WGPURenderPassColorAttachmentDescriptor *color_attachments;
  uintptr_t color_attachments_length;
  const WGPURenderPassDepthStencilAttachmentDescriptor *depth_stencil_attachment;
} WGPURenderPassDescriptor;

typedef struct {
  WGPUBufferId buffer;
  WGPUBufferAddress offset;
  uint32_t bytes_per_row;
  uint32_t rows_per_image;
} WGPUBufferCopyView;

typedef WGPUNonZeroU64 WGPUId_Texture_Dummy;

typedef WGPUId_Texture_Dummy WGPUTextureId;

typedef struct {
  uint32_t x;
  uint32_t y;
  uint32_t z;
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

typedef struct {
  uint32_t todo;
} WGPUCommandBufferDescriptor;

typedef WGPURawPass *WGPUComputePassId;

typedef const char *WGPURawString;

typedef uint32_t WGPUDynamicOffset;

typedef WGPUNonZeroU64 WGPUId_ComputePipeline_Dummy;

typedef WGPUId_ComputePipeline_Dummy WGPUComputePipelineId;

typedef WGPUNonZeroU64 WGPUId_Surface;

typedef WGPUId_Surface WGPUSurfaceId;

typedef WGPUNonZeroU64 WGPUId_BindGroupLayout_Dummy;

typedef WGPUId_BindGroupLayout_Dummy WGPUBindGroupLayoutId;

typedef struct {
  WGPUBufferId buffer;
  WGPUBufferAddress offset;
  WGPUBufferAddress size;
} WGPUBufferBinding;

typedef WGPUNonZeroU64 WGPUId_Sampler_Dummy;

typedef WGPUId_Sampler_Dummy WGPUSamplerId;

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
} WGPUBindGroupEntry;

typedef struct {
  const char *label;
  WGPUBindGroupLayoutId layout;
  const WGPUBindGroupEntry *entries;
  uintptr_t entries_length;
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
  bool multisampled;
  bool has_dynamic_offset;
  WGPUTextureViewDimension view_dimension;
  WGPUTextureComponentType texture_component_type;
  WGPUTextureFormat storage_texture_format;
} WGPUBindGroupLayoutEntry;

typedef struct {
  const char *label;
  const WGPUBindGroupLayoutEntry *entries;
  uintptr_t entries_length;
} WGPUBindGroupLayoutDescriptor;

typedef uint32_t WGPUBufferUsage;
#define WGPUBufferUsage_MAP_READ 1
#define WGPUBufferUsage_MAP_WRITE 2
#define WGPUBufferUsage_COPY_SRC 4
#define WGPUBufferUsage_COPY_DST 8
#define WGPUBufferUsage_INDEX 16
#define WGPUBufferUsage_VERTEX 32
#define WGPUBufferUsage_UNIFORM 64
#define WGPUBufferUsage_STORAGE 128
#define WGPUBufferUsage_INDIRECT 256
#define WGPUBufferUsage_STORAGE_READ 512
#define WGPUBufferUsage_NONE 0

typedef struct {
  const char *label;
  WGPUBufferAddress size;
  WGPUBufferUsage usage;
} WGPUBufferDescriptor;

typedef struct {
  const char *label;
} WGPUCommandEncoderDescriptor;

typedef WGPUNonZeroU64 WGPUId_PipelineLayout_Dummy;

typedef WGPUId_PipelineLayout_Dummy WGPUPipelineLayoutId;

typedef WGPUNonZeroU64 WGPUId_ShaderModule_Dummy;

typedef WGPUId_ShaderModule_Dummy WGPUShaderModuleId;

typedef struct {
  WGPUShaderModuleId module;
  WGPURawString entry_point;
} WGPUProgrammableStageDescriptor;

typedef struct {
  WGPUPipelineLayoutId layout;
  WGPUProgrammableStageDescriptor compute_stage;
} WGPUComputePipelineDescriptor;

typedef struct {
  const WGPUBindGroupLayoutId *bind_group_layouts;
  uintptr_t bind_group_layouts_length;
} WGPUPipelineLayoutDescriptor;

typedef WGPUNonZeroU64 WGPUId_RenderPipeline_Dummy;

typedef WGPUId_RenderPipeline_Dummy WGPURenderPipelineId;

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
  WGPUBufferAddress array_stride;
  WGPUInputStepMode step_mode;
  const WGPUVertexAttributeDescriptor *attributes;
  uintptr_t attributes_length;
} WGPUVertexBufferLayoutDescriptor;

typedef struct {
  WGPUIndexFormat index_format;
  const WGPUVertexBufferLayoutDescriptor *vertex_buffers;
  uintptr_t vertex_buffers_length;
} WGPUVertexStateDescriptor;

typedef struct {
  WGPUPipelineLayoutId layout;
  WGPUProgrammableStageDescriptor vertex_stage;
  const WGPUProgrammableStageDescriptor *fragment_stage;
  WGPUPrimitiveTopology primitive_topology;
  const WGPURasterizationStateDescriptor *rasterization_state;
  const WGPUColorStateDescriptor *color_states;
  uintptr_t color_states_length;
  const WGPUDepthStencilStateDescriptor *depth_stencil_state;
  WGPUVertexStateDescriptor vertex_state;
  uint32_t sample_count;
  uint32_t sample_mask;
  bool alpha_to_coverage_enabled;
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
  WGPUCompareFunction compare;
} WGPUSamplerDescriptor;

typedef struct {
  const uint32_t *bytes;
  uintptr_t length;
} WGPUU32Array;

typedef struct {
  WGPUU32Array code;
} WGPUShaderModuleDescriptor;

typedef WGPUNonZeroU64 WGPUId_SwapChain_Dummy;

typedef WGPUId_SwapChain_Dummy WGPUSwapChainId;

typedef uint32_t WGPUTextureUsage;
#define WGPUTextureUsage_COPY_SRC 1
#define WGPUTextureUsage_COPY_DST 2
#define WGPUTextureUsage_SAMPLED 4
#define WGPUTextureUsage_STORAGE 8
#define WGPUTextureUsage_OUTPUT_ATTACHMENT 16
#define WGPUTextureUsage_NONE 0
#define WGPUTextureUsage_UNINITIALIZED 65535

typedef struct {
  WGPUTextureUsage usage;
  WGPUTextureFormat format;
  uint32_t width;
  uint32_t height;
  WGPUPresentMode present_mode;
} WGPUSwapChainDescriptor;

typedef struct {
  const char *label;
  WGPUExtent3d size;
  uint32_t array_layer_count;
  uint32_t mip_level_count;
  uint32_t sample_count;
  WGPUTextureDimension dimension;
  WGPUTextureFormat format;
  WGPUTextureUsage usage;
} WGPUTextureDescriptor;

typedef WGPUDeviceId WGPUQueueId;

typedef WGPURawPass *WGPURenderPassId;

typedef WGPUNonZeroU64 WGPUId_RenderBundle_Dummy;

typedef WGPUId_RenderBundle_Dummy WGPURenderBundleId;

typedef struct {
  WGPUPowerPreference power_preference;
  WGPUOption_SurfaceId compatible_surface;
} WGPURequestAdapterOptions;

typedef uint32_t WGPUBackendBit;

typedef void (*WGPURequestAdapterCallback)(WGPUOption_AdapterId id, void *userdata);

typedef struct {
  WGPUOption_TextureViewId view_id;
} WGPUSwapChainOutput;

typedef struct {
  WGPUTextureFormat format;
  WGPUTextureViewDimension dimension;
  WGPUTextureAspect aspect;
  uint32_t base_mip_level;
  uint32_t level_count;
  uint32_t base_array_layer;
  uint32_t array_layer_count;
} WGPUTextureViewDescriptor;

void wgpu_adapter_destroy(WGPUAdapterId adapter_id);

WGPUDeviceId wgpu_adapter_request_device(WGPUAdapterId adapter_id,
                                         const WGPUDeviceDescriptor *desc);

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

void wgpu_command_buffer_destroy(WGPUCommandBufferId command_buffer_id);

/**
 * # Safety
 *
 * This function is unsafe because improper use may lead to memory
 * problems. For example, a double-free may occur if the function is called
 * twice on the same raw pointer.
 */
WGPURawPass *wgpu_command_encoder_begin_compute_pass(WGPUCommandEncoderId encoder_id,
                                                     const WGPUComputePassDescriptor *_desc);

/**
 * # Safety
 *
 * This function is unsafe because improper use may lead to memory
 * problems. For example, a double-free may occur if the function is called
 * twice on the same raw pointer.
 */
WGPURawPass *wgpu_command_encoder_begin_render_pass(WGPUCommandEncoderId encoder_id,
                                                    const WGPURenderPassDescriptor *desc);

void wgpu_command_encoder_copy_buffer_to_buffer(WGPUCommandEncoderId command_encoder_id,
                                                WGPUBufferId source,
                                                WGPUBufferAddress source_offset,
                                                WGPUBufferId destination,
                                                WGPUBufferAddress destination_offset,
                                                WGPUBufferAddress size);

void wgpu_command_encoder_copy_buffer_to_texture(WGPUCommandEncoderId command_encoder_id,
                                                 const WGPUBufferCopyView *source,
                                                 const WGPUTextureCopyView *destination,
                                                 WGPUExtent3d copy_size);

void wgpu_command_encoder_copy_texture_to_buffer(WGPUCommandEncoderId command_encoder_id,
                                                 const WGPUTextureCopyView *source,
                                                 const WGPUBufferCopyView *destination,
                                                 WGPUExtent3d copy_size);

void wgpu_command_encoder_copy_texture_to_texture(WGPUCommandEncoderId command_encoder_id,
                                                  const WGPUTextureCopyView *source,
                                                  const WGPUTextureCopyView *destination,
                                                  WGPUExtent3d copy_size);

void wgpu_command_encoder_destroy(WGPUCommandEncoderId command_encoder_id);

WGPUCommandBufferId wgpu_command_encoder_finish(WGPUCommandEncoderId encoder_id,
                                                const WGPUCommandBufferDescriptor *desc);

void wgpu_compute_pass_destroy(WGPURawPass *pass);

void wgpu_compute_pass_dispatch(WGPURawPass *pass,
                                uint32_t groups_x,
                                uint32_t groups_y,
                                uint32_t groups_z);

void wgpu_compute_pass_dispatch_indirect(WGPURawPass *pass,
                                         WGPUBufferId buffer_id,
                                         WGPUBufferAddress offset);

void wgpu_compute_pass_end_pass(WGPUComputePassId pass_id);

const uint8_t *wgpu_compute_pass_finish(WGPURawPass *pass, uintptr_t *length);

void wgpu_compute_pass_insert_debug_marker(WGPURawPass *_pass, WGPURawString _label);

void wgpu_compute_pass_pop_debug_group(WGPURawPass *_pass);

void wgpu_compute_pass_push_debug_group(WGPURawPass *_pass, WGPURawString _label);

/**
 * # Safety
 *
 * This function is unsafe as there is no guarantee that the given pointer is
 * valid for `offset_length` elements.
 */
void wgpu_compute_pass_set_bind_group(WGPURawPass *pass,
                                      uint32_t index,
                                      WGPUBindGroupId bind_group_id,
                                      const WGPUDynamicOffset *offsets,
                                      uintptr_t offset_length);

void wgpu_compute_pass_set_pipeline(WGPURawPass *pass, WGPUComputePipelineId pipeline_id);

WGPUSurfaceId wgpu_create_surface_from_metal_layer(void *layer);

WGPUSurfaceId wgpu_create_surface_from_wayland(void *surface, void *display);

WGPUSurfaceId wgpu_create_surface_from_windows_hwnd(void *_hinstance, void *hwnd);

WGPUSurfaceId wgpu_create_surface_from_xlib(const void **display, unsigned long window);

WGPUBindGroupId wgpu_device_create_bind_group(WGPUDeviceId device_id,
                                              const WGPUBindGroupDescriptor *desc);

WGPUBindGroupLayoutId wgpu_device_create_bind_group_layout(WGPUDeviceId device_id,
                                                           const WGPUBindGroupLayoutDescriptor *desc);

WGPUBufferId wgpu_device_create_buffer(WGPUDeviceId device_id, const WGPUBufferDescriptor *desc);

/**
 * # Safety
 *
 * This function is unsafe as there is no guarantee that the given pointer
 * dereferenced in this function is valid.
 */
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

WGPUQueueId wgpu_device_get_default_queue(WGPUDeviceId device_id);

void wgpu_device_get_limits(WGPUDeviceId _device_id, WGPULimits *limits);

void wgpu_device_poll(WGPUDeviceId device_id, bool force_wait);

/**
 * # Safety
 *
 * This function is unsafe as there is no guarantee that the given pointer is
 * valid for `command_buffers_length` elements.
 */
void wgpu_queue_submit(WGPUQueueId queue_id,
                       const WGPUCommandBufferId *command_buffers,
                       uintptr_t command_buffers_length);

void wgpu_render_pass_destroy(WGPURawPass *pass);

void wgpu_render_pass_draw(WGPURawPass *pass,
                           uint32_t vertex_count,
                           uint32_t instance_count,
                           uint32_t first_vertex,
                           uint32_t first_instance);

void wgpu_render_pass_draw_indexed(WGPURawPass *pass,
                                   uint32_t index_count,
                                   uint32_t instance_count,
                                   uint32_t first_index,
                                   int32_t base_vertex,
                                   uint32_t first_instance);

void wgpu_render_pass_draw_indexed_indirect(WGPURawPass *pass,
                                            WGPUBufferId buffer_id,
                                            WGPUBufferAddress offset);

void wgpu_render_pass_draw_indirect(WGPURawPass *pass,
                                    WGPUBufferId buffer_id,
                                    WGPUBufferAddress offset);

/**
 * # Safety
 *
 * This function is unsafe because improper use may lead to memory
 * problems. For example, a double-free may occur if the function is called
 * twice on the same raw pointer.
 */
void wgpu_render_pass_end_pass(WGPURenderPassId pass_id);

void wgpu_render_pass_execute_bundles(WGPURawPass *_pass,
                                      const WGPURenderBundleId *_bundles,
                                      uintptr_t _bundles_length);

const uint8_t *wgpu_render_pass_finish(WGPURawPass *pass, uintptr_t *length);

void wgpu_render_pass_insert_debug_marker(WGPURawPass *_pass, WGPURawString _label);

void wgpu_render_pass_pop_debug_group(WGPURawPass *_pass);

void wgpu_render_pass_push_debug_group(WGPURawPass *_pass, WGPURawString _label);

/**
 * # Safety
 *
 * This function is unsafe as there is no guarantee that the given pointer is
 * valid for `offset_length` elements.
 */
void wgpu_render_pass_set_bind_group(WGPURawPass *pass,
                                     uint32_t index,
                                     WGPUBindGroupId bind_group_id,
                                     const WGPUDynamicOffset *offsets,
                                     uintptr_t offset_length);

void wgpu_render_pass_set_blend_color(WGPURawPass *pass, const WGPUColor *color);

void wgpu_render_pass_set_index_buffer(WGPURawPass *pass,
                                       WGPUBufferId buffer_id,
                                       WGPUBufferAddress offset,
                                       WGPUBufferAddress size);

void wgpu_render_pass_set_pipeline(WGPURawPass *pass, WGPURenderPipelineId pipeline_id);

void wgpu_render_pass_set_scissor_rect(WGPURawPass *pass,
                                       uint32_t x,
                                       uint32_t y,
                                       uint32_t w,
                                       uint32_t h);

void wgpu_render_pass_set_stencil_reference(WGPURawPass *pass, uint32_t value);

void wgpu_render_pass_set_vertex_buffer(WGPURawPass *pass,
                                        uint32_t slot,
                                        WGPUBufferId buffer_id,
                                        WGPUBufferAddress offset,
                                        WGPUBufferAddress size);

void wgpu_render_pass_set_viewport(WGPURawPass *pass,
                                   float x,
                                   float y,
                                   float w,
                                   float h,
                                   float depth_min,
                                   float depth_max);

/**
 * # Safety
 *
 * This function is unsafe as it calls an unsafe extern callback.
 */
void wgpu_request_adapter_async(const WGPURequestAdapterOptions *desc,
                                WGPUBackendBit mask,
                                WGPURequestAdapterCallback callback,
                                void *userdata);

void wgpu_sampler_destroy(WGPUSamplerId sampler_id);

WGPUSwapChainOutput wgpu_swap_chain_get_next_texture(WGPUSwapChainId swap_chain_id);

void wgpu_swap_chain_present(WGPUSwapChainId swap_chain_id);

WGPUTextureViewId wgpu_texture_create_view(WGPUTextureId texture_id,
                                           const WGPUTextureViewDescriptor *desc);

void wgpu_texture_destroy(WGPUTextureId texture_id);

void wgpu_texture_view_destroy(WGPUTextureViewId texture_view_id);
