#define WGPU_LOCAL


/* Generated with cbindgen:0.9.1 */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

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
  WGPUBindingType_SampledTexture = 4,
  WGPUBindingType_StorageTexture = 5,
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
  WGPUStoreOp_Clear = 0,
  WGPUStoreOp_Store = 1,
} WGPUStoreOp;

typedef enum {
  WGPUTextureAspect_All,
  WGPUTextureAspect_StencilOnly,
  WGPUTextureAspect_DepthOnly,
} WGPUTextureAspect;

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
  WGPUTextureFormat_R16Unorm = 4,
  WGPUTextureFormat_R16Snorm = 5,
  WGPUTextureFormat_R16Uint = 6,
  WGPUTextureFormat_R16Sint = 7,
  WGPUTextureFormat_R16Float = 8,
  WGPUTextureFormat_Rg8Unorm = 9,
  WGPUTextureFormat_Rg8Snorm = 10,
  WGPUTextureFormat_Rg8Uint = 11,
  WGPUTextureFormat_Rg8Sint = 12,
  WGPUTextureFormat_R32Uint = 13,
  WGPUTextureFormat_R32Sint = 14,
  WGPUTextureFormat_R32Float = 15,
  WGPUTextureFormat_Rg16Unorm = 16,
  WGPUTextureFormat_Rg16Snorm = 17,
  WGPUTextureFormat_Rg16Uint = 18,
  WGPUTextureFormat_Rg16Sint = 19,
  WGPUTextureFormat_Rg16Float = 20,
  WGPUTextureFormat_Rgba8Unorm = 21,
  WGPUTextureFormat_Rgba8UnormSrgb = 22,
  WGPUTextureFormat_Rgba8Snorm = 23,
  WGPUTextureFormat_Rgba8Uint = 24,
  WGPUTextureFormat_Rgba8Sint = 25,
  WGPUTextureFormat_Bgra8Unorm = 26,
  WGPUTextureFormat_Bgra8UnormSrgb = 27,
  WGPUTextureFormat_Rgb10a2Unorm = 28,
  WGPUTextureFormat_Rg11b10Float = 29,
  WGPUTextureFormat_Rg32Uint = 30,
  WGPUTextureFormat_Rg32Sint = 31,
  WGPUTextureFormat_Rg32Float = 32,
  WGPUTextureFormat_Rgba16Unorm = 33,
  WGPUTextureFormat_Rgba16Snorm = 34,
  WGPUTextureFormat_Rgba16Uint = 35,
  WGPUTextureFormat_Rgba16Sint = 36,
  WGPUTextureFormat_Rgba16Float = 37,
  WGPUTextureFormat_Rgba32Uint = 38,
  WGPUTextureFormat_Rgba32Sint = 39,
  WGPUTextureFormat_Rgba32Float = 40,
  WGPUTextureFormat_Depth32Float = 41,
  WGPUTextureFormat_Depth24Plus = 42,
  WGPUTextureFormat_Depth24PlusStencil8 = 43,
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

typedef struct WGPUEventLoop WGPUEventLoop;

typedef uint64_t WGPUId_Device_Dummy;

typedef WGPUId_Device_Dummy WGPUDeviceId;

typedef uint64_t WGPUId_Adapter_Dummy;

typedef WGPUId_Adapter_Dummy WGPUAdapterId;

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

typedef uint64_t WGPUId_BindGroup_Dummy;

typedef WGPUId_BindGroup_Dummy WGPUBindGroupId;

typedef uint64_t WGPUId_Buffer_Dummy;

typedef WGPUId_Buffer_Dummy WGPUBufferId;

typedef uint64_t WGPUBufferAddress;

typedef void (*WGPUBufferMapReadCallback)(WGPUBufferMapAsyncStatus status, const uint8_t *data, uint8_t *userdata);

typedef WGPUEventLoop *WGPUEventLoopId;

typedef void (*WGPUBufferMapWriteCallback)(WGPUBufferMapAsyncStatus status, uint8_t *data, uint8_t *userdata);

typedef uint64_t WGPUId_ComputePass_Dummy;

typedef WGPUId_ComputePass_Dummy WGPUComputePassId;

typedef uint64_t WGPUId_CommandBuffer_Dummy;

typedef WGPUId_CommandBuffer_Dummy WGPUCommandBufferId;

typedef WGPUCommandBufferId WGPUCommandEncoderId;

typedef struct {
  uint32_t todo;
} WGPUComputePassDescriptor;

typedef uint64_t WGPUId_RenderPass_Dummy;

typedef WGPUId_RenderPass_Dummy WGPURenderPassId;

typedef uint64_t WGPUId_TextureView_Dummy;

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

typedef struct {
  WGPUBufferId buffer;
  WGPUBufferAddress offset;
  uint32_t row_pitch;
  uint32_t image_height;
} WGPUBufferCopyView;

typedef uint64_t WGPUId_Texture_Dummy;

typedef WGPUId_Texture_Dummy WGPUTextureId;

typedef struct {
  float x;
  float y;
  float z;
} WGPUOrigin3d;
#define WGPUOrigin3d_ZERO (WGPUOrigin3d){ .x = 0.0, .y = 0.0, .z = 0.0 }

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

typedef const char *WGPURawString;

typedef uint64_t WGPUId_ComputePipeline_Dummy;

typedef WGPUId_ComputePipeline_Dummy WGPUComputePipelineId;

typedef uint64_t WGPUId_Surface;

typedef WGPUId_Surface WGPUSurfaceId;

typedef uint64_t WGPUId_BindGroupLayout_Dummy;

typedef WGPUId_BindGroupLayout_Dummy WGPUBindGroupLayoutId;

typedef struct {
  WGPUBufferId buffer;
  WGPUBufferAddress offset;
  WGPUBufferAddress size;
} WGPUBufferBinding;

typedef uint64_t WGPUId_Sampler_Dummy;

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
  WGPUTextureViewDimension texture_dimension;
  bool multisampled;
  bool dynamic;
} WGPUBindGroupLayoutBinding;

typedef struct {
  const WGPUBindGroupLayoutBinding *bindings;
  uintptr_t bindings_length;
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
#define WGPUBufferUsage_STORAGE_READ 256
#define WGPUBufferUsage_INDIRECT 512
#define WGPUBufferUsage_NONE 0

typedef struct {
  WGPUBufferAddress size;
  WGPUBufferUsage usage;
} WGPUBufferDescriptor;

typedef struct {
  uint32_t todo;
} WGPUCommandEncoderDescriptor;

typedef uint64_t WGPUId_PipelineLayout_Dummy;

typedef WGPUId_PipelineLayout_Dummy WGPUPipelineLayoutId;

typedef uint64_t WGPUId_ShaderModule_Dummy;

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

typedef uint64_t WGPUId_RenderPipeline_Dummy;

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
  WGPUProgrammableStageDescriptor vertex_stage;
  const WGPUProgrammableStageDescriptor *fragment_stage;
  WGPUPrimitiveTopology primitive_topology;
  const WGPURasterizationStateDescriptor *rasterization_state;
  const WGPUColorStateDescriptor *color_states;
  uintptr_t color_states_length;
  const WGPUDepthStencilStateDescriptor *depth_stencil_state;
  WGPUVertexInputDescriptor vertex_input;
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
  WGPUCompareFunction compare_function;
} WGPUSamplerDescriptor;

typedef struct {
  const uint32_t *bytes;
  uintptr_t length;
} WGPUU32Array;

typedef struct {
  WGPUU32Array code;
} WGPUShaderModuleDescriptor;

typedef uint64_t WGPUId_SwapChain_Dummy;

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
  WGPUExtent3d size;
  uint32_t array_layer_count;
  uint32_t mip_level_count;
  uint32_t sample_count;
  WGPUTextureDimension dimension;
  WGPUTextureFormat format;
  WGPUTextureUsage usage;
} WGPUTextureDescriptor;

typedef WGPUDeviceId WGPUQueueId;

typedef uint64_t WGPUId_RenderBundle_Dummy;

typedef WGPUId_RenderBundle_Dummy WGPURenderBundleId;

typedef uint32_t WGPUBackendBit;

typedef struct {
  WGPUPowerPreference power_preference;
  WGPUBackendBit backends;
} WGPURequestAdapterOptions;

typedef void (*WGPURequestAdapterCallback)(const WGPUAdapterId *adapter, void *userdata);

typedef struct {
  WGPUTextureViewId view_id;
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

#if defined(WGPU_LOCAL)
WGPUDeviceId wgpu_adapter_request_device(WGPUAdapterId adapter_id,
                                         const WGPUDeviceDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
void wgpu_bind_group_destroy(WGPUBindGroupId bind_group_id);
#endif

#if defined(WGPU_LOCAL)
void wgpu_buffer_destroy(WGPUBufferId buffer_id);
#endif

#if defined(WGPU_LOCAL)
void wgpu_buffer_map_read_async(WGPUBufferId buffer_id,
                                WGPUBufferAddress start,
                                WGPUBufferAddress size,
                                WGPUBufferMapReadCallback callback,
                                uint8_t *userdata,
                                WGPUEventLoopId event_loop_id);
#endif

#if defined(WGPU_LOCAL)
void wgpu_buffer_map_write_async(WGPUBufferId buffer_id,
                                 WGPUBufferAddress start,
                                 WGPUBufferAddress size,
                                 WGPUBufferMapWriteCallback callback,
                                 uint8_t *userdata,
                                 WGPUEventLoopId event_loop_id);
#endif

#if defined(WGPU_LOCAL)
void wgpu_buffer_unmap(WGPUBufferId buffer_id);
#endif

#if defined(WGPU_LOCAL)
WGPUComputePassId wgpu_command_encoder_begin_compute_pass(WGPUCommandEncoderId encoder_id,
                                                          const WGPUComputePassDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
WGPURenderPassId wgpu_command_encoder_begin_render_pass(WGPUCommandEncoderId encoder_id,
                                                        const WGPURenderPassDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
void wgpu_command_encoder_copy_buffer_to_buffer(WGPUCommandEncoderId command_encoder_id,
                                                WGPUBufferId source,
                                                WGPUBufferAddress source_offset,
                                                WGPUBufferId destination,
                                                WGPUBufferAddress destination_offset,
                                                WGPUBufferAddress size);
#endif

#if defined(WGPU_LOCAL)
void wgpu_command_encoder_copy_buffer_to_texture(WGPUCommandEncoderId command_encoder_id,
                                                 const WGPUBufferCopyView *source,
                                                 const WGPUTextureCopyView *destination,
                                                 WGPUExtent3d copy_size);
#endif

#if defined(WGPU_LOCAL)
void wgpu_command_encoder_copy_texture_to_buffer(WGPUCommandEncoderId command_encoder_id,
                                                 const WGPUTextureCopyView *source,
                                                 const WGPUBufferCopyView *destination,
                                                 WGPUExtent3d copy_size);
#endif

#if defined(WGPU_LOCAL)
void wgpu_command_encoder_copy_texture_to_texture(WGPUCommandEncoderId command_encoder_id,
                                                  const WGPUTextureCopyView *source,
                                                  const WGPUTextureCopyView *destination,
                                                  WGPUExtent3d copy_size);
#endif

#if defined(WGPU_LOCAL)
WGPUCommandBufferId wgpu_command_encoder_finish(WGPUCommandEncoderId encoder_id,
                                                const WGPUCommandBufferDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
void wgpu_compute_pass_dispatch(WGPUComputePassId pass_id, uint32_t x, uint32_t y, uint32_t z);
#endif

#if defined(WGPU_LOCAL)
void wgpu_compute_pass_dispatch_indirect(WGPUComputePassId pass_id,
                                         WGPUBufferId indirect_buffer_id,
                                         WGPUBufferAddress indirect_offset);
#endif

#if defined(WGPU_LOCAL)
void wgpu_compute_pass_end_pass(WGPUComputePassId pass_id);
#endif

void wgpu_compute_pass_insert_debug_marker(WGPUComputePassId _pass_id, WGPURawString _label);

void wgpu_compute_pass_pop_debug_group(WGPUComputePassId _pass_id);

void wgpu_compute_pass_push_debug_group(WGPUComputePassId _pass_id, WGPURawString _label);

#if defined(WGPU_LOCAL)
void wgpu_compute_pass_set_bind_group(WGPUComputePassId pass_id,
                                      uint32_t index,
                                      WGPUBindGroupId bind_group_id,
                                      const WGPUBufferAddress *offsets,
                                      uintptr_t offsets_length);
#endif

#if defined(WGPU_LOCAL)
void wgpu_compute_pass_set_pipeline(WGPUComputePassId pass_id, WGPUComputePipelineId pipeline_id);
#endif

#if defined(WGPU_LOCAL)
WGPUEventLoopId wgpu_create_event_loop(void);
#endif

#if defined(WGPU_LOCAL)
WGPUSurfaceId wgpu_create_surface_from_metal_layer(void *layer);
#endif

#if defined(WGPU_LOCAL)
WGPUSurfaceId wgpu_create_surface_from_windows_hwnd(void *_hinstance, void *hwnd);
#endif

#if defined(WGPU_LOCAL)
WGPUSurfaceId wgpu_create_surface_from_xlib(const void **display, uint64_t window);
#endif

#if defined(WGPU_LOCAL)
void wgpu_destroy_event_loop(WGPUEventLoopId event_loop_id);
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

#if defined(WGPU_LOCAL)
void wgpu_device_destroy(WGPUDeviceId device_id, WGPUEventLoopId event_loop_id);
#endif

#if defined(WGPU_LOCAL)
void wgpu_device_get_limits(WGPUDeviceId _device_id, WGPULimits *limits);
#endif

#if defined(WGPU_LOCAL)
WGPUQueueId wgpu_device_get_queue(WGPUDeviceId device_id);
#endif

#if defined(WGPU_LOCAL)
void wgpu_process_events(WGPUEventLoopId event_loop_id);
#endif

#if defined(WGPU_LOCAL)
void wgpu_queue_submit(WGPUQueueId queue_id,
                       const WGPUCommandBufferId *command_buffers,
                       uintptr_t command_buffers_length,
                       WGPUEventLoopId event_loop_id);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_draw(WGPURenderPassId pass_id,
                           uint32_t vertex_count,
                           uint32_t instance_count,
                           uint32_t first_vertex,
                           uint32_t first_instance);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_draw_indexed(WGPURenderPassId pass_id,
                                   uint32_t index_count,
                                   uint32_t instance_count,
                                   uint32_t first_index,
                                   int32_t base_vertex,
                                   uint32_t first_instance);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_draw_indexed_indirect(WGPURenderPassId pass_id,
                                            WGPUBufferId indirect_buffer_id,
                                            WGPUBufferAddress indirect_offset);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_draw_indirect(WGPURenderPassId pass_id,
                                    WGPUBufferId indirect_buffer_id,
                                    WGPUBufferAddress indirect_offset);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_end_pass(WGPURenderPassId pass_id);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_execute_bundles(WGPURenderPassId _pass_id,
                                      const WGPURenderBundleId *_bundles,
                                      uintptr_t _bundles_length);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_insert_debug_marker(WGPURenderPassId _pass_id, WGPURawString _label);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_pop_debug_group(WGPURenderPassId _pass_id);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_push_debug_group(WGPURenderPassId _pass_id, WGPURawString _label);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_set_bind_group(WGPURenderPassId pass_id,
                                     uint32_t index,
                                     WGPUBindGroupId bind_group_id,
                                     const WGPUBufferAddress *offsets,
                                     uintptr_t offsets_length);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_set_blend_color(WGPURenderPassId pass_id, const WGPUColor *color);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_set_index_buffer(WGPURenderPassId pass_id,
                                       WGPUBufferId buffer_id,
                                       WGPUBufferAddress offset);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_set_pipeline(WGPURenderPassId pass_id, WGPURenderPipelineId pipeline_id);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_set_scissor_rect(WGPURenderPassId pass_id,
                                       uint32_t x,
                                       uint32_t y,
                                       uint32_t w,
                                       uint32_t h);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_set_stencil_reference(WGPURenderPassId pass_id, uint32_t value);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_set_vertex_buffers(WGPURenderPassId pass_id,
                                         uint32_t start_slot,
                                         const WGPUBufferId *buffers,
                                         const WGPUBufferAddress *offsets,
                                         uintptr_t length);
#endif

#if defined(WGPU_LOCAL)
void wgpu_render_pass_set_viewport(WGPURenderPassId pass_id,
                                   float x,
                                   float y,
                                   float w,
                                   float h,
                                   float min_depth,
                                   float max_depth);
#endif

#if defined(WGPU_LOCAL)
void wgpu_request_adapter_async(const WGPURequestAdapterOptions *desc,
                                WGPUEventLoopId event_loop_id,
                                WGPURequestAdapterCallback callback,
                                void *userdata);
#endif

#if defined(WGPU_LOCAL)
void wgpu_sampler_destroy(WGPUSamplerId sampler_id);
#endif

#if defined(WGPU_LOCAL)
WGPUSwapChainOutput wgpu_swap_chain_get_next_texture(WGPUSwapChainId swap_chain_id);
#endif

#if defined(WGPU_LOCAL)
void wgpu_swap_chain_present(WGPUSwapChainId swap_chain_id);
#endif

#if defined(WGPU_LOCAL)
WGPUTextureViewId wgpu_texture_create_view(WGPUTextureId texture_id,
                                           const WGPUTextureViewDescriptor *desc);
#endif

#if defined(WGPU_LOCAL)
void wgpu_texture_destroy(WGPUTextureId texture_id);
#endif

#if defined(WGPU_LOCAL)
void wgpu_texture_view_destroy(WGPUTextureViewId texture_view_id);
#endif
