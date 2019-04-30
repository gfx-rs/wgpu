#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define WGPUBITS_PER_BYTE 8

#define WGPUMAX_BIND_GROUPS 4

#define WGPUMAX_COLOR_TARGETS 4

typedef enum {
  WGPUBufferMapAsyncStatus_Success,
  WGPUBufferMapAsyncStatus_Error,
  WGPUBufferMapAsyncStatus_Unknown,
  WGPUBufferMapAsyncStatus_ContextLost,
} WGPUBufferMapAsyncStatus;

typedef enum {
  WGPUPowerPreference_Default = 0,
  WGPUPowerPreference_LowPower = 1,
  WGPUPowerPreference_HighPerformance = 2,
} WGPUPowerPreference;

typedef uint32_t WGPUIndex;

typedef uint32_t WGPUEpoch;

typedef struct {
  WGPUIndex _0;
  WGPUEpoch _1;
} WGPUId;

typedef WGPUId WGPUAdapterId;

typedef struct {
  WGPUPowerPreference power_preference;
} WGPUAdapterDescriptor;

typedef struct {
  bool anisotropic_filtering;
} WGPUExtensions;

typedef struct {
  WGPUExtensions extensions;
} WGPUDeviceDescriptor;

typedef struct {
  WGPUAdapterId adapter;
  WGPUAdapterDescriptor adapter_desc;
  WGPUDeviceDescriptor device_desc;
} WGPUForcedExports;

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

typedef WGPUCommandBufferId WGPUCommandEncoderId;

typedef WGPUId WGPUComputePassId;

typedef WGPUId WGPUComputePipelineId;

typedef WGPUId WGPUDeviceId;

typedef WGPUDeviceId WGPUQueueId;

typedef WGPUId WGPURenderPassId;

typedef struct {
  float r;
  float g;
  float b;
  float a;
} WGPUColor;

typedef WGPUId WGPURenderPipelineId;

typedef WGPUId WGPUTextureViewId;

typedef struct {
  WGPUTextureId texture_id;
  WGPUTextureViewId view_id;
} WGPUSwapChainOutput;

typedef WGPUId WGPUSurfaceId;

typedef WGPUSurfaceId WGPUSwapChainId;

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
