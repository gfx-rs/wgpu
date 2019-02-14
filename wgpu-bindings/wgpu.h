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

typedef WGPUId WGPUBufferId;

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

typedef WGPUId WGPUBindGroupId;

typedef WGPUId WGPUComputePipelineId;

typedef WGPUId WGPUDeviceId;

typedef WGPUDeviceId WGPUQueueId;

typedef WGPUId WGPURenderPassId;

typedef WGPUId WGPURenderPipelineId;

typedef WGPUId WGPUTextureViewId;

typedef struct {
  WGPUTextureId texture_id;
  WGPUTextureViewId view_id;
} WGPUSwapChainOutput;

typedef WGPUId WGPUSwapChainId;

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

void wgpu_buffer_destroy(WGPUBufferId buffer_id);

void wgpu_buffer_set_sub_data(WGPUBufferId buffer_id,
                              uint32_t start,
                              uint32_t count,
                              const uint8_t *data);

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

WGPUCommandBufferId wgpu_command_encoder_finish(WGPUCommandEncoderId command_encoder_id);

void wgpu_compute_pass_dispatch(WGPUComputePassId pass_id, uint32_t x, uint32_t y, uint32_t z);

WGPUCommandBufferId wgpu_compute_pass_end_pass(WGPUComputePassId pass_id);

void wgpu_compute_pass_set_bind_group(WGPUComputePassId pass_id,
                                      uint32_t index,
                                      WGPUBindGroupId bind_group_id);

void wgpu_compute_pass_set_pipeline(WGPUComputePassId pass_id, WGPUComputePipelineId pipeline_id);

WGPUQueueId wgpu_device_get_queue(WGPUDeviceId device_id);

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

void wgpu_texture_destroy(WGPUTextureId texture_id);

void wgpu_texture_view_destroy(WGPUTextureViewId _texture_view_id);
