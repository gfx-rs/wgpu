#include <stdio.h>
#include "./../../wgpu-bindings/wgpu.h"

#define WGPU_TARGET_MACOS 1
#define WGPU_TARGET_LINUX 2
#define WGPU_TARGET_WINDOWS 3

#if WGPU_TARGET == WGPU_TARGET_MACOS
#include <QuartzCore/CAMetalLayer.h>
#include <Foundation/Foundation.h>
#endif

#include <GLFW/glfw3.h>
#if WGPU_TARGET == WGPU_TARGET_MACOS
#define GLFW_EXPOSE_NATIVE_COCOA
#elif WGPU_TARGET == WGPU_TARGET_LINUX
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_WAYLAND
#elif WGPU_TARGET == WGPU_TARGET_WINDOWS
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include <GLFW/glfw3native.h>

#define STAGES_LENGTH (2)
#define BLEND_STATES_LENGTH (1)
#define ATTACHMENTS_LENGTH (1)
#define RENDER_PASS_ATTACHMENTS_LENGTH (1)
#define BIND_GROUP_LAYOUTS_LENGTH (1)

WGPUByteArray read_file(const char *name)
{
    FILE *file = fopen(name, "rb");
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    unsigned char *bytes = malloc(length);
    fseek(file, 0, SEEK_SET);
    fread(bytes, 1, length, file);
    fclose(file);
    WGPUByteArray ret = {
        .bytes = bytes,
        .length = length,
    };
    return ret;
}

int main()
{
    WGPUInstanceId instance = wgpu_create_instance();
    WGPUAdapterDescriptor adapter_desc = {
        .power_preference = WGPUPowerPreference_LowPower,
    };
    WGPUAdapterId adapter = wgpu_instance_get_adapter(instance, &adapter_desc);
    WGPUDeviceDescriptor device_desc = {
        .extensions = {
            .anisotropic_filtering = false,
        },
    };
    WGPUDeviceId device = wgpu_adapter_create_device(adapter, &device_desc);

    WGPUShaderModuleDescriptor vertex_shader_desc = {
        .code = read_file("./../../data/hello_triangle.vert.spv"),
    };
    WGPUShaderModuleId vertex_shader = wgpu_device_create_shader_module(device, &vertex_shader_desc);
    WGPUPipelineStageDescriptor vertex_stage = {
        .module = vertex_shader,
        .stage = WGPUShaderStage_Vertex,
        .entry_point = "main",
    };

    WGPUShaderModuleDescriptor fragment_shader_desc = {
        .code = read_file("./../../data/hello_triangle.frag.spv"),
    };
    WGPUShaderModuleId fragment_shader = wgpu_device_create_shader_module(device, &fragment_shader_desc);
    WGPUPipelineStageDescriptor fragment_stage = {
        .module = fragment_shader,
        .stage = WGPUShaderStage_Fragment,
        .entry_point = "main",
    };

    WGPUPipelineStageDescriptor stages[STAGES_LENGTH] = {vertex_stage, fragment_stage};

    WGPUBindGroupLayoutDescriptor bind_group_layout_desc = {
        .bindings = NULL,
        .bindings_length = 0,
    };
    WGPUBindGroupLayoutId bind_group_layout = wgpu_device_create_bind_group_layout(device, &bind_group_layout_desc);

    WGPUBindGroupLayoutId bind_group_layouts[BIND_GROUP_LAYOUTS_LENGTH] = { bind_group_layout };

    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
        .bind_group_layouts = bind_group_layouts,
        .bind_group_layouts_length = BIND_GROUP_LAYOUTS_LENGTH,
    };
    WGPUPipelineLayoutId pipeline_layout = wgpu_device_create_pipeline_layout(device, &pipeline_layout_desc);

    WGPUBlendDescriptor blend_alpha = {
        .src_factor = WGPUBlendFactor_One,
        .dst_factor = WGPUBlendFactor_Zero,
        .operation = WGPUBlendOperation_Add,
    };
    WGPUBlendDescriptor blend_color = {
        .src_factor = WGPUBlendFactor_One,
        .dst_factor = WGPUBlendFactor_Zero,
        .operation = WGPUBlendOperation_Add,
    };
    WGPUBlendStateDescriptor blend_state_0_desc = {
        .blend_enabled = false,
        .alpha = blend_alpha,
        .color = blend_color,
        .write_mask = WGPUColorWriteFlags_ALL,
    };
    WGPUBlendStateId blend_state_0 = wgpu_device_create_blend_state(device, &blend_state_0_desc);
    WGPUBlendStateId blend_state[BLEND_STATES_LENGTH] = {blend_state_0};

    WGPUStencilStateFaceDescriptor stencil_state_front = {
        .compare = WGPUCompareFunction_Never,
        .stencil_fail_op = WGPUStencilOperation_Keep,
        .depth_fail_op = WGPUStencilOperation_Keep,
        .pass_op = WGPUStencilOperation_Keep,
    };
    WGPUStencilStateFaceDescriptor stencil_state_back = {
        .compare = WGPUCompareFunction_Never,
        .stencil_fail_op = WGPUStencilOperation_Keep,
        .depth_fail_op = WGPUStencilOperation_Keep,
        .pass_op = WGPUStencilOperation_Keep,
    };
    WGPUDepthStencilStateDescriptor depth_stencil_state_desc = {
        .depth_write_enabled = false,
        .depth_compare = WGPUCompareFunction_Never,
        .front = stencil_state_front,
        .back = stencil_state_back,
        .stencil_read_mask = 0,
        .stencil_write_mask = 0,
    };
    WGPUDepthStencilStateId depth_stencil_state = wgpu_device_create_depth_stencil_state(device, &depth_stencil_state_desc);

    WGPUAttachment attachments[ATTACHMENTS_LENGTH] = {
        {
            .format = WGPUTextureFormat_Bgra8Unorm,
            .samples = 1,
        },
    };
    WGPUAttachmentsState attachment_state = {
        .color_attachments = attachments,
        .color_attachments_length = ATTACHMENTS_LENGTH,
        .depth_stencil_attachment = NULL,
    };

    WGPURenderPipelineDescriptor render_pipeline_desc = {
        .layout = pipeline_layout,
        .stages = stages,
        .stages_length = STAGES_LENGTH,
        .primitive_topology = WGPUPrimitiveTopology_TriangleList,
        .attachments_state = attachment_state,
        .blend_states = blend_state,
        .blend_states_length = BLEND_STATES_LENGTH,
        .depth_stencil_state = depth_stencil_state,
    };

    WGPURenderPipelineId render_pipeline = wgpu_device_create_render_pipeline(device, &render_pipeline_desc);

    if (!glfwInit())
    {
        printf("Cannot initialize glfw");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(640, 480, "wgpu with glfw", NULL, NULL);

    if (!window)
    {
        printf("Cannot create window");
        return 1;
    }

    WGPUSurfaceId surface = NULL;

#if WGPU_TARGET == WGPU_TARGET_MACOS
    {
        id metal_layer = NULL;
        NSWindow *ns_window = glfwGetCocoaWindow(window);
        CALayer *layer = ns_window.contentView.layer;
        [ns_window.contentView setWantsLayer:YES];
        metal_layer = [CAMetalLayer layer];
        [ns_window.contentView setLayer:metal_layer];
        surface = wgpu_instance_create_surface_from_macos_layer(instance, metal_layer);
    }
#elif WGPU_TARGET == WGPU_TARGET_LINUX
    {
        Display* x11_display = glfwGetX11Display();
        Window x11_window = glfwGetX11Window(window);
        surface = wgpu_instance_create_surface_from_xlib(instance, (const void**)x11_display, x11_window);
    }
#elif WGPU_TARGET == WGPU_TARGET_WINDOWS
    {
		HWND hwnd = glfwGetWin32Window(window);
		HINSTANCE hinstance = GetModuleHandle(NULL);
		surface = wgpu_instance_create_surface_from_windows_hwnd(instance, hinstance, hwnd);
    }
#endif

    WGPUSwapChainDescriptor swap_chain_desc = {
        .usage = WGPUTextureUsageFlags_OUTPUT_ATTACHMENT | WGPUTextureUsageFlags_PRESENT,
        .format = WGPUTextureFormat_Bgra8Unorm,
        .width = 640,
        .height = 480,
    };
    WGPUSwapChainId swap_chain = wgpu_device_create_swap_chain(device, surface, &swap_chain_desc);

    while (!glfwWindowShouldClose(window))
    {
        WGPUSwapChainOutput next_texture = wgpu_swap_chain_get_next_texture(swap_chain);
        WGPUCommandBufferDescriptor cmd_buf_desc = { .todo = 0 };
        WGPUCommandBufferId cmd_buf = wgpu_device_create_command_buffer(device, &cmd_buf_desc);
        WGPURenderPassColorAttachmentDescriptor_TextureViewId color_attachments[ATTACHMENTS_LENGTH] = {
            {
                .attachment = next_texture.view_id,
                .load_op = WGPULoadOp_Clear,
                .store_op = WGPUStoreOp_Store,
                .clear_color = WGPUColor_GREEN,
            },
        };
        WGPURenderPassDescriptor rpass_desc = {
            .color_attachments = color_attachments,
            .color_attachments_length = RENDER_PASS_ATTACHMENTS_LENGTH,
            .depth_stencil_attachment = NULL,
        };
        WGPURenderPassId rpass = wgpu_command_buffer_begin_render_pass(cmd_buf, rpass_desc);
        wgpu_render_pass_set_pipeline(rpass, render_pipeline);
        wgpu_render_pass_draw(rpass, 3, 1, 0, 0);
        wgpu_render_pass_end_pass(rpass);
        WGPUQueueId queue = wgpu_device_get_queue(device);
        wgpu_queue_submit(queue, &cmd_buf, 1);
        wgpu_swap_chain_present(swap_chain);

        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
