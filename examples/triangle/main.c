#ifndef WGPU_H
#define WGPU_H
#include "wgpu.h"
#endif

#include "framework.h"
#include <stdio.h>
#include <stdlib.h>

#define WGPU_TARGET_MACOS 1
#define WGPU_TARGET_LINUX 2
#define WGPU_TARGET_WINDOWS 3

#if WGPU_TARGET == WGPU_TARGET_MACOS
#include <Foundation/Foundation.h>
#include <QuartzCore/CAMetalLayer.h>
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

#define BLEND_STATES_LENGTH (1)
#define ATTACHMENTS_LENGTH (1)
#define RENDER_PASS_ATTACHMENTS_LENGTH (1)
#define BIND_GROUP_LAYOUTS_LENGTH (1)

typedef enum {
    ApplicationStatus_Initial,
    ApplicationStatus_WaitingForEvent,
    ApplicationStatus_ReceivedAdapter,
    ApplicationStatus_Rendering,
    ApplicationStatus_QuitRequested,
} ApplicationStatus;

typedef struct {
    ApplicationStatus status;
    WGPUEventLoopId event_loop;
    WGPUAdapterId adapter;
    int prev_width;
    int prev_height;
    GLFWwindow *window;
    WGPUDeviceId device;
    bool has_swap_chain;
    WGPUSwapChainId swap_chain;
    WGPUSurfaceId surface;
    WGPURenderPipelineId render_pipeline;
    WGPUBindGroupId bind_group;
} Application;

void received_adapter(WGPUAdapterId const *adapter, void *userdata) {
    Application *app = (Application *)userdata;
    app->adapter = *adapter;
    app->status = ApplicationStatus_ReceivedAdapter;
}

void get_adapter(Application *app) {
    app->status = ApplicationStatus_WaitingForEvent;
    wgpu_request_adapter_async(
        NULL, app->event_loop, received_adapter, (void *)app);
}

void setup_render_pipeline(Application *app) {
    app->device = wgpu_adapter_request_device(app->adapter,
        &(WGPUDeviceDescriptor){
            .extensions =
                {
                    .anisotropic_filtering = false,
                },
            .limits =
                {
                    .max_bind_groups = 1,
                },
        });

    WGPUShaderModuleId vertex_shader =
        wgpu_device_create_shader_module(app->device,
            &(WGPUShaderModuleDescriptor){
                .code = read_file("./../../data/triangle.vert.spv"),
            });

    WGPUShaderModuleId fragment_shader =
        wgpu_device_create_shader_module(app->device,
            &(WGPUShaderModuleDescriptor){
                .code = read_file("./../../data/triangle.frag.spv"),
            });

    WGPUBindGroupLayoutId bind_group_layout =
        wgpu_device_create_bind_group_layout(app->device,
            &(WGPUBindGroupLayoutDescriptor){
                .bindings = NULL,
                .bindings_length = 0,
            });
    app->bind_group = wgpu_device_create_bind_group(app->device,
        &(WGPUBindGroupDescriptor){
            .layout = bind_group_layout,
            .bindings = NULL,
            .bindings_length = 0,
        });

    WGPUBindGroupLayoutId bind_group_layouts[BIND_GROUP_LAYOUTS_LENGTH] = {
        bind_group_layout};

    WGPUPipelineLayoutId pipeline_layout =
        wgpu_device_create_pipeline_layout(app->device,
            &(WGPUPipelineLayoutDescriptor){
                .bind_group_layouts = bind_group_layouts,
                .bind_group_layouts_length = BIND_GROUP_LAYOUTS_LENGTH,
            });

    app->render_pipeline = wgpu_device_create_render_pipeline(app->device,
        &(WGPURenderPipelineDescriptor){
            .layout = pipeline_layout,
            .vertex_stage =
                (WGPUProgrammableStageDescriptor){
                    .module = vertex_shader,
                    .entry_point = "main",
                },
            .fragment_stage =
                &(WGPUProgrammableStageDescriptor){
                    .module = fragment_shader,
                    .entry_point = "main",
                },
            .rasterization_state =
                &(WGPURasterizationStateDescriptor){
                    .front_face = WGPUFrontFace_Ccw,
                    .cull_mode = WGPUCullMode_None,
                    .depth_bias = 0,
                    .depth_bias_slope_scale = 0.0,
                    .depth_bias_clamp = 0.0,
                },
            .primitive_topology = WGPUPrimitiveTopology_TriangleList,
            .color_states =
                &(WGPUColorStateDescriptor){
                    .format = WGPUTextureFormat_Bgra8Unorm,
                    .alpha_blend =
                        (WGPUBlendDescriptor){
                            .src_factor = WGPUBlendFactor_One,
                            .dst_factor = WGPUBlendFactor_Zero,
                            .operation = WGPUBlendOperation_Add,
                        },
                    .color_blend =
                        (WGPUBlendDescriptor){
                            .src_factor = WGPUBlendFactor_One,
                            .dst_factor = WGPUBlendFactor_Zero,
                            .operation = WGPUBlendOperation_Add,
                        },
                    .write_mask = WGPUColorWrite_ALL,
                },
            .color_states_length = 1,
            .depth_stencil_state = NULL,
            .vertex_input =
                (WGPUVertexInputDescriptor){
                    .index_format = WGPUIndexFormat_Uint16,
                    .vertex_buffers = NULL,
                    .vertex_buffers_length = 0,
                },
            .sample_count = 1,
        });

    if (!glfwInit()) {
        printf("Cannot initialize glfw");
        exit(1);
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    app->window = glfwCreateWindow(640, 480, "wgpu with glfw", NULL, NULL);

    if (!app->window) {
        printf("Cannot create window");
        exit(1);
    }

#if WGPU_TARGET == WGPU_TARGET_MACOS
    {
        id metal_layer = NULL;
        NSWindow *ns_window = glfwGetCocoaWindow(app->window);
        [ns_window.contentView setWantsLayer:YES];
        metal_layer = [CAMetalLayer layer];
        [ns_window.contentView setLayer:metal_layer];
        app->surface = wgpu_create_surface_from_metal_layer(metal_layer);
    }
#elif WGPU_TARGET == WGPU_TARGET_LINUX
    {
        Display *x11_display = glfwGetX11Display();
        Window x11_window = glfwGetX11Window(app->window);
        app->surface = wgpu_create_surface_from_xlib(
            (const void **)x11_display, x11_window);
    }
#elif WGPU_TARGET == WGPU_TARGET_WINDOWS
    {
        HWND hwnd = glfwGetWin32Window(app->window);
        HINSTANCE hinstance = GetModuleHandle(NULL);
        app->surface = wgpu_create_surface_from_windows_hwnd(hinstance, hwnd);
    }
#else
#error "Unsupported WGPU_TARGET"
#endif

    glfwGetWindowSize(app->window, &app->prev_width, &app->prev_height);
    app->status = ApplicationStatus_Rendering;
}

void render_frame(Application *app) {
    glfwPollEvents();
    if (glfwWindowShouldClose(app->window)) {
        app->status = ApplicationStatus_QuitRequested;
        return;
    }

    int width = 0;
    int height = 0;
    glfwGetWindowSize(app->window, &width, &height);
    if (!app->has_swap_chain || width != app->prev_width ||
        height != app->prev_height) {
        app->has_swap_chain = true;
        app->prev_width = width;
        app->prev_height = height;
        app->swap_chain =
            wgpu_device_create_swap_chain(app->device, app->surface,
                &(WGPUSwapChainDescriptor){
                    .usage = WGPUTextureUsage_OUTPUT_ATTACHMENT,
                    .format = WGPUTextureFormat_Bgra8Unorm,
                    .width = width,
                    .height = height,
                    .present_mode = WGPUPresentMode_Vsync,
                });
    }

    WGPUSwapChainOutput next_texture =
        wgpu_swap_chain_get_next_texture(app->swap_chain);
    if (!next_texture.view_id) {
        printf("Cannot acquire next swap chain texture");
        exit(1);
    }

    WGPUCommandEncoderId cmd_encoder = wgpu_device_create_command_encoder(
        app->device, &(WGPUCommandEncoderDescriptor){.todo = 0});

    WGPURenderPassColorAttachmentDescriptor
        color_attachments[ATTACHMENTS_LENGTH] = {
            {
                .attachment = next_texture.view_id,
                .load_op = WGPULoadOp_Clear,
                .store_op = WGPUStoreOp_Store,
                .clear_color = WGPUColor_GREEN,
            },
        };

    WGPURenderPassId rpass = wgpu_command_encoder_begin_render_pass(cmd_encoder,
        &(WGPURenderPassDescriptor){
            .color_attachments = color_attachments,
            .color_attachments_length = RENDER_PASS_ATTACHMENTS_LENGTH,
            .depth_stencil_attachment = NULL,
        });

    wgpu_render_pass_set_pipeline(rpass, app->render_pipeline);
    wgpu_render_pass_set_bind_group(rpass, 0, app->bind_group, NULL, 0);
    wgpu_render_pass_draw(rpass, 3, 1, 0, 0);
    WGPUQueueId queue = wgpu_device_get_queue(app->device);
    wgpu_render_pass_end_pass(rpass);
    WGPUCommandBufferId cmd_buf =
        wgpu_command_encoder_finish(cmd_encoder, NULL);
    wgpu_queue_submit(queue, &cmd_buf, 1, app->event_loop);
    wgpu_swap_chain_present(app->swap_chain);
}

void quit(Application *app) {
    wgpu_destroy_event_loop(app->event_loop);
    glfwDestroyWindow(app->window);
    glfwTerminate();
    exit(0);
}

int main() {
    Application app = {.status = ApplicationStatus_Initial,
        .event_loop = wgpu_create_event_loop()};

    while (true) {
        switch (app.status) {
        case ApplicationStatus_Initial:
            get_adapter(&app);
            break;
        case ApplicationStatus_ReceivedAdapter:
            setup_render_pipeline(&app);
            break;
        case ApplicationStatus_Rendering:
            render_frame(&app);
            break;
        case ApplicationStatus_QuitRequested:
            quit(&app);
            break;
        case ApplicationStatus_WaitingForEvent:
            break;
        }

        wgpu_process_events(app.event_loop);
    }
}
