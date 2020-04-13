/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef WGPU_H
#define WGPU_H
#include "wgpu.h"
#endif

#include "framework.h"
#include <stdio.h>
#include <stdlib.h>

#define WGPU_TARGET_MACOS 1
#define WGPU_TARGET_LINUX_X11 2
#define WGPU_TARGET_WINDOWS 3
#define WGPU_TARGET_LINUX_WAYLAND 4

#if WGPU_TARGET == WGPU_TARGET_MACOS
#include <Foundation/Foundation.h>
#include <QuartzCore/CAMetalLayer.h>
#endif

#include <GLFW/glfw3.h>
#if WGPU_TARGET == WGPU_TARGET_MACOS
#define GLFW_EXPOSE_NATIVE_COCOA
#elif WGPU_TARGET == WGPU_TARGET_LINUX_X11
#define GLFW_EXPOSE_NATIVE_X11
#elif WGPU_TARGET == WGPU_TARGET_LINUX_WAYLAND
#define GLFW_EXPOSE_NATIVE_WAYLAND
#elif WGPU_TARGET == WGPU_TARGET_WINDOWS
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include <GLFW/glfw3native.h>

#define BLEND_STATES_LENGTH (1)
#define ATTACHMENTS_LENGTH (1)
#define RENDER_PASS_ATTACHMENTS_LENGTH (1)
#define BIND_GROUP_LAYOUTS_LENGTH (1)

void request_adapter_callback(WGPUAdapterId received, void *userdata) {
    *(WGPUAdapterId*)userdata = received;
}

int main() {
    if (!glfwInit()) {
        printf("Cannot initialize glfw");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window =
        glfwCreateWindow(640, 480, "wgpu with glfw", NULL, NULL);

    if (!window) {
        printf("Cannot create window");
        return 1;
    }

    WGPUSurfaceId surface;

#if WGPU_TARGET == WGPU_TARGET_MACOS
    {
        id metal_layer = NULL;
        NSWindow *ns_window = glfwGetCocoaWindow(window);
        [ns_window.contentView setWantsLayer:YES];
        metal_layer = [CAMetalLayer layer];
        [ns_window.contentView setLayer:metal_layer];
        surface = wgpu_create_surface_from_metal_layer(metal_layer);
    }
#elif WGPU_TARGET == WGPU_TARGET_LINUX_X11
    {
        Display *x11_display = glfwGetX11Display();
        Window x11_window = glfwGetX11Window(window);
        surface = wgpu_create_surface_from_xlib((const void **)x11_display, x11_window);
    }
#elif WGPU_TARGET == WGPU_TARGET_LINUX_WAYLAND
    {
        struct wl_display *wayland_display = glfwGetWaylandDisplay();
        struct wl_surface *wayland_surface = glfwGetWaylandWindow(window);
        surface = wgpu_create_surface_from_wayland(wayland_surface, wayland_display);
    }
#elif WGPU_TARGET == WGPU_TARGET_WINDOWS
    {
        HWND hwnd = glfwGetWin32Window(window);
        HINSTANCE hinstance = GetModuleHandle(NULL);
        surface = wgpu_create_surface_from_windows_hwnd(hinstance, hwnd);
    }
#else
    #error "Unsupported WGPU_TARGET"
#endif

    WGPUAdapterId adapter = { 0 };
    wgpu_request_adapter_async(
        &(WGPURequestAdapterOptions){
            .power_preference = WGPUPowerPreference_LowPower,
            .compatible_surface = surface,
        },
        2 | 4 | 8,
        request_adapter_callback,
        (void *) &adapter
    );

    WGPUDeviceId device = wgpu_adapter_request_device(adapter,
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

    WGPUShaderModuleId vertex_shader = wgpu_device_create_shader_module(device,
        &(WGPUShaderModuleDescriptor){
            .code = read_file("./../../data/triangle.vert.spv"),
        });

    WGPUShaderModuleId fragment_shader =
        wgpu_device_create_shader_module(device,
            &(WGPUShaderModuleDescriptor){
                .code = read_file("./../../data/triangle.frag.spv"),
            });

    WGPUBindGroupLayoutId bind_group_layout =
        wgpu_device_create_bind_group_layout(device,
            &(WGPUBindGroupLayoutDescriptor){
                .label = "bind group layout",
                .entries = NULL,
                .entries_length = 0,
            });
    WGPUBindGroupId bind_group =
        wgpu_device_create_bind_group(device,
            &(WGPUBindGroupDescriptor){
                .label = "bind group",
                .layout = bind_group_layout,
                .entries = NULL,
                .entries_length = 0,
            });

    WGPUBindGroupLayoutId bind_group_layouts[BIND_GROUP_LAYOUTS_LENGTH] = {
        bind_group_layout};

    WGPUPipelineLayoutId pipeline_layout =
        wgpu_device_create_pipeline_layout(device,
            &(WGPUPipelineLayoutDescriptor){
                .bind_group_layouts = bind_group_layouts,
                .bind_group_layouts_length = BIND_GROUP_LAYOUTS_LENGTH,
            });

    WGPURenderPipelineId render_pipeline =
        wgpu_device_create_render_pipeline(device,
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
                .vertex_state =
                    (WGPUVertexStateDescriptor){
                        .index_format = WGPUIndexFormat_Uint16,
                        .vertex_buffers = NULL,
                        .vertex_buffers_length = 0,
                    },
                .sample_count = 1,
            });

    int prev_width = 0;
    int prev_height = 0;
    glfwGetWindowSize(window, &prev_width, &prev_height);

    WGPUSwapChainId swap_chain = wgpu_device_create_swap_chain(device, surface,
        &(WGPUSwapChainDescriptor){
            .usage = WGPUTextureUsage_OUTPUT_ATTACHMENT,
            .format = WGPUTextureFormat_Bgra8Unorm,
            .width = prev_width,
            .height = prev_height,
            .present_mode = WGPUPresentMode_Fifo,
        });

    while (!glfwWindowShouldClose(window)) {
        int width = 0;
        int height = 0;
        glfwGetWindowSize(window, &width, &height);
        if (width != prev_width || height != prev_height) {
            prev_width = width;
            prev_height = height;

            swap_chain = wgpu_device_create_swap_chain(device, surface,
                &(WGPUSwapChainDescriptor){
                    .usage = WGPUTextureUsage_OUTPUT_ATTACHMENT,
                    .format = WGPUTextureFormat_Bgra8Unorm,
                    .width = width,
                    .height = height,
                    .present_mode = WGPUPresentMode_Fifo,
                });
        }

        WGPUSwapChainOutput next_texture =
            wgpu_swap_chain_get_next_texture(swap_chain);
        if (!next_texture.view_id) {
            printf("Cannot acquire next swap chain texture");
            return 1;
        }

        WGPUCommandEncoderId cmd_encoder = wgpu_device_create_command_encoder(
            device, &(WGPUCommandEncoderDescriptor){.label = "command encoder"});

        WGPURenderPassColorAttachmentDescriptor
            color_attachments[ATTACHMENTS_LENGTH] = {
                {
                    .attachment = next_texture.view_id,
                    .load_op = WGPULoadOp_Clear,
                    .store_op = WGPUStoreOp_Store,
                    .clear_color = WGPUColor_GREEN,
                },
            };

        WGPURenderPassId rpass =
            wgpu_command_encoder_begin_render_pass(cmd_encoder,
                &(WGPURenderPassDescriptor){
                    .color_attachments = color_attachments,
                    .color_attachments_length = RENDER_PASS_ATTACHMENTS_LENGTH,
                    .depth_stencil_attachment = NULL,
                });

        wgpu_render_pass_set_pipeline(rpass, render_pipeline);
        wgpu_render_pass_set_bind_group(rpass, 0, bind_group, NULL, 0);
        wgpu_render_pass_draw(rpass, 3, 1, 0, 0);
        WGPUQueueId queue = wgpu_device_get_default_queue(device);
        wgpu_render_pass_end_pass(rpass);
        WGPUCommandBufferId cmd_buf =  wgpu_command_encoder_finish(cmd_encoder, NULL);
        wgpu_queue_submit(queue, &cmd_buf, 1);
        wgpu_swap_chain_present(swap_chain);

        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
