/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use core::id;

pub type FactoryParam = *mut std::ffi::c_void;

#[derive(Debug)]
pub struct IdentityRecycler<I> {
    fun: extern "C" fn(I, FactoryParam),
    param: FactoryParam,
    kind: &'static str,
}

impl<I: id::TypedId + Clone + std::fmt::Debug> core::hub::IdentityHandler<I>
    for IdentityRecycler<I>
{
    type Input = I;
    fn process(&self, id: I, _backend: wgt::Backend) -> I {
        log::debug!("process {} {:?}", self.kind, id);
        //debug_assert_eq!(id.unzip().2, backend);
        id
    }
    fn free(&self, id: I) {
        log::debug!("free {} {:?}", self.kind, id);
        (self.fun)(id, self.param);
    }
}

#[repr(C)]
pub struct IdentityRecyclerFactory {
    param: FactoryParam,
    free_adapter: extern "C" fn(id::AdapterId, FactoryParam),
    free_device: extern "C" fn(id::DeviceId, FactoryParam),
    free_swap_chain: extern "C" fn(id::SwapChainId, FactoryParam),
    free_pipeline_layout: extern "C" fn(id::PipelineLayoutId, FactoryParam),
    free_shader_module: extern "C" fn(id::ShaderModuleId, FactoryParam),
    free_bind_group_layout: extern "C" fn(id::BindGroupLayoutId, FactoryParam),
    free_bind_group: extern "C" fn(id::BindGroupId, FactoryParam),
    free_command_buffer: extern "C" fn(id::CommandBufferId, FactoryParam),
    free_render_pipeline: extern "C" fn(id::RenderPipelineId, FactoryParam),
    free_compute_pipeline: extern "C" fn(id::ComputePipelineId, FactoryParam),
    free_buffer: extern "C" fn(id::BufferId, FactoryParam),
    free_texture: extern "C" fn(id::TextureId, FactoryParam),
    free_texture_view: extern "C" fn(id::TextureViewId, FactoryParam),
    free_sampler: extern "C" fn(id::SamplerId, FactoryParam),
    free_surface: extern "C" fn(id::SurfaceId, FactoryParam),
}

impl core::hub::IdentityHandlerFactory<id::AdapterId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::AdapterId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_adapter,
            param: self.param,
            kind: "adapter",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::DeviceId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::DeviceId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_device,
            param: self.param,
            kind: "device",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::SwapChainId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::SwapChainId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_swap_chain,
            param: self.param,
            kind: "swap_chain",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::PipelineLayoutId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::PipelineLayoutId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_pipeline_layout,
            param: self.param,
            kind: "pipeline_layout",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::ShaderModuleId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::ShaderModuleId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_shader_module,
            param: self.param,
            kind: "shader_module",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::BindGroupLayoutId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::BindGroupLayoutId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_bind_group_layout,
            param: self.param,
            kind: "bind_group_layout",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::BindGroupId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::BindGroupId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_bind_group,
            param: self.param,
            kind: "bind_group",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::CommandBufferId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::CommandBufferId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_command_buffer,
            param: self.param,
            kind: "command_buffer",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::RenderPipelineId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::RenderPipelineId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_render_pipeline,
            param: self.param,
            kind: "render_pipeline",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::ComputePipelineId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::ComputePipelineId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_compute_pipeline,
            param: self.param,
            kind: "compute_pipeline",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::BufferId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::BufferId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_buffer,
            param: self.param,
            kind: "buffer",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::TextureId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::TextureId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_texture,
            param: self.param,
            kind: "texture",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::TextureViewId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::TextureViewId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_texture_view,
            param: self.param,
            kind: "texture_view",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::SamplerId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::SamplerId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_sampler,
            param: self.param,
            kind: "sampler",
        }
    }
}
impl core::hub::IdentityHandlerFactory<id::SurfaceId> for IdentityRecyclerFactory {
    type Filter = IdentityRecycler<id::SurfaceId>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityRecycler {
            fun: self.free_surface,
            param: self.param,
            kind: "surface",
        }
    }
}

impl core::hub::GlobalIdentityHandlerFactory for IdentityRecyclerFactory {}
