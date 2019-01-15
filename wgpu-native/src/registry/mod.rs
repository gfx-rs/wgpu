use crate::{
    AdapterHandle, BindGroupLayoutHandle, BindGroupHandle,
    BlendStateHandle, CommandBufferHandle, DepthStencilStateHandle, DeviceHandle, InstanceHandle,
    RenderPassHandle, ComputePassHandle,
    PipelineLayoutHandle, RenderPipelineHandle, ComputePipelineHandle, ShaderModuleHandle,
    BufferHandle, TextureHandle, TextureViewHandle,
    SurfaceHandle, SwapChainHandle,
};

use lazy_static::lazy_static;
use parking_lot::RwLock;

use std::sync::Arc;


#[cfg(not(feature = "remote"))]
mod local;
#[cfg(feature = "remote")]
mod remote;

#[cfg(not(feature = "remote"))]
pub use self::local::{Id, Items as ConcreteItems};
#[cfg(feature = "remote")]
pub use self::remote::{Id, Items as ConcreteItems};

pub trait Items<T>: Default {
    fn register(&mut self, handle: T) -> Id;
    fn get(&self, id: Id) -> &T;
    fn get_mut(&mut self, id: Id) -> &mut T;
    fn take(&mut self, id: Id) -> T;
}

pub type ConcreteRegistry<T> = Arc<RwLock<ConcreteItems<T>>>;

#[derive(Default)]
pub struct Hub {
    pub(crate) instances: ConcreteRegistry<InstanceHandle>,
    pub(crate) adapters: ConcreteRegistry<AdapterHandle>,
    pub(crate) devices: ConcreteRegistry<DeviceHandle>,
    pub(crate) pipeline_layouts: ConcreteRegistry<PipelineLayoutHandle>,
    pub(crate) bind_group_layouts: ConcreteRegistry<BindGroupLayoutHandle>,
    pub(crate) bind_groups: ConcreteRegistry<BindGroupHandle>,
    pub(crate) blend_states: ConcreteRegistry<BlendStateHandle>,
    pub(crate) depth_stencil_states: ConcreteRegistry<DepthStencilStateHandle>,
    pub(crate) shader_modules: ConcreteRegistry<ShaderModuleHandle>,
    pub(crate) command_buffers: ConcreteRegistry<CommandBufferHandle>,
    pub(crate) render_pipelines: ConcreteRegistry<RenderPipelineHandle>,
    pub(crate) compute_pipelines: ConcreteRegistry<ComputePipelineHandle>,
    pub(crate) render_passes: ConcreteRegistry<RenderPassHandle>,
    pub(crate) compute_passes: ConcreteRegistry<ComputePassHandle>,
    pub(crate) buffers: ConcreteRegistry<BufferHandle>,
    pub(crate) textures: ConcreteRegistry<TextureHandle>,
    pub(crate) texture_views: ConcreteRegistry<TextureViewHandle>,
    pub(crate) surfaces: ConcreteRegistry<SurfaceHandle>,
    pub(crate) swap_chains: ConcreteRegistry<SwapChainHandle>,
}

lazy_static! {
    pub static ref HUB: Hub = Hub::default();
}
