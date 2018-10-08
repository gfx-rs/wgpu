#[cfg(not(feature = "remote"))]
mod local;
#[cfg(feature = "remote")]
mod remote;

#[cfg(not(feature = "remote"))]
pub use self::local::{Id, ItemsGuard, Registry as ConcreteRegistry};
#[cfg(feature = "remote")]
pub use self::remote::{Id, ItemsGuard, Registry as ConcreteRegistry};

use {
    AdapterHandle, AttachmentStateHandle, BindGroupLayoutHandle, BindGroupHandle,
    BlendStateHandle, CommandBufferHandle, DepthStencilStateHandle, DeviceHandle, InstanceHandle,
    RenderPassHandle, ComputePassHandle,
    PipelineLayoutHandle, RenderPipelineHandle, ComputePipelineHandle, ShaderModuleHandle,
    TextureHandle,
};


type Item<'a, T> = &'a T;
type ItemMut<'a, T> = &'a mut T;

pub trait Registry<T>: Default {
    fn lock(&self) -> ItemsGuard<T>;
}

pub trait Items<T> {
    fn register(&mut self, handle: T) -> Id;
    fn get(&self, id: Id) -> Item<T>;
    fn get_mut(&mut self, id: Id) -> ItemMut<T>;
    fn take(&mut self, id: Id) -> T;
}

#[derive(Default)]
pub struct Hub {
    pub(crate) instances: ConcreteRegistry<InstanceHandle>,
    pub(crate) adapters: ConcreteRegistry<AdapterHandle>,
    pub(crate) devices: ConcreteRegistry<DeviceHandle>,
    pub(crate) pipeline_layouts: ConcreteRegistry<PipelineLayoutHandle>,
    pub(crate) bind_group_layouts: ConcreteRegistry<BindGroupLayoutHandle>,
    pub(crate) bind_groups: ConcreteRegistry<BindGroupHandle>,
    pub(crate) attachment_states: ConcreteRegistry<AttachmentStateHandle>,
    pub(crate) blend_states: ConcreteRegistry<BlendStateHandle>,
    pub(crate) depth_stencil_states: ConcreteRegistry<DepthStencilStateHandle>,
    pub(crate) shader_modules: ConcreteRegistry<ShaderModuleHandle>,
    pub(crate) command_buffers: ConcreteRegistry<CommandBufferHandle>,
    pub(crate) render_pipelines: ConcreteRegistry<RenderPipelineHandle>,
    pub(crate) compute_pipelines: ConcreteRegistry<ComputePipelineHandle>,
    pub(crate) render_passes: ConcreteRegistry<RenderPassHandle>,
    pub(crate) compute_passes: ConcreteRegistry<ComputePassHandle>,
    pub(crate) textures: ConcreteRegistry<TextureHandle>,
}

lazy_static! {
    pub static ref HUB: Hub = Hub::default();
}
