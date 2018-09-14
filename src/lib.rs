extern crate gfx_hal as hal;
#[cfg(feature = "gfx-backend-vulkan")]
extern crate gfx_backend_vulkan as back;
#[cfg(feature = "gfx-backend-dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "gfx-backend-metal")]
extern crate gfx_backend_metal as back;

mod command;
mod device;
mod handle;
mod instance;

pub use self::command::*;
pub use self::device::*;
pub use self::instance::*;

use back::Backend as B;
use handle::Handle;

pub type InstanceHandle = Handle<back::Instance>;
pub type AdapterHandle = Handle<hal::Adapter<B>>;
pub type DeviceHandle = Handle<Device<B>>;
pub type BufferHandle = Handle<Buffer<B>>;
pub type CommandBufferHandle = Handle<CommandBuffer<B>>;
pub type RenderPassHandle = Handle<RenderPass<B>>;
pub type ComputePassHandle = Handle<ComputePass<B>>;
