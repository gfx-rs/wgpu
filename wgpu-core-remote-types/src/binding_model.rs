use alloc::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::id::{BindGroupLayoutId, BufferId, ExternalTextureId, SamplerId, TextureViewId};
use crate::Label;

#[repr(C)]
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct BufferBinding {
    pub buffer: BufferId,
    pub offset: wgt::BufferAddress,

    /// Size of the binding. If `None`, the binding spans from `offset` to the
    /// end of the buffer.
    ///
    /// We use `BufferAddress` to allow a size of zero on this `wgpu_core` type,
    /// because JavaScript bindings cannot readily express `Option<NonZeroU64>`.
    /// The `wgpu` API uses `Option<BufferSize>` (i.e. `NonZeroU64`) for this
    /// field.
    pub size: Option<wgt::BufferAddress>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BindingResource {
    Buffer(BufferBinding),
    Sampler(SamplerId),
    TextureView(TextureViewId),
    ExternalTexture(ExternalTextureId),
}

/// Bindable resource and the slot to bind it to.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BindGroupEntry {
    /// Slot for which binding provides resource. Corresponds to an entry of the same
    /// binding index in the [`BindGroupLayoutDescriptor`].
    pub binding: u32,
    /// Resource to attach to the binding
    pub resource: BindingResource,
}

/// Describes a group of bindings and the resources to be bound.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BindGroupDescriptor<'a> {
    /// Debug label of the bind group.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The BindGroupLayout that corresponds to this bind group.
    pub layout: BindGroupLayoutId,
    /// The resources to bind to this bind group.
    pub entries: Cow<'a, [BindGroupEntry]>,
}

/// Describes a [`BindGroupLayout`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BindGroupLayoutDescriptor<'a> {
    /// Debug label of the bind group layout.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Array of entries in this BindGroupLayout
    pub entries: Cow<'a, [wgt::BindGroupLayoutEntry]>,
}
