/// Marker trait used to determine which types uniquely identify a resource.
///
/// For example, `Device<A>` will have the same type of identifier as
/// `Device<B>` because `Device<T>` for any `T` defines the same maker type.
pub trait Marker: 'static + crate::WasmNotSendSync {
    const TYPE: &'static str;
}

// This allows `()` to be used as a marker type for tests.
//
// We don't want these in production code, since they essentially remove type
// safety, like how identifiers across different types can be compared.
#[cfg(test)]
impl Marker for () {
    const TYPE: &'static str = "Untyped";
}

/// Define markers for each resource.
macro_rules! ids {
    ($(
        $(#[$($meta:meta)*])*
        pub type $marker:ident;
    )*) => {
        /// Marker types for each resource.
        pub mod markers {
            $(
                #[derive(Debug)]
                pub enum $marker {}
                impl super::Marker for $marker {
                    const TYPE: &'static str = stringify!($marker);
                }
            )*
        }
    }
}

ids! {
    pub type Adapter;
    pub type Surface;
    pub type Device;
    pub type Queue;
    pub type Buffer;
    pub type StagingBuffer;
    pub type TextureView;
    pub type Texture;
    pub type ExternalTexture;
    pub type Sampler;
    pub type BindGroupLayout;
    pub type PipelineLayout;
    pub type BindGroup;
    pub type ShaderModule;
    pub type RenderPipeline;
    pub type ComputePipeline;
    pub type PipelineCache;
    pub type CommandEncoder;
    pub type CommandBuffer;
    pub type RenderPassEncoder;
    pub type ComputePassEncoder;
    pub type RenderBundleEncoder;
    pub type RenderBundle;
    pub type QuerySet;
    pub type Blas;
    pub type Tlas;
}
