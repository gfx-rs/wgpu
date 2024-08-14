//! Types and functions which define our public api and their
//! helper functionality.
//!
//! # Conventions
//!
//! Each major type gets its own module. The module is laid out as follows:
//!
//! - The type itself
//! - `impl` block for the type
//! - `Drop` implementation for the type (if needed)
//! - Descriptor types and their subtypes.
//! - Any non-public helper types or functions.
//!
//! # Imports
//!
//! Because our public api is "flat" (i.e. all types are directly under the `wgpu` module),
//! we use a single `crate::*` import at the top of each module to bring in all the types in
//! the public api. This is done to:
//! - Avoid having to write out a long list of imports for each module.
//! - Allow docs to be written naturally, without needing to worry about needing dedicated doc imports.
//! - Treat wgpu-types types and wgpu-core types as a single set.
//!

mod adapter;
mod bind_group;
mod bind_group_layout;
mod buffer;
mod command_buffer;
mod command_encoder;
// Not a root type, but common descriptor types for pipelines.
mod common_pipeline;
mod compute_pass;
mod compute_pipeline;
mod device;
mod id;
mod instance;
mod pipeline_cache;
mod pipeline_layout;
mod query_set;
mod queue;
mod render_bundle;
mod render_bundle_encoder;
mod render_pass;
mod render_pipeline;
mod sampler;
mod shader_module;
mod surface;
mod surface_texture;
mod texture;
mod texture_view;

pub use adapter::*;
pub use bind_group::*;
pub use bind_group_layout::*;
pub use buffer::*;
pub use command_buffer::*;
pub use command_encoder::*;
pub use common_pipeline::*;
pub use compute_pass::*;
pub use compute_pipeline::*;
pub use device::*;
pub use id::*;
pub use instance::*;
pub use pipeline_cache::*;
pub use pipeline_layout::*;
pub use query_set::*;
pub use queue::*;
pub use render_bundle::*;
pub use render_bundle_encoder::*;
pub use render_pass::*;
pub use render_pipeline::*;
pub use sampler::*;
pub use shader_module::*;
pub use surface::*;
pub use surface_texture::*;
pub use texture::*;
pub use texture_view::*;

/// Object debugging label.
pub type Label<'a> = Option<&'a str>;
