#[cfg(webgpu)]
mod web;
#[cfg(webgpu)]
pub(crate) use web::Context;

#[cfg(not(webgpu))]
mod direct;
#[cfg(not(webgpu))]
pub(crate) use direct::Context;
