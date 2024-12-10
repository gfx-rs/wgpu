use std::{error, fmt, thread};

use crate::*;

/// Surface texture that can be rendered to.
/// Result of a successful call to [`Surface::get_current_texture`].
///
/// This type is unique to the Rust API of `wgpu`. In the WebGPU specification,
/// the [`GPUCanvasContext`](https://gpuweb.github.io/gpuweb/#canvas-context) provides
/// a texture without any additional information.
#[derive(Debug)]
pub struct SurfaceTexture {
    /// Accessible view of the frame.
    pub texture: Texture,
    /// `true` if the acquired buffer can still be used for rendering,
    /// but should be recreated for maximum performance.
    pub suboptimal: bool,
    pub(crate) presented: bool,
    pub(crate) detail: dispatch::DispatchSurfaceOutputDetail,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(SurfaceTexture: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(SurfaceTexture => .texture.inner);

impl SurfaceTexture {
    /// Schedule this texture to be presented on the owning surface.
    ///
    /// Needs to be called after any work on the texture is scheduled via [`Queue::submit`].
    ///
    /// # Platform dependent behavior
    ///
    /// On Wayland, `present` will attach a `wl_buffer` to the underlying `wl_surface` and commit the new surface
    /// state. If it is desired to do things such as request a frame callback, scale the surface using the viewporter
    /// or synchronize other double buffered state, then these operations should be done before the call to `present`.
    pub fn present(mut self) {
        self.presented = true;
        self.detail.present();
    }
}

impl Drop for SurfaceTexture {
    fn drop(&mut self) {
        if !self.presented && !thread::panicking() {
            self.detail.texture_discard();
        }
    }
}

/// Result of an unsuccessful call to [`Surface::get_current_texture`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SurfaceError {
    /// A timeout was encountered while trying to acquire the next frame.
    Timeout,
    /// The underlying surface has changed, and therefore the swap chain must be updated.
    Outdated,
    /// The swap chain has been lost and needs to be recreated.
    Lost,
    /// There is no more memory left to allocate a new frame.
    OutOfMemory,
    /// Acquiring a texture failed with a generic error. Check error callbacks for more information.
    Other,
}
static_assertions::assert_impl_all!(SurfaceError: Send, Sync);

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            Self::Timeout => "A timeout was encountered while trying to acquire the next frame",
            Self::Outdated => "The underlying surface has changed, and therefore the swap chain must be updated",
            Self::Lost =>  "The swap chain has been lost and needs to be recreated",
            Self::OutOfMemory => "There is no more memory left to allocate a new frame",
            Self::Other => "Acquiring a texture failed with a generic error. Check error callbacks for more information",
        })
    }
}

impl error::Error for SurfaceError {}
