use core::{ffi::c_void, fmt};

pub const LINUX_DMA_BUF_EXT: khronos_egl::Enum = 0x3270;
pub const LINUX_DRM_FOURCC_EXT: khronos_egl::Int = 0x3271;
pub const DMA_BUF_PLANE0_FD_EXT: khronos_egl::Int = 0x3272;
pub const DMA_BUF_PLANE0_OFFSET_EXT: khronos_egl::Int = 0x3273;
pub const DMA_BUF_PLANE0_PITCH_EXT: khronos_egl::Int = 0x3274;
pub const DMA_BUF_PLANE0_MODIFIER_LO_EXT: khronos_egl::Int = 0x3443;
pub const DMA_BUF_PLANE0_MODIFIER_HI_EXT: khronos_egl::Int = 0x3444;
pub const EXT_IMAGE_DMA_BUF_IMPORT: &str = "EGL_EXT_image_dma_buf_import";
pub const EXT_IMAGE_DMA_BUF_IMPORT_MODIFIERS: &str = "EGL_EXT_image_dma_buf_import_modifiers";

pub type EglCreateImageKhr = unsafe extern "system" fn(
    display: *mut c_void,
    context: *mut c_void,
    target: u32,
    buffer: *mut c_void,
    attributes: *const i32,
) -> *mut c_void;

pub type EglDestroyImageKhr =
    unsafe extern "system" fn(display: *mut c_void, image: *mut c_void) -> u32;

pub type GlEglImageTargetTexture2DOes = unsafe extern "system" fn(target: u32, image: *mut c_void);

pub struct DmabufImage {
    display: usize,
    image: usize,
    destroy_image: EglDestroyImageKhr,
}

impl DmabufImage {
    pub(super) fn new(
        display: khronos_egl::Display,
        image: *mut c_void,
        destroy_image: EglDestroyImageKhr,
    ) -> Self {
        Self {
            display: display.as_ptr() as usize,
            image: image as usize,
            destroy_image,
        }
    }

    /// # Safety
    ///
    /// The EGLDisplay and EGLImage must remain valid, and this function must
    /// be called exactly once after all GL siblings have been deleted.
    pub unsafe fn destroy(self) -> bool {
        let result =
            unsafe { (self.destroy_image)(self.display as *mut c_void, self.image as *mut c_void) };

        result != 0
    }
}

impl fmt::Debug for DmabufImage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DmabufImage")
            .field("display", &self.display)
            .field("image", &self.image)
            .finish_non_exhaustive()
    }
}
