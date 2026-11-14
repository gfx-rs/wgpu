use core::ffi::c_void;

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
