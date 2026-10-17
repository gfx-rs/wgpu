use alloc::vec::Vec;
use core::{ffi::c_void, fmt, ptr};

/// A DRM pixel format supported by the single-plane 2D DMA-BUF importer.
/// Only formats with an unambiguous wgpu mapping and enabled format features are listed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DmabufFormat {
    /// DRM FourCC, preserving the driver's format code.
    pub fourcc: u32,
    /// Matching wgpu format, without channel swizzling or forced alpha.
    pub texture_format: wgt::TextureFormat,
    /// Explicit modifiers supported by the importer; currently only LINEAR.
    pub modifiers: Vec<DmabufModifier>,
}

/// A DMA-BUF memory layout and its EGL sampling restriction.
/// Query results exclude external-only modifiers, so `external_only` is always false there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DmabufModifier {
    /// DRM format modifier describing the memory layout.
    pub modifier: u64,
    /// True if EGL restricts this pair to external-texture sampling.
    pub external_only: bool,
}

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

pub type QueryFormats = unsafe extern "system" fn(*mut c_void, i32, *mut i32, *mut i32) -> u32;

pub type QueryModifiers =
    unsafe extern "system" fn(*mut c_void, i32, i32, *mut u64, *mut u32, *mut i32) -> u32;

pub type QueryDisplay = unsafe extern "system" fn(*mut c_void, i32, *mut isize) -> u32;

pub type QueryDeviceString =
    unsafe extern "system" fn(*mut c_void, i32) -> *const core::ffi::c_char;

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

pub(super) fn has_extension(extensions: &str, name: &str) -> bool {
    extensions
        .split_ascii_whitespace()
        .any(|extension| extension == name)
}

pub(super) unsafe fn query_modifiers(
    display: *mut c_void,
    query_modifiers: QueryModifiers,
    format: i32,
    modifiers: &mut Vec<u64>,
    external: &mut Vec<u32>,
) -> Result<Vec<DmabufModifier>, crate::DeviceError> {
    const DRM_FORMAT_MOD_LINEAR: u64 = 0;

    let mut count = 0;
    let query_succeeded = unsafe {
        query_modifiers(
            display,
            format,
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut count,
        )
    } != 0;

    if !query_succeeded || count < 0 {
        return Err(crate::DeviceError::Unexpected);
    }

    if count == 0 {
        return Ok(Vec::new());
    }

    let capacity = count as usize;
    modifiers.resize(capacity, 0);
    external.resize(capacity, 0);

    let query_succeeded = unsafe {
        query_modifiers(
            display,
            format,
            count,
            modifiers.as_mut_ptr(),
            external.as_mut_ptr(),
            &mut count,
        )
    } != 0;

    if !query_succeeded || count < 0 || count as usize > modifiers.len() {
        return Err(crate::DeviceError::Unexpected);
    }

    let returned_count = count as usize;
    let mut supported = Vec::new();
    for index in 0..returned_count {
        let modifier = modifiers[index];
        let external_only = external[index] != 0;
        // EGL exposes no plane count; only known LINEAR layouts are promised.
        if modifier != DRM_FORMAT_MOD_LINEAR || external_only {
            continue;
        }

        supported.push(DmabufModifier {
            modifier,
            external_only,
        });
    }

    Ok(supported)
}

pub(super) unsafe fn query_device_id(
    display: *mut c_void,
    query_display: QueryDisplay,
    query_device: QueryDeviceString,
) -> Result<Option<u64>, crate::DeviceError> {
    use std::os::unix::fs::{FileTypeExt as _, MetadataExt as _};
    let mut device = 0;
    // EGL_DEVICE_EXT
    if unsafe { query_display(display, 0x322C, &mut device) } == 0 || device == 0 {
        return Ok(None);
    }
    let device = device as *mut c_void;
    let extensions = unsafe { query_device(device, khronos_egl::EXTENSIONS) };
    if extensions.is_null() {
        return Ok(None);
    }
    let extensions = unsafe { core::ffi::CStr::from_ptr(extensions) }.to_string_lossy();
    if !has_extension(&extensions, "EGL_EXT_device_drm_render_node") {
        return Ok(None);
    }
    // EGL_DRM_RENDER_NODE_FILE_EXT
    let path = unsafe { query_device(device, 0x3377) };
    if path.is_null() {
        return Ok(None);
    }
    use std::os::unix::ffi::OsStrExt as _;
    let path = std::ffi::OsStr::from_bytes(unsafe { core::ffi::CStr::from_ptr(path) }.to_bytes());
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.file_type().is_char_device() => Ok(Some(metadata.rdev())),
        _ => Ok(None),
    }
}
