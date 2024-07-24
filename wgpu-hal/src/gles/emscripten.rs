extern "C" {
    /// returns 1 if success. 0 if failure. extension name must be null terminated
    fn emscripten_webgl_enable_extension(
        context: std::ffi::c_int,
        extension: *const std::ffi::c_char,
    ) -> std::ffi::c_int;
    fn emscripten_webgl_get_current_context() -> std::ffi::c_int;
}
/// Webgl requires extensions to be enabled before using them.
/// This function can be used to enable webgl extension on emscripten target.
///
/// returns true on success
///
/// # Safety
///
/// - opengl context MUST BE current
/// - extension_name_null_terminated argument must be a valid string with null terminator.
/// - extension must be present. check `glow_context.supported_extensions()`
pub unsafe fn enable_extension(extension_name_null_terminated: &str) -> bool {
    unsafe {
        emscripten_webgl_enable_extension(
            emscripten_webgl_get_current_context(),
            extension_name_null_terminated.as_ptr().cast(),
        ) == 1
    }
}
