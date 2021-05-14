/// Config
pub struct OpenXROptions {

}

impl Default for OpenXROptions {
    fn default() -> Self {
        OpenXROptions {}
    }
}

/// FIXME FIXME FIXME is this okay?
pub struct OpenXRHandles {
    /// TODO: Documentation
    pub session: openxr::Session<openxr::Vulkan>,
    /// TODO: Documentation
    pub frame_waiter: openxr::FrameWaiter,
    /// TODO: Documentation
    pub frame_stream: openxr::FrameStream<openxr::Vulkan>,
    /// TODO: Documentation
    pub space: openxr::Space,
    /// TODO: Documentation
    pub system: openxr::SystemId,
}
