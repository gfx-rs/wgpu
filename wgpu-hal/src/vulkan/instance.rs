use std::{
    ffi::{c_void, CStr, CString},
    slice,
    str::FromStr,
    sync::Arc,
    thread,
};

use ash::{
    extensions::{ext, khr},
    vk,
};
use parking_lot::RwLock;

unsafe extern "system" fn debug_utils_messenger_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data_ptr: *const vk::DebugUtilsMessengerCallbackDataEXT,
    user_data: *mut c_void,
) -> vk::Bool32 {
    use std::borrow::Cow;

    if thread::panicking() {
        return vk::FALSE;
    }

    let cd = unsafe { &*callback_data_ptr };
    let user_data = unsafe { &*(user_data as *mut super::DebugUtilsMessengerUserData) };

    const VUID_VKCMDENDDEBUGUTILSLABELEXT_COMMANDBUFFER_01912: i32 = 0x56146426;
    if cd.message_id_number == VUID_VKCMDENDDEBUGUTILSLABELEXT_COMMANDBUFFER_01912 {
        // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/5671
        // Versions 1.3.240 through 1.3.250 return a spurious error here if
        // the debug range start and end appear in different command buffers.
        let khronos_validation_layer =
            std::ffi::CStr::from_bytes_with_nul(b"Khronos Validation Layer\0").unwrap();
        if user_data.validation_layer_description.as_ref() == khronos_validation_layer
            && user_data.validation_layer_spec_version >= vk::make_api_version(0, 1, 3, 240)
            && user_data.validation_layer_spec_version <= vk::make_api_version(0, 1, 3, 250)
        {
            return vk::FALSE;
        }
    }

    // Silence Vulkan Validation error "VUID-VkSwapchainCreateInfoKHR-pNext-07781"
    // This happens when a surface is configured with a size outside the allowed extent.
    // It's s false positive due to the inherent racy-ness of surface resizing.
    const VUID_VKSWAPCHAINCREATEINFOKHR_PNEXT_07781: i32 = 0x4c8929c1;
    if cd.message_id_number == VUID_VKSWAPCHAINCREATEINFOKHR_PNEXT_07781 {
        return vk::FALSE;
    }

    // Silence Vulkan Validation error "VUID-VkRenderPassBeginInfo-framebuffer-04627"
    // if the OBS layer is enabled. This is a bug in the OBS layer. As the OBS layer
    // does not have a version number they increment, there is no way to qualify the
    // supression of the error to a specific version of the OBS layer.
    //
    // See https://github.com/obsproject/obs-studio/issues/9353
    const VUID_VKRENDERPASSBEGININFO_FRAMEBUFFER_04627: i32 = 0x45125641;
    if cd.message_id_number == VUID_VKRENDERPASSBEGININFO_FRAMEBUFFER_04627
        && user_data.has_obs_layer
    {
        return vk::FALSE;
    }

    let level = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
        _ => log::Level::Warn,
    };

    let message_id_name = if cd.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        unsafe { CStr::from_ptr(cd.p_message_id_name) }.to_string_lossy()
    };
    let message = if cd.p_message.is_null() {
        Cow::from("")
    } else {
        unsafe { CStr::from_ptr(cd.p_message) }.to_string_lossy()
    };

    let _ = std::panic::catch_unwind(|| {
        log::log!(
            level,
            "{:?} [{} (0x{:x})]\n\t{}",
            message_type,
            message_id_name,
            cd.message_id_number,
            message,
        );
    });

    if cd.queue_label_count != 0 {
        let labels =
            unsafe { slice::from_raw_parts(cd.p_queue_labels, cd.queue_label_count as usize) };
        let names = labels
            .iter()
            .flat_map(|dul_obj| {
                unsafe { dul_obj.p_label_name.as_ref() }
                    .map(|lbl| unsafe { CStr::from_ptr(lbl) }.to_string_lossy())
            })
            .collect::<Vec<_>>();

        let _ = std::panic::catch_unwind(|| {
            log::log!(level, "\tqueues: {}", names.join(", "));
        });
    }

    if cd.cmd_buf_label_count != 0 {
        let labels =
            unsafe { slice::from_raw_parts(cd.p_cmd_buf_labels, cd.cmd_buf_label_count as usize) };
        let names = labels
            .iter()
            .flat_map(|dul_obj| {
                unsafe { dul_obj.p_label_name.as_ref() }
                    .map(|lbl| unsafe { CStr::from_ptr(lbl) }.to_string_lossy())
            })
            .collect::<Vec<_>>();

        let _ = std::panic::catch_unwind(|| {
            log::log!(level, "\tcommand buffers: {}", names.join(", "));
        });
    }

    if cd.object_count != 0 {
        let labels = unsafe { slice::from_raw_parts(cd.p_objects, cd.object_count as usize) };
        //TODO: use color fields of `vk::DebugUtilsLabelExt`?
        let names = labels
            .iter()
            .map(|obj_info| {
                let name = unsafe { obj_info.p_object_name.as_ref() }
                    .map(|name| unsafe { CStr::from_ptr(name) }.to_string_lossy())
                    .unwrap_or(Cow::Borrowed("?"));

                format!(
                    "(type: {:?}, hndl: 0x{:x}, name: {})",
                    obj_info.object_type, obj_info.object_handle, name
                )
            })
            .collect::<Vec<_>>();
        let _ = std::panic::catch_unwind(|| {
            log::log!(level, "\tobjects: {}", names.join(", "));
        });
    }

    if cfg!(debug_assertions) && level == log::Level::Error {
        // Set canary and continue
        crate::VALIDATION_CANARY.add(message.to_string());
    }

    vk::FALSE
}

impl super::DebugUtilsCreateInfo {
    fn to_vk_create_info(&self) -> vk::DebugUtilsMessengerCreateInfoEXTBuilder<'_> {
        let user_data_ptr: *const super::DebugUtilsMessengerUserData = &*self.callback_data;
        vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(self.severity)
            .message_type(self.message_type)
            .user_data(user_data_ptr as *mut _)
            .pfn_user_callback(Some(debug_utils_messenger_callback))
    }
}

impl super::Swapchain {
    /// # Safety
    ///
    /// - The device must have been made idle before calling this function.
    unsafe fn release_resources(self, device: &ash::Device) -> Self {
        profiling::scope!("Swapchain::release_resources");
        {
            profiling::scope!("vkDeviceWaitIdle");
            // We need to also wait until all presentation work is done. Because there is no way to portably wait until
            // the presentation work is done, we are forced to wait until the device is idle.
            let _ = unsafe { device.device_wait_idle() };
        };
        unsafe { device.destroy_fence(self.fence, None) };
        self
    }
}

impl super::InstanceShared {
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub fn raw_instance(&self) -> &ash::Instance {
        &self.raw
    }

    pub fn instance_api_version(&self) -> u32 {
        self.instance_api_version
    }

    pub fn extensions(&self) -> &[&'static CStr] {
        &self.extensions[..]
    }
}

impl super::Instance {
    pub fn shared_instance(&self) -> &super::InstanceShared {
        &self.shared
    }

    /// Return the instance extension names wgpu would like to enable.
    ///
    /// Return a vector of the names of instance extensions actually available
    /// on `entry` that wgpu would like to enable.
    ///
    /// The `instance_api_version` argument should be the instance's Vulkan API
    /// version, as obtained from `vkEnumerateInstanceVersion`. This is the same
    /// space of values as the `VK_API_VERSION` constants.
    ///
    /// Note that wgpu can function without many of these extensions (for
    /// example, `VK_KHR_wayland_surface` is certainly not going to be available
    /// everywhere), but if one of these extensions is available at all, wgpu
    /// assumes that it has been enabled.
    pub fn desired_extensions(
        entry: &ash::Entry,
        _instance_api_version: u32,
        flags: wgt::InstanceFlags,
    ) -> Result<Vec<&'static CStr>, crate::InstanceError> {
        let instance_extensions = {
            profiling::scope!("vkEnumerateInstanceExtensionProperties");
            entry.enumerate_instance_extension_properties(None)
        };
        let instance_extensions = instance_extensions.map_err(|e| {
            crate::InstanceError::with_source(
                String::from("enumerate_instance_extension_properties() failed"),
                e,
            )
        })?;

        // Check our extensions against the available extensions
        let mut extensions: Vec<&'static CStr> = Vec::new();

        // VK_KHR_surface
        extensions.push(khr::Surface::name());

        // Platform-specific WSI extensions
        if cfg!(all(
            unix,
            not(target_os = "android"),
            not(target_os = "macos")
        )) {
            // VK_KHR_xlib_surface
            extensions.push(khr::XlibSurface::name());
            // VK_KHR_xcb_surface
            extensions.push(khr::XcbSurface::name());
            // VK_KHR_wayland_surface
            extensions.push(khr::WaylandSurface::name());
        }
        if cfg!(target_os = "android") {
            // VK_KHR_android_surface
            extensions.push(khr::AndroidSurface::name());
        }
        if cfg!(target_os = "windows") {
            // VK_KHR_win32_surface
            extensions.push(khr::Win32Surface::name());
        }
        if cfg!(target_os = "macos") {
            // VK_EXT_metal_surface
            extensions.push(ext::MetalSurface::name());
            extensions.push(ash::vk::KhrPortabilityEnumerationFn::name());
        }

        if flags.contains(wgt::InstanceFlags::DEBUG) {
            // VK_EXT_debug_utils
            extensions.push(ext::DebugUtils::name());
        }

        // VK_EXT_swapchain_colorspace
        // Provid wide color gamut
        extensions.push(vk::ExtSwapchainColorspaceFn::name());

        // VK_KHR_get_physical_device_properties2
        // Even though the extension was promoted to Vulkan 1.1, we still require the extension
        // so that we don't have to conditionally use the functions provided by the 1.1 instance
        extensions.push(vk::KhrGetPhysicalDeviceProperties2Fn::name());

        // Only keep available extensions.
        extensions.retain(|&ext| {
            if instance_extensions.iter().any(|inst_ext| {
                crate::auxil::cstr_from_bytes_until_nul(&inst_ext.extension_name) == Some(ext)
            }) {
                true
            } else {
                log::warn!("Unable to find extension: {}", ext.to_string_lossy());
                false
            }
        });
        Ok(extensions)
    }

    /// # Safety
    ///
    /// - `raw_instance` must be created from `entry`
    /// - `raw_instance` must be created respecting `instance_api_version`, `extensions` and `flags`
    /// - `extensions` must be a superset of `desired_extensions()` and must be created from the
    ///   same entry, `instance_api_version`` and flags.
    /// - `android_sdk_version` is ignored and can be `0` for all platforms besides Android
    ///
    /// If `debug_utils_user_data` is `Some`, then the validation layer is
    /// available, so create a [`vk::DebugUtilsMessengerEXT`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_raw(
        entry: ash::Entry,
        raw_instance: ash::Instance,
        instance_api_version: u32,
        android_sdk_version: u32,
        debug_utils_create_info: Option<super::DebugUtilsCreateInfo>,
        extensions: Vec<&'static CStr>,
        flags: wgt::InstanceFlags,
        has_nv_optimus: bool,
        drop_guard: Option<crate::DropGuard>,
    ) -> Result<Self, crate::InstanceError> {
        log::debug!("Instance version: 0x{:x}", instance_api_version);

        let debug_utils = if let Some(debug_utils_create_info) = debug_utils_create_info {
            if extensions.contains(&ext::DebugUtils::name()) {
                log::info!("Enabling debug utils");

                let extension = ext::DebugUtils::new(&entry, &raw_instance);
                let vk_info = debug_utils_create_info.to_vk_create_info();
                let messenger =
                    unsafe { extension.create_debug_utils_messenger(&vk_info, None) }.unwrap();

                Some(super::DebugUtils {
                    extension,
                    messenger,
                    callback_data: debug_utils_create_info.callback_data,
                })
            } else {
                log::info!("Debug utils not enabled: extension not listed");
                None
            }
        } else {
            log::info!(
                "Debug utils not enabled: \
                        debug_utils_user_data not passed to Instance::from_raw"
            );
            None
        };

        let get_physical_device_properties =
            if extensions.contains(&khr::GetPhysicalDeviceProperties2::name()) {
                log::debug!("Enabling device properties2");
                Some(khr::GetPhysicalDeviceProperties2::new(
                    &entry,
                    &raw_instance,
                ))
            } else {
                None
            };

        Ok(Self {
            shared: Arc::new(super::InstanceShared {
                raw: raw_instance,
                extensions,
                drop_guard,
                flags,
                debug_utils,
                get_physical_device_properties,
                entry,
                has_nv_optimus,
                instance_api_version,
                android_sdk_version,
            }),
        })
    }

    #[allow(dead_code)]
    fn create_surface_from_xlib(
        &self,
        dpy: *mut vk::Display,
        window: vk::Window,
    ) -> Result<super::Surface, crate::InstanceError> {
        if !self.shared.extensions.contains(&khr::XlibSurface::name()) {
            return Err(crate::InstanceError::new(String::from(
                "Vulkan driver does not support VK_KHR_xlib_surface",
            )));
        }

        let surface = {
            let xlib_loader = khr::XlibSurface::new(&self.shared.entry, &self.shared.raw);
            let info = vk::XlibSurfaceCreateInfoKHR::builder()
                .flags(vk::XlibSurfaceCreateFlagsKHR::empty())
                .window(window)
                .dpy(dpy);

            unsafe { xlib_loader.create_xlib_surface(&info, None) }
                .expect("XlibSurface::create_xlib_surface() failed")
        };

        Ok(self.create_surface_from_vk_surface_khr(surface))
    }

    #[allow(dead_code)]
    fn create_surface_from_xcb(
        &self,
        connection: *mut vk::xcb_connection_t,
        window: vk::xcb_window_t,
    ) -> Result<super::Surface, crate::InstanceError> {
        if !self.shared.extensions.contains(&khr::XcbSurface::name()) {
            return Err(crate::InstanceError::new(String::from(
                "Vulkan driver does not support VK_KHR_xcb_surface",
            )));
        }

        let surface = {
            let xcb_loader = khr::XcbSurface::new(&self.shared.entry, &self.shared.raw);
            let info = vk::XcbSurfaceCreateInfoKHR::builder()
                .flags(vk::XcbSurfaceCreateFlagsKHR::empty())
                .window(window)
                .connection(connection);

            unsafe { xcb_loader.create_xcb_surface(&info, None) }
                .expect("XcbSurface::create_xcb_surface() failed")
        };

        Ok(self.create_surface_from_vk_surface_khr(surface))
    }

    #[allow(dead_code)]
    fn create_surface_from_wayland(
        &self,
        display: *mut c_void,
        surface: *mut c_void,
    ) -> Result<super::Surface, crate::InstanceError> {
        if !self
            .shared
            .extensions
            .contains(&khr::WaylandSurface::name())
        {
            return Err(crate::InstanceError::new(String::from(
                "Vulkan driver does not support VK_KHR_wayland_surface",
            )));
        }

        let surface = {
            let w_loader = khr::WaylandSurface::new(&self.shared.entry, &self.shared.raw);
            let info = vk::WaylandSurfaceCreateInfoKHR::builder()
                .flags(vk::WaylandSurfaceCreateFlagsKHR::empty())
                .display(display)
                .surface(surface);

            unsafe { w_loader.create_wayland_surface(&info, None) }.expect("WaylandSurface failed")
        };

        Ok(self.create_surface_from_vk_surface_khr(surface))
    }

    #[allow(dead_code)]
    fn create_surface_android(
        &self,
        window: *const c_void,
    ) -> Result<super::Surface, crate::InstanceError> {
        if !self
            .shared
            .extensions
            .contains(&khr::AndroidSurface::name())
        {
            return Err(crate::InstanceError::new(String::from(
                "Vulkan driver does not support VK_KHR_android_surface",
            )));
        }

        let surface = {
            let a_loader = khr::AndroidSurface::new(&self.shared.entry, &self.shared.raw);
            let info = vk::AndroidSurfaceCreateInfoKHR::builder()
                .flags(vk::AndroidSurfaceCreateFlagsKHR::empty())
                .window(window as *mut _);

            unsafe { a_loader.create_android_surface(&info, None) }.expect("AndroidSurface failed")
        };

        Ok(self.create_surface_from_vk_surface_khr(surface))
    }

    #[allow(dead_code)]
    fn create_surface_from_hwnd(
        &self,
        hinstance: *mut c_void,
        hwnd: *mut c_void,
    ) -> Result<super::Surface, crate::InstanceError> {
        if !self.shared.extensions.contains(&khr::Win32Surface::name()) {
            return Err(crate::InstanceError::new(String::from(
                "Vulkan driver does not support VK_KHR_win32_surface",
            )));
        }

        let surface = {
            let info = vk::Win32SurfaceCreateInfoKHR::builder()
                .flags(vk::Win32SurfaceCreateFlagsKHR::empty())
                .hinstance(hinstance)
                .hwnd(hwnd);
            let win32_loader = khr::Win32Surface::new(&self.shared.entry, &self.shared.raw);
            unsafe {
                win32_loader
                    .create_win32_surface(&info, None)
                    .expect("Unable to create Win32 surface")
            }
        };

        Ok(self.create_surface_from_vk_surface_khr(surface))
    }

    #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
    fn create_surface_from_view(
        &self,
        view: *mut c_void,
    ) -> Result<super::Surface, crate::InstanceError> {
        if !self.shared.extensions.contains(&ext::MetalSurface::name()) {
            return Err(crate::InstanceError::new(String::from(
                "Vulkan driver does not support VK_EXT_metal_surface",
            )));
        }

        let layer = unsafe {
            crate::metal::Surface::get_metal_layer(view as *mut objc::runtime::Object, None)
        };

        let surface = {
            let metal_loader = ext::MetalSurface::new(&self.shared.entry, &self.shared.raw);
            let vk_info = vk::MetalSurfaceCreateInfoEXT::builder()
                .flags(vk::MetalSurfaceCreateFlagsEXT::empty())
                .layer(layer as *mut _)
                .build();

            unsafe { metal_loader.create_metal_surface(&vk_info, None).unwrap() }
        };

        Ok(self.create_surface_from_vk_surface_khr(surface))
    }

    fn create_surface_from_vk_surface_khr(&self, surface: vk::SurfaceKHR) -> super::Surface {
        let functor = khr::Surface::new(&self.shared.entry, &self.shared.raw);
        super::Surface {
            raw: surface,
            functor,
            instance: Arc::clone(&self.shared),
            swapchain: RwLock::new(None),
        }
    }
}

impl Drop for super::InstanceShared {
    fn drop(&mut self) {
        unsafe {
            // Keep du alive since destroy_instance may also log
            let _du = self.debug_utils.take().map(|du| {
                du.extension
                    .destroy_debug_utils_messenger(du.messenger, None);
                du
            });
            if let Some(_drop_guard) = self.drop_guard.take() {
                self.raw.destroy_instance(None);
            }
        }
    }
}

impl crate::Instance<super::Api> for super::Instance {
    unsafe fn init(desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        profiling::scope!("Init Vulkan Backend");
        use crate::auxil::cstr_from_bytes_until_nul;

        let entry = unsafe {
            profiling::scope!("Load vk library");
            ash::Entry::load()
        }
        .map_err(|err| {
            crate::InstanceError::with_source(String::from("missing Vulkan entry points"), err)
        })?;
        let version = {
            profiling::scope!("vkEnumerateInstanceVersion");
            entry.try_enumerate_instance_version()
        };
        let instance_api_version = match version {
            // Vulkan 1.1+
            Ok(Some(version)) => version,
            Ok(None) => vk::API_VERSION_1_0,
            Err(err) => {
                return Err(crate::InstanceError::with_source(
                    String::from("try_enumerate_instance_version() failed"),
                    err,
                ));
            }
        };

        let app_name = CString::new(desc.name).unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(1)
            .engine_name(CStr::from_bytes_with_nul(b"wgpu-hal\0").unwrap())
            .engine_version(2)
            .api_version(
                // Vulkan 1.0 doesn't like anything but 1.0 passed in here...
                if instance_api_version < vk::API_VERSION_1_1 {
                    vk::API_VERSION_1_0
                } else {
                    // This is the max Vulkan API version supported by `wgpu-hal`.
                    //
                    // If we want to increment this, there are some things that must be done first:
                    //  - Audit the behavioral differences between the previous and new API versions.
                    //  - Audit all extensions used by this backend:
                    //    - If any were promoted in the new API version and the behavior has changed, we must handle the new behavior in addition to the old behavior.
                    //    - If any were obsoleted in the new API version, we must implement a fallback for the new API version
                    //    - If any are non-KHR-vendored, we must ensure the new behavior is still correct (since backwards-compatibility is not guaranteed).
                    vk::API_VERSION_1_3
                },
            );

        let extensions = Self::desired_extensions(&entry, instance_api_version, desc.flags)?;

        let instance_layers = {
            profiling::scope!("vkEnumerateInstanceLayerProperties");
            entry.enumerate_instance_layer_properties()
        };
        let instance_layers = instance_layers.map_err(|e| {
            log::debug!("enumerate_instance_layer_properties: {:?}", e);
            crate::InstanceError::with_source(
                String::from("enumerate_instance_layer_properties() failed"),
                e,
            )
        })?;

        fn find_layer<'layers>(
            instance_layers: &'layers [vk::LayerProperties],
            name: &CStr,
        ) -> Option<&'layers vk::LayerProperties> {
            instance_layers
                .iter()
                .find(|inst_layer| cstr_from_bytes_until_nul(&inst_layer.layer_name) == Some(name))
        }

        let nv_optimus_layer = CStr::from_bytes_with_nul(b"VK_LAYER_NV_optimus\0").unwrap();
        let has_nv_optimus = find_layer(&instance_layers, nv_optimus_layer).is_some();

        let obs_layer = CStr::from_bytes_with_nul(b"VK_LAYER_OBS_HOOK\0").unwrap();
        let has_obs_layer = find_layer(&instance_layers, obs_layer).is_some();

        let mut layers: Vec<&'static CStr> = Vec::new();

        // Request validation layer if asked.
        let mut debug_utils = None;
        if desc.flags.intersects(wgt::InstanceFlags::VALIDATION) {
            let validation_layer_name =
                CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap();
            if let Some(layer_properties) = find_layer(&instance_layers, validation_layer_name) {
                layers.push(validation_layer_name);

                if extensions.contains(&ext::DebugUtils::name()) {
                    // Put the callback data on the heap, to ensure it will never be
                    // moved.
                    let callback_data = Box::new(super::DebugUtilsMessengerUserData {
                        validation_layer_description: cstr_from_bytes_until_nul(
                            &layer_properties.description,
                        )
                        .unwrap()
                        .to_owned(),
                        validation_layer_spec_version: layer_properties.spec_version,
                        has_obs_layer,
                    });

                    // having ERROR unconditionally because Vk doesn't like empty flags
                    let mut severity = vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
                    if log::max_level() >= log::LevelFilter::Debug {
                        severity |= vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE;
                    }
                    if log::max_level() >= log::LevelFilter::Info {
                        severity |= vk::DebugUtilsMessageSeverityFlagsEXT::INFO;
                    }
                    if log::max_level() >= log::LevelFilter::Warn {
                        severity |= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING;
                    }

                    let message_type = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE;

                    let create_info = super::DebugUtilsCreateInfo {
                        severity,
                        message_type,
                        callback_data,
                    };

                    let vk_create_info = create_info.to_vk_create_info().build();

                    debug_utils = Some((create_info, vk_create_info));
                }
            } else {
                log::warn!(
                    "InstanceFlags::VALIDATION requested, but unable to find layer: {}",
                    validation_layer_name.to_string_lossy()
                );
            }
        }

        #[cfg(target_os = "android")]
        let android_sdk_version = {
            let properties = android_system_properties::AndroidSystemProperties::new();
            // See: https://developer.android.com/reference/android/os/Build.VERSION_CODES
            if let Some(val) = properties.get("ro.build.version.sdk") {
                match val.parse::<u32>() {
                    Ok(sdk_ver) => sdk_ver,
                    Err(err) => {
                        log::error!(
                            "Couldn't parse Android's ro.build.version.sdk system property ({val}): {err}"
                        );
                        0
                    }
                }
            } else {
                log::error!("Couldn't read Android's ro.build.version.sdk system property");
                0
            }
        };
        #[cfg(not(target_os = "android"))]
        let android_sdk_version = 0;

        let mut flags = vk::InstanceCreateFlags::empty();

        // Avoid VUID-VkInstanceCreateInfo-flags-06559: Only ask the instance to
        // enumerate incomplete Vulkan implementations (which we need on Mac) if
        // we managed to find the extension that provides the flag.
        if extensions.contains(&ash::vk::KhrPortabilityEnumerationFn::name()) {
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }
        let vk_instance = {
            let str_pointers = layers
                .iter()
                .chain(extensions.iter())
                .map(|&s: &&'static _| {
                    // Safe because `layers` and `extensions` entries have static lifetime.
                    s.as_ptr()
                })
                .collect::<Vec<_>>();

            let mut create_info = vk::InstanceCreateInfo::builder()
                .flags(flags)
                .application_info(&app_info)
                .enabled_layer_names(&str_pointers[..layers.len()])
                .enabled_extension_names(&str_pointers[layers.len()..]);

            if let Some(&mut (_, ref mut vk_create_info)) = debug_utils.as_mut() {
                create_info = create_info.push_next(vk_create_info);
            }

            unsafe {
                profiling::scope!("vkCreateInstance");
                entry.create_instance(&create_info, None)
            }
            .map_err(|e| {
                crate::InstanceError::with_source(
                    String::from("Entry::create_instance() failed"),
                    e,
                )
            })?
        };

        unsafe {
            Self::from_raw(
                entry,
                vk_instance,
                instance_api_version,
                android_sdk_version,
                debug_utils.map(|(i, _)| i),
                extensions,
                desc.flags,
                has_nv_optimus,
                Some(Box::new(())), // `Some` signals that wgpu-hal is in charge of destroying vk_instance
            )
        }
    }

    unsafe fn create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<super::Surface, crate::InstanceError> {
        use raw_window_handle::{RawDisplayHandle as Rdh, RawWindowHandle as Rwh};

        match (window_handle, display_handle) {
            (Rwh::Wayland(handle), Rdh::Wayland(display)) => {
                self.create_surface_from_wayland(display.display.as_ptr(), handle.surface.as_ptr())
            }
            (Rwh::Xlib(handle), Rdh::Xlib(display)) => {
                let display = display.display.expect("Display pointer is not set.");
                self.create_surface_from_xlib(display.as_ptr() as *mut *const c_void, handle.window)
            }
            (Rwh::Xcb(handle), Rdh::Xcb(display)) => {
                let connection = display.connection.expect("Pointer to X-Server is not set.");
                self.create_surface_from_xcb(connection.as_ptr(), handle.window.get())
            }
            (Rwh::AndroidNdk(handle), _) => {
                self.create_surface_android(handle.a_native_window.as_ptr())
            }
            #[cfg(windows)]
            (Rwh::Win32(handle), _) => {
                use winapi::um::libloaderapi::GetModuleHandleW;

                let hinstance = unsafe { GetModuleHandleW(std::ptr::null()) };
                self.create_surface_from_hwnd(hinstance as *mut _, handle.hwnd.get() as *mut _)
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            (Rwh::AppKit(handle), _)
                if self.shared.extensions.contains(&ext::MetalSurface::name()) =>
            {
                self.create_surface_from_view(handle.ns_view.as_ptr())
            }
            #[cfg(all(target_os = "ios", feature = "metal"))]
            (Rwh::UiKit(handle), _)
                if self.shared.extensions.contains(&ext::MetalSurface::name()) =>
            {
                self.create_surface_from_view(handle.ui_view.as_ptr())
            }
            (_, _) => Err(crate::InstanceError::new(format!(
                "window handle {window_handle:?} is not a Vulkan-compatible handle"
            ))),
        }
    }

    unsafe fn destroy_surface(&self, surface: super::Surface) {
        unsafe { surface.functor.destroy_surface(surface.raw, None) };
    }

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<super::Api>> {
        use crate::auxil::db;

        let raw_devices = match unsafe { self.shared.raw.enumerate_physical_devices() } {
            Ok(devices) => devices,
            Err(err) => {
                log::error!("enumerate_adapters: {}", err);
                Vec::new()
            }
        };

        let mut exposed_adapters = raw_devices
            .into_iter()
            .flat_map(|device| self.expose_adapter(device))
            .collect::<Vec<_>>();

        // Detect if it's an Intel + NVidia configuration with Optimus
        let has_nvidia_dgpu = exposed_adapters.iter().any(|exposed| {
            exposed.info.device_type == wgt::DeviceType::DiscreteGpu
                && exposed.info.vendor == db::nvidia::VENDOR
        });
        if cfg!(target_os = "linux") && has_nvidia_dgpu && self.shared.has_nv_optimus {
            for exposed in exposed_adapters.iter_mut() {
                if exposed.info.device_type == wgt::DeviceType::IntegratedGpu
                    && exposed.info.vendor == db::intel::VENDOR
                {
                    // Check if mesa driver and version less than 21.2
                    if let Some(version) = exposed.info.driver_info.split_once("Mesa ").map(|s| {
                        let mut components = s.1.split('.');
                        let major = components.next().and_then(|s| u8::from_str(s).ok());
                        let minor = components.next().and_then(|s| u8::from_str(s).ok());
                        if let (Some(major), Some(minor)) = (major, minor) {
                            (major, minor)
                        } else {
                            (0, 0)
                        }
                    }) {
                        if version < (21, 2) {
                            // See https://gitlab.freedesktop.org/mesa/mesa/-/issues/4688
                            log::warn!(
                                "Disabling presentation on '{}' (id {:?}) due to NV Optimus and Intel Mesa < v21.2",
                                exposed.info.name,
                                exposed.adapter.raw
                            );
                            exposed.adapter.private_caps.can_present = false;
                        }
                    }
                }
            }
        }

        exposed_adapters
    }
}

impl crate::Surface<super::Api> for super::Surface {
    unsafe fn configure(
        &self,
        device: &super::Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        // Safety: `configure`'s contract guarantees there are no resources derived from the swapchain in use.
        let mut swap_chain = self.swapchain.write();
        let old = swap_chain
            .take()
            .map(|sc| unsafe { sc.release_resources(&device.shared.raw) });

        let swapchain = unsafe { device.create_swapchain(self, config, old)? };
        *swap_chain = Some(swapchain);

        Ok(())
    }

    unsafe fn unconfigure(&self, device: &super::Device) {
        if let Some(sc) = self.swapchain.write().take() {
            // Safety: `unconfigure`'s contract guarantees there are no resources derived from the swapchain in use.
            let swapchain = unsafe { sc.release_resources(&device.shared.raw) };
            unsafe { swapchain.functor.destroy_swapchain(swapchain.raw, None) };
        }
    }

    unsafe fn acquire_texture(
        &self,
        timeout: Option<std::time::Duration>,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<super::Api>>, crate::SurfaceError> {
        let mut swapchain = self.swapchain.write();
        let sc = swapchain.as_mut().unwrap();

        let mut timeout_ns = match timeout {
            Some(duration) => duration.as_nanos() as u64,
            None => u64::MAX,
        };

        // AcquireNextImageKHR on Android (prior to Android 11) doesn't support timeouts
        // and will also log verbose warnings if tying to use a timeout.
        //
        // Android 10 implementation for reference:
        // https://android.googlesource.com/platform/frameworks/native/+/refs/tags/android-mainline-10.0.0_r13/vulkan/libvulkan/swapchain.cpp#1426
        // Android 11 implementation for reference:
        // https://android.googlesource.com/platform/frameworks/native/+/refs/tags/android-mainline-11.0.0_r45/vulkan/libvulkan/swapchain.cpp#1438
        //
        // Android 11 corresponds to an SDK_INT/ro.build.version.sdk of 30
        if cfg!(target_os = "android") && self.instance.android_sdk_version < 30 {
            timeout_ns = u64::MAX;
        }

        // will block if no image is available
        let (index, suboptimal) = match unsafe {
            sc.functor
                .acquire_next_image(sc.raw, timeout_ns, vk::Semaphore::null(), sc.fence)
        } {
            // We treat `VK_SUBOPTIMAL_KHR` as `VK_SUCCESS` on Android.
            // See the comment in `Queue::present`.
            #[cfg(target_os = "android")]
            Ok((index, _)) => (index, false),
            #[cfg(not(target_os = "android"))]
            Ok(pair) => pair,
            Err(error) => {
                return match error {
                    vk::Result::TIMEOUT => Ok(None),
                    vk::Result::NOT_READY | vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        Err(crate::SurfaceError::Outdated)
                    }
                    vk::Result::ERROR_SURFACE_LOST_KHR => Err(crate::SurfaceError::Lost),
                    other => Err(crate::DeviceError::from(other).into()),
                }
            }
        };

        // special case for Intel Vulkan returning bizzare values (ugh)
        if sc.device.vendor_id == crate::auxil::db::intel::VENDOR && index > 0x100 {
            return Err(crate::SurfaceError::Outdated);
        }

        let fences = &[sc.fence];

        unsafe { sc.device.raw.wait_for_fences(fences, true, !0) }
            .map_err(crate::DeviceError::from)?;
        unsafe { sc.device.raw.reset_fences(fences) }.map_err(crate::DeviceError::from)?;

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkRenderPassBeginInfo.html#VUID-VkRenderPassBeginInfo-framebuffer-03209
        let raw_flags = if sc
            .raw_flags
            .contains(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT)
        {
            vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE
        } else {
            vk::ImageCreateFlags::empty()
        };

        let texture = super::SurfaceTexture {
            index,
            texture: super::Texture {
                raw: sc.images[index as usize],
                drop_guard: None,
                block: None,
                usage: sc.config.usage,
                format: sc.config.format,
                raw_flags,
                copy_size: crate::CopyExtent {
                    width: sc.config.extent.width,
                    height: sc.config.extent.height,
                    depth: 1,
                },
                view_formats: sc.view_formats.clone(),
            },
        };
        Ok(Some(crate::AcquiredSurfaceTexture {
            texture,
            suboptimal,
        }))
    }

    unsafe fn discard_texture(&self, _texture: super::SurfaceTexture) {}
}
