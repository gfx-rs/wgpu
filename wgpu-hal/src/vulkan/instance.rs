use std::{
    ffi::{c_void, CStr, CString},
    slice,
    sync::Arc,
    thread,
};

use ash::{
    extensions::{ext, khr},
    vk,
};

unsafe extern "system" fn debug_utils_messenger_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data_ptr: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    const VUID_VKSWAPCHAINCREATEINFOKHR_IMAGEEXTENT_01274: i32 = 0x7cd0911d;
    use std::borrow::Cow;

    if thread::panicking() {
        return vk::FALSE;
    }

    let level = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
        _ => log::Level::Warn,
    };

    let cd = unsafe { &*callback_data_ptr };

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

    // Silence Vulkan Validation error "VUID-VkSwapchainCreateInfoKHR-imageExtent-01274"
    // - it's a false positive due to the inherent racy-ness of surface resizing
    if cd.message_id_number == VUID_VKSWAPCHAINCREATEINFOKHR_IMAGEEXTENT_01274 {
        return vk::FALSE;
    }

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
        crate::VALIDATION_CANARY.set();
    }

    vk::FALSE
}

impl super::Swapchain {
    unsafe fn release_resources(self, device: &ash::Device) -> Self {
        profiling::scope!("Swapchain::release_resources");
        {
            profiling::scope!("vkDeviceWaitIdle");
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

    pub fn driver_api_version(&self) -> u32 {
        self.driver_api_version
    }

    pub fn extensions(&self) -> &[&'static CStr] {
        &self.extensions[..]
    }
}

impl super::Instance {
    pub fn shared_instance(&self) -> &super::InstanceShared {
        &self.shared
    }

    pub fn required_extensions(
        entry: &ash::Entry,
        _driver_api_version: u32,
        flags: crate::InstanceFlags,
    ) -> Result<Vec<&'static CStr>, crate::InstanceError> {
        let instance_extensions = entry
            .enumerate_instance_extension_properties(None)
            .map_err(|e| {
                log::info!("enumerate_instance_extension_properties: {:?}", e);
                crate::InstanceError
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
        }

        if flags.contains(crate::InstanceFlags::DEBUG) {
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
                log::info!("Unable to find extension: {}", ext.to_string_lossy());
                false
            }
        });
        Ok(extensions)
    }

    /// # Safety
    ///
    /// - `raw_instance` must be created from `entry`
    /// - `raw_instance` must be created respecting `driver_api_version`, `extensions` and `flags`
    /// - `extensions` must be a superset of `required_extensions()` and must be created from the
    ///   same entry, driver_api_version and flags.
    /// - `android_sdk_version` is ignored and can be `0` for all platforms besides Android
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_raw(
        entry: ash::Entry,
        raw_instance: ash::Instance,
        driver_api_version: u32,
        android_sdk_version: u32,
        extensions: Vec<&'static CStr>,
        flags: crate::InstanceFlags,
        has_nv_optimus: bool,
        drop_guard: Option<crate::DropGuard>,
    ) -> Result<Self, crate::InstanceError> {
        log::info!("Instance version: 0x{:x}", driver_api_version);

        let debug_utils = if extensions.contains(&ext::DebugUtils::name()) {
            log::info!("Enabling debug utils");
            let extension = ext::DebugUtils::new(&entry, &raw_instance);
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
            let vk_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
                .message_severity(severity)
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_utils_messenger_callback));
            let messenger =
                unsafe { extension.create_debug_utils_messenger(&vk_info, None) }.unwrap();
            Some(super::DebugUtils {
                extension,
                messenger,
            })
        } else {
            None
        };

        let get_physical_device_properties =
            if extensions.contains(&khr::GetPhysicalDeviceProperties2::name()) {
                log::info!("Enabling device properties2");
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
                driver_api_version,
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
            log::warn!("Vulkan driver does not support VK_KHR_xlib_surface");
            return Err(crate::InstanceError);
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
            log::warn!("Vulkan driver does not support VK_KHR_xcb_surface");
            return Err(crate::InstanceError);
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
            log::debug!("Vulkan driver does not support VK_KHR_wayland_surface");
            return Err(crate::InstanceError);
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
            log::warn!("Vulkan driver does not support VK_KHR_android_surface");
            return Err(crate::InstanceError);
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
            log::debug!("Vulkan driver does not support VK_KHR_win32_surface");
            return Err(crate::InstanceError);
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

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn create_surface_from_view(
        &self,
        view: *mut c_void,
    ) -> Result<super::Surface, crate::InstanceError> {
        if !self.shared.extensions.contains(&ext::MetalSurface::name()) {
            log::warn!("Vulkan driver does not support VK_EXT_metal_surface");
            return Err(crate::InstanceError);
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
            swapchain: None,
        }
    }
}

impl Drop for super::InstanceShared {
    fn drop(&mut self) {
        unsafe {
            if let Some(du) = self.debug_utils.take() {
                du.extension
                    .destroy_debug_utils_messenger(du.messenger, None);
            }
            if let Some(_drop_guard) = self.drop_guard.take() {
                self.raw.destroy_instance(None);
            }
        }
    }
}

impl crate::Instance<super::Api> for super::Instance {
    unsafe fn init(desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        use crate::auxil::cstr_from_bytes_until_nul;

        let entry = match unsafe { ash::Entry::load() } {
            Ok(entry) => entry,
            Err(err) => {
                log::info!("Missing Vulkan entry points: {:?}", err);
                return Err(crate::InstanceError);
            }
        };
        let driver_api_version = match entry.try_enumerate_instance_version() {
            // Vulkan 1.1+
            Ok(Some(version)) => version,
            Ok(None) => vk::API_VERSION_1_0,
            Err(err) => {
                log::warn!("try_enumerate_instance_version: {:?}", err);
                return Err(crate::InstanceError);
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
                if driver_api_version < vk::API_VERSION_1_1 {
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
                    vk::HEADER_VERSION_COMPLETE
                },
            );

        let extensions = Self::required_extensions(&entry, driver_api_version, desc.flags)?;

        let instance_layers = entry.enumerate_instance_layer_properties().map_err(|e| {
            log::info!("enumerate_instance_layer_properties: {:?}", e);
            crate::InstanceError
        })?;

        let nv_optimus_layer = CStr::from_bytes_with_nul(b"VK_LAYER_NV_optimus\0").unwrap();
        let has_nv_optimus = instance_layers.iter().any(|inst_layer| {
            cstr_from_bytes_until_nul(&inst_layer.layer_name) == Some(nv_optimus_layer)
        });

        // Check requested layers against the available layers
        let layers = {
            let mut layers: Vec<&'static CStr> = Vec::new();
            if desc.flags.contains(crate::InstanceFlags::VALIDATION) {
                layers.push(CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap());
            }

            // Only keep available layers.
            layers.retain(|&layer| {
                if instance_layers.iter().any(|inst_layer| {
                    cstr_from_bytes_until_nul(&inst_layer.layer_name) == Some(layer)
                }) {
                    true
                } else {
                    log::warn!("Unable to find layer: {}", layer.to_string_lossy());
                    false
                }
            });
            layers
        };

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

        let vk_instance = {
            let str_pointers = layers
                .iter()
                .chain(extensions.iter())
                .map(|&s| {
                    // Safe because `layers` and `extensions` entries have static lifetime.
                    s.as_ptr()
                })
                .collect::<Vec<_>>();

            let create_info = vk::InstanceCreateInfo::builder()
                .flags(vk::InstanceCreateFlags::empty())
                .application_info(&app_info)
                .enabled_layer_names(&str_pointers[..layers.len()])
                .enabled_extension_names(&str_pointers[layers.len()..]);

            unsafe { entry.create_instance(&create_info, None) }.map_err(|e| {
                log::warn!("create_instance: {:?}", e);
                crate::InstanceError
            })?
        };

        unsafe {
            Self::from_raw(
                entry,
                vk_instance,
                driver_api_version,
                android_sdk_version,
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
                self.create_surface_from_wayland(display.display, handle.surface)
            }
            (Rwh::Xlib(handle), Rdh::Xlib(display)) => {
                self.create_surface_from_xlib(display.display as *mut _, handle.window)
            }
            (Rwh::Xcb(handle), Rdh::Xcb(display)) => {
                self.create_surface_from_xcb(display.connection, handle.window)
            }
            (Rwh::AndroidNdk(handle), _) => self.create_surface_android(handle.a_native_window),
            #[cfg(windows)]
            (Rwh::Win32(handle), _) => {
                use winapi::um::libloaderapi::GetModuleHandleW;

                let hinstance = unsafe { GetModuleHandleW(std::ptr::null()) };
                self.create_surface_from_hwnd(hinstance as *mut _, handle.hwnd)
            }
            #[cfg(target_os = "macos")]
            (Rwh::AppKit(handle), _)
                if self.shared.extensions.contains(&ext::MetalSurface::name()) =>
            {
                self.create_surface_from_view(handle.ns_view)
            }
            #[cfg(target_os = "ios")]
            (Rwh::UiKit(handle), _)
                if self.shared.extensions.contains(&ext::MetalSurface::name()) =>
            {
                self.create_surface_from_view(handle.ui_view)
            }
            (_, _) => Err(crate::InstanceError),
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
                && exposed.info.vendor == db::nvidia::VENDOR as usize
        });
        if cfg!(target_os = "linux") && has_nvidia_dgpu && self.shared.has_nv_optimus {
            for exposed in exposed_adapters.iter_mut() {
                if exposed.info.device_type == wgt::DeviceType::IntegratedGpu
                    && exposed.info.vendor == db::intel::VENDOR as usize
                {
                    // See https://gitlab.freedesktop.org/mesa/mesa/-/issues/4688
                    log::warn!(
                        "Disabling presentation on '{}' (id {:?}) because of NV Optimus (on Linux)",
                        exposed.info.name,
                        exposed.adapter.raw
                    );
                    exposed.adapter.private_caps.can_present = false;
                }
            }
        }

        exposed_adapters
    }
}

impl crate::Surface<super::Api> for super::Surface {
    unsafe fn configure(
        &mut self,
        device: &super::Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        let old = self
            .swapchain
            .take()
            .map(|sc| unsafe { sc.release_resources(&device.shared.raw) });

        let swapchain = unsafe { device.create_swapchain(self, config, old)? };
        self.swapchain = Some(swapchain);

        Ok(())
    }

    unsafe fn unconfigure(&mut self, device: &super::Device) {
        if let Some(sc) = self.swapchain.take() {
            let swapchain = unsafe { sc.release_resources(&device.shared.raw) };
            unsafe { swapchain.functor.destroy_swapchain(swapchain.raw, None) };
        }
    }

    unsafe fn acquire_texture(
        &mut self,
        timeout: Option<std::time::Duration>,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<super::Api>>, crate::SurfaceError> {
        let sc = self.swapchain.as_mut().unwrap();

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

    unsafe fn discard_texture(&mut self, _texture: super::SurfaceTexture) {}
}
