#![cfg(drm)]

use alloc::{string::ToString, vec::Vec};
use core::{mem::MaybeUninit, num::NonZeroU32};
use std::os::fd::{AsFd, BorrowedFd};

use ash::{ext, khr, vk};

use drm::{
    self,
    control::{self, connector, Device as _},
};

struct Card(i32);
impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        unsafe { BorrowedFd::borrow_raw(self.0) }
    }
}
impl control::Device for Card {}
impl drm::Device for Card {}

macro_rules! to_u64 {
    ($expr:expr) => {{
        #[allow(trivial_numeric_casts)]
        let expr = $expr as u64;
        assert!(size_of_val(&expr) <= size_of::<u64>());
        expr
    }};
}

impl super::Instance {
    /// Creates a new surface from the given drm fd and plane, deriving the connector and mode.
    ///
    /// # Safety
    ///
    /// - All parameters must point to valid DRM values.
    pub fn create_surface_from_drm_plane(
        &self,
        fd: i32,
        plane: u32,
    ) -> Result<super::Surface, crate::InstanceError> {
        let card = Card(fd);
        let plane_info = card
            .get_plane(
                NonZeroU32::new(plane)
                    .ok_or(crate::InstanceError::new("Invalid drm plane".to_string()))?
                    .into(),
            )
            .map_err(|e| crate::InstanceError::with_source("drm plane not found".to_string(), e))?;
        let crtc_handle = plane_info.crtc().ok_or(crate::InstanceError::new(
            "No CRTC for drm plane".to_string(),
        ))?;
        let crtc_info = card
            .get_crtc(crtc_handle)
            .map_err(|e| crate::InstanceError::with_source("drm CRTC not found".to_string(), e))?;
        let mode = crtc_info.mode().ok_or(crate::InstanceError::new(
            "No mode for drm CRTC".to_string(),
        ))?;

        let mut connector_and_mode = None;
        let resources = card.resource_handles().map_err(|e| {
            crate::InstanceError::with_source("No drm resource handles found".to_string(), e)
        })?;
        for connector_handle in resources.connectors() {
            let Ok(connector_info) = card.get_connector(*connector_handle, false) else {
                continue;
            };
            if connector_info.state() != connector::State::Connected {
                continue;
            }

            if let Some(encoder_handle) = connector_info.current_encoder() {
                if let Ok(encoder_info) = card.get_encoder(encoder_handle) {
                    if encoder_info.crtc() == Some(crtc_handle) {
                        connector_and_mode = Some((*connector_handle, mode));
                        break;
                    }
                }
            }
        }

        let (connector_handle, mode) = connector_and_mode.ok_or(crate::InstanceError::new(
            "Failed to derive drm connector and mode for plane".to_string(),
        ))?;
        let (width, height) = mode.size();
        // Rate in millihertz
        let refresh_rate = (((mode.clock() as f64 * 1000.0)
            / (mode.hsync().2 as f64 * mode.vsync().2 as f64))
            * 1000.0)
            .round() as u32;
        unsafe {
            self.create_surface_from_drm(
                fd,
                plane,
                connector_handle.into(),
                width as u32,
                height as u32,
                refresh_rate,
            )
        }
    }

    /// Creates a new surface from the given drm configuration.
    ///
    /// # Safety
    ///
    /// - All parameters must point to valid DRM values.
    pub unsafe fn create_surface_from_drm(
        &self,
        fd: i32,
        plane: u32,
        connector_id: u32,
        width: u32,
        height: u32,
        refresh_rate: u32,
    ) -> Result<super::Surface, crate::InstanceError> {
        if !self
            .shared
            .extensions
            .contains(&ext::acquire_drm_display::NAME)
        {
            return Err(crate::InstanceError::new(
                "Vulkan driver does not support VK_EXT_acquire_drm_display".to_string(),
            ));
        }

        let drm_stat = {
            let mut stat = MaybeUninit::<libc::stat>::uninit();

            if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } != 0 {
                return Err(crate::InstanceError::new(
                    "Unable to fstat drm device".to_string(),
                ));
            }

            unsafe { stat.assume_init() }
        };

        let raw_devices = match unsafe { self.shared.raw.enumerate_physical_devices() } {
            Ok(devices) => devices,
            Err(err) => {
                log::error!("enumerate_adapters: {err}");
                Vec::new()
            }
        };

        let mut physical_device = None;

        for device in raw_devices {
            let properties2 = vk::PhysicalDeviceProperties2KHR::default();

            let mut drm_props = vk::PhysicalDeviceDrmPropertiesEXT::default();
            let mut properties2 = properties2.push_next(&mut drm_props);

            unsafe {
                self.shared
                    .raw
                    .get_physical_device_properties2(device, &mut properties2)
            };

            /*
                The makedev call is just bit manipulation to combine major and minor device numbers into a Unix device ID.
                It doesn't perform any filesystem operations, only bitshifting.
                See: https://github.com/rust-lang/libc/blob/268e1b3810ac07ed637d9005bc1a54e49218c958/src/unix/linux_like/linux/mod.rs#L6049
                We use the resulting device IDs to check if the Vulkan raw device from enumerate_physical_devices
                matches the DRM device referred to by our file descriptor.
            */

            let primary_devid =
                libc::makedev(drm_props.primary_major as _, drm_props.primary_minor as _);
            let render_devid =
                libc::makedev(drm_props.render_major as _, drm_props.render_minor as _);

            // On most platforms, both `*_devid`s and `st_rdev` are `dev_t`s (which is generally
            // observed to be an unsigned integral type no greater than 64 bits). However, on some
            // platforms, there divergences from this pattern:
            //
            // - `armv7-linux-androideabi`: `dev_t` is `c_ulong`, and `*_devid`s are `dev_t`, but
            //   `st_rdev` is `c_ulonglong`. So, we can't just do a `==` comparison.
            // - OpenBSD has `dev_t` on both sides, but is `i32` (N.B., unsigned). Therefore, we
            //   can't just use `u64::from`.
            #[allow(clippy::useless_conversion)]
            if [primary_devid, render_devid]
                .map(|devid| to_u64!(devid))
                .contains(&to_u64!(drm_stat.st_rdev))
            {
                physical_device = Some(device)
            }
        }

        let physical_device = physical_device.ok_or(crate::InstanceError::new(
            "Failed to find suitable drm device".to_string(),
        ))?;

        let acquire_drm_display_instance =
            ext::acquire_drm_display::Instance::new(&self.shared.entry, &self.shared.raw);

        let display = unsafe {
            acquire_drm_display_instance
                .get_drm_display(physical_device, fd, connector_id)
                .expect("Failed to get drm display")
        };

        unsafe {
            acquire_drm_display_instance
                .acquire_drm_display(physical_device, fd, display)
                .expect("Failed to acquire drm display")
        }

        let display_instance = khr::display::Instance::new(&self.shared.entry, &self.shared.raw);

        let modes = unsafe {
            display_instance
                .get_display_mode_properties(physical_device, display)
                .expect("Failed to get display modes")
        };

        let mut mode = None;

        for current_mode in modes {
            log::trace!(
                "Comparing mode {}x{}@{} with {width}x{height}@{refresh_rate}",
                current_mode.parameters.visible_region.width,
                current_mode.parameters.visible_region.height,
                current_mode.parameters.refresh_rate
            );
            if current_mode.parameters.refresh_rate == refresh_rate
                && current_mode.parameters.visible_region.width == width
                && current_mode.parameters.visible_region.height == height
            {
                mode = Some(current_mode)
            }
        }

        let mode = mode.ok_or(crate::InstanceError::new(
            "Failed to find suitable display mode".to_string(),
        ))?;

        let create_info = vk::DisplaySurfaceCreateInfoKHR::default()
            .display_mode(mode.display_mode)
            .image_extent(mode.parameters.visible_region)
            .transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .alpha_mode(vk::DisplayPlaneAlphaFlagsKHR::OPAQUE)
            .plane_index(plane);

        let surface = unsafe { display_instance.create_display_plane_surface(&create_info, None) }
            .expect("Failed to create DRM surface");

        Ok(self.create_surface_from_vk_surface_khr(surface))
    }
}
