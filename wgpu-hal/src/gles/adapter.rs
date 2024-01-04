use glow::HasContext;
use parking_lot::Mutex;
use std::sync::{atomic::AtomicU8, Arc};
use wgt::AstcChannel;

use crate::auxil::db;

// https://webgl2fundamentals.org/webgl/lessons/webgl-data-textures.html

const GL_UNMASKED_VENDOR_WEBGL: u32 = 0x9245;
const GL_UNMASKED_RENDERER_WEBGL: u32 = 0x9246;

impl super::Adapter {
    /// Note that this function is intentionally lenient in regards to parsing,
    /// and will try to recover at least the first two version numbers without
    /// resulting in an `Err`.
    /// # Notes
    /// `WebGL 2` version returned as `OpenGL ES 3.0`
    fn parse_version(mut src: &str) -> Result<(u8, u8), crate::InstanceError> {
        let webgl_sig = "WebGL ";
        // According to the WebGL specification
        // VERSION  WebGL<space>1.0<space><vendor-specific information>
        // SHADING_LANGUAGE_VERSION WebGL<space>GLSL<space>ES<space>1.0<space><vendor-specific information>
        let is_webgl = src.starts_with(webgl_sig);
        if is_webgl {
            let pos = src.rfind(webgl_sig).unwrap_or(0);
            src = &src[pos + webgl_sig.len()..];
        } else {
            let es_sig = " ES ";
            match src.rfind(es_sig) {
                Some(pos) => {
                    src = &src[pos + es_sig.len()..];
                }
                None => {
                    return Err(crate::InstanceError::new(format!(
                        "OpenGL version {src:?} does not contain 'ES'"
                    )));
                }
            }
        };

        let glsl_es_sig = "GLSL ES ";
        let is_glsl = match src.find(glsl_es_sig) {
            Some(pos) => {
                src = &src[pos + glsl_es_sig.len()..];
                true
            }
            None => false,
        };

        Self::parse_full_version(src).map(|(major, minor)| {
            (
                // Return WebGL 2.0 version as OpenGL ES 3.0
                if is_webgl && !is_glsl {
                    major + 1
                } else {
                    major
                },
                minor,
            )
        })
    }

    /// According to the OpenGL specification, the version information is
    /// expected to follow the following syntax:
    ///
    /// ~~~bnf
    /// <major>       ::= <number>
    /// <minor>       ::= <number>
    /// <revision>    ::= <number>
    /// <vendor-info> ::= <string>
    /// <release>     ::= <major> "." <minor> ["." <release>]
    /// <version>     ::= <release> [" " <vendor-info>]
    /// ~~~
    ///
    /// Note that this function is intentionally lenient in regards to parsing,
    /// and will try to recover at least the first two version numbers without
    /// resulting in an `Err`.
    pub(super) fn parse_full_version(src: &str) -> Result<(u8, u8), crate::InstanceError> {
        let (version, _vendor_info) = match src.find(' ') {
            Some(i) => (&src[..i], src[i + 1..].to_string()),
            None => (src, String::new()),
        };

        // TODO: make this even more lenient so that we can also accept
        // `<major> "." <minor> [<???>]`
        let mut it = version.split('.');
        let major = it.next().and_then(|s| s.parse().ok());
        let minor = it.next().and_then(|s| {
            let trimmed = if s.starts_with('0') {
                "0"
            } else {
                s.trim_end_matches('0')
            };
            trimmed.parse().ok()
        });

        match (major, minor) {
            (Some(major), Some(minor)) => Ok((major, minor)),
            _ => Err(crate::InstanceError::new(format!(
                "unable to extract OpenGL version from {version:?}"
            ))),
        }
    }

    fn make_info(vendor_orig: String, renderer_orig: String) -> wgt::AdapterInfo {
        let vendor = vendor_orig.to_lowercase();
        let renderer = renderer_orig.to_lowercase();

        // opengl has no way to discern device_type, so we can try to infer it from the renderer string
        let strings_that_imply_integrated = [
            " xpress", // space here is on purpose so we don't match express
            "amd renoir",
            "radeon hd 4200",
            "radeon hd 4250",
            "radeon hd 4290",
            "radeon hd 4270",
            "radeon hd 4225",
            "radeon hd 3100",
            "radeon hd 3200",
            "radeon hd 3000",
            "radeon hd 3300",
            "radeon(tm) r4 graphics",
            "radeon(tm) r5 graphics",
            "radeon(tm) r6 graphics",
            "radeon(tm) r7 graphics",
            "radeon r7 graphics",
            "nforce", // all nvidia nforce are integrated
            "tegra",  // all nvidia tegra are integrated
            "shield", // all nvidia shield are integrated
            "igp",
            "mali",
            "intel",
            "v3d",
            "apple m", // all apple m are integrated
        ];
        let strings_that_imply_cpu = ["mesa offscreen", "swiftshader", "llvmpipe"];

        //TODO: handle Intel Iris XE as discreet
        let inferred_device_type = if vendor.contains("qualcomm")
            || vendor.contains("intel")
            || strings_that_imply_integrated
                .iter()
                .any(|&s| renderer.contains(s))
        {
            wgt::DeviceType::IntegratedGpu
        } else if strings_that_imply_cpu.iter().any(|&s| renderer.contains(s)) {
            wgt::DeviceType::Cpu
        } else {
            // At this point the Device type is Unknown.
            // It's most likely DiscreteGpu, but we do not know for sure.
            // Use "Other" to avoid possibly making incorrect assumptions.
            // Note that if this same device is available under some other API (ex: Vulkan),
            // It will mostly likely get a different device type (probably DiscreteGpu).
            wgt::DeviceType::Other
        };

        // source: Sascha Willems at Vulkan
        let vendor_id = if vendor.contains("amd") {
            db::amd::VENDOR
        } else if vendor.contains("imgtec") {
            db::imgtec::VENDOR
        } else if vendor.contains("nvidia") {
            db::nvidia::VENDOR
        } else if vendor.contains("arm") {
            db::arm::VENDOR
        } else if vendor.contains("qualcomm") {
            db::qualcomm::VENDOR
        } else if vendor.contains("intel") {
            db::intel::VENDOR
        } else if vendor.contains("broadcom") {
            db::broadcom::VENDOR
        } else if vendor.contains("mesa") {
            db::mesa::VENDOR
        } else if vendor.contains("apple") {
            db::apple::VENDOR
        } else {
            0
        };

        wgt::AdapterInfo {
            name: renderer_orig,
            vendor: vendor_id,
            device: 0,
            device_type: inferred_device_type,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgt::Backend::Gl,
        }
    }

    pub(super) unsafe fn expose(
        context: super::AdapterContext,
    ) -> Option<crate::ExposedAdapter<super::Api>> {
        let gl = context.lock();
        let extensions = gl.supported_extensions();

        let (vendor_const, renderer_const) = if extensions.contains("WEBGL_debug_renderer_info") {
            // emscripten doesn't enable "WEBGL_debug_renderer_info" extension by default. so, we do it manually.
            // See https://github.com/gfx-rs/wgpu/issues/3245 for context
            #[cfg(target_os = "emscripten")]
            if unsafe { super::emscripten::enable_extension("WEBGL_debug_renderer_info\0") } {
                (GL_UNMASKED_VENDOR_WEBGL, GL_UNMASKED_RENDERER_WEBGL)
            } else {
                (glow::VENDOR, glow::RENDERER)
            }
            // glow already enables WEBGL_debug_renderer_info on wasm32-unknown-unknown target by default.
            #[cfg(not(target_os = "emscripten"))]
            (GL_UNMASKED_VENDOR_WEBGL, GL_UNMASKED_RENDERER_WEBGL)
        } else {
            (glow::VENDOR, glow::RENDERER)
        };

        let vendor = unsafe { gl.get_parameter_string(vendor_const) };
        let renderer = unsafe { gl.get_parameter_string(renderer_const) };
        let version = unsafe { gl.get_parameter_string(glow::VERSION) };
        log::debug!("Vendor: {}", vendor);
        log::debug!("Renderer: {}", renderer);
        log::debug!("Version: {}", version);

        let full_ver = Self::parse_full_version(&version).ok();
        let es_ver = full_ver.map_or_else(|| Self::parse_version(&version).ok(), |_| None);

        if let Some(full_ver) = full_ver {
            let core_profile = (full_ver >= (3, 2)).then_some(unsafe {
                gl.get_parameter_i32(glow::CONTEXT_PROFILE_MASK)
                    & glow::CONTEXT_CORE_PROFILE_BIT as i32
                    != 0
            });
            log::trace!(
                "Profile: {}",
                core_profile
                    .map(|core_profile| if core_profile {
                        "Core"
                    } else {
                        "Compatibility"
                    })
                    .unwrap_or("Legacy")
            );
        }

        if es_ver.is_none() && full_ver.is_none() {
            log::warn!("Unable to parse OpenGL version");
            return None;
        }

        if let Some(es_ver) = es_ver {
            if es_ver < (3, 0) {
                log::warn!(
                    "Returned GLES context is {}.{}, when 3.0+ was requested",
                    es_ver.0,
                    es_ver.1
                );
                return None;
            }
        }

        if let Some(full_ver) = full_ver {
            if full_ver < (3, 3) {
                log::warn!(
                    "Returned GL context is {}.{}, when 3.3+ is needed",
                    full_ver.0,
                    full_ver.1
                );
                return None;
            }
        }

        let shading_language_version = {
            let sl_version = unsafe { gl.get_parameter_string(glow::SHADING_LANGUAGE_VERSION) };
            log::debug!("SL version: {}", &sl_version);
            if full_ver.is_some() {
                let (sl_major, sl_minor) = Self::parse_full_version(&sl_version).ok()?;
                let mut value = sl_major as u16 * 100 + sl_minor as u16 * 10;
                // Naga doesn't think it supports GL 460+, so we cap it at 450
                if value > 450 {
                    value = 450;
                }
                naga::back::glsl::Version::Desktop(value)
            } else {
                let (sl_major, sl_minor) = Self::parse_version(&sl_version).ok()?;
                let value = sl_major as u16 * 100 + sl_minor as u16 * 10;
                naga::back::glsl::Version::Embedded {
                    version: value,
                    is_webgl: cfg!(target_arch = "wasm32"),
                }
            }
        };

        log::debug!("Supported GL Extensions: {:#?}", extensions);

        let supported = |(req_es_major, req_es_minor), (req_full_major, req_full_minor)| {
            let es_supported = es_ver
                .map(|es_ver| es_ver >= (req_es_major, req_es_minor))
                .unwrap_or_default();

            let full_supported = full_ver
                .map(|full_ver| full_ver >= (req_full_major, req_full_minor))
                .unwrap_or_default();

            es_supported || full_supported
        };

        let supports_storage =
            supported((3, 1), (4, 3)) || extensions.contains("GL_ARB_shader_storage_buffer_object");
        let supports_compute =
            supported((3, 1), (4, 3)) || extensions.contains("GL_ARB_compute_shader");
        let supports_work_group_params = supports_compute;

        // ANGLE provides renderer strings like: "ANGLE (Apple, Apple M1 Pro, OpenGL 4.1)"
        let is_angle = renderer.contains("ANGLE");

        let vertex_shader_storage_blocks = if supports_storage {
            let value =
                (unsafe { gl.get_parameter_i32(glow::MAX_VERTEX_SHADER_STORAGE_BLOCKS) } as u32);

            if value == 0 && extensions.contains("GL_ARB_shader_storage_buffer_object") {
                // The driver for AMD Radeon HD 5870 returns zero here, so assume the value matches the compute shader storage block count.
                // Windows doesn't recognize `GL_MAX_VERTEX_ATTRIB_STRIDE`.
                let new = (unsafe { gl.get_parameter_i32(glow::MAX_COMPUTE_SHADER_STORAGE_BLOCKS) }
                    as u32);
                log::warn!("Max vertex shader storage blocks is zero, but GL_ARB_shader_storage_buffer_object is specified. Assuming the compute value {new}");
                new
            } else {
                value
            }
        } else {
            0
        };
        let fragment_shader_storage_blocks = if supports_storage {
            (unsafe { gl.get_parameter_i32(glow::MAX_FRAGMENT_SHADER_STORAGE_BLOCKS) } as u32)
        } else {
            0
        };
        let vertex_shader_storage_textures = if supports_storage {
            (unsafe { gl.get_parameter_i32(glow::MAX_VERTEX_IMAGE_UNIFORMS) } as u32)
        } else {
            0
        };
        let fragment_shader_storage_textures = if supports_storage {
            (unsafe { gl.get_parameter_i32(glow::MAX_FRAGMENT_IMAGE_UNIFORMS) } as u32)
        } else {
            0
        };
        let max_storage_block_size = if supports_storage {
            (unsafe { gl.get_parameter_i32(glow::MAX_SHADER_STORAGE_BLOCK_SIZE) } as u32)
        } else {
            0
        };
        let max_element_index = unsafe { gl.get_parameter_i32(glow::MAX_ELEMENT_INDEX) } as u32;

        // WORKAROUND: In order to work around an issue with GL on RPI4 and similar, we ignore a
        // zero vertex ssbo count if there are vertex sstos. (more info:
        // https://github.com/gfx-rs/wgpu/pull/1607#issuecomment-874938961) The hardware does not
        // want us to write to these SSBOs, but GLES cannot express that. We detect this case and
        // disable writing to SSBOs.
        let vertex_ssbo_false_zero =
            vertex_shader_storage_blocks == 0 && vertex_shader_storage_textures != 0;
        if vertex_ssbo_false_zero {
            // We only care about fragment here as the 0 is a lie.
            log::warn!("Max vertex shader SSBO == 0 and SSTO != 0. Interpreting as false zero.");
        }

        let max_storage_buffers_per_shader_stage = if vertex_shader_storage_blocks == 0 {
            fragment_shader_storage_blocks
        } else {
            vertex_shader_storage_blocks.min(fragment_shader_storage_blocks)
        };
        let max_storage_textures_per_shader_stage = if vertex_shader_storage_textures == 0 {
            fragment_shader_storage_textures
        } else {
            vertex_shader_storage_textures.min(fragment_shader_storage_textures)
        };

        let mut downlevel_flags = wgt::DownlevelFlags::empty()
            | wgt::DownlevelFlags::NON_POWER_OF_TWO_MIPMAPPED_TEXTURES
            | wgt::DownlevelFlags::CUBE_ARRAY_TEXTURES
            | wgt::DownlevelFlags::COMPARISON_SAMPLERS
            | wgt::DownlevelFlags::VERTEX_AND_INSTANCE_INDEX_RESPECTS_RESPECTIVE_FIRST_VALUE_IN_INDIRECT_DRAW;
        downlevel_flags.set(wgt::DownlevelFlags::COMPUTE_SHADERS, supports_compute);
        downlevel_flags.set(
            wgt::DownlevelFlags::FRAGMENT_WRITABLE_STORAGE,
            max_storage_block_size != 0,
        );
        downlevel_flags.set(
            wgt::DownlevelFlags::INDIRECT_EXECUTION,
            supported((3, 1), (4, 3)) || extensions.contains("GL_ARB_multi_draw_indirect"),
        );
        downlevel_flags.set(wgt::DownlevelFlags::BASE_VERTEX, supported((3, 2), (3, 2)));
        downlevel_flags.set(
            wgt::DownlevelFlags::INDEPENDENT_BLEND,
            supported((3, 2), (4, 0)) || extensions.contains("GL_EXT_draw_buffers_indexed"),
        );
        downlevel_flags.set(
            wgt::DownlevelFlags::VERTEX_STORAGE,
            max_storage_block_size != 0
                && max_storage_buffers_per_shader_stage != 0
                && (vertex_shader_storage_blocks != 0 || vertex_ssbo_false_zero),
        );
        downlevel_flags.set(wgt::DownlevelFlags::FRAGMENT_STORAGE, supports_storage);
        if extensions.contains("EXT_texture_filter_anisotropic")
            || extensions.contains("GL_EXT_texture_filter_anisotropic")
        {
            let max_aniso =
                unsafe { gl.get_parameter_i32(glow::MAX_TEXTURE_MAX_ANISOTROPY_EXT) } as u32;
            downlevel_flags.set(wgt::DownlevelFlags::ANISOTROPIC_FILTERING, max_aniso >= 16);
        }
        downlevel_flags.set(
            wgt::DownlevelFlags::BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED,
            !(cfg!(target_arch = "wasm32") || is_angle),
        );
        // see https://registry.khronos.org/webgl/specs/latest/2.0/#BUFFER_OBJECT_BINDING
        downlevel_flags.set(
            wgt::DownlevelFlags::UNRESTRICTED_INDEX_BUFFER,
            !cfg!(target_arch = "wasm32"),
        );
        downlevel_flags.set(
            wgt::DownlevelFlags::UNRESTRICTED_EXTERNAL_TEXTURE_COPIES,
            !cfg!(target_arch = "wasm32"),
        );
        downlevel_flags.set(
            wgt::DownlevelFlags::FULL_DRAW_INDEX_UINT32,
            max_element_index == u32::MAX,
        );
        downlevel_flags.set(
            wgt::DownlevelFlags::MULTISAMPLED_SHADING,
            supported((3, 2), (4, 0)) || extensions.contains("OES_sample_variables"),
        );
        let query_buffers = extensions.contains("GL_ARB_query_buffer_object")
            || extensions.contains("GL_AMD_query_buffer_object");
        if query_buffers {
            downlevel_flags.set(wgt::DownlevelFlags::NONBLOCKING_QUERY_RESOLVE, true);
        }

        let mut features = wgt::Features::empty()
            | wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgt::Features::CLEAR_TEXTURE
            | wgt::Features::PUSH_CONSTANTS;
        features.set(
            wgt::Features::ADDRESS_MODE_CLAMP_TO_BORDER | wgt::Features::ADDRESS_MODE_CLAMP_TO_ZERO,
            extensions.contains("GL_EXT_texture_border_clamp")
                || extensions.contains("GL_ARB_texture_border_clamp"),
        );
        features.set(
            wgt::Features::DEPTH_CLIP_CONTROL,
            extensions.contains("GL_EXT_depth_clamp") || extensions.contains("GL_ARB_depth_clamp"),
        );
        features.set(
            wgt::Features::VERTEX_WRITABLE_STORAGE,
            downlevel_flags.contains(wgt::DownlevelFlags::VERTEX_STORAGE)
                && vertex_shader_storage_textures != 0,
        );
        features.set(
            wgt::Features::MULTIVIEW,
            extensions.contains("OVR_multiview2") || extensions.contains("GL_OVR_multiview2"),
        );
        features.set(
            wgt::Features::DUAL_SOURCE_BLENDING,
            extensions.contains("GL_EXT_blend_func_extended")
                || extensions.contains("GL_ARB_blend_func_extended"),
        );
        features.set(
            wgt::Features::SHADER_PRIMITIVE_INDEX,
            supported((3, 2), (3, 2))
                || extensions.contains("OES_geometry_shader")
                || extensions.contains("GL_ARB_geometry_shader4"),
        );
        features.set(
            wgt::Features::SHADER_EARLY_DEPTH_TEST,
            supported((3, 1), (4, 2)) || extensions.contains("GL_ARB_shader_image_load_store"),
        );
        features.set(wgt::Features::SHADER_UNUSED_VERTEX_OUTPUT, true);
        if extensions.contains("GL_ARB_timer_query") {
            features.set(wgt::Features::TIMESTAMP_QUERY, true);
            features.set(wgt::Features::TIMESTAMP_QUERY_INSIDE_PASSES, true);
        }
        let gl_bcn_exts = [
            "GL_EXT_texture_compression_s3tc",
            "GL_EXT_texture_compression_rgtc",
            "GL_ARB_texture_compression_bptc",
        ];
        let gles_bcn_exts = [
            "GL_EXT_texture_compression_s3tc_srgb",
            "GL_EXT_texture_compression_rgtc",
            "GL_EXT_texture_compression_bptc",
        ];
        let webgl_bcn_exts = [
            "WEBGL_compressed_texture_s3tc",
            "WEBGL_compressed_texture_s3tc_srgb",
            "EXT_texture_compression_rgtc",
            "EXT_texture_compression_bptc",
        ];
        let bcn_exts = if cfg!(target_arch = "wasm32") {
            &webgl_bcn_exts[..]
        } else if es_ver.is_some() {
            &gles_bcn_exts[..]
        } else {
            &gl_bcn_exts[..]
        };
        features.set(
            wgt::Features::TEXTURE_COMPRESSION_BC,
            bcn_exts.iter().all(|&ext| extensions.contains(ext)),
        );
        let has_etc = if cfg!(target_arch = "wasm32") {
            extensions.contains("WEBGL_compressed_texture_etc")
        } else {
            // This is a required part of GLES3, but not part of Desktop GL at all.
            es_ver.is_some()
        };
        features.set(wgt::Features::TEXTURE_COMPRESSION_ETC2, has_etc);

        // `OES_texture_compression_astc` provides 2D + 3D, LDR + HDR support
        if extensions.contains("WEBGL_compressed_texture_astc")
            || extensions.contains("GL_OES_texture_compression_astc")
        {
            #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
            {
                if context
                    .glow_context
                    .compressed_texture_astc_supports_ldr_profile()
                {
                    features.insert(wgt::Features::TEXTURE_COMPRESSION_ASTC);
                }
                if context
                    .glow_context
                    .compressed_texture_astc_supports_hdr_profile()
                {
                    features.insert(wgt::Features::TEXTURE_COMPRESSION_ASTC_HDR);
                }
            }

            #[cfg(any(not(target_arch = "wasm32"), target_os = "emscripten"))]
            {
                features.insert(wgt::Features::TEXTURE_COMPRESSION_ASTC);
                features.insert(wgt::Features::TEXTURE_COMPRESSION_ASTC_HDR);
            }
        } else {
            features.set(
                wgt::Features::TEXTURE_COMPRESSION_ASTC,
                extensions.contains("GL_KHR_texture_compression_astc_ldr"),
            );
            features.set(
                wgt::Features::TEXTURE_COMPRESSION_ASTC_HDR,
                extensions.contains("GL_KHR_texture_compression_astc_hdr"),
            );
        }

        features.set(
            wgt::Features::FLOAT32_FILTERABLE,
            extensions.contains("GL_ARB_color_buffer_float")
                || extensions.contains("GL_EXT_color_buffer_float")
                || extensions.contains("OES_texture_float_linear"),
        );

        if es_ver.is_none() {
            features |= wgt::Features::POLYGON_MODE_LINE | wgt::Features::POLYGON_MODE_POINT;
        }

        // We *might* be able to emulate bgra8unorm-storage but currently don't attempt to.

        let mut private_caps = super::PrivateCapabilities::empty();
        private_caps.set(
            super::PrivateCapabilities::BUFFER_ALLOCATION,
            extensions.contains("GL_EXT_buffer_storage")
                || extensions.contains("GL_ARB_buffer_storage"),
        );
        private_caps.set(
            super::PrivateCapabilities::SHADER_BINDING_LAYOUT,
            supports_compute,
        );
        private_caps.set(
            super::PrivateCapabilities::SHADER_TEXTURE_SHADOW_LOD,
            extensions.contains("GL_EXT_texture_shadow_lod"),
        );
        private_caps.set(
            super::PrivateCapabilities::MEMORY_BARRIERS,
            supported((3, 1), (4, 2)),
        );
        private_caps.set(
            super::PrivateCapabilities::VERTEX_BUFFER_LAYOUT,
            supported((3, 1), (4, 3)) || extensions.contains("GL_ARB_vertex_attrib_binding"),
        );
        private_caps.set(
            super::PrivateCapabilities::INDEX_BUFFER_ROLE_CHANGE,
            !cfg!(target_arch = "wasm32"),
        );
        private_caps.set(
            super::PrivateCapabilities::GET_BUFFER_SUB_DATA,
            cfg!(target_arch = "wasm32") || full_ver.is_some(),
        );
        let color_buffer_float = extensions.contains("GL_EXT_color_buffer_float")
            || extensions.contains("GL_ARB_color_buffer_float")
            || extensions.contains("EXT_color_buffer_float");
        let color_buffer_half_float = extensions.contains("GL_EXT_color_buffer_half_float")
            || extensions.contains("GL_ARB_half_float_pixel");
        private_caps.set(
            super::PrivateCapabilities::COLOR_BUFFER_HALF_FLOAT,
            color_buffer_half_float || color_buffer_float,
        );
        private_caps.set(
            super::PrivateCapabilities::COLOR_BUFFER_FLOAT,
            color_buffer_float,
        );
        private_caps.set(super::PrivateCapabilities::QUERY_BUFFERS, query_buffers);
        private_caps.set(super::PrivateCapabilities::QUERY_64BIT, full_ver.is_some());
        private_caps.set(
            super::PrivateCapabilities::TEXTURE_STORAGE,
            supported((3, 0), (4, 2)),
        );
        private_caps.set(super::PrivateCapabilities::DEBUG_FNS, gl.supports_debug());
        private_caps.set(
            super::PrivateCapabilities::INVALIDATE_FRAMEBUFFER,
            supported((3, 0), (4, 3)),
        );
        if let Some(full_ver) = full_ver {
            let supported =
                full_ver >= (4, 2) && extensions.contains("GL_ARB_shader_draw_parameters");
            private_caps.set(
                super::PrivateCapabilities::FULLY_FEATURED_INSTANCING,
                supported,
            );
            // Desktop 4.2 and greater specify the first instance parameter.
            //
            // For all other versions, the behavior is undefined.
            //
            // We only support indirect first instance when we also have ARB_shader_draw_parameters as
            // that's the only way to get gl_InstanceID to work correctly.
            features.set(wgt::Features::INDIRECT_FIRST_INSTANCE, supported);
        }

        let max_texture_size = unsafe { gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE) } as u32;
        let max_texture_3d_size = unsafe { gl.get_parameter_i32(glow::MAX_3D_TEXTURE_SIZE) } as u32;

        let min_uniform_buffer_offset_alignment =
            (unsafe { gl.get_parameter_i32(glow::UNIFORM_BUFFER_OFFSET_ALIGNMENT) } as u32);
        let min_storage_buffer_offset_alignment = if supports_storage {
            (unsafe { gl.get_parameter_i32(glow::SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT) } as u32)
        } else {
            256
        };
        let max_uniform_buffers_per_shader_stage =
            unsafe { gl.get_parameter_i32(glow::MAX_VERTEX_UNIFORM_BLOCKS) }
                .min(unsafe { gl.get_parameter_i32(glow::MAX_FRAGMENT_UNIFORM_BLOCKS) })
                as u32;

        let max_compute_workgroups_per_dimension = if supports_work_group_params {
            unsafe { gl.get_parameter_indexed_i32(glow::MAX_COMPUTE_WORK_GROUP_COUNT, 0) }
                .min(unsafe { gl.get_parameter_indexed_i32(glow::MAX_COMPUTE_WORK_GROUP_COUNT, 1) })
                .min(unsafe { gl.get_parameter_indexed_i32(glow::MAX_COMPUTE_WORK_GROUP_COUNT, 2) })
                as u32
        } else {
            0
        };

        let limits = wgt::Limits {
            max_texture_dimension_1d: max_texture_size,
            max_texture_dimension_2d: max_texture_size,
            max_texture_dimension_3d: max_texture_3d_size,
            max_texture_array_layers: unsafe {
                gl.get_parameter_i32(glow::MAX_ARRAY_TEXTURE_LAYERS)
            } as u32,
            max_bind_groups: crate::MAX_BIND_GROUPS as u32,
            max_bindings_per_bind_group: 65535,
            max_dynamic_uniform_buffers_per_pipeline_layout: max_uniform_buffers_per_shader_stage,
            max_dynamic_storage_buffers_per_pipeline_layout: max_storage_buffers_per_shader_stage,
            max_sampled_textures_per_shader_stage: super::MAX_TEXTURE_SLOTS as u32,
            max_samplers_per_shader_stage: super::MAX_SAMPLERS as u32,
            max_storage_buffers_per_shader_stage,
            max_storage_textures_per_shader_stage,
            max_uniform_buffers_per_shader_stage,
            max_uniform_buffer_binding_size: unsafe {
                gl.get_parameter_i32(glow::MAX_UNIFORM_BLOCK_SIZE)
            } as u32,
            max_storage_buffer_binding_size: if supports_storage {
                unsafe { gl.get_parameter_i32(glow::MAX_SHADER_STORAGE_BLOCK_SIZE) }
            } else {
                0
            } as u32,
            max_vertex_buffers: if private_caps
                .contains(super::PrivateCapabilities::VERTEX_BUFFER_LAYOUT)
            {
                (unsafe { gl.get_parameter_i32(glow::MAX_VERTEX_ATTRIB_BINDINGS) } as u32)
            } else {
                16 // should this be different?
            }
            .min(crate::MAX_VERTEX_BUFFERS as u32),
            max_vertex_attributes: (unsafe { gl.get_parameter_i32(glow::MAX_VERTEX_ATTRIBS) }
                as u32)
                .min(super::MAX_VERTEX_ATTRIBUTES as u32),
            max_vertex_buffer_array_stride: if private_caps
                .contains(super::PrivateCapabilities::VERTEX_BUFFER_LAYOUT)
            {
                if let Some(full_ver) = full_ver {
                    if full_ver >= (4, 4) {
                        // We can query `GL_MAX_VERTEX_ATTRIB_STRIDE` in OpenGL 4.4+
                        let value =
                            (unsafe { gl.get_parameter_i32(glow::MAX_VERTEX_ATTRIB_STRIDE) })
                                as u32;

                        if value == 0 {
                            // This should be at least 2048, but the driver for AMD Radeon HD 5870 on
                            // Windows doesn't recognize `GL_MAX_VERTEX_ATTRIB_STRIDE`.

                            log::warn!("Max vertex attribute stride is 0. Assuming it is 2048");
                            2048
                        } else {
                            value
                        }
                    } else {
                        log::warn!("Max vertex attribute stride unknown. Assuming it is 2048");
                        2048
                    }
                } else {
                    (unsafe { gl.get_parameter_i32(glow::MAX_VERTEX_ATTRIB_STRIDE) }) as u32
                }
            } else {
                !0
            },
            max_push_constant_size: super::MAX_PUSH_CONSTANTS as u32 * 4,
            min_uniform_buffer_offset_alignment,
            min_storage_buffer_offset_alignment,
            max_inter_stage_shader_components: unsafe {
                gl.get_parameter_i32(glow::MAX_VARYING_COMPONENTS)
            } as u32,
            max_compute_workgroup_storage_size: if supports_work_group_params {
                (unsafe { gl.get_parameter_i32(glow::MAX_COMPUTE_SHARED_MEMORY_SIZE) } as u32)
            } else {
                0
            },
            max_compute_invocations_per_workgroup: if supports_work_group_params {
                (unsafe { gl.get_parameter_i32(glow::MAX_COMPUTE_WORK_GROUP_INVOCATIONS) } as u32)
            } else {
                0
            },
            max_compute_workgroup_size_x: if supports_work_group_params {
                (unsafe { gl.get_parameter_indexed_i32(glow::MAX_COMPUTE_WORK_GROUP_SIZE, 0) }
                    as u32)
            } else {
                0
            },
            max_compute_workgroup_size_y: if supports_work_group_params {
                (unsafe { gl.get_parameter_indexed_i32(glow::MAX_COMPUTE_WORK_GROUP_SIZE, 1) }
                    as u32)
            } else {
                0
            },
            max_compute_workgroup_size_z: if supports_work_group_params {
                (unsafe { gl.get_parameter_indexed_i32(glow::MAX_COMPUTE_WORK_GROUP_SIZE, 2) }
                    as u32)
            } else {
                0
            },
            max_compute_workgroups_per_dimension,
            max_buffer_size: i32::MAX as u64,
            max_non_sampler_bindings: std::u32::MAX,
        };

        let mut workarounds = super::Workarounds::empty();

        workarounds.set(
            super::Workarounds::EMULATE_BUFFER_MAP,
            cfg!(target_arch = "wasm32"),
        );

        let r = renderer.to_lowercase();
        // Check for Mesa sRGB clear bug. See
        // [`super::PrivateCapabilities::MESA_I915_SRGB_SHADER_CLEAR`].
        if context.is_owned()
            && r.contains("mesa")
            && r.contains("intel")
            && r.split(&[' ', '(', ')'][..])
                .any(|substr| substr.len() == 3 && substr.chars().nth(2) == Some('l'))
        {
            log::warn!(
                "Detected skylake derivative running on mesa i915. Clears to srgb textures will \
                use manual shader clears."
            );
            workarounds.set(super::Workarounds::MESA_I915_SRGB_SHADER_CLEAR, true);
        }

        let downlevel_defaults = wgt::DownlevelLimits {};

        // Drop the GL guard so we can move the context into AdapterShared
        // ( on Wasm the gl handle is just a ref so we tell clippy to allow
        // dropping the ref )
        #[cfg_attr(target_arch = "wasm32", allow(dropping_references))]
        drop(gl);

        Some(crate::ExposedAdapter {
            adapter: super::Adapter {
                shared: Arc::new(super::AdapterShared {
                    context,
                    private_caps,
                    workarounds,
                    features,
                    shading_language_version,
                    next_shader_id: Default::default(),
                    program_cache: Default::default(),
                    es: es_ver.is_some(),
                }),
            },
            info: Self::make_info(vendor, renderer),
            features,
            capabilities: crate::Capabilities {
                limits,
                downlevel: wgt::DownlevelCapabilities {
                    flags: downlevel_flags,
                    limits: downlevel_defaults,
                    shader_model: wgt::ShaderModel::Sm5,
                },
                alignments: crate::Alignments {
                    buffer_copy_offset: wgt::BufferSize::new(4).unwrap(),
                    buffer_copy_pitch: wgt::BufferSize::new(4).unwrap(),
                },
            },
        })
    }

    unsafe fn compile_shader(
        source: &str,
        gl: &glow::Context,
        shader_type: u32,
        es: bool,
    ) -> Option<glow::Shader> {
        let source = if es {
            format!("#version 300 es\nprecision lowp float;\n{source}")
        } else {
            format!("#version 130\n{source}")
        };
        let shader = unsafe { gl.create_shader(shader_type) }.expect("Could not create shader");
        unsafe { gl.shader_source(shader, &source) };
        unsafe { gl.compile_shader(shader) };

        if !unsafe { gl.get_shader_compile_status(shader) } {
            let msg = unsafe { gl.get_shader_info_log(shader) };
            if !msg.is_empty() {
                log::error!("\tShader compile error: {}", msg);
            }
            unsafe { gl.delete_shader(shader) };
            None
        } else {
            Some(shader)
        }
    }

    unsafe fn create_shader_clear_program(
        gl: &glow::Context,
        es: bool,
    ) -> Option<(glow::Program, glow::UniformLocation)> {
        let program = unsafe { gl.create_program() }.expect("Could not create shader program");
        let vertex = unsafe {
            Self::compile_shader(
                include_str!("./shaders/clear.vert"),
                gl,
                glow::VERTEX_SHADER,
                es,
            )?
        };
        let fragment = unsafe {
            Self::compile_shader(
                include_str!("./shaders/clear.frag"),
                gl,
                glow::FRAGMENT_SHADER,
                es,
            )?
        };
        unsafe { gl.attach_shader(program, vertex) };
        unsafe { gl.attach_shader(program, fragment) };
        unsafe { gl.link_program(program) };

        let linked_ok = unsafe { gl.get_program_link_status(program) };
        let msg = unsafe { gl.get_program_info_log(program) };
        if !msg.is_empty() {
            log::warn!("Shader link error: {}", msg);
        }
        if !linked_ok {
            return None;
        }

        let color_uniform_location = unsafe { gl.get_uniform_location(program, "color") }
            .expect("Could not find color uniform in shader clear shader");
        unsafe { gl.delete_shader(vertex) };
        unsafe { gl.delete_shader(fragment) };

        Some((program, color_uniform_location))
    }
}

impl crate::Adapter<super::Api> for super::Adapter {
    unsafe fn open(
        &self,
        features: wgt::Features,
        _limits: &wgt::Limits,
    ) -> Result<crate::OpenDevice<super::Api>, crate::DeviceError> {
        let gl = &self.shared.context.lock();
        unsafe { gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1) };
        unsafe { gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1) };
        let main_vao =
            unsafe { gl.create_vertex_array() }.map_err(|_| crate::DeviceError::OutOfMemory)?;
        unsafe { gl.bind_vertex_array(Some(main_vao)) };

        let zero_buffer =
            unsafe { gl.create_buffer() }.map_err(|_| crate::DeviceError::OutOfMemory)?;
        unsafe { gl.bind_buffer(glow::COPY_READ_BUFFER, Some(zero_buffer)) };
        let zeroes = vec![0u8; super::ZERO_BUFFER_SIZE];
        unsafe { gl.buffer_data_u8_slice(glow::COPY_READ_BUFFER, &zeroes, glow::STATIC_DRAW) };

        // Compile the shader program we use for doing manual clears to work around Mesa fastclear
        // bug.

        let (shader_clear_program, shader_clear_program_color_uniform_location) = unsafe {
            Self::create_shader_clear_program(gl, self.shared.es)
                .ok_or(crate::DeviceError::ResourceCreationFailed)?
        };

        Ok(crate::OpenDevice {
            device: super::Device {
                shared: Arc::clone(&self.shared),
                main_vao,
                #[cfg(all(not(target_arch = "wasm32"), feature = "renderdoc"))]
                render_doc: Default::default(),
            },
            queue: super::Queue {
                shared: Arc::clone(&self.shared),
                features,
                draw_fbo: unsafe { gl.create_framebuffer() }
                    .map_err(|_| crate::DeviceError::OutOfMemory)?,
                copy_fbo: unsafe { gl.create_framebuffer() }
                    .map_err(|_| crate::DeviceError::OutOfMemory)?,
                shader_clear_program,
                shader_clear_program_color_uniform_location,
                zero_buffer,
                temp_query_results: Mutex::new(Vec::new()),
                draw_buffer_count: AtomicU8::new(1),
                current_index_buffer: Mutex::new(None),
            },
        })
    }

    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> crate::TextureFormatCapabilities {
        use crate::TextureFormatCapabilities as Tfc;
        use wgt::TextureFormat as Tf;

        let sample_count = {
            let max_samples = unsafe {
                self.shared
                    .context
                    .lock()
                    .get_parameter_i32(glow::MAX_SAMPLES)
            };
            if max_samples >= 16 {
                Tfc::MULTISAMPLE_X2
                    | Tfc::MULTISAMPLE_X4
                    | Tfc::MULTISAMPLE_X8
                    | Tfc::MULTISAMPLE_X16
            } else if max_samples >= 8 {
                Tfc::MULTISAMPLE_X2 | Tfc::MULTISAMPLE_X4 | Tfc::MULTISAMPLE_X8
            } else {
                // The lowest supported level in GLE3.0/WebGL2 is 4X
                // (see GL_MAX_SAMPLES in https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glGet.xhtml).
                // On some platforms, like iOS Safari, `get_parameter_i32(MAX_SAMPLES)` returns 0,
                // so we always fall back to supporting 4x here.
                Tfc::MULTISAMPLE_X2 | Tfc::MULTISAMPLE_X4
            }
        };

        // Base types are pulled from the table in the OpenGLES 3.0 spec in section 3.8.
        //
        // The storage types are based on table 8.26, in section
        // "TEXTURE IMAGE LOADS AND STORES" of OpenGLES-3.2 spec.
        let empty = Tfc::empty();
        let base = Tfc::COPY_SRC | Tfc::COPY_DST;
        let unfilterable = base | Tfc::SAMPLED;
        let depth = base | Tfc::SAMPLED | sample_count | Tfc::DEPTH_STENCIL_ATTACHMENT;
        let filterable = unfilterable | Tfc::SAMPLED_LINEAR;
        let renderable =
            unfilterable | Tfc::COLOR_ATTACHMENT | sample_count | Tfc::MULTISAMPLE_RESOLVE;
        let filterable_renderable = filterable | renderable | Tfc::COLOR_ATTACHMENT_BLEND;
        let storage = base | Tfc::STORAGE | Tfc::STORAGE_READ_WRITE;

        let feature_fn = |f, caps| {
            if self.shared.features.contains(f) {
                caps
            } else {
                empty
            }
        };

        let bcn_features = feature_fn(wgt::Features::TEXTURE_COMPRESSION_BC, filterable);
        let etc2_features = feature_fn(wgt::Features::TEXTURE_COMPRESSION_ETC2, filterable);
        let astc_features = feature_fn(wgt::Features::TEXTURE_COMPRESSION_ASTC, filterable);
        let astc_hdr_features = feature_fn(wgt::Features::TEXTURE_COMPRESSION_ASTC_HDR, filterable);

        let private_caps_fn = |f, caps| {
            if self.shared.private_caps.contains(f) {
                caps
            } else {
                empty
            }
        };

        let half_float_renderable = private_caps_fn(
            super::PrivateCapabilities::COLOR_BUFFER_HALF_FLOAT,
            Tfc::COLOR_ATTACHMENT
                | Tfc::COLOR_ATTACHMENT_BLEND
                | sample_count
                | Tfc::MULTISAMPLE_RESOLVE,
        );

        let float_renderable = private_caps_fn(
            super::PrivateCapabilities::COLOR_BUFFER_FLOAT,
            Tfc::COLOR_ATTACHMENT
                | Tfc::COLOR_ATTACHMENT_BLEND
                | sample_count
                | Tfc::MULTISAMPLE_RESOLVE,
        );

        let texture_float_linear = feature_fn(wgt::Features::FLOAT32_FILTERABLE, filterable);

        match format {
            Tf::R8Unorm => filterable_renderable,
            Tf::R8Snorm => filterable,
            Tf::R8Uint => renderable,
            Tf::R8Sint => renderable,
            Tf::R16Uint => renderable,
            Tf::R16Sint => renderable,
            Tf::R16Unorm => empty,
            Tf::R16Snorm => empty,
            Tf::R16Float => filterable | half_float_renderable,
            Tf::Rg8Unorm => filterable_renderable,
            Tf::Rg8Snorm => filterable,
            Tf::Rg8Uint => renderable,
            Tf::Rg8Sint => renderable,
            Tf::R32Uint => renderable | storage,
            Tf::R32Sint => renderable | storage,
            Tf::R32Float => unfilterable | storage | float_renderable | texture_float_linear,
            Tf::Rg16Uint => renderable,
            Tf::Rg16Sint => renderable,
            Tf::Rg16Unorm => empty,
            Tf::Rg16Snorm => empty,
            Tf::Rg16Float => filterable | half_float_renderable,
            Tf::Rgba8Unorm => filterable_renderable | storage,
            Tf::Rgba8UnormSrgb => filterable_renderable,
            Tf::Bgra8Unorm | Tf::Bgra8UnormSrgb => filterable_renderable,
            Tf::Rgba8Snorm => filterable | storage,
            Tf::Rgba8Uint => renderable | storage,
            Tf::Rgba8Sint => renderable | storage,
            Tf::Rgb10a2Uint => renderable,
            Tf::Rgb10a2Unorm => filterable_renderable,
            Tf::Rg11b10Float => filterable | float_renderable,
            Tf::Rg32Uint => renderable,
            Tf::Rg32Sint => renderable,
            Tf::Rg32Float => unfilterable | float_renderable | texture_float_linear,
            Tf::Rgba16Uint => renderable | storage,
            Tf::Rgba16Sint => renderable | storage,
            Tf::Rgba16Unorm => empty,
            Tf::Rgba16Snorm => empty,
            Tf::Rgba16Float => filterable | storage | half_float_renderable,
            Tf::Rgba32Uint => renderable | storage,
            Tf::Rgba32Sint => renderable | storage,
            Tf::Rgba32Float => unfilterable | storage | float_renderable | texture_float_linear,
            Tf::Stencil8
            | Tf::Depth16Unorm
            | Tf::Depth32Float
            | Tf::Depth32FloatStencil8
            | Tf::Depth24Plus
            | Tf::Depth24PlusStencil8 => depth,
            Tf::NV12 => empty,
            Tf::Rgb9e5Ufloat => filterable,
            Tf::Bc1RgbaUnorm
            | Tf::Bc1RgbaUnormSrgb
            | Tf::Bc2RgbaUnorm
            | Tf::Bc2RgbaUnormSrgb
            | Tf::Bc3RgbaUnorm
            | Tf::Bc3RgbaUnormSrgb
            | Tf::Bc4RUnorm
            | Tf::Bc4RSnorm
            | Tf::Bc5RgUnorm
            | Tf::Bc5RgSnorm
            | Tf::Bc6hRgbFloat
            | Tf::Bc6hRgbUfloat
            | Tf::Bc7RgbaUnorm
            | Tf::Bc7RgbaUnormSrgb => bcn_features,
            Tf::Etc2Rgb8Unorm
            | Tf::Etc2Rgb8UnormSrgb
            | Tf::Etc2Rgb8A1Unorm
            | Tf::Etc2Rgb8A1UnormSrgb
            | Tf::Etc2Rgba8Unorm
            | Tf::Etc2Rgba8UnormSrgb
            | Tf::EacR11Unorm
            | Tf::EacR11Snorm
            | Tf::EacRg11Unorm
            | Tf::EacRg11Snorm => etc2_features,
            Tf::Astc {
                block: _,
                channel: AstcChannel::Unorm | AstcChannel::UnormSrgb,
            } => astc_features,
            Tf::Astc {
                block: _,
                channel: AstcChannel::Hdr,
            } => astc_hdr_features,
        }
    }

    unsafe fn surface_capabilities(
        &self,
        surface: &super::Surface,
    ) -> Option<crate::SurfaceCapabilities> {
        if surface.presentable {
            let mut formats = vec![
                wgt::TextureFormat::Rgba8Unorm,
                #[cfg(not(target_arch = "wasm32"))]
                wgt::TextureFormat::Bgra8Unorm,
            ];
            if surface.supports_srgb() {
                formats.extend([
                    wgt::TextureFormat::Rgba8UnormSrgb,
                    #[cfg(not(target_arch = "wasm32"))]
                    wgt::TextureFormat::Bgra8UnormSrgb,
                ])
            }
            if self
                .shared
                .private_caps
                .contains(super::PrivateCapabilities::COLOR_BUFFER_HALF_FLOAT)
            {
                formats.push(wgt::TextureFormat::Rgba16Float)
            }

            Some(crate::SurfaceCapabilities {
                formats,
                present_modes: if cfg!(windows) {
                    vec![wgt::PresentMode::Fifo, wgt::PresentMode::Immediate]
                } else {
                    vec![wgt::PresentMode::Fifo] //TODO
                },
                composite_alpha_modes: vec![wgt::CompositeAlphaMode::Opaque], //TODO
                swap_chain_sizes: 2..=2,
                current_extent: None,
                usage: crate::TextureUses::COLOR_TARGET,
            })
        } else {
            None
        }
    }

    unsafe fn get_presentation_timestamp(&self) -> wgt::PresentationTimestamp {
        wgt::PresentationTimestamp::INVALID_TIMESTAMP
    }
}

impl super::AdapterShared {
    pub(super) unsafe fn get_buffer_sub_data(
        &self,
        gl: &glow::Context,
        target: u32,
        offset: i32,
        dst_data: &mut [u8],
    ) {
        if self
            .private_caps
            .contains(super::PrivateCapabilities::GET_BUFFER_SUB_DATA)
        {
            unsafe { gl.get_buffer_sub_data(target, offset, dst_data) };
        } else {
            log::error!("Fake map");
            let length = dst_data.len();
            let buffer_mapping =
                unsafe { gl.map_buffer_range(target, offset, length as _, glow::MAP_READ_BIT) };

            unsafe { std::ptr::copy_nonoverlapping(buffer_mapping, dst_data.as_mut_ptr(), length) };

            unsafe { gl.unmap_buffer(target) };
        }
    }
}

#[cfg(all(
    target_arch = "wasm32",
    feature = "fragile-send-sync-non-atomic-wasm",
    not(target_feature = "atomics")
))]
unsafe impl Sync for super::Adapter {}
#[cfg(all(
    target_arch = "wasm32",
    feature = "fragile-send-sync-non-atomic-wasm",
    not(target_feature = "atomics")
))]
unsafe impl Send for super::Adapter {}

#[cfg(test)]
mod tests {
    use super::super::Adapter;

    #[test]
    fn test_version_parse() {
        Adapter::parse_version("1").unwrap_err();
        Adapter::parse_version("1.").unwrap_err();
        Adapter::parse_version("1 h3l1o. W0rld").unwrap_err();
        Adapter::parse_version("1. h3l1o. W0rld").unwrap_err();
        Adapter::parse_version("1.2.3").unwrap_err();

        assert_eq!(Adapter::parse_version("OpenGL ES 3.1").unwrap(), (3, 1));
        assert_eq!(
            Adapter::parse_version("OpenGL ES 2.0 Google Nexus").unwrap(),
            (2, 0)
        );
        assert_eq!(Adapter::parse_version("GLSL ES 1.1").unwrap(), (1, 1));
        assert_eq!(
            Adapter::parse_version("OpenGL ES GLSL ES 3.20").unwrap(),
            (3, 2)
        );
        assert_eq!(
            // WebGL 2.0 should parse as OpenGL ES 3.0
            Adapter::parse_version("WebGL 2.0 (OpenGL ES 3.0 Chromium)").unwrap(),
            (3, 0)
        );
        assert_eq!(
            Adapter::parse_version("WebGL GLSL ES 3.00 (OpenGL ES GLSL ES 3.0 Chromium)").unwrap(),
            (3, 0)
        );
    }
}
