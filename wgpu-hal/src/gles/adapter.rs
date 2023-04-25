use glow::HasContext;
use std::sync::Arc;
use wgt::AstcChannel;

use crate::auxil::db;

// https://webgl2fundamentals.org/webgl/lessons/webgl-data-textures.html

const GL_UNMASKED_VENDOR_WEBGL: u32 = 0x9245;
const GL_UNMASKED_RENDERER_WEBGL: u32 = 0x9246;

impl super::Adapter {
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
                    log::warn!("ES not found in '{}'", src);
                    return Err(crate::InstanceError);
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
            (Some(major), Some(minor)) => Ok((
                // Return WebGL 2.0 version as OpenGL ES 3.0
                if is_webgl && !is_glsl {
                    major + 1
                } else {
                    major
                },
                minor,
            )),
            _ => {
                log::warn!("Unable to extract the version from '{}'", version);
                Err(crate::InstanceError)
            }
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
            vendor: vendor_id as usize,
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

        let (vendor, renderer) = {
            let vendor = unsafe { gl.get_parameter_string(vendor_const) };
            let renderer = unsafe { gl.get_parameter_string(renderer_const) };

            (vendor, renderer)
        };
        let version = unsafe { gl.get_parameter_string(glow::VERSION) };
        log::info!("Vendor: {}", vendor);
        log::info!("Renderer: {}", renderer);
        log::info!("Version: {}", version);

        log::debug!("Extensions: {:#?}", extensions);

        let ver = Self::parse_version(&version).ok()?;
        if ver < (3, 0) {
            log::warn!(
                "Returned GLES context is {}.{}, when 3.0+ was requested",
                ver.0,
                ver.1
            );
            return None;
        }

        let supports_storage = ver >= (3, 1);
        let supports_work_group_params = ver >= (3, 1);

        let shading_language_version = {
            let sl_version = unsafe { gl.get_parameter_string(glow::SHADING_LANGUAGE_VERSION) };
            log::info!("SL version: {}", &sl_version);
            let (sl_major, sl_minor) = Self::parse_version(&sl_version).ok()?;
            let value = sl_major as u16 * 100 + sl_minor as u16 * 10;
            naga::back::glsl::Version::Embedded {
                version: value,
                is_webgl: cfg!(target_arch = "wasm32"),
            }
        };

        // ANGLE provides renderer strings like: "ANGLE (Apple, Apple M1 Pro, OpenGL 4.1)"
        let is_angle = renderer.contains("ANGLE");

        let vertex_shader_storage_blocks = if supports_storage {
            (unsafe { gl.get_parameter_i32(glow::MAX_VERTEX_SHADER_STORAGE_BLOCKS) } as u32)
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
            | wgt::DownlevelFlags::COMPARISON_SAMPLERS;
        downlevel_flags.set(wgt::DownlevelFlags::COMPUTE_SHADERS, ver >= (3, 1));
        downlevel_flags.set(
            wgt::DownlevelFlags::FRAGMENT_WRITABLE_STORAGE,
            max_storage_block_size != 0,
        );
        downlevel_flags.set(wgt::DownlevelFlags::INDIRECT_EXECUTION, ver >= (3, 1));
        //TODO: we can actually support positive `base_vertex` in the same way
        // as we emulate the `start_instance`. But we can't deal with negatives...
        downlevel_flags.set(wgt::DownlevelFlags::BASE_VERTEX, ver >= (3, 2));
        downlevel_flags.set(
            wgt::DownlevelFlags::INDEPENDENT_BLEND,
            ver >= (3, 2) || extensions.contains("GL_EXT_draw_buffers_indexed"),
        );
        downlevel_flags.set(
            wgt::DownlevelFlags::VERTEX_STORAGE,
            max_storage_block_size != 0
                && max_storage_buffers_per_shader_stage != 0
                && (vertex_shader_storage_blocks != 0 || vertex_ssbo_false_zero),
        );
        downlevel_flags.set(wgt::DownlevelFlags::FRAGMENT_STORAGE, supports_storage);
        if extensions.contains("EXT_texture_filter_anisotropic") {
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
            ver >= (3, 2) || extensions.contains("OES_sample_variables"),
        );

        let mut features = wgt::Features::empty()
            | wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgt::Features::CLEAR_TEXTURE
            | wgt::Features::PUSH_CONSTANTS;
        features.set(
            wgt::Features::ADDRESS_MODE_CLAMP_TO_BORDER | wgt::Features::ADDRESS_MODE_CLAMP_TO_ZERO,
            extensions.contains("GL_EXT_texture_border_clamp"),
        );
        features.set(
            wgt::Features::DEPTH_CLIP_CONTROL,
            extensions.contains("GL_EXT_depth_clamp"),
        );
        features.set(
            wgt::Features::VERTEX_WRITABLE_STORAGE,
            downlevel_flags.contains(wgt::DownlevelFlags::VERTEX_STORAGE)
                && vertex_shader_storage_textures != 0,
        );
        features.set(
            wgt::Features::MULTIVIEW,
            extensions.contains("OVR_multiview2"),
        );
        features.set(
            wgt::Features::SHADER_PRIMITIVE_INDEX,
            ver >= (3, 2) || extensions.contains("OES_geometry_shader"),
        );
        features.set(wgt::Features::SHADER_EARLY_DEPTH_TEST, ver >= (3, 1));
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
        } else {
            &gles_bcn_exts[..]
        };
        features.set(
            wgt::Features::TEXTURE_COMPRESSION_BC,
            bcn_exts.iter().all(|&ext| extensions.contains(ext)),
        );
        features.set(
            wgt::Features::TEXTURE_COMPRESSION_ETC2,
            // This is a part of GLES-3 but not WebGL2 core
            !cfg!(target_arch = "wasm32") || extensions.contains("WEBGL_compressed_texture_etc"),
        );
        // `OES_texture_compression_astc` provides 2D + 3D, LDR + HDR support
        if extensions.contains("WEBGL_compressed_texture_astc")
            || extensions.contains("GL_OES_texture_compression_astc")
        {
            features.insert(wgt::Features::TEXTURE_COMPRESSION_ASTC);
            features.insert(wgt::Features::TEXTURE_COMPRESSION_ASTC_HDR);
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

        let mut private_caps = super::PrivateCapabilities::empty();
        private_caps.set(
            super::PrivateCapabilities::BUFFER_ALLOCATION,
            extensions.contains("GL_EXT_buffer_storage"),
        );
        private_caps.set(
            super::PrivateCapabilities::SHADER_BINDING_LAYOUT,
            ver >= (3, 1),
        );
        private_caps.set(
            super::PrivateCapabilities::SHADER_TEXTURE_SHADOW_LOD,
            extensions.contains("GL_EXT_texture_shadow_lod"),
        );
        private_caps.set(super::PrivateCapabilities::MEMORY_BARRIERS, ver >= (3, 1));
        private_caps.set(
            super::PrivateCapabilities::VERTEX_BUFFER_LAYOUT,
            ver >= (3, 1),
        );
        private_caps.set(
            super::PrivateCapabilities::INDEX_BUFFER_ROLE_CHANGE,
            !cfg!(target_arch = "wasm32"),
        );
        private_caps.set(
            super::PrivateCapabilities::CAN_DISABLE_DRAW_BUFFER,
            !cfg!(target_arch = "wasm32"),
        );
        private_caps.set(
            super::PrivateCapabilities::GET_BUFFER_SUB_DATA,
            cfg!(target_arch = "wasm32"),
        );
        let color_buffer_float = extensions.contains("GL_EXT_color_buffer_float")
            || extensions.contains("EXT_color_buffer_float");
        let color_buffer_half_float = extensions.contains("GL_EXT_color_buffer_half_float");
        private_caps.set(
            super::PrivateCapabilities::COLOR_BUFFER_HALF_FLOAT,
            color_buffer_half_float || color_buffer_float,
        );
        private_caps.set(
            super::PrivateCapabilities::COLOR_BUFFER_FLOAT,
            color_buffer_float,
        );
        private_caps.set(
            super::PrivateCapabilities::TEXTURE_FLOAT_LINEAR,
            extensions.contains("OES_texture_float_linear"),
        );

        let max_texture_size = unsafe { gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE) } as u32;
        let max_texture_3d_size = unsafe { gl.get_parameter_i32(glow::MAX_3D_TEXTURE_SIZE) } as u32;

        let min_uniform_buffer_offset_alignment =
            (unsafe { gl.get_parameter_i32(glow::UNIFORM_BUFFER_OFFSET_ALIGNMENT) } as u32);
        let min_storage_buffer_offset_alignment = if ver >= (3, 1) {
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
            max_storage_buffer_binding_size: if ver >= (3, 1) {
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
            },
            max_vertex_attributes: (unsafe { gl.get_parameter_i32(glow::MAX_VERTEX_ATTRIBS) }
                as u32)
                .min(super::MAX_VERTEX_ATTRIBUTES as u32),
            max_vertex_buffer_array_stride: if private_caps
                .contains(super::PrivateCapabilities::VERTEX_BUFFER_LAYOUT)
            {
                (unsafe { gl.get_parameter_i32(glow::MAX_VERTEX_ATTRIB_STRIDE) } as u32)
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
        // ( on WASM the gl handle is just a ref so we tell clippy to allow
        // dropping the ref )
        #[allow(clippy::drop_ref)]
        drop(gl);

        Some(crate::ExposedAdapter {
            adapter: super::Adapter {
                shared: Arc::new(super::AdapterShared {
                    context,
                    private_caps,
                    workarounds,
                    features,
                    shading_language_version,
                    max_texture_size,
                    next_shader_id: Default::default(),
                    program_cache: Default::default(),
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

    unsafe fn create_shader_clear_program(
        gl: &glow::Context,
    ) -> (glow::Program, glow::UniformLocation) {
        let program = unsafe { gl.create_program() }.expect("Could not create shader program");
        let vertex =
            unsafe { gl.create_shader(glow::VERTEX_SHADER) }.expect("Could not create shader");
        unsafe { gl.shader_source(vertex, include_str!("./shaders/clear.vert")) };
        unsafe { gl.compile_shader(vertex) };
        let fragment =
            unsafe { gl.create_shader(glow::FRAGMENT_SHADER) }.expect("Could not create shader");
        unsafe { gl.shader_source(fragment, include_str!("./shaders/clear.frag")) };
        unsafe { gl.compile_shader(fragment) };
        unsafe { gl.attach_shader(program, vertex) };
        unsafe { gl.attach_shader(program, fragment) };
        unsafe { gl.link_program(program) };
        let color_uniform_location = unsafe { gl.get_uniform_location(program, "color") }
            .expect("Could not find color uniform in shader clear shader");
        unsafe { gl.delete_shader(vertex) };
        unsafe { gl.delete_shader(fragment) };

        (program, color_uniform_location)
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
        let (shader_clear_program, shader_clear_program_color_uniform_location) =
            unsafe { Self::create_shader_clear_program(gl) };

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
                temp_query_results: Vec::new(),
                draw_buffer_count: 1,
                current_index_buffer: None,
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
            } else if max_samples >= 4 {
                Tfc::MULTISAMPLE_X2 | Tfc::MULTISAMPLE_X4
            } else {
                Tfc::MULTISAMPLE_X2
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

        let texture_float_linear =
            private_caps_fn(super::PrivateCapabilities::TEXTURE_FLOAT_LINEAR, filterable);

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
            Tf::Rgba8Unorm | Tf::Rgba8UnormSrgb => filterable_renderable | storage,
            Tf::Bgra8Unorm | Tf::Bgra8UnormSrgb => filterable_renderable,
            Tf::Rgba8Snorm => filterable,
            Tf::Rgba8Uint => renderable | storage,
            Tf::Rgba8Sint => renderable | storage,
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
                present_modes: vec![wgt::PresentMode::Fifo], //TODO
                composite_alpha_modes: vec![wgt::CompositeAlphaMode::Opaque], //TODO
                swap_chain_sizes: 2..=2,
                current_extent: None,
                extents: wgt::Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                }..=wgt::Extent3d {
                    width: self.shared.max_texture_size,
                    height: self.shared.max_texture_size,
                    depth_or_array_layers: 1,
                },
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

// SAFE: WASM doesn't have threads
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for super::Adapter {}
#[cfg(target_arch = "wasm32")]
unsafe impl Send for super::Adapter {}

#[cfg(test)]
mod tests {
    use super::super::Adapter;

    #[test]
    fn test_version_parse() {
        let error = Err(crate::InstanceError);
        assert_eq!(Adapter::parse_version("1"), error);
        assert_eq!(Adapter::parse_version("1."), error);
        assert_eq!(Adapter::parse_version("1 h3l1o. W0rld"), error);
        assert_eq!(Adapter::parse_version("1. h3l1o. W0rld"), error);
        assert_eq!(Adapter::parse_version("1.2.3"), error);
        assert_eq!(Adapter::parse_version("OpenGL ES 3.1"), Ok((3, 1)));
        assert_eq!(
            Adapter::parse_version("OpenGL ES 2.0 Google Nexus"),
            Ok((2, 0))
        );
        assert_eq!(Adapter::parse_version("GLSL ES 1.1"), Ok((1, 1)));
        assert_eq!(Adapter::parse_version("OpenGL ES GLSL ES 3.20"), Ok((3, 2)));
        assert_eq!(
            // WebGL 2.0 should parse as OpenGL ES 3.0
            Adapter::parse_version("WebGL 2.0 (OpenGL ES 3.0 Chromium)"),
            Ok((3, 0))
        );
        assert_eq!(
            Adapter::parse_version("WebGL GLSL ES 3.00 (OpenGL ES GLSL ES 3.0 Chromium)"),
            Ok((3, 0))
        );
    }
}
