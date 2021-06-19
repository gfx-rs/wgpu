use glow::HasContext;
use std::sync::Arc;

// https://webgl2fundamentals.org/webgl/lessons/webgl-data-textures.html

impl super::Adapter {
    fn make_info(vendor_orig: String, renderer_orig: String) -> wgt::AdapterInfo {
        let vendor = vendor_orig.to_lowercase();
        let renderer = renderer_orig.to_lowercase();

        // opengl has no way to discern device_type, so we can try to infer it from the renderer string
        let strings_that_imply_integrated = [
            " xpress", // space here is on purpose so we don't match express
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
        ];
        let strings_that_imply_cpu = ["mesa offscreen", "swiftshader", "lavapipe"];

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
            wgt::DeviceType::DiscreteGpu
        };

        // source: Sascha Willems at Vulkan
        let vendor_id = if vendor.contains("amd") {
            0x1002
        } else if vendor.contains("imgtec") {
            0x1010
        } else if vendor.contains("nvidia") {
            0x10DE
        } else if vendor.contains("arm") {
            0x13B5
        } else if vendor.contains("qualcomm") {
            0x5143
        } else if vendor.contains("intel") {
            0x8086
        } else {
            0
        };

        wgt::AdapterInfo {
            name: renderer_orig,
            vendor: vendor_id,
            device: 0,
            device_type: inferred_device_type,
            backend: wgt::Backend::Gl,
        }
    }

    pub(super) unsafe fn expose(gl: glow::Context) -> crate::ExposedAdapter<super::Api> {
        let vendor = gl.get_parameter_string(glow::VENDOR);
        let renderer = gl.get_parameter_string(glow::RENDERER);

        let min_uniform_buffer_offset_alignment =
            gl.get_parameter_i32(glow::UNIFORM_BUFFER_OFFSET_ALIGNMENT);
        let min_storage_buffer_offset_alignment = if super::is_webgl() {
            256
        } else {
            gl.get_parameter_i32(glow::SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT)
        };

        crate::ExposedAdapter {
            adapter: super::Adapter {
                shared: Arc::new(super::AdapterShared {
                    context: gl,
                    private_caps: super::PrivateCapabilities {},
                }),
            },
            info: Self::make_info(vendor, renderer),
            features: wgt::Features::empty(), //TODO
            capabilities: crate::Capabilities {
                limits: wgt::Limits::default(),                   //TODO
                downlevel: wgt::DownlevelCapabilities::default(), //TODO
                alignments: crate::Alignments {
                    buffer_copy_offset: wgt::BufferSize::new(4).unwrap(),
                    buffer_copy_pitch: wgt::BufferSize::new(4).unwrap(),
                    uniform_buffer_offset: wgt::BufferSize::new(
                        min_storage_buffer_offset_alignment as u64,
                    )
                    .unwrap(),
                    storage_buffer_offset: wgt::BufferSize::new(
                        min_uniform_buffer_offset_alignment as u64,
                    )
                    .unwrap(),
                },
            },
        }
    }
}
