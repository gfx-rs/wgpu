#[allow(dead_code)]
fn load_test_data(name: &str) -> String {
    let path = format!("{}/test-data/{}", env!("CARGO_MANIFEST_DIR"), name);
    std::fs::read_to_string(path).unwrap()
}

#[cfg(feature = "wgsl-in")]
fn load_wgsl(name: &str) -> naga::Module {
    let input = load_test_data(name);
    naga::front::wgsl::parse_str(&input).unwrap()
}

#[cfg(feature = "spv-in")]
fn load_spv(name: &str) -> naga::Module {
    let path = format!("{}/test-data/spv/{}", env!("CARGO_MANIFEST_DIR"), name);
    let input = std::fs::read(path).unwrap();
    naga::front::spv::parse_u8_slice(&input, &Default::default()).unwrap()
}

#[cfg(feature = "glsl-in")]
fn load_glsl(name: &str, entry: &str, stage: naga::ShaderStage) -> naga::Module {
    let input = load_test_data(name);
    naga::front::glsl::parse_str(&input, entry, stage, Default::default()).unwrap()
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_quad() {
    let module = load_wgsl("quad.wgsl");
    naga::proc::Validator::new().validate(&module).unwrap();
    #[cfg(feature = "msl-out")]
    {
        use naga::back::msl;
        let mut binding_map = msl::BindingMap::default();
        binding_map.insert(
            msl::BindSource {
                stage: naga::ShaderStage::Fragment,
                group: 0,
                binding: 0,
            },
            msl::BindTarget {
                buffer: None,
                texture: Some(1),
                sampler: None,
                mutable: false,
            },
        );
        binding_map.insert(
            msl::BindSource {
                stage: naga::ShaderStage::Fragment,
                group: 0,
                binding: 1,
            },
            msl::BindTarget {
                buffer: None,
                texture: None,
                sampler: Some(1),
                mutable: false,
            },
        );
        let options = msl::Options {
            lang_version: (1, 0),
            spirv_cross_compatibility: false,
            binding_map,
        };
        msl::write_string(&module, &options).unwrap();
    }
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_boids() {
    let module = load_wgsl("boids.wgsl");
    naga::proc::Validator::new().validate(&module).unwrap();
    #[cfg(feature = "msl-out")]
    {
        use naga::back::msl;
        let mut binding_map = msl::BindingMap::default();
        binding_map.insert(
            msl::BindSource {
                stage: naga::ShaderStage::Compute,
                group: 0,
                binding: 0,
            },
            msl::BindTarget {
                buffer: Some(0),
                texture: None,
                sampler: None,
                mutable: false,
            },
        );
        binding_map.insert(
            msl::BindSource {
                stage: naga::ShaderStage::Compute,
                group: 0,
                binding: 1,
            },
            msl::BindTarget {
                buffer: Some(1),
                texture: None,
                sampler: Some(1),
                mutable: false,
            },
        );
        binding_map.insert(
            msl::BindSource {
                stage: naga::ShaderStage::Compute,
                group: 0,
                binding: 2,
            },
            msl::BindTarget {
                buffer: Some(2),
                texture: None,
                sampler: Some(1),
                mutable: false,
            },
        );
        let options = msl::Options {
            lang_version: (1, 0),
            spirv_cross_compatibility: false,
            binding_map,
        };
        msl::write_string(&module, &options).unwrap();
    }
}

#[cfg(feature = "spv-in")]
#[test]
fn convert_cube() {
    let mut validator = naga::proc::Validator::new();
    let vs = load_spv("cube.vert.spv");
    validator.validate(&vs).unwrap();
    let fs = load_spv("cube.frag.spv");
    validator.validate(&fs).unwrap();
    #[cfg(feature = "msl-out")]
    {
        use naga::back::msl;
        let mut binding_map = msl::BindingMap::default();
        binding_map.insert(
            msl::BindSource {
                stage: naga::ShaderStage::Vertex,
                group: 0,
                binding: 0,
            },
            msl::BindTarget {
                buffer: Some(0),
                texture: None,
                sampler: None,
                mutable: false,
            },
        );
        binding_map.insert(
            msl::BindSource {
                stage: naga::ShaderStage::Fragment,
                group: 0,
                binding: 1,
            },
            msl::BindTarget {
                buffer: None,
                texture: Some(1),
                sampler: None,
                mutable: false,
            },
        );
        binding_map.insert(
            msl::BindSource {
                stage: naga::ShaderStage::Fragment,
                group: 0,
                binding: 2,
            },
            msl::BindTarget {
                buffer: None,
                texture: None,
                sampler: Some(1),
                mutable: false,
            },
        );
        let options = msl::Options {
            lang_version: (1, 0),
            spirv_cross_compatibility: false,
            binding_map,
        };
        msl::write_string(&vs, &options).unwrap();
        msl::write_string(&fs, &options).unwrap();
    }
}

#[cfg(feature = "glsl-in")]
#[test]
#[ignore]
fn convert_phong_lighting() {
    let module = load_glsl(
        "glsl_phong_lighting.frag",
        "main",
        naga::ShaderStage::Fragment,
    );
    naga::proc::Validator::new().validate(&module).unwrap();

    #[cfg(feaure = "spv-out")]
    {
        let header = naga::Header {
            version: (1, 0, 0),
            generator: 1234,
        };
        let writer_flags = naga::back::spv::WriterFlags::empty();
        let mut w = naga::back::spv::Writer::new(&header, writer_flags);
        w.write(&module);
    }
}

//TODO: get this working again (glsl-new)
// #[cfg(feature = "glsl-in")]
// #[test]
// fn constant_expressions() {
//     let module = load_glsl(
//         "glsl_constant_expression.vert",
//         "main",
//         naga::ShaderStage::Fragment,
//     );
//     naga::proc::Validator::new().validate(&module).unwrap();
// }
