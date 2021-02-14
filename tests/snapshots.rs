bitflags::bitflags! {
    struct Language: u32 {
        const SPIRV = 0x1;
        const METAL = 0x2;
        const GLSL = 0x4;
    }
}

#[derive(Hash, PartialEq, Eq, serde::Deserialize)]
enum Stage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Hash, PartialEq, Eq, serde::Deserialize)]
struct BindSource {
    stage: Stage,
    group: u32,
    binding: u32,
}

#[derive(serde::Deserialize)]
struct BindTarget {
    #[cfg_attr(not(feature = "msl-out"), allow(dead_code))]
    #[serde(default)]
    buffer: Option<u8>,
    #[cfg_attr(not(feature = "msl-out"), allow(dead_code))]
    #[serde(default)]
    texture: Option<u8>,
    #[cfg_attr(not(feature = "msl-out"), allow(dead_code))]
    #[serde(default)]
    sampler: Option<u8>,
    #[cfg_attr(not(feature = "msl-out"), allow(dead_code))]
    #[serde(default)]
    mutable: bool,
}

#[derive(Default, serde::Deserialize)]
struct Parameters {
    #[serde(default)]
    #[allow(dead_code)]
    spv_flow_dump_prefix: String,
    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    spv_version: (u8, u8),
    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    spv_capabilities: naga::FastHashSet<spirv::Capability>,
    #[cfg_attr(not(feature = "msl-out"), allow(dead_code))]
    mtl_bindings: naga::FastHashMap<BindSource, BindTarget>,
}

fn with_snapshot_settings<F: FnOnce() -> ()>(snapshot_assertion: F) {
    let mut settings = insta::Settings::new();
    settings.set_snapshot_path("out");
    settings.set_prepend_module_to_snapshot(false);
    settings.bind(|| snapshot_assertion());
}

#[cfg(feature = "spv-out")]
fn check_output_spv(module: &naga::Module, name: &str, params: &Parameters) {
    use naga::back::spv;
    use rspirv::binary::Disassemble;

    let options = spv::Options {
        lang_version: params.spv_version,
        flags: spv::WriterFlags::DEBUG,
        capabilities: params.spv_capabilities.clone(),
    };

    let spv = spv::write_vec(&module, &options).unwrap();

    let dis = rspirv::dr::load_words(spv)
        .expect("Produced invalid SPIR-V")
        .disassemble();
    with_snapshot_settings(|| {
        insta::assert_snapshot!(format!("{}.spvasm", name), dis);
    });
}

#[cfg(feature = "msl-out")]
fn check_output_msl(
    module: &naga::Module,
    analysis: &naga::proc::analyzer::Analysis,
    name: &str,
    params: &Parameters,
) {
    use naga::back::msl;

    let mut binding_map = msl::BindingMap::default();
    for (key, value) in params.mtl_bindings.iter() {
        binding_map.insert(
            msl::BindSource {
                stage: match key.stage {
                    Stage::Vertex => naga::ShaderStage::Vertex,
                    Stage::Fragment => naga::ShaderStage::Fragment,
                    Stage::Compute => naga::ShaderStage::Compute,
                },
                group: key.group,
                binding: key.binding,
            },
            msl::BindTarget {
                buffer: value.buffer,
                texture: value.texture,
                sampler: value.sampler,
                mutable: value.mutable,
            },
        );
    }
    let options = msl::Options {
        lang_version: (1, 0),
        spirv_cross_compatibility: false,
        binding_map,
    };

    let (msl, _) = msl::write_string(module, analysis, &options).unwrap();

    with_snapshot_settings(|| {
        insta::assert_snapshot!(format!("{}.msl", name), msl);
    });
}

#[cfg(feature = "glsl-out")]
fn check_output_glsl(
    module: &naga::Module,
    analysis: &naga::proc::analyzer::Analysis,
    name: &str,
    stage: naga::ShaderStage,
    ep_name: &str,
) {
    use naga::back::glsl;

    let options = glsl::Options {
        version: glsl::Version::Embedded(310),
        entry_point: (stage, ep_name.to_string()),
    };

    let mut buffer = Vec::new();
    let mut writer = glsl::Writer::new(&mut buffer, module, analysis, &options).unwrap();
    writer.write().unwrap();

    let string = String::from_utf8(buffer).unwrap();

    with_snapshot_settings(|| {
        insta::assert_snapshot!(format!("{}-{:?}.glsl", name, stage), string);
    });
}

#[cfg(feature = "wgsl-in")]
fn convert_wgsl(name: &str, language: Language) {
    let params = match std::fs::read_to_string(format!("tests/in/{}{}", name, ".param.ron")) {
        Ok(string) => ron::de::from_str(&string).expect("Couldn't find param file"),
        Err(_) => Parameters::default(),
    };

    let module = naga::front::wgsl::parse_str(
        &std::fs::read_to_string(format!("tests/in/{}{}", name, ".wgsl"))
            .expect("Couldn't find wgsl file"),
    )
    .unwrap();
    let analysis = naga::proc::Validator::new().validate(&module).unwrap();

    #[cfg(feature = "spv-out")]
    {
        if language.contains(Language::SPIRV) {
            check_output_spv(&module, name, &params);
        }
    }
    #[cfg(feature = "msl-out")]
    {
        if language.contains(Language::METAL) {
            check_output_msl(&module, &analysis, name, &params);
        }
    }
    #[cfg(feature = "glsl-out")]
    {
        if language.contains(Language::GLSL) {
            for &(stage, ref ep_name) in module.entry_points.keys() {
                check_output_glsl(&module, &analysis, name, stage, ep_name);
            }
        }
    }
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_quad() {
    convert_wgsl("quad", Language::all());
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_empty() {
    convert_wgsl("empty", Language::all());
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_boids() {
    convert_wgsl("boids", Language::METAL | Language::SPIRV);
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_skybox() {
    convert_wgsl("skybox", Language::all());
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_collatz() {
    convert_wgsl("collatz", Language::METAL | Language::SPIRV);
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_shadow() {
    convert_wgsl("shadow", Language::METAL | Language::SPIRV);
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_texture_array() {
    convert_wgsl("texture-array", Language::SPIRV);
}

#[cfg(feature = "spv-in")]
fn convert_spv(name: &str) {
    let module = naga::front::spv::parse_u8_slice(
        &std::fs::read(format!("tests/in/{}{}", name, ".spv")).expect("Couldn't find spv file"),
        &Default::default(),
    )
    .unwrap();
    naga::proc::Validator::new().validate(&module).unwrap();

    #[cfg(feature = "serialize")]
    {
        let config = ron::ser::PrettyConfig::default().with_new_line("\n".to_string());
        let output = ron::ser::to_string_pretty(&module, config).unwrap();
        with_snapshot_settings(|| {
            insta::assert_snapshot!(format!("{}.ron", name), output);
        });
    }
}

#[cfg(feature = "spv-in")]
#[test]
fn convert_spv_shadow() {
    convert_spv("shadow");
}
