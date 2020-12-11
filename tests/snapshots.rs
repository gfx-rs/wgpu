use serde::Deserialize;

#[derive(Hash, PartialEq, Eq, Deserialize)]
enum Stage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Hash, PartialEq, Eq, Deserialize)]
struct BindSource {
    stage: Stage,
    group: u32,
    binding: u32,
}

#[derive(Deserialize)]
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

#[derive(Default, Deserialize)]
struct Parameters {
    #[serde(default)]
    #[allow(dead_code)]
    spv_flow_dump_prefix: String,
    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    spv_capabilities: naga::FastHashSet<spirv::Capability>,
    #[cfg_attr(not(feature = "msl-out"), allow(dead_code))]
    mtl_bindings: naga::FastHashMap<BindSource, BindTarget>,
}

bitflags::bitflags! {
    struct Language: u32 {
        const SPIRV = 0x1;
        const METAL = 0x2;
        const GLSL = 0x4;
    }
}

#[cfg(feature = "spv-out")]
fn check_output_spv(module: &naga::Module, name: &str, params: &Parameters) {
    use naga::back::spv;
    use rspirv::binary::Disassemble;

    let spv = spv::write_vec(
        &module,
        spv::WriterFlags::NONE,
        params.spv_capabilities.clone(),
    )
    .unwrap();

    let dis = rspirv::dr::load_words(spv)
        .expect("Produced invalid SPIR-V")
        .disassemble();
    insta::assert_snapshot!(format!("{}.spvasm", name), dis);
}

#[cfg(feature = "msl-out")]
fn check_output_msl(module: &naga::Module, name: &str, params: &Parameters) {
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

    let (msl, _) = msl::write_string(&module, &options).unwrap();
    insta::assert_snapshot!(format!("{}.msl", name), msl);
}

#[cfg(feature = "glsl-out")]
fn check_output_glsl(module: &naga::Module, name: &str, stage: naga::ShaderStage, ep_name: &str) {
    use naga::back::glsl;

    let options = glsl::Options {
        version: glsl::Version::Embedded(310),
        entry_point: (stage, ep_name.to_string()),
    };

    let mut buffer = Vec::new();
    let mut writer = glsl::Writer::new(&mut buffer, &module, &options).unwrap();
    writer.write().unwrap();

    let string = String::from_utf8(buffer).unwrap();
    insta::assert_snapshot!(format!("{}-{:?}.glsl", name, stage), string);
}

#[cfg(feature = "wgsl-in")]
fn convert_wgsl(name: &str, language: Language) {
    let params =
        match std::fs::read_to_string(format!("tests/snapshots/in/{}{}", name, ".param.ron")) {
            Ok(string) => ron::de::from_str(&string).expect("Couldn't find param file"),
            Err(_) => Parameters::default(),
        };

    let module = naga::front::wgsl::parse_str(
        &std::fs::read_to_string(format!("tests/snapshots/in/{}{}", name, ".wgsl"))
            .expect("Couldn't find wgsl file"),
    )
    .unwrap();
    naga::proc::Validator::new().validate(&module).unwrap();

    #[cfg(feature = "spv-out")]
    {
        if language.contains(Language::SPIRV) {
            check_output_spv(&module, name, &params);
        }
    }
    #[cfg(feature = "msl-out")]
    {
        if language.contains(Language::METAL) {
            check_output_msl(&module, name, &params);
        }
    }
    #[cfg(feature = "glsl-out")]
    {
        if language.contains(Language::GLSL) {
            for &(stage, ref ep_name) in module.entry_points.keys() {
                check_output_glsl(&module, name, stage, ep_name);
            }
        }
    }
}

#[cfg(feature = "wgsl-in")]
#[test]
fn converts_wgsl_quad() {
    convert_wgsl("quad", Language::all());
}

#[cfg(feature = "wgsl-in")]
#[test]
fn converts_wgsl_simple() {
    convert_wgsl("simple", Language::all());
}

#[cfg(feature = "wgsl-in")]
#[test]
fn converts_wgsl_boids() {
    convert_wgsl("boids", Language::METAL);
}
