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
    #[serde(default)]
    buffer: Option<u8>,
    #[serde(default)]
    texture: Option<u8>,
    #[serde(default)]
    sampler: Option<u8>,
    #[serde(default)]
    mutable: bool,
}

#[derive(Default, Deserialize)]
struct Parameters {
    #[serde(default)]
    spv_flow_dump_prefix: String,
    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    spv_capabilities: naga::FastHashSet<spirv::Capability>,
    #[cfg_attr(not(feature = "msl-out"), allow(dead_code))]
    mtl_bindings: naga::FastHashMap<BindSource, BindTarget>,
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
    insta::assert_snapshot!(format!("{}{}", name, ".msl"), msl);
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
    insta::assert_snapshot!(format!("{}{}", name, ".spvasm"), dis);
}

#[cfg(feature = "wgsl-in")]
fn convert_wgsl(name: &str) {
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

    #[cfg(feature = "msl-out")]
    check_output_msl(&module, name, &params);
    #[cfg(feature = "spv-out")]
    check_output_spv(&module, name, &params);
}

#[cfg(feature = "wgsl-in")]
#[test]
fn converts_wgsl_quad() {
    convert_wgsl("quad");
}
