bitflags::bitflags! {
    struct Targets: u32 {
        const IR = 0x1;
        const ANALYSIS = 0x2;
        const SPIRV = 0x4;
        const METAL = 0x8;
        const GLSL = 0x10;
        const DOT = 0x20;
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
    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    spv_version: (u8, u8),
    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    spv_capabilities: naga::FastHashSet<spirv::Capability>,
    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    spv_debug: bool,
    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    spv_adjust_coordinate_space: bool,
    #[cfg_attr(not(feature = "msl-out"), allow(dead_code))]
    mtl_bindings: naga::FastHashMap<BindSource, BindTarget>,
}

#[allow(dead_code)]
fn with_snapshot_settings<F: FnOnce() -> ()>(snapshot_assertion: F) {
    let mut settings = insta::Settings::new();
    settings.set_snapshot_path("out");
    settings.set_prepend_module_to_snapshot(false);
    settings.bind(|| snapshot_assertion());
}

#[allow(dead_code, unused_variables)]
fn check_targets(module: &naga::Module, name: &str, targets: Targets) {
    let params = match std::fs::read_to_string(format!("tests/in/{}{}", name, ".param.ron")) {
        Ok(string) => ron::de::from_str(&string).expect("Couldn't find param file"),
        Err(_) => Parameters::default(),
    };
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all())
        .validate(module)
        .unwrap();

    #[cfg(feature = "serialize")]
    {
        if targets.contains(Targets::IR) {
            let config = ron::ser::PrettyConfig::default().with_new_line("\n".to_string());
            let output = ron::ser::to_string_pretty(module, config).unwrap();
            with_snapshot_settings(|| {
                insta::assert_snapshot!(format!("{}.ron", name), output);
            });
        }
        if targets.contains(Targets::ANALYSIS) {
            let config = ron::ser::PrettyConfig::default().with_new_line("\n".to_string());
            let output = ron::ser::to_string_pretty(&info, config).unwrap();
            with_snapshot_settings(|| {
                insta::assert_snapshot!(format!("{}.info.ron", name), output);
            });
        }
    }

    #[cfg(feature = "spv-out")]
    {
        if targets.contains(Targets::SPIRV) {
            check_output_spv(module, &info, name, &params);
        }
    }
    #[cfg(feature = "msl-out")]
    {
        if targets.contains(Targets::METAL) {
            check_output_msl(module, &info, name, &params);
        }
    }
    #[cfg(feature = "glsl-out")]
    {
        if targets.contains(Targets::GLSL) {
            for ep in module.entry_points.iter() {
                check_output_glsl(module, &info, name, ep.stage, &ep.name);
            }
        }
    }
    #[cfg(feature = "dot-out")]
    {
        if targets.contains(Targets::DOT) {
            let string = naga::back::dot::write(module, Some(&info)).unwrap();
            with_snapshot_settings(|| {
                insta::assert_snapshot!(format!("{}.dot", name), string);
            });
        }
    }
}

#[cfg(feature = "spv-out")]
fn check_output_spv(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    name: &str,
    params: &Parameters,
) {
    use naga::back::spv;
    use rspirv::binary::Disassemble;

    let mut flags = spv::WriterFlags::empty();
    if params.spv_debug {
        flags |= spv::WriterFlags::DEBUG;
    }
    if params.spv_adjust_coordinate_space {
        flags |= spv::WriterFlags::ADJUST_COORDINATE_SPACE;
    }
    let options = spv::Options {
        lang_version: params.spv_version,
        flags,
        capabilities: params.spv_capabilities.clone(),
    };

    let spv = spv::write_vec(module, info, &options).unwrap();

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
    info: &naga::valid::ModuleInfo,
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
        binding_map,
        spirv_cross_compatibility: false,
        fake_missing_bindings: false,
        allow_point_size: true,
    };

    let (msl, _) = msl::write_string(module, info, &options).unwrap();

    with_snapshot_settings(|| {
        insta::assert_snapshot!(format!("{}.msl", name), msl);
    });
}

#[cfg(feature = "glsl-out")]
fn check_output_glsl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    name: &str,
    stage: naga::ShaderStage,
    ep_name: &str,
) {
    use naga::back::glsl;

    let options = glsl::Options {
        version: glsl::Version::Embedded(310),
        shader_stage: stage,
        entry_point: ep_name.to_string(),
    };

    let mut buffer = Vec::new();
    let mut writer = glsl::Writer::new(&mut buffer, module, info, &options).unwrap();
    writer.write().unwrap();

    let string = String::from_utf8(buffer).unwrap();

    with_snapshot_settings(|| {
        insta::assert_snapshot!(format!("{}-{:?}.glsl", name, stage), string);
    });
}

#[cfg(feature = "wgsl-in")]
fn convert_wgsl(name: &str, targets: Targets) {
    let module = naga::front::wgsl::parse_str(
        &std::fs::read_to_string(format!("tests/in/{}{}", name, ".wgsl"))
            .expect("Couldn't find wgsl file"),
    )
    .unwrap();
    check_targets(&module, name, targets);
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_quad() {
    convert_wgsl(
        "quad",
        Targets::SPIRV | Targets::METAL | Targets::GLSL | Targets::DOT,
    );
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_empty() {
    convert_wgsl("empty", Targets::SPIRV | Targets::METAL | Targets::GLSL);
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_boids() {
    convert_wgsl("boids", Targets::SPIRV | Targets::METAL);
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_skybox() {
    convert_wgsl("skybox", Targets::SPIRV | Targets::METAL | Targets::GLSL);
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_collatz() {
    convert_wgsl(
        "collatz",
        Targets::SPIRV | Targets::METAL | Targets::IR | Targets::ANALYSIS,
    );
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_shadow() {
    convert_wgsl("shadow", Targets::SPIRV | Targets::METAL | Targets::GLSL);
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_image_copy() {
    //SPIR-V is blocked by https://github.com/gfx-rs/naga/issues/646
    convert_wgsl("image-copy", Targets::METAL);
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl_texture_array() {
    convert_wgsl("texture-array", Targets::SPIRV);
}

#[cfg(feature = "spv-in")]
fn convert_spv(name: &str, adjust_coordinate_space: bool, targets: Targets) {
    let module = naga::front::spv::parse_u8_slice(
        &std::fs::read(format!("tests/in/{}{}", name, ".spv")).expect("Couldn't find spv file"),
        &naga::front::spv::Options {
            adjust_coordinate_space,
            flow_graph_dump_prefix: None,
        },
    )
    .unwrap();
    check_targets(&module, name, targets);
    naga::valid::Validator::new(naga::valid::ValidationFlags::all())
        .validate(&module)
        .unwrap();
}

#[cfg(feature = "spv-in")]
#[test]
fn convert_spv_quad_vert() {
    convert_spv("quad-vert", false, Targets::METAL);
}

#[cfg(feature = "spv-in")]
#[test]
fn convert_spv_shadow() {
    convert_spv("shadow", true, Targets::IR | Targets::ANALYSIS);
}

#[cfg(feature = "glsl-in")]
fn convert_glsl(
    name: &str,
    entry_points: naga::FastHashMap<String, naga::ShaderStage>,
    _targets: Targets,
) {
    let _module = naga::front::glsl::parse_str(
        &std::fs::read_to_string(format!("tests/in/{}{}", name, ".glsl"))
            .expect("Couldn't find glsl file"),
        &naga::front::glsl::Options {
            entry_points,
            defines: Default::default(),
        },
    )
    .unwrap();
    //TODO
    //check_targets(&module, name, targets);
}

#[cfg(feature = "glsl-in")]
#[test]
fn convert_glsl_quad() {
    let mut entry_points = naga::FastHashMap::default();
    entry_points.insert("vert_main".to_string(), naga::ShaderStage::Vertex);
    entry_points.insert("frag_main".to_string(), naga::ShaderStage::Fragment);
    convert_glsl("quad-glsl", entry_points, Targets::SPIRV | Targets::IR);
}
