//TODO: move this to a binary target once Rust supports
// binary-specific dependencies.

use std::{fs, path::PathBuf};

const DIR_IN: &str = "tests/in";
const DIR_OUT: &str = "tests/out";

bitflags::bitflags! {
    struct Targets: u32 {
        const IR = 0x1;
        const ANALYSIS = 0x2;
        const SPIRV = 0x4;
        const METAL = 0x8;
        const GLSL = 0x10;
        const DOT = 0x20;
        const HLSL = 0x40;
        const WGSL = 0x80;
    }
}

#[derive(Default, serde::Deserialize)]
struct Parameters {
    #[serde(default)]
    god_mode: bool,
    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    spv_version: (u8, u8),
    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    spv_capabilities: naga::FastHashSet<spirv::Capability>,
    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    #[serde(default)]
    spv_debug: bool,
    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    #[serde(default)]
    spv_adjust_coordinate_space: bool,
    #[cfg(all(feature = "deserialize", feature = "msl-out"))]
    #[serde(default)]
    msl: naga::back::msl::Options,
    #[cfg(all(not(feature = "deserialize"), feature = "msl-out"))]
    #[serde(default)]
    msl_custom: bool,
    #[cfg_attr(not(feature = "glsl-out"), allow(dead_code))]
    #[serde(default)]
    glsl_desktop_version: Option<u16>,
}

#[allow(dead_code, unused_variables)]
fn check_targets(module: &naga::Module, name: &str, targets: Targets) {
    let root = env!("CARGO_MANIFEST_DIR");
    let params = match fs::read_to_string(format!("{}/{}/{}.param.ron", root, DIR_IN, name)) {
        Ok(string) => ron::de::from_str(&string).expect("Couldn't find param file"),
        Err(_) => Parameters::default(),
    };
    let capabilities = if params.god_mode {
        naga::valid::Capabilities::all()
    } else {
        naga::valid::Capabilities::empty()
    };
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
        .validate(module)
        .unwrap();

    let dest = PathBuf::from(root).join(DIR_OUT).join(name);

    #[cfg(feature = "serialize")]
    {
        if targets.contains(Targets::IR) {
            let config = ron::ser::PrettyConfig::default().with_new_line("\n".to_string());
            let string = ron::ser::to_string_pretty(module, config).unwrap();
            fs::write(dest.with_extension("ron"), string).unwrap();
        }
        if targets.contains(Targets::ANALYSIS) {
            let config = ron::ser::PrettyConfig::default().with_new_line("\n".to_string());
            let string = ron::ser::to_string_pretty(&info, config).unwrap();
            fs::write(dest.with_extension("info.ron"), string).unwrap();
        }
    }

    #[cfg(feature = "spv-out")]
    {
        if targets.contains(Targets::SPIRV) {
            check_output_spv(module, &info, &dest, &params);
        }
    }
    #[cfg(feature = "msl-out")]
    {
        if targets.contains(Targets::METAL) {
            check_output_msl(module, &info, &dest, &params);
        }
    }
    #[cfg(feature = "glsl-out")]
    {
        if targets.contains(Targets::GLSL) {
            for ep in module.entry_points.iter() {
                check_output_glsl(module, &info, &dest, ep.stage, &ep.name, &params);
            }
        }
    }
    #[cfg(feature = "dot-out")]
    {
        if targets.contains(Targets::DOT) {
            let string = naga::back::dot::write(module, Some(&info)).unwrap();
            fs::write(dest.with_extension("dot"), string).unwrap();
        }
    }
    #[cfg(feature = "hlsl-out")]
    {
        if targets.contains(Targets::HLSL) {
            for ep in module.entry_points.iter() {
                check_output_hlsl(module, &dest, ep.stage);
            }
        }
    }
    #[cfg(feature = "wgsl-out")]
    {
        if targets.contains(Targets::WGSL) {
            check_output_wgsl(module, &info, &dest);
        }
    }
}

#[cfg(feature = "spv-out")]
fn check_output_spv(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    destination: &PathBuf,
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
        capabilities: Some(params.spv_capabilities.clone()),
    };

    let spv = spv::write_vec(module, info, &options).unwrap();

    let dis = rspirv::dr::load_words(spv)
        .expect("Produced invalid SPIR-V")
        .disassemble();

    fs::write(destination.with_extension("spvasm"), dis).unwrap();
}

#[cfg(feature = "msl-out")]
fn check_output_msl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    destination: &PathBuf,
    params: &Parameters,
) {
    use naga::back::msl;

    #[cfg_attr(feature = "deserialize", allow(unused_variables))]
    let default_options = msl::Options::default();
    #[cfg(feature = "deserialize")]
    let options = &params.msl;
    #[cfg(not(feature = "deserialize"))]
    let options = if params.msl_custom {
        println!("Skipping {}", destination.display());
        return;
    } else {
        &default_options
    };

    let pipeline_options = msl::PipelineOptions {
        allow_point_size: true,
    };

    let (string, tr_info) = msl::write_string(module, info, options, &pipeline_options).unwrap();

    for (ep, result) in module.entry_points.iter().zip(tr_info.entry_point_names) {
        if let Err(error) = result {
            panic!("Failed to translate '{}': {}", ep.name, error);
        }
    }

    fs::write(destination.with_extension("msl"), string).unwrap();
}

#[cfg(feature = "glsl-out")]
fn check_output_glsl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    destination: &PathBuf,
    stage: naga::ShaderStage,
    ep_name: &str,
    params: &Parameters,
) {
    use naga::back::glsl;

    let options = glsl::Options {
        version: match params.glsl_desktop_version {
            Some(v) => glsl::Version::Desktop(v),
            None => glsl::Version::Embedded(310),
        },
        shader_stage: stage,
        entry_point: ep_name.to_string(),
    };

    let mut buffer = String::new();
    let mut writer = glsl::Writer::new(&mut buffer, module, info, &options).unwrap();
    writer.write().unwrap();

    let ext = format!("{:?}.glsl", stage);
    fs::write(destination.with_extension(&ext), buffer).unwrap();
}

#[cfg(feature = "hlsl-out")]
fn check_output_hlsl(module: &naga::Module, destination: &PathBuf, stage: naga::ShaderStage) {
    use naga::back::hlsl;

    let string = hlsl::write_string(module).unwrap();

    let ext = format!("{:?}.hlsl", stage);
    fs::write(destination.with_extension(&ext), string).unwrap();
}

#[cfg(feature = "wgsl-out")]
fn check_output_wgsl(module: &naga::Module, info: &naga::valid::ModuleInfo, destination: &PathBuf) {
    use naga::back::wgsl;

    let string = wgsl::write_string(module, info).unwrap();

    fs::write(destination.with_extension("wgsl"), string).unwrap();
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_wgsl() {
    let root = env!("CARGO_MANIFEST_DIR");
    let inputs = [
        (
            "empty",
            Targets::SPIRV | Targets::METAL | Targets::GLSL | Targets::HLSL | Targets::WGSL,
        ),
        (
            "quad",
            Targets::SPIRV | Targets::METAL | Targets::GLSL | Targets::DOT | Targets::WGSL,
        ),
        ("boids", Targets::SPIRV | Targets::METAL | Targets::GLSL),
        ("skybox", Targets::SPIRV | Targets::METAL | Targets::GLSL),
        (
            "collatz",
            Targets::SPIRV | Targets::METAL | Targets::IR | Targets::ANALYSIS,
        ),
        ("shadow", Targets::SPIRV | Targets::METAL | Targets::GLSL),
        ("image", Targets::SPIRV | Targets::METAL),
        ("extra", Targets::SPIRV | Targets::METAL),
        ("operators", Targets::SPIRV | Targets::METAL | Targets::GLSL),
        (
            "interpolate",
            Targets::SPIRV | Targets::METAL | Targets::GLSL,
        ),
        ("access", Targets::SPIRV | Targets::METAL),
        (
            "control-flow",
            Targets::SPIRV | Targets::METAL | Targets::GLSL,
        ),
    ];

    for &(name, targets) in inputs.iter() {
        println!("Processing '{}'", name);
        let file = fs::read_to_string(format!("{}/{}/{}.wgsl", root, DIR_IN, name))
            .expect("Couldn't find wgsl file");
        match naga::front::wgsl::parse_str(&file) {
            Ok(module) => check_targets(&module, name, targets),
            Err(e) => panic!("{}", e),
        }
    }
}

#[cfg(feature = "spv-in")]
fn convert_spv(name: &str, adjust_coordinate_space: bool, targets: Targets) {
    let root = env!("CARGO_MANIFEST_DIR");
    let module = naga::front::spv::parse_u8_slice(
        &fs::read(format!("{}/{}/{}.spv", root, DIR_IN, name)).expect("Couldn't find spv file"),
        &naga::front::spv::Options {
            adjust_coordinate_space,
            strict_capabilities: false,
            flow_graph_dump_prefix: None,
        },
    )
    .unwrap();
    check_targets(&module, name, targets);
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
}

#[cfg(feature = "spv-in")]
#[test]
fn convert_spv_quad_vert() {
    convert_spv(
        "quad-vert",
        false,
        Targets::METAL | Targets::GLSL | Targets::WGSL,
    );
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
    let root = env!("CARGO_MANIFEST_DIR");
    let _module = naga::front::glsl::parse_str(
        &fs::read_to_string(format!("{}/{}/{}.glsl", root, DIR_IN, name))
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
