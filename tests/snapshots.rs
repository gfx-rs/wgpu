//TODO: move this to a binary target once Rust supports
// binary-specific dependencies.

use std::{fs, path::PathBuf};

const BASE_DIR_IN: &str = "tests/in";
const BASE_DIR_OUT: &str = "tests/out";

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

    // We can only deserialize `IndexBoundsCheckPolicy` values if `deserialize`
    // feature was enabled, but features should not affect snapshot contents, so
    // just take the policy as booleans instead.
    #[serde(default)]
    bounds_check_read_zero_skip_write: bool,
    #[serde(default)]
    bounds_check_restrict: bool,

    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    spv_version: (u8, u8),
    #[cfg_attr(not(feature = "spv-out"), allow(dead_code))]
    #[serde(default)]
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
    #[cfg_attr(not(feature = "glsl-out"), allow(dead_code))]
    #[serde(default)]
    glsl_vert_ep_name: Option<String>,
    #[cfg_attr(not(feature = "glsl-out"), allow(dead_code))]
    #[serde(default)]
    glsl_frag_ep_name: Option<String>,
    #[cfg_attr(not(feature = "glsl-out"), allow(dead_code))]
    #[serde(default)]
    glsl_comp_ep_name: Option<String>,
}

#[allow(dead_code, unused_variables)]
fn check_targets(module: &naga::Module, name: &str, targets: Targets) {
    let root = env!("CARGO_MANIFEST_DIR");
    let params = match fs::read_to_string(format!("{}/{}/{}.param.ron", root, BASE_DIR_IN, name)) {
        Ok(string) => ron::de::from_str(&string).expect("Couldn't find param file"),
        Err(_) => Parameters::default(),
    };
    if params.bounds_check_restrict && params.bounds_check_read_zero_skip_write {
        panic!("select only one bounds check policy");
    }
    let capabilities = if params.god_mode {
        naga::valid::Capabilities::all()
    } else {
        naga::valid::Capabilities::empty()
    };
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
        .validate(module)
        .unwrap();

    let dest = PathBuf::from(root).join(BASE_DIR_OUT);

    #[cfg(feature = "serialize")]
    {
        if targets.contains(Targets::IR) {
            let config = ron::ser::PrettyConfig::default().with_new_line("\n".to_string());
            let string = ron::ser::to_string_pretty(module, config).unwrap();
            fs::write(dest.join(format!("ir/{}.ron", name)), string).unwrap();
        }
        if targets.contains(Targets::ANALYSIS) {
            let config = ron::ser::PrettyConfig::default().with_new_line("\n".to_string());
            let string = ron::ser::to_string_pretty(&info, config).unwrap();
            fs::write(dest.join(format!("analysis/{}.info.ron", name)), string).unwrap();
        }
    }

    #[cfg(feature = "spv-out")]
    {
        if targets.contains(Targets::SPIRV) {
            write_output_spv(module, &info, &dest, name, &params);
        }
    }
    #[cfg(feature = "msl-out")]
    {
        if targets.contains(Targets::METAL) {
            write_output_msl(module, &info, &dest, name, &params);
        }
    }
    #[cfg(feature = "glsl-out")]
    {
        if targets.contains(Targets::GLSL) {
            for ep in module.entry_points.iter() {
                write_output_glsl(module, &info, &dest, name, ep.stage, &ep.name, &params);
            }
        }
    }
    #[cfg(feature = "dot-out")]
    {
        if targets.contains(Targets::DOT) {
            let string = naga::back::dot::write(module, Some(&info)).unwrap();
            fs::write(dest.join(format!("dot/{}.dot", name)), string).unwrap();
        }
    }
    #[cfg(feature = "hlsl-out")]
    {
        if targets.contains(Targets::HLSL) {
            write_output_hlsl(module, &info, &dest, name);
        }
    }
    #[cfg(feature = "wgsl-out")]
    {
        if targets.contains(Targets::WGSL) {
            write_output_wgsl(module, &info, &dest, name);
        }
    }
}

#[cfg(feature = "spv-out")]
fn write_output_spv(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    destination: &PathBuf,
    file_name: &str,
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
        capabilities: if params.spv_capabilities.is_empty() {
            None
        } else {
            Some(params.spv_capabilities.clone())
        },
        index_bounds_check_policy: if params.bounds_check_restrict {
            naga::back::IndexBoundsCheckPolicy::Restrict
        } else if params.bounds_check_read_zero_skip_write {
            naga::back::IndexBoundsCheckPolicy::ReadZeroSkipWrite
        } else {
            naga::back::IndexBoundsCheckPolicy::UndefinedBehavior
        },
        ..spv::Options::default()
    };

    let spv = spv::write_vec(module, info, &options).unwrap();

    let dis = rspirv::dr::load_words(spv)
        .expect("Produced invalid SPIR-V")
        .disassemble();

    fs::write(destination.join(format!("spv/{}.spvasm", file_name)), dis).unwrap();
}

#[cfg(feature = "msl-out")]
fn write_output_msl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    destination: &PathBuf,
    file_name: &str,
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

    fs::write(destination.join(format!("msl/{}.msl", file_name)), string).unwrap();
}

#[cfg(feature = "glsl-out")]
fn write_output_glsl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    destination: &PathBuf,
    file_name: &str,
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

    fs::write(
        destination.join(format!("glsl/{}.{}.{:?}.glsl", file_name, ep_name, stage)),
        buffer,
    )
    .unwrap();
}

#[cfg(feature = "hlsl-out")]
fn write_output_hlsl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    destination: &PathBuf,
    file_name: &str,
) {
    use naga::back::hlsl;
    let mut buffer = String::new();
    let options = hlsl::Options::default();
    let mut writer = hlsl::Writer::new(&mut buffer, &options);
    let reflection_info = writer.write(module, info).unwrap();

    fs::write(destination.join(format!("hlsl/{}.hlsl", file_name)), buffer).unwrap();

    // We need a config file for validation script
    // This file contains an info about profiles (shader stages) contains inside generated shader
    // This info will be passed to dxc
    let mut config_str = String::from("");
    for (stage, name) in reflection_info.entry_points.iter() {
        let stage_str = match stage {
            naga::ShaderStage::Vertex => "vertex",
            naga::ShaderStage::Fragment => "fragment",
            naga::ShaderStage::Compute => "compute",
        };
        config_str = format!(
            "{}{}={}\n{}_name={}\n",
            config_str,
            stage_str,
            options.shader_model.to_profile_string(*stage),
            stage_str,
            name
        );
    }
    fs::write(
        destination.join(format!("hlsl/{}.hlsl.config", file_name)),
        config_str,
    )
    .unwrap();
}

#[cfg(feature = "wgsl-out")]
fn write_output_wgsl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    destination: &PathBuf,
    file_name: &str,
) {
    use naga::back::wgsl;

    let string = wgsl::write_string(module, info).unwrap();

    fs::write(destination.join(format!("wgsl/{}.wgsl", file_name)), string).unwrap();
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
            Targets::SPIRV
                | Targets::METAL
                | Targets::GLSL
                | Targets::DOT
                | Targets::HLSL
                | Targets::WGSL,
        ),
        (
            "boids",
            Targets::SPIRV | Targets::METAL | Targets::GLSL | Targets::WGSL,
        ),
        (
            "skybox",
            Targets::SPIRV | Targets::METAL | Targets::GLSL | Targets::WGSL,
        ),
        (
            "collatz",
            Targets::SPIRV | Targets::METAL | Targets::IR | Targets::ANALYSIS | Targets::WGSL,
        ),
        (
            "shadow",
            Targets::SPIRV | Targets::METAL | Targets::GLSL | Targets::WGSL,
        ),
        ("image", Targets::SPIRV | Targets::METAL | Targets::WGSL),
        ("extra", Targets::SPIRV | Targets::METAL | Targets::WGSL),
        (
            "operators",
            Targets::SPIRV | Targets::METAL | Targets::GLSL | Targets::WGSL,
        ),
        (
            "interpolate",
            Targets::SPIRV | Targets::METAL | Targets::GLSL | Targets::WGSL,
        ),
        ("access", Targets::SPIRV | Targets::METAL | Targets::WGSL),
        (
            "control-flow",
            Targets::SPIRV | Targets::METAL | Targets::GLSL | Targets::WGSL,
        ),
        (
            "standard",
            Targets::SPIRV | Targets::METAL | Targets::GLSL | Targets::WGSL,
        ),
        (
            "standard",
            Targets::SPIRV | Targets::METAL | Targets::GLSL | Targets::WGSL,
        ),
        //TODO: GLSL https://github.com/gfx-rs/naga/issues/874
        ("interface", Targets::SPIRV | Targets::METAL | Targets::WGSL),
        (
            "globals",
            Targets::SPIRV | Targets::METAL | Targets::GLSL | Targets::WGSL,
        ),
        ("bounds-check-zero", Targets::SPIRV),
    ];

    for &(name, targets) in inputs.iter() {
        println!("Processing '{}'", name);
        // WGSL shaders lives in root dir as a privileged.
        let file = fs::read_to_string(format!("{}/{}/{}.wgsl", root, BASE_DIR_IN, name))
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
        &fs::read(format!("{}/{}/spv/{}.spv", root, BASE_DIR_IN, name))
            .expect("Couldn't find spv file"),
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
        Targets::METAL | Targets::GLSL | Targets::HLSL | Targets::WGSL,
    );
}

#[cfg(feature = "spv-in")]
#[test]
fn convert_spv_shadow() {
    convert_spv("shadow", true, Targets::IR | Targets::ANALYSIS);
}

#[cfg(all(feature = "spv-in", feature = "spv-out"))]
#[test]
fn convert_spv_pointer_access() {
    convert_spv("pointer-access", true, Targets::SPIRV);
}

#[cfg(feature = "glsl-in")]
#[allow(unused_variables)]
#[test]
fn convert_glsl_folder() {
    let root = env!("CARGO_MANIFEST_DIR");

    for entry in std::fs::read_dir(format!("{}/{}/glsl", root, BASE_DIR_IN)).unwrap() {
        let entry = entry.unwrap();
        let file_name = entry.file_name().into_string().unwrap();
        if file_name.ends_with(".ron") {
            // No needed to validate ron files
            continue;
        }
        let params_path = format!(
            "{}/{}/glsl/{}.params.ron",
            root,
            BASE_DIR_IN,
            PathBuf::from(&file_name).with_extension("").display()
        );
        let is_params_used = PathBuf::from(&params_path).exists();
        println!("Processing {}", file_name);

        let mut entry_points = naga::FastHashMap::default();
        if is_params_used {
            let params: Parameters = match fs::read_to_string(&params_path) {
                Ok(string) => ron::de::from_str(&string).expect("Couldn't find param file"),
                Err(_) => panic!("Can't parse glsl params ron file: {:?}", &params_path),
            };

            if let Some(vert) = params.glsl_vert_ep_name {
                entry_points.insert(vert, naga::ShaderStage::Vertex);
            };

            if let Some(frag) = params.glsl_frag_ep_name {
                entry_points.insert(frag, naga::ShaderStage::Fragment);
            };

            if let Some(comp) = params.glsl_comp_ep_name {
                entry_points.insert(comp, naga::ShaderStage::Compute);
            };
        } else {
            let stage = match entry.path().extension().and_then(|s| s.to_str()).unwrap() {
                "vert" => naga::ShaderStage::Vertex,
                "frag" => naga::ShaderStage::Fragment,
                "comp" => naga::ShaderStage::Compute,
                ext => panic!("Unknown extension for glsl file {}", ext),
            };
            entry_points.insert("main".to_string(), stage);
        }

        let module = naga::front::glsl::parse_str(
            &fs::read_to_string(entry.path()).expect("Couldn't find glsl file"),
            &naga::front::glsl::Options {
                entry_points,
                defines: Default::default(),
            },
        )
        .unwrap();

        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();

        #[cfg(feature = "wgsl-out")]
        {
            let dest = PathBuf::from(root).join(BASE_DIR_OUT);
            write_output_wgsl(&module, &info, &dest, &file_name.replace(".", "-"));
        }
    }
}
