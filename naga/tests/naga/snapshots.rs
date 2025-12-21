use naga::compact::KeepUnused;
use naga_test::*;

const DIR_IN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/in");
const DIR_OUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/out");

#[allow(unused_variables)]
fn check_targets(input: &Input, module: &mut naga::Module, source_code: Option<&str>) {
    let params = input.read_parameters(DIR_IN);
    let name = input.file_name.display().to_string();

    let targets = params.targets.unwrap();

    let capabilities = params.capabilities.unwrap_or_default();

    {
        if targets.contains(Targets::IR) {
            let config = ron::ser::PrettyConfig::default().new_line("\n".to_string());
            let string = ron::ser::to_string_pretty(module, config).unwrap();
            input.write_output_file("ir", "ron", string, DIR_OUT);
        }
    }

    let validation_flags = if targets.contains(Targets::NO_VALIDATION) {
        naga::valid::ValidationFlags::empty()
    } else {
        naga::valid::ValidationFlags::all()
    };

    let info = naga::valid::Validator::new(validation_flags, capabilities)
        .validate(module)
        .unwrap_or_else(|err| {
            panic!("Naga module validation failed on test `{name}`:\n{err:?}");
        });

    let info = {
        // Our backends often generate temporary names based on handle indices,
        // which means that adding or removing unused arena entries can affect
        // the output even though they have no semantic effect. Such
        // meaningless changes add noise to snapshot diffs, making accurate
        // patch review difficult. Compacting the modules before generating
        // snapshots makes the output independent of unused arena entries.
        naga::compact::compact(module, KeepUnused::No);

        {
            if targets.contains(Targets::IR) {
                let config = ron::ser::PrettyConfig::default().new_line("\n".to_string());
                let string = ron::ser::to_string_pretty(module, config).unwrap();
                input.write_output_file("ir", "compact.ron", string, DIR_OUT);
            }
        }

        naga::valid::Validator::new(validation_flags, capabilities)
            .validate(module)
            .unwrap_or_else(|err| {
                panic!("Post-compaction module validation failed on test '{name}':\n<{err:?}")
            })
    };

    {
        if targets.contains(Targets::ANALYSIS) {
            let config = ron::ser::PrettyConfig::default().new_line("\n".to_string());
            let string = ron::ser::to_string_pretty(&info, config).unwrap();
            input.write_output_file("analysis", "info.ron", string, DIR_OUT);
        }
    }

    if targets.contains(Targets::SPIRV) {
        let mut debug_info = None;
        if let Some(source_code) = source_code {
            debug_info = Some(naga::back::spv::DebugInfo {
                source_code,
                file_name: &name,
                // wgpu#6266: we technically know all the information here to
                // produce the valid language but it's not too important for
                // validation purposes
                language: naga::back::spv::SourceLanguage::Unknown,
            })
        }

        write_output_spv(
            input,
            module,
            &info,
            debug_info,
            &params.spv,
            params.bounds_check_policies,
            &params.pipeline_constants,
        );
    }

    if targets.contains(Targets::METAL) {
        write_output_msl(
            input,
            module,
            &info,
            &params.msl,
            &params.msl_pipeline,
            params.bounds_check_policies,
            &params.pipeline_constants,
        );
    }

    if targets.contains(Targets::GLSL) {
        for ep in module.entry_points.iter() {
            if params.glsl_exclude_list.contains(&ep.name) {
                continue;
            }
            write_output_glsl(
                input,
                module,
                &info,
                ep.stage,
                &ep.name,
                &params.glsl,
                params.bounds_check_policies,
                params.glsl_multiview,
                &params.pipeline_constants,
            );
        }
    }

    if targets.contains(Targets::DOT) {
        let string = naga::back::dot::write(module, Some(&info), Default::default()).unwrap();
        input.write_output_file("dot", "dot", string, DIR_OUT);
    }

    if targets.contains(Targets::HLSL) {
        let frag_module;
        let mut frag_ep = None;
        if let Some(ref module_spec) = params.fragment_module {
            let full_path = input.input_directory(DIR_IN).join(&module_spec.path);

            assert_eq!(
                full_path.extension().unwrap().to_string_lossy(),
                "wgsl",
                "Currently all fragment modules must be in WGSL"
            );

            let frag_src = std::fs::read_to_string(full_path).unwrap();

            frag_module =
                naga::front::wgsl::parse_str(&frag_src).expect("Failed to parse fragment module");

            frag_ep = Some(
                naga::back::hlsl::FragmentEntryPoint::new(&frag_module, &module_spec.entry_point)
                    .expect("Could not find fragment entry point"),
            );
        }

        write_output_hlsl(
            input,
            module,
            &info,
            &params.hlsl,
            &params.pipeline_constants,
            frag_ep,
        );
    }

    if targets.contains(Targets::WGSL) {
        write_output_wgsl(input, module, &info, &params.wgsl);
    }
}

fn write_output_spv(
    input: &Input,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    debug_info: Option<naga::back::spv::DebugInfo>,
    params: &SpirvOutParameters,
    bounds_check_policies: naga::proc::BoundsCheckPolicies,
    pipeline_constants: &naga::back::PipelineConstants,
) {
    use naga::back::spv;

    let options = params.to_options(bounds_check_policies, debug_info);

    let (module, info) =
        naga::back::pipeline_constants::process_overrides(module, info, None, pipeline_constants)
            .expect("override evaluation failed");

    if params.separate_entry_points {
        for ep in module.entry_points.iter() {
            let pipeline_options = spv::PipelineOptions {
                entry_point: ep.name.clone(),
                shader_stage: ep.stage,
            };
            write_output_spv_inner(
                input,
                &module,
                &info,
                &options,
                Some(&pipeline_options),
                &format!("{}.spvasm", ep.name),
            );
        }
    } else {
        write_output_spv_inner(input, &module, &info, &options, None, "spvasm");
    }
}

fn write_output_spv_inner(
    input: &Input,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    options: &naga::back::spv::Options<'_>,
    pipeline_options: Option<&naga::back::spv::PipelineOptions>,
    extension: &str,
) {
    use naga::back::spv;
    use rspirv::binary::Disassemble;
    println!("Generating SPIR-V for {:?}", input.file_name);
    let spv = spv::write_vec(module, info, options, pipeline_options).unwrap();
    let dis = rspirv::dr::load_words(spv)
        .expect("Produced invalid SPIR-V")
        .disassemble();
    // HACK escape CR/LF if source code is in side.
    let dis = if options.debug_info.is_some() {
        let dis = dis.replace("\\r", "\r");
        dis.replace("\\n", "\n")
    } else {
        dis
    };
    input.write_output_file("spv", extension, dis, DIR_OUT);
}

fn write_output_msl(
    input: &Input,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    options: &naga::back::msl::Options,
    pipeline_options: &naga::back::msl::PipelineOptions,
    bounds_check_policies: naga::proc::BoundsCheckPolicies,
    pipeline_constants: &naga::back::PipelineConstants,
) {
    use naga::back::msl;

    println!("generating MSL");

    let (module, info) =
        naga::back::pipeline_constants::process_overrides(module, info, None, pipeline_constants)
            .expect("override evaluation failed");

    let mut options = options.clone();
    options.bounds_check_policies = bounds_check_policies;
    let (string, tr_info) = msl::write_string(&module, &info, &options, pipeline_options)
        .unwrap_or_else(|err| panic!("Metal write failed: {err}"));

    for (ep, result) in module.entry_points.iter().zip(tr_info.entry_point_names) {
        if let Err(error) = result {
            panic!("Failed to translate '{}': {}", ep.name, error);
        }
    }

    input.write_output_file("msl", "msl", string, DIR_OUT);
}

#[allow(clippy::too_many_arguments)]
fn write_output_glsl(
    input: &Input,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    stage: naga::ShaderStage,
    ep_name: &str,
    options: &naga::back::glsl::Options,
    bounds_check_policies: naga::proc::BoundsCheckPolicies,
    multiview: Option<core::num::NonZeroU32>,
    pipeline_constants: &naga::back::PipelineConstants,
) {
    use naga::back::glsl;

    println!("generating GLSL");

    let pipeline_options = glsl::PipelineOptions {
        shader_stage: stage,
        entry_point: ep_name.to_string(),
        multiview,
    };

    let mut buffer = String::new();
    let (module, info) =
        naga::back::pipeline_constants::process_overrides(module, info, None, pipeline_constants)
            .expect("override evaluation failed");
    let mut writer = glsl::Writer::new(
        &mut buffer,
        &module,
        &info,
        options,
        &pipeline_options,
        bounds_check_policies,
    )
    .expect("GLSL init failed");
    writer.write().expect("GLSL write failed");

    let extension = format!("{ep_name}.{stage:?}.glsl");
    input.write_output_file("glsl", &extension, buffer, DIR_OUT);
}

fn write_output_hlsl(
    input: &Input,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    options: &naga::back::hlsl::Options,
    pipeline_constants: &naga::back::PipelineConstants,
    frag_ep: Option<naga::back::hlsl::FragmentEntryPoint>,
) {
    use naga::back::hlsl;

    println!("generating HLSL");

    let (module, info) =
        naga::back::pipeline_constants::process_overrides(module, info, None, pipeline_constants)
            .expect("override evaluation failed");

    let mut buffer = String::new();
    let pipeline_options = Default::default();
    let mut writer = hlsl::Writer::new(&mut buffer, options, &pipeline_options);
    let reflection_info = writer
        .write(&module, &info, frag_ep.as_ref())
        .expect("HLSL write failed");

    input.write_output_file("hlsl", "hlsl", buffer, DIR_OUT);

    // We need a config file for validation script
    // This file contains an info about profiles (shader stages) contains inside generated shader
    // This info will be passed to dxc
    let mut config = hlsl_snapshots::Config::empty();
    for (index, ep) in module.entry_points.iter().enumerate() {
        let name = match reflection_info.entry_point_names[index] {
            Ok(ref name) => name,
            Err(_) => continue,
        };
        match ep.stage {
            naga::ShaderStage::Vertex => &mut config.vertex,
            naga::ShaderStage::Fragment => &mut config.fragment,
            naga::ShaderStage::Compute => &mut config.compute,
            naga::ShaderStage::Task | naga::ShaderStage::Mesh => unreachable!(),
        }
        .push(hlsl_snapshots::ConfigItem {
            entry_point: name.clone(),
            target_profile: format!(
                "{}_{}",
                ep.stage.to_hlsl_str(),
                options.shader_model.to_str()
            ),
        });
    }

    config
        .to_file(input.output_path("hlsl", "ron", DIR_OUT))
        .unwrap();
}

fn write_output_wgsl(
    input: &Input,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    params: &WgslOutParameters,
) {
    use naga::back::wgsl;

    println!("generating WGSL");

    let string = wgsl::write_string(module, info, params.into()).expect("WGSL write failed");

    input.write_output_file("wgsl", "wgsl", string, DIR_OUT);
}

// While we _can_ run this test under miri, it is extremely slow (>5 minutes),
// and naga isn't the primary target for miri testing, so we disable it.
#[cfg_attr(miri, ignore)]
#[test]
fn convert_snapshots_wgsl() {
    let _ = env_logger::try_init();

    for input in Input::files_in_dir("wgsl", &["wgsl"], DIR_IN) {
        let source = input.read_source(DIR_IN, true);
        // crlf will make the large split output different on different platform
        let source = source.replace('\r', "");

        let params = input.read_parameters(DIR_IN);

        let mut frontend = naga::front::wgsl::Frontend::new_with_options((&params.wgsl_in).into());
        match frontend.parse(&source) {
            Ok(mut module) => check_targets(&input, &mut module, Some(&source)),
            Err(e) => panic!(
                "{}",
                e.emit_to_string_with_path(&source, input.input_path(DIR_IN))
            ),
        }
    }
}

// miri doesn't allow us to shell out to `spirv-as`
#[cfg_attr(miri, ignore)]
#[test]
fn convert_snapshots_spv() {
    use std::process::Command;

    let _ = env_logger::try_init();

    for input in Input::files_in_dir("spv", &["spvasm"], DIR_IN) {
        println!("Assembling '{}'", input.file_name.display());

        let command = Command::new("spirv-as")
            .arg(input.input_path(DIR_IN))
            .arg("-o")
            .arg("-")
            .output()
            .expect(
                "Failed to execute spirv-as. It can be installed \
            by installing the Vulkan SDK and adding it to your path.",
            );

        println!("Processing '{}'", input.file_name.display());

        if !command.status.success() {
            panic!(
                "spirv-as failed: {}\n{}",
                String::from_utf8_lossy(&command.stdout),
                String::from_utf8_lossy(&command.stderr)
            );
        }

        let params = input.read_parameters(DIR_IN);

        let mut module =
            naga::front::spv::parse_u8_slice(&command.stdout, &(&params.spv_in).into()).unwrap();

        check_targets(&input, &mut module, None);
    }
}

// While we _can_ run this test under miri, it is extremely slow (>5 minutes),
// and naga isn't the primary target for miri testing, so we disable it.
#[cfg_attr(miri, ignore)]
#[allow(unused_variables)]
#[test]
fn convert_snapshots_glsl() {
    let _ = env_logger::try_init();

    for input in Input::files_in_dir("glsl", &["vert", "frag", "comp"], DIR_IN) {
        let input = Input {
            keep_input_extension: true,
            ..input
        };
        let file_name = &input.file_name;

        let stage = match file_name.extension().and_then(|s| s.to_str()).unwrap() {
            "vert" => naga::ShaderStage::Vertex,
            "frag" => naga::ShaderStage::Fragment,
            "comp" => naga::ShaderStage::Compute,
            ext => panic!("Unknown extension for glsl file {ext}"),
        };

        let mut parser = naga::front::glsl::Frontend::default();
        let mut module = parser
            .parse(
                &naga::front::glsl::Options {
                    stage,
                    defines: Default::default(),
                },
                &input.read_source(DIR_IN, true),
            )
            .unwrap();

        check_targets(&input, &mut module, None);
    }
}
