use std::{fs, process::Command};
use wgpu_benchmark::{iter_auto, BenchmarkContext, SubBenchResult};

const DIR_IN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../naga/tests/in");

use naga_test::*;

struct InputWithInfo {
    inner: Input,
    data: Vec<u8>,
    string: Option<String>,
    options: Parameters,
    module: Option<naga::Module>,
    module_info: Option<naga::valid::ModuleInfo>,
}
impl From<Input> for InputWithInfo {
    fn from(value: Input) -> Self {
        let mut options = value.read_parameters(DIR_IN);
        options.targets = Some(options.targets.unwrap_or(Targets::all()));
        Self {
            options,
            inner: value,
            data: Vec::new(),
            string: None,
            module: None,
            module_info: None,
        }
    }
}
impl InputWithInfo {
    fn filename(&self) -> &str {
        self.inner.file_name.file_name().unwrap().to_str().unwrap()
    }
}

struct Inputs {
    inner: Vec<InputWithInfo>,
}

impl Inputs {
    #[track_caller]
    fn from_dir(folder: &str, extension: &str) -> Self {
        let inputs: Vec<InputWithInfo> = Input::files_in_dir(folder, &[extension], DIR_IN)
            .map(|a| a.into())
            .collect();

        Self { inner: inputs }
    }
    fn bytes(&self) -> u64 {
        self.inner
            .iter()
            .map(|input| input.inner.bytes(DIR_IN))
            .sum()
    }

    fn load(&mut self) {
        for input in &mut self.inner {
            if !input.data.is_empty() {
                continue;
            }

            input.data = fs::read(input.inner.input_path(DIR_IN)).unwrap_or_default();
        }
    }

    fn load_utf8(&mut self) {
        self.load();

        for input in &mut self.inner {
            if input.string.is_some() {
                continue;
            }

            input.string = Some(std::str::from_utf8(&input.data).unwrap().to_string());
        }
    }

    fn parse(&mut self) {
        self.load_utf8();

        let mut parser = naga::front::wgsl::Frontend::new();
        for input in &mut self.inner {
            if input.module.is_some() {
                continue;
            }

            parser.set_options((&input.options.wgsl_in).into());

            input.module = Some(parser.parse(input.string.as_ref().unwrap()).unwrap());
        }
    }

    fn validate(&mut self) {
        self.parse();

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            // Note, this is empty, to let all backends work.
            naga::valid::Capabilities::empty(),
        );

        for input in &mut self.inner {
            if input.module_info.is_some() {
                continue;
            }

            input.module_info = validator.validate(input.module.as_ref().unwrap()).ok();
        }

        self.inner.retain(|input| input.module_info.is_some());
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

fn parse_glsl(stage: naga::ShaderStage, inputs: &Inputs) {
    let mut parser = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options {
        stage,
        defines: Default::default(),
    };
    for input in &inputs.inner {
        parser
            .parse(&options, &input.inner.read_source(DIR_IN, false))
            .unwrap();
    }
}

fn get_wgsl_inputs() -> Inputs {
    let mut inputs: Vec<InputWithInfo> = Input::files_in_dir("wgsl", &["wgsl"], DIR_IN)
        .map(|a| a.into())
        .collect();

    // remove "large-source" tests, they skew the results
    inputs.retain(|input| !input.filename().contains("large-source"));

    assert!(!inputs.is_empty());

    Inputs { inner: inputs }
}

pub fn frontends(ctx: BenchmarkContext) -> anyhow::Result<Vec<SubBenchResult>> {
    let mut results = Vec::new();

    let mut inputs_wgsl = get_wgsl_inputs();

    inputs_wgsl.parse();
    inputs_wgsl.load_utf8();

    let inputs_bin = inputs_wgsl
        .inner
        .iter()
        .map(|input| {
            bincode::serde::encode_to_vec(
                input.module.as_ref().unwrap(),
                bincode::config::standard(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    results.push(iter_auto(
        &ctx,
        "bincode decode",
        "bytes",
        inputs_wgsl.bytes() as u32,
        move || {
            for input in inputs_bin.iter() {
                bincode::serde::decode_from_slice::<naga::Module, _>(
                    input,
                    bincode::config::standard(),
                )
                .unwrap();
            }
        },
    ));

    let mut frontend = naga::front::wgsl::Frontend::new();

    results.push(iter_auto(
        &ctx,
        "wgsl",
        "bytes",
        inputs_wgsl.bytes() as u32,
        || {
            for input in &inputs_wgsl.inner {
                frontend.set_options((&input.options.wgsl_in).into());
                frontend.parse(input.string.as_ref().unwrap()).unwrap();
            }
        },
    ));

    let inputs_spirv = Inputs::from_dir("spv", "spvasm");
    assert!(!inputs_spirv.is_empty());

    // Assemble all the SPIR-V assembly.
    let mut assembled_spirv = Vec::<Vec<u32>>::new();
    'spirv: for input in &inputs_spirv.inner {
        let output = match Command::new("spirv-as")
            .arg(input.inner.input_path(DIR_IN))
            .arg("-o")
            .arg("-")
            .output()
        {
            Ok(output) => output,
            Err(e) => {
                eprintln!(
                    "Failed to execute spirv-as: {e}\n\
                    spvasm benchmarks will be skipped.\n\
                    spirv-as can be installed by installing the Vulkan SDK and adding \
                        it to your PATH.",
                );
                break 'spirv;
            }
        };

        if !output.status.success() {
            panic!(
                "spirv-as failed: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        assembled_spirv.push(bytemuck::pod_collect_to_vec(&output.stdout));
    }

    let total_bytes: u64 = assembled_spirv.iter().map(|spv| spv.len() as u64).sum();

    assert!(assembled_spirv.len() == inputs_spirv.inner.len() || assembled_spirv.is_empty());

    results.push(iter_auto(
        &ctx,
        "spv parse",
        "bytes",
        total_bytes as u32,
        || {
            for (i, input) in assembled_spirv.iter().enumerate() {
                let params = &inputs_spirv.inner[i].options;
                let SpirvInParameters {
                    adjust_coordinate_space,
                } = params.spv_in;

                let parser = naga::front::spv::Frontend::new(
                    input.iter().cloned(),
                    &naga::front::spv::Options {
                        adjust_coordinate_space,
                        strict_capabilities: true,
                        ..Default::default()
                    },
                );
                parser.parse().unwrap();
            }
        },
    ));

    let mut inputs_vertex = Inputs::from_dir("glsl", "vert");
    let mut inputs_fragment = Inputs::from_dir("glsl", "frag");
    let mut inputs_compute = Inputs::from_dir("glsl", "comp");
    assert!(!inputs_vertex.is_empty());
    assert!(!inputs_fragment.is_empty());
    assert!(!inputs_compute.is_empty());

    inputs_vertex.load_utf8();
    inputs_fragment.load_utf8();
    inputs_compute.load_utf8();

    results.push(iter_auto(
        &ctx,
        "glsl parse",
        "bytes",
        (inputs_vertex.bytes() + inputs_fragment.bytes() + inputs_compute.bytes()) as u32,
        || {
            parse_glsl(naga::ShaderStage::Vertex, &inputs_vertex);
            parse_glsl(naga::ShaderStage::Fragment, &inputs_fragment);
            parse_glsl(naga::ShaderStage::Compute, &inputs_compute);
        },
    ));

    Ok(results)
}

pub fn validation(ctx: BenchmarkContext) -> anyhow::Result<Vec<SubBenchResult>> {
    let mut results = Vec::new();

    let mut inputs = get_wgsl_inputs();

    inputs.parse();

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all());

    results.push(iter_auto(
        &ctx,
        "validation",
        "bytes",
        inputs.bytes() as u32,
        || {
            for input in &inputs.inner {
                validator.validate(input.module.as_ref().unwrap()).unwrap();
            }
        },
    ));

    Ok(results)
}

pub fn compact(ctx: BenchmarkContext) -> anyhow::Result<Vec<SubBenchResult>> {
    use naga::compact::{compact, KeepUnused};

    let mut results = Vec::new();

    let mut inputs = get_wgsl_inputs();

    inputs.validate();
    assert!(!inputs.is_empty());

    results.push(iter_auto(
        &ctx,
        "compact",
        "bytes",
        inputs.bytes() as u32,
        || {
            for input in &mut inputs.inner {
                compact(input.module.as_mut().unwrap(), KeepUnused::No);
            }
        },
    ));

    Ok(results)
}

pub fn backends(ctx: BenchmarkContext) -> anyhow::Result<Vec<SubBenchResult>> {
    let mut results = Vec::new();

    let mut inputs = get_wgsl_inputs();

    inputs.validate();
    assert!(!inputs.is_empty());

    let total_bytes = inputs.bytes() as u32;

    results.push(iter_auto(&ctx, "wgsl", "bytes", total_bytes, || {
        let mut string = String::new();
        for input in &inputs.inner {
            if input.options.targets.unwrap().contains(Targets::WGSL) {
                let mut writer =
                    naga::back::wgsl::Writer::new(&mut string, (&input.options.wgsl).into());
                let _ = writer.write(
                    input.module.as_ref().unwrap(),
                    input.module_info.as_ref().unwrap(),
                );
                string.clear();
            }
        }
    }));

    results.push(iter_auto(&ctx, "spv", "bytes", total_bytes, || {
        let mut data = Vec::new();
        let mut writer = naga::back::spv::Writer::new(&Default::default()).unwrap();
        for input in &inputs.inner {
            let shared_info = WriterSharedOptions {
                mesh_output_validation: input.options.mesh_output_validation,
                task_limits: input.options.task_limits,
                bounds_checks_policies: input.options.bounds_check_policies,
            };
            if input.options.targets.unwrap().contains(Targets::SPIRV) {
                if input.filename().contains("pointer-function-arg") {
                    continue;
                }
                let opt = input.options.spv.to_options(&shared_info, None);
                if writer.set_options(&opt).is_ok() {
                    let _ = writer.write(
                        input.module.as_ref().unwrap(),
                        input.module_info.as_ref().unwrap(),
                        None,
                        &None,
                        &mut data,
                    );
                    data.clear();
                }
            }
        }
    }));

    results.push(iter_auto(
        &ctx,
        "spv multiple entrypoints",
        "bytes",
        total_bytes,
        || {
            let mut data = Vec::new();
            let options = naga::back::spv::Options::default();
            for input in &inputs.inner {
                if input.options.targets.unwrap().contains(Targets::SPIRV) {
                    if input.filename().contains("pointer-function-arg") {
                        continue;
                    }
                    let mut writer = naga::back::spv::Writer::new(&options).unwrap();
                    let module = input.module.as_ref().unwrap();
                    for ep in module.entry_points.iter() {
                        let pipeline_options = naga::back::spv::PipelineOptions {
                            shader_stage: ep.stage,
                            entry_point: ep.name.clone(),
                        };
                        let _ = writer.write(
                            input.module.as_ref().unwrap(),
                            input.module_info.as_ref().unwrap(),
                            Some(&pipeline_options),
                            &None,
                            &mut data,
                        );
                        data.clear();
                    }
                }
            }
        },
    ));

    results.push(iter_auto(&ctx, "msl", "bytes", total_bytes, || {
        let mut string = String::new();
        let options = naga::back::msl::Options::default();
        for input in &inputs.inner {
            if input.options.targets.unwrap().contains(Targets::METAL) {
                let pipeline_options = naga::back::msl::PipelineOptions::default();
                let mut writer = naga::back::msl::Writer::new(&mut string);
                let _ = writer.write(
                    input.module.as_ref().unwrap(),
                    input.module_info.as_ref().unwrap(),
                    &options,
                    &pipeline_options,
                );
                string.clear();
            }
        }
    }));

    results.push(iter_auto(&ctx, "hlsl", "bytes", total_bytes, || {
        let options = naga::back::hlsl::Options::default();
        let mut string = String::new();
        for input in &inputs.inner {
            if input.options.targets.unwrap().contains(Targets::HLSL) {
                let pipeline_options = Default::default();
                let mut writer =
                    naga::back::hlsl::Writer::new(&mut string, &options, &pipeline_options);
                let _ = writer.write(
                    input.module.as_ref().unwrap(),
                    input.module_info.as_ref().unwrap(),
                    None,
                );
                string.clear();
            }
        }
    }));

    results.push(iter_auto(
        &ctx,
        "glsl multiple entrypoints",
        "bytes",
        total_bytes,
        || {
            let mut string = String::new();
            let options = naga::back::glsl::Options {
                version: naga::back::glsl::Version::new_gles(320),
                writer_flags: naga::back::glsl::WriterFlags::empty(),
                binding_map: Default::default(),
                zero_initialize_workgroup_memory: true,
            };
            for input in &inputs.inner {
                if !input.options.targets.unwrap().contains(Targets::GLSL) {
                    continue;
                }
                let module = input.module.as_ref().unwrap();
                let info = input.module_info.as_ref().unwrap();
                for ep in module.entry_points.iter() {
                    let pipeline_options = naga::back::glsl::PipelineOptions {
                        shader_stage: ep.stage,
                        entry_point: ep.name.clone(),
                        multiview: None,
                    };

                    if let Ok(mut writer) = naga::back::glsl::Writer::new(
                        &mut string,
                        module,
                        info,
                        &options,
                        &pipeline_options,
                        naga::proc::BoundsCheckPolicies::default(),
                    ) {
                        let _ = writer.write();
                    }

                    string.clear();
                }
            }
        },
    ));

    Ok(results)
}
