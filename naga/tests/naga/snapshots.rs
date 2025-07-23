// A lot of the code can be unused based on configuration flags,
// the corresponding warnings aren't helpful.
#![allow(dead_code, unused_imports)]

use core::fmt::Write;

use std::{
    fs,
    path::{Path, PathBuf},
};

use naga::compact::KeepUnused;
use ron::de;

const CRATE_ROOT: &str = env!("CARGO_MANIFEST_DIR");
const BASE_DIR_IN: &str = "tests/in";
const BASE_DIR_OUT: &str = "tests/out";

bitflags::bitflags! {
    #[derive(Clone, Copy, serde::Deserialize)]
    #[serde(transparent)]
    #[derive(Debug, Eq, PartialEq)]
    struct Targets: u32 {
        /// A serialization of the `naga::Module`, in RON format.
        const IR = 1;

        /// A serialization of the `naga::valid::ModuleInfo`, in RON format.
        const ANALYSIS = 1 << 1;

        const SPIRV = 1 << 2;
        const METAL = 1 << 3;
        const GLSL = 1 << 4;
        const DOT = 1 << 5;
        const HLSL = 1 << 6;
        const WGSL = 1 << 7;
        const NO_VALIDATION = 1 << 8;
    }
}

impl Targets {
    /// Defaults for `spv` and `glsl` snapshots.
    fn non_wgsl_default() -> Self {
        Targets::WGSL
    }

    /// Defaults for `wgsl` snapshots.
    fn wgsl_default() -> Self {
        Targets::HLSL | Targets::SPIRV | Targets::GLSL | Targets::METAL | Targets::WGSL
    }
}

#[derive(serde::Deserialize)]
struct SpvOutVersion(u8, u8);
impl Default for SpvOutVersion {
    fn default() -> Self {
        SpvOutVersion(1, 1)
    }
}

#[cfg(all(feature = "deserialize", spv_out))]
#[derive(serde::Deserialize)]
struct BindingMapSerialization {
    resource_binding: naga::ResourceBinding,
    bind_target: naga::back::spv::BindingInfo,
}

#[cfg(all(feature = "deserialize", spv_out))]
fn deserialize_binding_map<'de, D>(deserializer: D) -> Result<naga::back::spv::BindingMap, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;

    let vec = Vec::<BindingMapSerialization>::deserialize(deserializer)?;
    let mut map = naga::back::spv::BindingMap::default();
    for item in vec {
        map.insert(item.resource_binding, item.bind_target);
    }
    Ok(map)
}

#[derive(Default, serde::Deserialize)]
#[serde(default)]
struct WgslInParameters {
    parse_doc_comments: bool,
}

#[derive(Default, serde::Deserialize)]
#[serde(default)]
struct SpirvInParameters {
    adjust_coordinate_space: bool,
}

#[derive(Default, serde::Deserialize)]
#[serde(default)]
struct SpirvOutParameters {
    version: SpvOutVersion,
    capabilities: naga::FastHashSet<spirv::Capability>,
    debug: bool,
    adjust_coordinate_space: bool,
    force_point_size: bool,
    clamp_frag_depth: bool,
    separate_entry_points: bool,
    #[cfg(all(feature = "deserialize", spv_out))]
    #[serde(deserialize_with = "deserialize_binding_map")]
    binding_map: naga::back::spv::BindingMap,
}

#[derive(Default, serde::Deserialize)]
#[serde(default)]
struct WgslOutParameters {
    explicit_types: bool,
}

#[derive(Default, serde::Deserialize)]
struct FragmentModule {
    path: String,
    entry_point: String,
}

#[derive(Default, serde::Deserialize)]
#[serde(default)]
struct Parameters {
    // -- GOD MODE --
    god_mode: bool,

    // -- wgsl-in options --
    #[serde(rename = "wgsl-in")]
    wgsl_in: WgslInParameters,

    // -- spirv-in options --
    #[serde(rename = "spv-in")]
    spv_in: SpirvInParameters,

    // -- SPIR-V options --
    spv: SpirvOutParameters,

    /// Defaults to [`Targets::non_wgsl_default()`] for `spv` and `glsl` snapshots,
    /// and [`Targets::wgsl_default()`] for `wgsl` snapshots.
    targets: Option<Targets>,

    // -- MSL options --
    #[cfg(all(feature = "deserialize", msl_out))]
    msl: naga::back::msl::Options,
    #[cfg(all(feature = "deserialize", msl_out))]
    #[serde(default)]
    msl_pipeline: naga::back::msl::PipelineOptions,

    // -- GLSL options --
    #[cfg(all(feature = "deserialize", glsl_out))]
    glsl: naga::back::glsl::Options,
    glsl_exclude_list: naga::FastHashSet<String>,
    #[cfg(all(feature = "deserialize", glsl_out))]
    glsl_multiview: Option<core::num::NonZeroU32>,

    // -- HLSL options --
    #[cfg(all(feature = "deserialize", hlsl_out))]
    hlsl: naga::back::hlsl::Options,

    // -- WGSL options --
    wgsl: WgslOutParameters,

    // -- General options --

    // Allow backends to be aware of the fragment module.
    // Is the name of a WGSL file in the same directory as the test file.
    fragment_module: Option<FragmentModule>,

    #[cfg(feature = "deserialize")]
    bounds_check_policies: naga::proc::BoundsCheckPolicies,

    #[cfg(all(feature = "deserialize", any(hlsl_out, msl_out, spv_out, glsl_out)))]
    pipeline_constants: naga::back::PipelineConstants,
}

/// Information about a shader input file.
#[derive(Debug)]
struct Input {
    /// The subdirectory of `tests/in` to which this input belongs, if any.
    ///
    /// If the subdirectory is omitted, we assume that the output goes
    /// to "wgsl".
    subdirectory: PathBuf,

    /// The input filename name, without a directory.
    file_name: PathBuf,

    /// True if output filenames should add the output extension on top of
    /// `file_name`'s existing extension, rather than replacing it.
    ///
    /// This is used by `convert_snapshots_glsl`, which wants to take input files
    /// like `210-bevy-2d-shader.frag` and just add `.wgsl` to it, producing
    /// `210-bevy-2d-shader.frag.wgsl`.
    keep_input_extension: bool,
}

impl Input {
    /// Read an input file and its corresponding parameters file.
    ///
    /// Given `input`, the relative path of a shader input file, return
    /// a `Source` value containing its path, code, and parameters.
    ///
    /// The `input` path is interpreted relative to the `BASE_DIR_IN`
    /// subdirectory of the directory given by the `CARGO_MANIFEST_DIR`
    /// environment variable.
    fn new(subdirectory: &str, name: &str, extension: &str) -> Input {
        Input {
            subdirectory: PathBuf::from(subdirectory),
            // Don't wipe out any extensions on `name`, as
            // `with_extension` would do.
            file_name: PathBuf::from(format!("{name}.{extension}")),
            keep_input_extension: false,
        }
    }

    /// Return an iterator that produces an `Input` for each entry in `subdirectory`.
    fn files_in_dir(
        subdirectory: &'static str,
        file_extensions: &'static [&'static str],
    ) -> impl Iterator<Item = Input> + 'static {
        let input_directory = Path::new(CRATE_ROOT).join(BASE_DIR_IN).join(subdirectory);

        let entries = match std::fs::read_dir(&input_directory) {
            Ok(entries) => entries,
            Err(err) => panic!(
                "Error opening directory '{}': {}",
                input_directory.display(),
                err
            ),
        };

        entries.filter_map(move |result| {
            let entry = result.expect("error reading directory");
            if !entry.file_type().unwrap().is_file() {
                return None;
            }

            let file_name = PathBuf::from(entry.file_name());
            let extension = file_name
                .extension()
                .expect("all files in snapshot input directory should have extensions");

            if !file_extensions.contains(&extension.to_str().unwrap()) {
                return None;
            }

            if let Ok(pat) = std::env::var("NAGA_SNAPSHOT") {
                if !file_name.to_string_lossy().contains(&pat) {
                    return None;
                }
            }

            let input = Input::new(
                subdirectory,
                file_name.file_stem().unwrap().to_str().unwrap(),
                extension.to_str().unwrap(),
            );
            Some(input)
        })
    }

    /// Return the path to the input directory.
    fn input_directory(&self) -> PathBuf {
        let mut dir = Path::new(CRATE_ROOT).join(BASE_DIR_IN);
        dir.push(&self.subdirectory);
        dir
    }

    /// Return the path to the output directory.
    fn output_directory(subdirectory: &str) -> PathBuf {
        let mut dir = Path::new(CRATE_ROOT).join(BASE_DIR_OUT);
        dir.push(subdirectory);
        dir
    }

    /// Return the path to the input file.
    fn input_path(&self) -> PathBuf {
        let mut input = self.input_directory();
        input.push(&self.file_name);
        input
    }

    fn output_path(&self, subdirectory: &str, extension: &str) -> PathBuf {
        let mut output = Self::output_directory(subdirectory);
        if self.keep_input_extension {
            let file_name = format!(
                "{}-{}.{}",
                self.subdirectory.display(),
                self.file_name.display(),
                extension
            );

            output.push(&file_name);
        } else {
            let file_name = format!(
                "{}-{}",
                self.subdirectory.display(),
                self.file_name.display()
            );

            output.push(&file_name);
            output.set_extension(extension);
        }
        output
    }

    /// Return the contents of the input file as a string.
    fn read_source(&self) -> String {
        println!("Processing '{}'", self.file_name.display());
        let input_path = self.input_path();
        match fs::read_to_string(&input_path) {
            Ok(source) => source,
            Err(err) => {
                panic!(
                    "Couldn't read shader input file `{}`: {}",
                    input_path.display(),
                    err
                );
            }
        }
    }

    /// Return the contents of the input file as a vector of bytes.
    fn read_bytes(&self) -> Vec<u8> {
        println!("Processing '{}'", self.file_name.display());
        let input_path = self.input_path();
        match fs::read(&input_path) {
            Ok(bytes) => bytes,
            Err(err) => {
                panic!(
                    "Couldn't read shader input file `{}`: {}",
                    input_path.display(),
                    err
                );
            }
        }
    }

    /// Return this input's parameter file, parsed.
    fn read_parameters(&self) -> Parameters {
        let mut param_path = self.input_path();
        param_path.set_extension("toml");
        let mut params = match fs::read_to_string(&param_path) {
            Ok(string) => match toml::de::from_str(&string) {
                Ok(params) => params,
                Err(e) => panic!(
                    "Couldn't parse param file: {} due to: {e}",
                    param_path.display()
                ),
            },
            Err(_) => Parameters::default(),
        };

        if params.targets.is_none() {
            match self.input_path().extension().unwrap().to_str().unwrap() {
                "wgsl" => params.targets = Some(Targets::wgsl_default()),
                "spvasm" => params.targets = Some(Targets::non_wgsl_default()),
                "vert" | "frag" | "comp" => params.targets = Some(Targets::non_wgsl_default()),
                e => {
                    panic!("Unknown extension: {e}");
                }
            }
        }

        params
    }

    /// Write `data` to a file corresponding to this input file in
    /// `subdirectory`, with `extension`.
    fn write_output_file(&self, subdirectory: &str, extension: &str, data: impl AsRef<[u8]>) {
        let output_path = self.output_path(subdirectory, extension);
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        if let Err(err) = fs::write(&output_path, data) {
            panic!("Error writing {}: {}", output_path.display(), err);
        }
    }
}

#[cfg(hlsl_out)]
type FragmentEntryPoint<'a> = naga::back::hlsl::FragmentEntryPoint<'a>;
#[cfg(not(hlsl_out))]
type FragmentEntryPoint<'a> = ();

#[allow(unused_variables)]
fn check_targets(input: &Input, module: &mut naga::Module, source_code: Option<&str>) {
    let params = input.read_parameters();
    let name = &input.file_name;

    let targets = params.targets.unwrap();

    let (capabilities, subgroup_stages, subgroup_operations) = if params.god_mode {
        (
            naga::valid::Capabilities::all(),
            naga::valid::ShaderStages::all(),
            naga::valid::SubgroupOperationSet::all(),
        )
    } else {
        (
            naga::valid::Capabilities::default(),
            naga::valid::ShaderStages::empty(),
            naga::valid::SubgroupOperationSet::empty(),
        )
    };

    #[cfg(feature = "serialize")]
    {
        if targets.contains(Targets::IR) {
            let config = ron::ser::PrettyConfig::default().new_line("\n".to_string());
            let string = ron::ser::to_string_pretty(module, config).unwrap();
            input.write_output_file("ir", "ron", string);
        }
    }

    let validation_flags = if targets.contains(Targets::NO_VALIDATION) {
        naga::valid::ValidationFlags::empty()
    } else {
        naga::valid::ValidationFlags::all()
    };

    let info = naga::valid::Validator::new(validation_flags, capabilities)
        .subgroup_stages(subgroup_stages)
        .subgroup_operations(subgroup_operations)
        .validate(module)
        .unwrap_or_else(|err| {
            panic!(
                "Naga module validation failed on test `{}`:\n{:?}",
                name.display(),
                err
            );
        });

    let info = {
        // Our backends often generate temporary names based on handle indices,
        // which means that adding or removing unused arena entries can affect
        // the output even though they have no semantic effect. Such
        // meaningless changes add noise to snapshot diffs, making accurate
        // patch review difficult. Compacting the modules before generating
        // snapshots makes the output independent of unused arena entries.
        naga::compact::compact(module, KeepUnused::No);

        #[cfg(feature = "serialize")]
        {
            if targets.contains(Targets::IR) {
                let config = ron::ser::PrettyConfig::default().new_line("\n".to_string());
                let string = ron::ser::to_string_pretty(module, config).unwrap();
                input.write_output_file("ir", "compact.ron", string);
            }
        }

        naga::valid::Validator::new(validation_flags, capabilities)
            .subgroup_stages(subgroup_stages)
            .subgroup_operations(subgroup_operations)
            .validate(module)
            .unwrap_or_else(|err| {
                panic!(
                    "Post-compaction module validation failed on test '{}':\n<{:?}",
                    name.display(),
                    err,
                )
            })
    };

    #[cfg(feature = "serialize")]
    {
        if targets.contains(Targets::ANALYSIS) {
            let config = ron::ser::PrettyConfig::default().new_line("\n".to_string());
            let string = ron::ser::to_string_pretty(&info, config).unwrap();
            input.write_output_file("analysis", "info.ron", string);
        }
    }

    #[cfg(all(feature = "deserialize", spv_out))]
    {
        if targets.contains(Targets::SPIRV) {
            let mut debug_info = None;
            if let Some(source_code) = source_code {
                debug_info = Some(naga::back::spv::DebugInfo {
                    source_code,
                    file_name: name.as_path().into(),
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
    }
    #[cfg(all(feature = "deserialize", msl_out))]
    {
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
    }
    #[cfg(all(feature = "deserialize", glsl_out))]
    {
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
    }
    #[cfg(dot_out)]
    {
        if targets.contains(Targets::DOT) {
            let string = naga::back::dot::write(module, Some(&info), Default::default()).unwrap();
            input.write_output_file("dot", "dot", string);
        }
    }
    #[cfg(all(feature = "deserialize", hlsl_out))]
    {
        if targets.contains(Targets::HLSL) {
            let frag_module;
            let mut frag_ep = None;
            if let Some(ref module_spec) = params.fragment_module {
                let full_path = input.input_directory().join(&module_spec.path);

                assert_eq!(
                    full_path.extension().unwrap().to_string_lossy(),
                    "wgsl",
                    "Currently all fragment modules must be in WGSL"
                );

                let frag_src = fs::read_to_string(full_path).unwrap();

                frag_module = naga::front::wgsl::parse_str(&frag_src)
                    .expect("Failed to parse fragment module");

                frag_ep = Some(
                    naga::back::hlsl::FragmentEntryPoint::new(
                        &frag_module,
                        &module_spec.entry_point,
                    )
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
    }
    #[cfg(all(feature = "deserialize", wgsl_out))]
    {
        if targets.contains(Targets::WGSL) {
            write_output_wgsl(input, module, &info, &params.wgsl);
        }
    }
}

#[cfg(spv_out)]
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
    use rspirv::binary::Disassemble;

    let mut flags = spv::WriterFlags::LABEL_VARYINGS;
    flags.set(spv::WriterFlags::DEBUG, params.debug);
    flags.set(
        spv::WriterFlags::ADJUST_COORDINATE_SPACE,
        params.adjust_coordinate_space,
    );
    flags.set(spv::WriterFlags::FORCE_POINT_SIZE, params.force_point_size);
    flags.set(spv::WriterFlags::CLAMP_FRAG_DEPTH, params.clamp_frag_depth);

    let options = spv::Options {
        lang_version: (params.version.0, params.version.1),
        flags,
        capabilities: if params.capabilities.is_empty() {
            None
        } else {
            Some(params.capabilities.clone())
        },
        bounds_check_policies,
        binding_map: params.binding_map.clone(),
        zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::Polyfill,
        force_loop_bounding: true,
        debug_info,
    };

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

#[cfg(spv_out)]
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
    input.write_output_file("spv", extension, dis);
}

#[cfg(msl_out)]
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

    input.write_output_file("msl", "msl", string);
}

#[cfg(glsl_out)]
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
    input.write_output_file("glsl", &extension, buffer);
}

#[cfg(hlsl_out)]
fn write_output_hlsl(
    input: &Input,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    options: &naga::back::hlsl::Options,
    pipeline_constants: &naga::back::PipelineConstants,
    frag_ep: Option<naga::back::hlsl::FragmentEntryPoint>,
) {
    use core::fmt::Write as _;
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

    input.write_output_file("hlsl", "hlsl", buffer);

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

    config.to_file(input.output_path("hlsl", "ron")).unwrap();
}

#[cfg(wgsl_out)]
fn write_output_wgsl(
    input: &Input,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    params: &WgslOutParameters,
) {
    use naga::back::wgsl;

    println!("generating WGSL");

    let mut flags = wgsl::WriterFlags::empty();
    flags.set(wgsl::WriterFlags::EXPLICIT_TYPES, params.explicit_types);

    let string = wgsl::write_string(module, info, flags).expect("WGSL write failed");

    input.write_output_file("wgsl", "wgsl", string);
}

#[cfg(feature = "wgsl-in")]
#[test]
fn convert_snapshots_wgsl() {
    let _ = env_logger::try_init();

    for input in Input::files_in_dir("wgsl", &["wgsl"]) {
        let source = input.read_source();
        // crlf will make the large split output different on different platform
        let source = source.replace('\r', "");

        let params = input.read_parameters();
        let WgslInParameters { parse_doc_comments } = params.wgsl_in;

        let options = naga::front::wgsl::Options { parse_doc_comments };
        let mut frontend = naga::front::wgsl::Frontend::new_with_options(options);
        match frontend.parse(&source) {
            Ok(mut module) => check_targets(&input, &mut module, Some(&source)),
            Err(e) => panic!(
                "{}",
                e.emit_to_string_with_path(&source, input.input_path())
            ),
        }
    }
}

#[cfg(feature = "spv-in")]
#[test]
fn convert_snapshots_spv() {
    use std::process::Command;

    let _ = env_logger::try_init();

    for input in Input::files_in_dir("spv", &["spvasm"]) {
        println!("Assembling '{}'", input.file_name.display());

        let command = Command::new("spirv-as")
            .arg(input.input_path())
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

        let params = input.read_parameters();
        let SpirvInParameters {
            adjust_coordinate_space,
        } = params.spv_in;

        let mut module = naga::front::spv::parse_u8_slice(
            &command.stdout,
            &naga::front::spv::Options {
                adjust_coordinate_space,
                strict_capabilities: true,
                ..Default::default()
            },
        )
        .unwrap();

        check_targets(&input, &mut module, None);
    }
}

#[cfg(feature = "glsl-in")]
#[allow(unused_variables)]
#[test]
fn convert_snapshots_glsl() {
    let _ = env_logger::try_init();

    for input in Input::files_in_dir("glsl", &["vert", "frag", "comp"]) {
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
                &input.read_source(),
            )
            .unwrap();

        check_targets(&input, &mut module, None);
    }
}
