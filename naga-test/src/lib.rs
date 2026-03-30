#![allow(
    dead_code,
    unused_imports,
    reason = "A lot of the code can be unused based on configuration flags; \
        the corresponding warnings aren't helpful."
)]

use core::fmt::Write;

use std::{
    fs,
    path::{Path, PathBuf},
};

use naga::compact::KeepUnused;
use ron::de;

bitflags::bitflags! {
    #[derive(Clone, Copy, serde::Deserialize)]
    #[serde(transparent)]
    #[derive(Debug, Eq, PartialEq)]
    pub struct Targets: u32 {
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
    pub fn non_wgsl_default() -> Self {
        Targets::WGSL
    }

    /// Defaults for `wgsl` snapshots.
    pub fn wgsl_default() -> Self {
        Targets::HLSL | Targets::SPIRV | Targets::GLSL | Targets::METAL | Targets::WGSL
    }
}

#[derive(serde::Deserialize)]
pub struct SpvOutVersion(pub u8, pub u8);
impl Default for SpvOutVersion {
    fn default() -> Self {
        SpvOutVersion(1, 1)
    }
}

#[derive(serde::Deserialize)]
pub struct BindingMapSerialization {
    pub resource_binding: naga::ResourceBinding,
    pub bind_target: naga::back::spv::BindingInfo,
}

pub fn deserialize_binding_map<'de, D>(
    deserializer: D,
) -> Result<naga::back::spv::BindingMap, D::Error>
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
pub struct WriterSharedOptions {
    pub mesh_output_validation: bool,
    pub task_limits: Option<naga::back::TaskDispatchLimits>,
    pub bounds_checks_policies: naga::proc::BoundsCheckPolicies,
}

#[derive(Default, serde::Deserialize)]
#[serde(default)]
pub struct WgslInParameters {
    pub parse_doc_comments: bool,
}
impl From<&WgslInParameters> for naga::front::wgsl::Options {
    fn from(value: &WgslInParameters) -> Self {
        Self {
            parse_doc_comments: value.parse_doc_comments,
            capabilities: naga::valid::Capabilities::all(),
        }
    }
}

#[derive(Default, serde::Deserialize)]
#[serde(default)]
pub struct SpirvInParameters {
    pub adjust_coordinate_space: bool,
}
impl From<&SpirvInParameters> for naga::front::spv::Options {
    fn from(value: &SpirvInParameters) -> Self {
        Self {
            adjust_coordinate_space: value.adjust_coordinate_space,
            ..Default::default()
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(default)]
pub struct SpirvOutParameters {
    pub version: SpvOutVersion,
    pub capabilities: naga::FastHashSet<spirv::Capability>,
    pub debug: bool,
    pub adjust_coordinate_space: bool,
    pub force_point_size: bool,
    pub clamp_frag_depth: bool,
    pub separate_entry_points: bool,
    #[serde(deserialize_with = "deserialize_binding_map")]
    pub binding_map: naga::back::spv::BindingMap,
    pub ray_query_initialization_tracking: bool,
    pub use_storage_input_output_16: bool,
}
impl Default for SpirvOutParameters {
    fn default() -> Self {
        Self {
            version: SpvOutVersion::default(),
            capabilities: naga::FastHashSet::default(),
            debug: false,
            adjust_coordinate_space: false,
            force_point_size: false,
            clamp_frag_depth: false,
            separate_entry_points: false,
            ray_query_initialization_tracking: true,
            use_storage_input_output_16: true,
            binding_map: naga::back::spv::BindingMap::default(),
        }
    }
}
impl SpirvOutParameters {
    pub fn to_options<'a>(
        &'a self,
        shared_info: &WriterSharedOptions,
        debug_info: Option<naga::back::spv::DebugInfo<'a>>,
    ) -> naga::back::spv::Options<'a> {
        use naga::back::spv;
        let mut flags = spv::WriterFlags::LABEL_VARYINGS;
        flags.set(spv::WriterFlags::DEBUG, self.debug);
        flags.set(
            spv::WriterFlags::ADJUST_COORDINATE_SPACE,
            self.adjust_coordinate_space,
        );
        flags.set(spv::WriterFlags::FORCE_POINT_SIZE, self.force_point_size);
        flags.set(spv::WriterFlags::CLAMP_FRAG_DEPTH, self.clamp_frag_depth);
        naga::back::spv::Options {
            lang_version: (self.version.0, self.version.1),
            flags,
            capabilities: if self.capabilities.is_empty() {
                None
            } else {
                Some(self.capabilities.clone())
            },
            bounds_check_policies: shared_info.bounds_checks_policies,
            fake_missing_bindings: true,
            binding_map: self.binding_map.clone(),
            zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::Polyfill,
            force_loop_bounding: true,
            ray_query_initialization_tracking: true,
            debug_info,
            use_storage_input_output_16: self.use_storage_input_output_16,
            task_dispatch_limits: shared_info.task_limits,
            mesh_shader_primitive_indices_clamp: shared_info.mesh_output_validation,
        }
    }
}

#[derive(Default, serde::Deserialize)]
#[serde(default)]
pub struct WgslOutParameters {
    pub explicit_types: bool,
}
impl From<&WgslOutParameters> for naga::back::wgsl::WriterFlags {
    fn from(value: &WgslOutParameters) -> Self {
        let mut flags = Self::empty();
        flags.set(Self::EXPLICIT_TYPES, value.explicit_types);
        flags
    }
}

#[derive(Default, serde::Deserialize)]
pub struct FragmentModule {
    pub path: String,
    pub entry_point: String,
}

#[derive(Default, serde::Deserialize)]
#[serde(default)]
pub struct Parameters {
    // -- validation options --
    //
    // Capabilities to enable. Defaults to `Capabilities::default()`.
    pub capabilities: Option<naga::valid::Capabilities>,

    // -- wgsl-in options --
    #[serde(rename = "wgsl-in")]
    pub wgsl_in: WgslInParameters,

    // -- spirv-in options --
    #[serde(rename = "spv-in")]
    pub spv_in: SpirvInParameters,

    // -- SPIR-V options --
    pub spv: SpirvOutParameters,

    /// Defaults to [`Targets::non_wgsl_default()`] for `spv` and `glsl` snapshots,
    /// and [`Targets::wgsl_default()`] for `wgsl` snapshots.
    pub targets: Option<Targets>,

    // -- MSL options --
    pub msl: naga::back::msl::Options,
    #[serde(default)]
    pub msl_pipeline: naga::back::msl::PipelineOptions,

    // -- GLSL options --
    pub glsl: naga::back::glsl::Options,
    pub glsl_exclude_list: naga::FastHashSet<String>,
    pub glsl_multiview: Option<core::num::NonZeroU32>,

    // -- HLSL options --
    pub hlsl: naga::back::hlsl::Options,

    // -- WGSL options --
    pub wgsl: WgslOutParameters,

    // -- General options --

    // Allow backends to be aware of the fragment module.
    // Is the name of a WGSL file in the same directory as the test file.
    pub fragment_module: Option<FragmentModule>,

    pub bounds_check_policies: naga::proc::BoundsCheckPolicies,
    pub pipeline_constants: naga::back::PipelineConstants,

    pub mesh_output_validation: bool,
    #[serde(default = "default_task_limits")]
    pub task_limits: Option<naga::back::TaskDispatchLimits>,
}

fn default_task_limits() -> Option<naga::back::TaskDispatchLimits> {
    Some(naga::back::TaskDispatchLimits {
        max_mesh_workgroups_per_dim: 256,
        max_mesh_workgroups_total: 1024,
    })
}

/// Information about a shader input file.
#[derive(Debug)]
pub struct Input {
    /// The subdirectory of `tests/in` to which this input belongs, if any.
    ///
    /// If the subdirectory is omitted, we assume that the output goes
    /// to "wgsl".
    pub subdirectory: PathBuf,

    /// The input filename name, without a directory.
    pub file_name: PathBuf,

    /// True if output filenames should add the output extension on top of
    /// `file_name`'s existing extension, rather than replacing it.
    ///
    /// This is used by `convert_snapshots_glsl`, which wants to take input files
    /// like `210-bevy-2d-shader.frag` and just add `.wgsl` to it, producing
    /// `210-bevy-2d-shader.frag.wgsl`.
    pub keep_input_extension: bool,
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
    pub fn new(subdirectory: &str, name: &str, extension: &str) -> Input {
        Input {
            subdirectory: PathBuf::from(subdirectory),
            // Don't wipe out any extensions on `name`, as
            // `with_extension` would do.
            file_name: PathBuf::from(format!("{name}.{extension}")),
            keep_input_extension: false,
        }
    }

    /// Return an iterator that produces an `Input` for each entry in `subdirectory`.
    pub fn files_in_dir<'a>(
        subdirectory: &'a str,
        file_extensions: &'a [&'a str],
        dir_in: &str,
    ) -> impl Iterator<Item = Input> + 'a {
        let input_directory = Path::new(dir_in).join(subdirectory);

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
    pub fn input_directory(&self, dir_in: &str) -> PathBuf {
        Path::new(dir_in).join(&self.subdirectory)
    }

    /// Return the path to the output directory.
    pub fn output_directory(subdirectory: &str, dir_out: &str) -> PathBuf {
        Path::new(dir_out).join(subdirectory)
    }

    /// Return the path to the input file.
    pub fn input_path(&self, dir_in: &str) -> PathBuf {
        let mut input = self.input_directory(dir_in);
        input.push(&self.file_name);
        input
    }

    pub fn output_path(&self, subdirectory: &str, extension: &str, dir_out: &str) -> PathBuf {
        let mut output = Self::output_directory(subdirectory, dir_out);
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
    pub fn read_source(&self, dir_in: &str, print: bool) -> String {
        if print {
            println!("Processing '{}'", self.file_name.display());
        }
        let input_path = self.input_path(dir_in);
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
    pub fn read_bytes(&self, dir_in: &str, print: bool) -> Vec<u8> {
        if print {
            println!("Processing '{}'", self.file_name.display());
        }
        let input_path = self.input_path(dir_in);
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

    pub fn bytes(&self, dir_in: &str) -> u64 {
        let input_path = self.input_path(dir_in);
        std::fs::metadata(input_path).unwrap().len()
    }

    /// Return this input's parameter file, parsed.
    pub fn read_parameters(&self, dir_in: &str) -> Parameters {
        let mut param_path = self.input_path(dir_in);
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
            match self
                .input_path(dir_in)
                .extension()
                .unwrap()
                .to_str()
                .unwrap()
            {
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
    pub fn write_output_file(
        &self,
        subdirectory: &str,
        extension: &str,
        data: impl AsRef<[u8]>,
        dir_out: &str,
    ) {
        let output_path = self.output_path(subdirectory, extension, dir_out);
        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        if let Err(err) = fs::write(&output_path, data) {
            panic!("Error writing {}: {}", output_path.display(), err);
        }
    }
}
