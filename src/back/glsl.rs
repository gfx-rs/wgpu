//! OpenGL shading language backend
//!
//! The main structure is [`Writer`](struct.Writer.html), it maintains internal state that is used
//! to output a `Module` into glsl
//!
//! # Supported versions
//! ### Core
//! - 330
//! - 400
//! - 410
//! - 420
//! - 430
//! - 450
//! - 460
//!
//! ### ES
//! - 300
//! - 310
//!

use crate::{
    proc::{
        CallGraph, CallGraphBuilder, Interface, NameKey, Namer, ResolveContext, ResolveError,
        Typifier, Visitor,
    },
    Arena, ArraySize, BinaryOperator, BuiltIn, ConservativeDepth, Constant, ConstantInner,
    DerivativeAxis, Expression, FastHashMap, Function, FunctionOrigin, GlobalVariable, Handle,
    ImageClass, Interpolation, IntrinsicFunction, LocalVariable, Module, ScalarKind, ShaderStage,
    Statement, StorageAccess, StorageClass, StorageFormat, StructMember, Type, TypeInner,
    UnaryOperator,
};
use std::{
    cmp::Ordering,
    fmt::{self, Error as FmtError},
    io::{Error as IoError, Write},
};

/// Contains a constant with a slice of all the reserved keywords RESERVED_KEYWORDS
mod keywords;

const SUPPORTED_CORE_VERSIONS: &[u16] = &[330, 400, 410, 420, 430, 440, 450];
const SUPPORTED_ES_VERSIONS: &[u16] = &[300, 310];

#[derive(Debug)]
pub enum Error {
    FormatError(FmtError),
    IoError(IoError),
    Type(ResolveError),
    Custom(String),
}

impl From<FmtError> for Error {
    fn from(err: FmtError) -> Self {
        Error::FormatError(err)
    }
}

impl From<IoError> for Error {
    fn from(err: IoError) -> Self {
        Error::IoError(err)
    }
}

impl From<ResolveError> for Error {
    fn from(err: ResolveError) -> Self {
        Error::Type(err)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::FormatError(err) => write!(f, "Formatting error {}", err),
            Error::IoError(err) => write!(f, "Io error: {}", err),
            Error::Type(err) => write!(f, "Type error: {:?}", err),
            Error::Custom(err) => write!(f, "{}", err),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Version {
    Desktop(u16),
    Embedded(u16),
}

impl Version {
    fn is_es(&self) -> bool {
        match self {
            Version::Desktop(_) => false,
            Version::Embedded(_) => true,
        }
    }

    fn is_supported(&self) -> bool {
        match self {
            Version::Desktop(v) => SUPPORTED_CORE_VERSIONS.contains(v),
            Version::Embedded(v) => SUPPORTED_ES_VERSIONS.contains(v),
        }
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (*self, *other) {
            (Version::Desktop(x), Version::Desktop(y)) => Some(x.cmp(&y)),
            (Version::Embedded(x), Version::Embedded(y)) => Some(x.cmp(&y)),
            _ => None,
        }
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Version::Desktop(v) => write!(f, "{} core", v),
            Version::Embedded(v) => write!(f, "{} es", v),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub version: Version,
    pub entry_point: (ShaderStage, String),
}

#[derive(Debug, Clone)]
pub struct TextureMapping {
    pub texture: Handle<GlobalVariable>,
    pub sampler: Option<Handle<GlobalVariable>>,
}

bitflags::bitflags! {
    struct Features: u32 {
        const BUFFER_STORAGE = 1;
        const ARRAY_OF_ARRAYS = 1 << 1;
        const DOUBLE_TYPE = 1 << 2;
        const FULL_IMAGE_FORMATS = 1 << 3;
        const MULTISAMPLED_TEXTURES = 1 << 4;
        const MULTISAMPLED_TEXTURE_ARRAYS = 1 << 5;
        const CUBE_TEXTURES_ARRAY = 1 << 6;
        const COMPUTE_SHADER = 1 << 7;
        const IMAGE_LOAD_STORE = 1 << 8;
        const CONSERVATIVE_DEPTH = 1 << 9;
        const TEXTURE_1D = 1 << 10;
        const PUSH_CONSTANT = 1 << 11;
    }
}

struct FeaturesManager(Features);

impl FeaturesManager {
    pub fn new() -> Self {
        Self(Features::empty())
    }

    pub fn request(&mut self, features: Features) {
        self.0 |= features
    }

    #[allow(clippy::collapsible_if)]
    pub fn write(&self, version: Version, mut out: impl Write) -> Result<(), Error> {
        if self.0.contains(Features::COMPUTE_SHADER) {
            if version < Version::Embedded(310) || version < Version::Desktop(420) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support compute shaders",
                    version
                )));
            }

            if !version.is_es() {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_compute_shader.txt
                writeln!(out, "#extension GL_ARB_compute_shader : require")?;
            }
        }

        if self.0.contains(Features::BUFFER_STORAGE) {
            if version < Version::Embedded(310) || version < Version::Desktop(400) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support buffer storage class",
                    version
                )));
            }

            if let Version::Desktop(_) = version {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_storage_buffer_object.txt
                writeln!(
                    out,
                    "#extension GL_ARB_shader_storage_buffer_object : require"
                )?;
            }
        }

        if self.0.contains(Features::DOUBLE_TYPE) {
            if version.is_es() || version < Version::Desktop(150) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support doubles",
                    version
                )));
            }

            if version < Version::Desktop(400) {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_gpu_shader_fp64.txt
                writeln!(out, "#extension GL_ARB_gpu_shader_fp64 : require")?;
            }
        }

        if self.0.contains(Features::CUBE_TEXTURES_ARRAY) {
            if version < Version::Embedded(310) || version < Version::Desktop(130) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support cube map array textures",
                    version
                )));
            }

            if version.is_es() {
                // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_cube_map_array.txt
                writeln!(out, "#extension GL_EXT_texture_cube_map_array : require")?;
            } else if version < Version::Desktop(400) {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_texture_cube_map_array.txt
                writeln!(out, "#extension GL_ARB_texture_cube_map_array : require")?;
            }
        }

        if self.0.contains(Features::MULTISAMPLED_TEXTURES) {
            if version < Version::Embedded(300) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support multi sampled textures",
                    version
                )));
            }
        }

        if self.0.contains(Features::MULTISAMPLED_TEXTURE_ARRAYS) {
            if version < Version::Embedded(310) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support multi sampled texture arrays",
                    version
                )));
            }

            if version.is_es() {
                // https://www.khronos.org/registry/OpenGL/extensions/OES/OES_texture_storage_multisample_2d_array.txt
                writeln!(
                    out,
                    "#extension GL_OES_texture_storage_multisample_2d_array : require"
                )?;
            }
        }

        if self.0.contains(Features::ARRAY_OF_ARRAYS) {
            if version < Version::Embedded(310) || version < Version::Desktop(120) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't arrays of arrays",
                    version
                )));
            }

            if version < Version::Desktop(430) {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_arrays_of_arrays.txt
                writeln!(out, "#extension ARB_arrays_of_arrays : require")?;
            }
        }

        if self.0.contains(Features::IMAGE_LOAD_STORE) {
            if version < Version::Embedded(310) || version < Version::Desktop(130) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support images load/stores",
                    version
                )));
            }

            if self.0.contains(Features::FULL_IMAGE_FORMATS) && version.is_es() {
                // https://www.khronos.org/registry/OpenGL/extensions/NV/NV_image_formats.txt
                writeln!(out, "#extension GL_NV_image_formats : require")?;
            }

            if version < Version::Desktop(420) {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_image_load_store.txt
                writeln!(out, "#extension GL_ARB_shader_image_load_store : require")?;
            }
        }

        if self.0.contains(Features::CONSERVATIVE_DEPTH) {
            if version < Version::Embedded(300) || version < Version::Desktop(130) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support conservative depth",
                    version
                )));
            }

            if version.is_es() {
                // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_conservative_depth.txt
                writeln!(out, "#extension GL_EXT_conservative_depth : require")?;
            }

            if version < Version::Desktop(420) {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_conservative_depth.txt
                writeln!(out, "#extension GL_ARB_conservative_depth : require")?;
            }
        }

        if self.0.contains(Features::TEXTURE_1D) {
            if version.is_es() {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support 1d textures",
                    version
                )));
            }
        }

        Ok(())
    }
}

enum FunctionType {
    Function(Handle<Function>),
    EntryPoint(crate::proc::EntryPointIndex),
}

struct FunctionCtx<'a, 'b> {
    func: FunctionType,
    expressions: &'a Arena<Expression>,
    typifier: &'b Typifier,
}

impl<'a, 'b> FunctionCtx<'a, 'b> {
    fn name_key(&self, local: Handle<LocalVariable>) -> NameKey {
        match self.func {
            FunctionType::Function(handle) => NameKey::FunctionLocal(handle, local),
            FunctionType::EntryPoint(idx) => NameKey::EntryPointLocal(idx, local),
        }
    }

    fn get_arg<'c>(&self, arg: u32, names: &'c FastHashMap<NameKey, String>) -> &'c str {
        match self.func {
            FunctionType::Function(handle) => &names[&NameKey::FunctionArgument(handle, arg)],
            FunctionType::EntryPoint(_) => unreachable!(),
        }
    }
}

/// Helper structure that generates a number
#[derive(Default)]
struct IdGenerator(u32);

impl IdGenerator {
    fn generate(&mut self) -> u32 {
        let ret = self.0;
        self.0 += 1;
        ret
    }
}

/// Main structure of the glsl backend responsible for all code generation
pub struct Writer<'a, W> {
    // Inputs
    module: &'a Module,
    out: W,
    options: &'a Options,

    // Internal State
    features: FeaturesManager,
    names: FastHashMap<NameKey, String>,
    entry_point: &'a crate::EntryPoint,
    entry_point_idx: crate::proc::EntryPointIndex,
    call_graph: CallGraph,

    /// Used to generate a unique number for blocks
    block_id: IdGenerator,
}

impl<'a, W: Write> Writer<'a, W> {
    pub fn new(out: W, module: &'a Module, options: &'a Options) -> Result<Self, Error> {
        if !options.version.is_supported() {
            return Err(Error::Custom(format!(
                "Version not supported {}",
                options.version
            )));
        }

        let (ep_idx, ep) = module
            .entry_points
            .iter()
            .enumerate()
            .find_map(|(i, (key, entry_point))| {
                Some((i as u16, entry_point)).filter(|_| &options.entry_point == key)
            })
            .ok_or_else(|| Error::Custom(String::from("Entry point not found")))?;

        let mut names = FastHashMap::default();
        Namer::process(module, keywords::RESERVED_KEYWORDS, &mut names);

        let call_graph = CallGraphBuilder {
            functions: &module.functions,
        }
        .process(&ep.function);

        let mut this = Self {
            module,
            out,
            options,

            features: FeaturesManager::new(),
            names,
            entry_point: ep,
            entry_point_idx: ep_idx,
            call_graph,

            block_id: IdGenerator::default(),
        };

        this.collect_required_features()?;

        Ok(this)
    }

    fn collect_required_features(&mut self) -> Result<(), Error> {
        let stage = self.options.entry_point.0;

        if let Some(depth_test) = self.entry_point.early_depth_test {
            self.features.request(Features::IMAGE_LOAD_STORE);

            if depth_test.conservative.is_some() {
                self.features.request(Features::CONSERVATIVE_DEPTH);
            }
        }

        if let ShaderStage::Compute = stage {
            self.features.request(Features::COMPUTE_SHADER)
        }

        for (_, ty) in self.module.types.iter() {
            match ty.inner {
                TypeInner::Scalar { kind, width } => self.scalar_required_features(kind, width),
                TypeInner::Vector { kind, width, .. } => self.scalar_required_features(kind, width),
                TypeInner::Matrix { .. } => self.scalar_required_features(ScalarKind::Float, 8),
                TypeInner::Array { base, .. } => {
                    if let TypeInner::Array { .. } = self.module.types[base].inner {
                        self.features.request(Features::ARRAY_OF_ARRAYS)
                    }
                }
                TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => {
                    if arrayed && dim == crate::ImageDimension::Cube {
                        self.features.request(Features::CUBE_TEXTURES_ARRAY)
                    } else if dim == crate::ImageDimension::D1 {
                        self.features.request(Features::TEXTURE_1D)
                    }

                    match class {
                        ImageClass::Sampled { multi: true, .. } => {
                            self.features.request(Features::MULTISAMPLED_TEXTURES);
                            if arrayed {
                                self.features.request(Features::MULTISAMPLED_TEXTURE_ARRAYS);
                            }
                        }
                        ImageClass::Storage(format) => match format {
                            StorageFormat::R8Unorm
                            | StorageFormat::R8Snorm
                            | StorageFormat::R8Uint
                            | StorageFormat::R8Sint
                            | StorageFormat::R16Uint
                            | StorageFormat::R16Sint
                            | StorageFormat::R16Float
                            | StorageFormat::Rg8Unorm
                            | StorageFormat::Rg8Snorm
                            | StorageFormat::Rg8Uint
                            | StorageFormat::Rg8Sint
                            | StorageFormat::Rg16Uint
                            | StorageFormat::Rg16Sint
                            | StorageFormat::Rg16Float
                            | StorageFormat::Rgb10a2Unorm
                            | StorageFormat::Rg11b10Float
                            | StorageFormat::Rg32Uint
                            | StorageFormat::Rg32Sint
                            | StorageFormat::Rg32Float => {
                                self.features.request(Features::FULL_IMAGE_FORMATS)
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        for (_, global) in self.module.global_variables.iter() {
            match global.class {
                StorageClass::WorkGroup => self.features.request(Features::COMPUTE_SHADER),
                StorageClass::Storage => self.features.request(Features::BUFFER_STORAGE),
                StorageClass::PushConstant => self.features.request(Features::PUSH_CONSTANT),
                _ => {}
            }
        }

        Ok(())
    }

    fn scalar_required_features(&mut self, kind: ScalarKind, width: crate::Bytes) {
        if kind == ScalarKind::Float && width == 8 {
            self.features.request(Features::DOUBLE_TYPE);
        }
    }

    pub fn write(&mut self) -> Result<FastHashMap<String, TextureMapping>, Error> {
        let es = self.options.version.is_es();

        writeln!(self.out, "#version {}", self.options.version)?;
        self.features.write(self.options.version, &mut self.out)?;
        writeln!(self.out)?;

        if es {
            writeln!(self.out, "precision highp float;\n")?;
        }

        if let Some(depth_test) = self.entry_point.early_depth_test {
            writeln!(self.out, "layout(early_fragment_tests) in;\n")?;

            if let Some(conservative) = depth_test.conservative {
                writeln!(
                    self.out,
                    "layout (depth_{}) out float gl_FragDepth;\n",
                    match conservative {
                        ConservativeDepth::GreaterEqual => "greater",
                        ConservativeDepth::LessEqual => "less",
                        ConservativeDepth::Unchanged => "unchanged",
                    }
                )?;
            }
        }

        for (handle, ty) in self.module.types.iter() {
            if let TypeInner::Struct { ref members } = ty.inner {
                self.write_struct(handle, members)?
            }
        }

        writeln!(self.out)?;

        let texture_mappings = self.collect_texture_mapping(
            self.call_graph
                .raw_nodes()
                .iter()
                .map(|node| &self.module.functions[node.weight])
                .chain(std::iter::once(&self.entry_point.function)),
        )?;

        for (handle, global) in self
            .module
            .global_variables
            .iter()
            .zip(&self.entry_point.function.global_usage)
            .filter_map(|(global, usage)| Some(global).filter(|_| !usage.is_empty()))
        {
            if let Some(crate::Binding::BuiltIn(_)) = global.binding {
                continue;
            }

            match self.module.types[global.ty].inner {
                TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => {
                    if let TypeInner::Image {
                        class: ImageClass::Storage(format),
                        ..
                    } = self.module.types[global.ty].inner
                    {
                        write!(self.out, "layout({}) ", glsl_storage_format(format))?;
                    }

                    if global.storage_access == StorageAccess::LOAD {
                        write!(self.out, "readonly ")?;
                    } else if global.storage_access == StorageAccess::STORE {
                        write!(self.out, "writeonly ")?;
                    }

                    write!(self.out, "uniform ")?;

                    self.write_image_type(dim, arrayed, class)?;

                    writeln!(
                        self.out,
                        " {};",
                        self.names[&NameKey::GlobalVariable(handle)]
                    )?
                }
                TypeInner::Sampler { .. } => continue,
                _ => self.write_global(handle, global)?,
            }
        }

        writeln!(self.out)?;

        // Sort the graph topologically so that functions calls are valid
        // It's impossible for this to panic because the IR forbids cycles
        let functions = petgraph::algo::toposort(&self.call_graph, None).unwrap();

        for node in functions {
            let handle = self.call_graph[node];
            let name = self.names[&NameKey::Function(handle)].clone();
            self.write_function(
                FunctionType::Function(handle),
                &self.module.functions[handle],
                name,
            )?;
        }

        self.write_function(
            FunctionType::EntryPoint(self.entry_point_idx),
            &self.entry_point.function,
            "main",
        )?;

        Ok(texture_mappings)
    }

    fn write_global(
        &mut self,
        handle: Handle<GlobalVariable>,
        global: &GlobalVariable,
    ) -> Result<(), Error> {
        if global.storage_access == StorageAccess::LOAD {
            write!(self.out, "readonly ")?;
        } else if global.storage_access == StorageAccess::STORE {
            write!(self.out, "writeonly ")?;
        }

        if let Some(interpolation) = global.interpolation {
            match (self.options.entry_point.0, global.class) {
                (ShaderStage::Fragment, StorageClass::Input)
                | (ShaderStage::Vertex, StorageClass::Output) => {
                    write!(self.out, "{} ", glsl_interpolation(interpolation)?)?;
                }
                _ => (),
            };
        }

        let block = match global.class {
            StorageClass::Storage | StorageClass::Uniform => {
                let block_name = self.names[&NameKey::Type(global.ty)].clone();

                Some(block_name)
            }
            _ => None,
        };

        write!(self.out, "{} ", glsl_storage_class(global.class))?;

        self.write_type(global.ty, block)?;

        let name = &self.names[&NameKey::GlobalVariable(handle)];
        writeln!(self.out, " {};", name)?;

        Ok(())
    }

    fn write_function<N: AsRef<str>>(
        &mut self,
        ty: FunctionType,
        func: &Function,
        name: N,
    ) -> Result<(), Error> {
        let mut typifier = Typifier::new();

        typifier.resolve_all(
            &func.expressions,
            &self.module.types,
            &ResolveContext {
                constants: &self.module.constants,
                global_vars: &self.module.global_variables,
                local_vars: &func.local_variables,
                functions: &self.module.functions,
                arguments: &func.arguments,
            },
        )?;

        let ctx = FunctionCtx {
            func: ty,
            expressions: &func.expressions,
            typifier: &typifier,
        };

        self.write_fn_header(name.as_ref(), func, &ctx)?;
        writeln!(self.out, " {{",)?;

        for (handle, local) in func.local_variables.iter() {
            write!(self.out, "\t")?;
            self.write_type(local.ty, None)?;

            write!(self.out, " {}", self.names[&ctx.name_key(handle)])?;

            if let Some(init) = local.init {
                write!(self.out, " = ",)?;

                self.write_constant(&self.module.constants[init])?;
            }

            writeln!(self.out, ";")?
        }

        writeln!(self.out)?;

        for sta in func.body.iter() {
            self.write_stmt(sta, &ctx, 1)?;
        }

        Ok(writeln!(self.out, "}}")?)
    }

    fn write_slice<T, F: FnMut(&mut Self, u32, &T) -> Result<(), Error>>(
        &mut self,
        data: &[T],
        mut f: F,
    ) -> Result<(), Error> {
        for (i, item) in data.iter().enumerate() {
            f(self, i as u32, item)?;

            if i != data.len().saturating_sub(1) {
                write!(self.out, ",")?;
            }
        }

        Ok(())
    }

    fn write_fn_header(
        &mut self,
        name: &str,
        func: &Function,
        ctx: &FunctionCtx<'_, '_>,
    ) -> Result<(), Error> {
        if let Some(ty) = func.return_type {
            self.write_type(ty, None)?;
        } else {
            write!(self.out, "void")?;
        }

        write!(self.out, " {}(", name)?;

        self.write_slice(&func.arguments, |this, i, arg| {
            this.write_type(arg.ty, None)?;

            let name = ctx.get_arg(i, &this.names);

            Ok(write!(this.out, " {}", name)?)
        })?;

        write!(self.out, ")")?;

        Ok(())
    }

    fn write_type(&mut self, ty: Handle<Type>, block: Option<String>) -> Result<(), Error> {
        match self.module.types[ty].inner {
            TypeInner::Scalar { kind, width } => {
                write!(self.out, "{}", glsl_scalar(kind, width)?.full)?
            }
            TypeInner::Vector { size, kind, width } => write!(
                self.out,
                "{}vec{}",
                glsl_scalar(kind, width)?.prefix,
                size as u8
            )?,
            TypeInner::Matrix {
                columns,
                rows,
                width,
            } => write!(
                self.out,
                "{}mat{}x{}",
                glsl_scalar(ScalarKind::Float, width)?.prefix,
                columns as u8,
                rows as u8
            )?,
            TypeInner::Pointer { base, .. } => self.write_type(base, None)?,
            TypeInner::Array { base, size, .. } => {
                self.write_type(base, None)?;

                write!(self.out, "[")?;
                self.write_array_size(size)?;
                write!(self.out, "]")?
            }
            TypeInner::Struct { ref members } => {
                if let Some(name) = block {
                    writeln!(self.out, "{}_block_{} {{", name, self.block_id.generate())?;

                    for (idx, member) in members.iter().enumerate() {
                        self.write_type(member.ty, None)?;

                        writeln!(
                            self.out,
                            " {};",
                            &self.names[&NameKey::StructMember(ty, idx as u32)]
                        )?;
                    }

                    write!(self.out, "}}")?
                } else {
                    write!(self.out, "{}", &self.names[&NameKey::Type(ty)])?
                }
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    fn write_image_type(
        &mut self,
        dim: crate::ImageDimension,
        arrayed: bool,
        class: ImageClass,
    ) -> Result<(), Error> {
        let (base, kind, ms, comparison) = match class {
            ImageClass::Sampled { kind, multi: true } => ("sampler", kind, "MS", ""),
            ImageClass::Sampled { kind, multi: false } => ("sampler", kind, "", ""),
            ImageClass::Depth => ("sampler", crate::ScalarKind::Float, "", "Shadow"),
            ImageClass::Storage(format) => ("image", format.into(), "", ""),
        };

        Ok(write!(
            self.out,
            "{}{}{}{}{}{}",
            glsl_scalar(kind, 4)?.prefix,
            base,
            ImageDimension(dim),
            ms,
            if arrayed { "Array" } else { "" },
            comparison
        )?)
    }

    fn write_array_size(&mut self, size: ArraySize) -> Result<(), Error> {
        match size {
            ArraySize::Constant(const_handle) => match self.module.constants[const_handle].inner {
                ConstantInner::Uint(size) => write!(self.out, "{}", size)?,
                _ => unreachable!(),
            },
            ArraySize::Dynamic => (),
        }

        Ok(())
    }

    fn collect_texture_mapping(
        &self,
        functions: impl Iterator<Item = &'a Function>,
    ) -> Result<FastHashMap<String, TextureMapping>, Error> {
        let mut mappings = FastHashMap::default();

        for func in functions {
            let mut interface = Interface {
                expressions: &func.expressions,
                local_variables: &func.local_variables,
                visitor: TextureMappingVisitor {
                    names: &self.names,
                    expressions: &func.expressions,
                    map: &mut mappings,
                    error: None,
                },
            };
            interface.traverse(&func.body);

            if let Some(error) = interface.visitor.error {
                return Err(error);
            }
        }

        Ok(mappings)
    }

    fn write_struct(
        &mut self,
        handle: Handle<Type>,
        members: &[StructMember],
    ) -> Result<(), Error> {
        writeln!(self.out, "struct {} {{", self.names[&NameKey::Type(handle)])?;

        for (idx, member) in members.iter().enumerate() {
            write!(self.out, "\t")?;
            self.write_type(member.ty, None)?;
            writeln!(
                self.out,
                " {};",
                self.names[&NameKey::StructMember(handle, idx as u32)]
            )?;
        }

        writeln!(self.out, "}};")?;
        Ok(())
    }

    fn write_stmt(
        &mut self,
        sta: &Statement,
        ctx: &FunctionCtx<'_, '_>,
        indent: usize,
    ) -> Result<(), Error> {
        write!(self.out, "{}", "\t".repeat(indent))?;

        match sta {
            Statement::Block(block) => {
                writeln!(self.out, "{{")?;
                for sta in block.iter() {
                    self.write_stmt(sta, ctx, indent + 1)?
                }
                writeln!(self.out, "{}}}", "\t".repeat(indent))?
            }
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                write!(self.out, "if(")?;
                self.write_expr(*condition, ctx)?;
                writeln!(self.out, ") {{")?;

                for sta in accept {
                    self.write_stmt(sta, ctx, indent + 1)?;
                }

                if !reject.is_empty() {
                    writeln!(self.out, "{}}} else {{", "\t".repeat(indent))?;

                    for sta in reject {
                        self.write_stmt(sta, ctx, indent + 1)?;
                    }
                }

                writeln!(self.out, "{}}}", "\t".repeat(indent))?
            }
            Statement::Switch {
                selector,
                cases,
                default,
            } => {
                write!(self.out, "switch(")?;
                self.write_expr(*selector, ctx)?;
                writeln!(self.out, ") {{")?;

                for (label, (block, fallthrough)) in cases {
                    writeln!(self.out, "{}case {}:", "\t".repeat(indent + 1), label)?;

                    for sta in block {
                        self.write_stmt(sta, ctx, indent + 2)?;
                    }

                    if fallthrough.is_none() {
                        writeln!(self.out, "{}break;", "\t".repeat(indent + 2))?;
                    }
                }

                if !default.is_empty() {
                    writeln!(self.out, "{}default:", "\t".repeat(indent + 1))?;

                    for sta in default {
                        self.write_stmt(sta, ctx, indent + 2)?;
                    }
                }

                writeln!(self.out, "{}}}", "\t".repeat(indent))?
            }
            Statement::Loop { body, continuing } => {
                writeln!(self.out, "while(true) {{")?;

                for sta in body.iter().chain(continuing.iter()) {
                    self.write_stmt(sta, ctx, indent + 1)?;
                }

                writeln!(self.out, "{}}}", "\t".repeat(indent))?
            }
            Statement::Break => writeln!(self.out, "break;")?,
            Statement::Continue => writeln!(self.out, "continue;")?,
            Statement::Return { value } => {
                write!(self.out, "return")?;
                if let Some(expr) = value {
                    write!(self.out, " ")?;
                    self.write_expr(*expr, ctx)?;
                }
                writeln!(self.out, ";")?;
            }
            Statement::Kill => writeln!(self.out, "discard;")?,
            Statement::Store { pointer, value } => {
                self.write_expr(*pointer, ctx)?;
                write!(self.out, " = ")?;
                self.write_expr(*value, ctx)?;
                writeln!(self.out, ";")?
            }
        }

        Ok(())
    }

    fn write_expr(
        &mut self,
        expr: Handle<Expression>,
        ctx: &FunctionCtx<'_, '_>,
    ) -> Result<(), Error> {
        match ctx.expressions[expr] {
            Expression::Access { base, index } => {
                self.write_expr(base, ctx)?;
                write!(self.out, "[")?;
                self.write_expr(index, ctx)?;
                write!(self.out, "]")?
            }
            Expression::AccessIndex { base, index } => {
                self.write_expr(base, ctx)?;

                match ctx.typifier.get(base, &self.module.types) {
                    TypeInner::Vector { .. }
                    | TypeInner::Matrix { .. }
                    | TypeInner::Array { .. } => write!(self.out, "[{}]", index)?,
                    TypeInner::Struct { .. } => {
                        let ty = ctx.typifier.get_handle(base).unwrap();

                        write!(
                            self.out,
                            ".{}",
                            &self.names[&NameKey::StructMember(ty, index)]
                        )?
                    }
                    ref other => return Err(Error::Custom(format!("Cannot index {:?}", other))),
                }
            }
            Expression::Constant(constant) => {
                self.write_constant(&self.module.constants[constant])?
            }
            Expression::Compose { ty, ref components } => {
                match self.module.types[ty].inner {
                    TypeInner::Vector { .. }
                    | TypeInner::Matrix { .. }
                    | TypeInner::Array { .. }
                    | TypeInner::Struct { .. } => self.write_type(ty, None)?,
                    _ => unreachable!(),
                }

                write!(self.out, "(")?;
                self.write_slice(components, |this, _, arg| this.write_expr(*arg, ctx))?;
                write!(self.out, ")")?
            }
            Expression::FunctionArgument(pos) => {
                write!(self.out, "{}", ctx.get_arg(pos, &self.names))?
            }
            Expression::GlobalVariable(handle) => {
                if let Some(crate::Binding::BuiltIn(built_in)) =
                    self.module.global_variables[handle].binding
                {
                    write!(self.out, "{}", glsl_built_in(built_in))?
                } else {
                    write!(
                        self.out,
                        "{}",
                        &self.names[&NameKey::GlobalVariable(handle)]
                    )?
                }
            }
            Expression::LocalVariable(handle) => {
                write!(self.out, "{}", self.names[&ctx.name_key(handle)])?
            }
            Expression::Load { pointer } => self.write_expr(pointer, ctx)?,
            Expression::ImageSample {
                image,
                coordinate,
                level,
                depth_ref,
                ..
            } => {
                //TODO: handle MS
                write!(
                    self.out,
                    "{}(",
                    match level {
                        crate::SampleLevel::Auto | crate::SampleLevel::Bias(_) => "texture",
                        crate::SampleLevel::Zero | crate::SampleLevel::Exact(_) => "textureLod",
                    }
                )?;
                self.write_expr(image, ctx)?;
                write!(self.out, ", ")?;

                let size = match *ctx.typifier.get(coordinate, &self.module.types) {
                    TypeInner::Vector { size, .. } => size,
                    ref other => {
                        return Err(Error::Custom(format!(
                            "Cannot sample with coordinates of type {:?}",
                            other
                        )))
                    }
                };

                if let Some(depth_ref) = depth_ref {
                    write!(self.out, "vec{}(", size as u8 + 1)?;
                    self.write_expr(coordinate, ctx)?;
                    write!(self.out, ", ")?;
                    self.write_expr(depth_ref, ctx)?;
                    write!(self.out, ")")?
                } else {
                    self.write_expr(coordinate, ctx)?
                }

                match level {
                    crate::SampleLevel::Auto => (),
                    crate::SampleLevel::Zero => write!(self.out, ", 0")?,
                    crate::SampleLevel::Exact(expr) | crate::SampleLevel::Bias(expr) => {
                        write!(self.out, ", ")?;
                        self.write_expr(expr, ctx)?;
                    }
                }

                write!(self.out, ")")?
            }
            Expression::ImageLoad {
                image,
                coordinate,
                index,
            } => {
                let class = match ctx.typifier.get(image, &self.module.types) {
                    TypeInner::Image { class, .. } => class,
                    _ => unreachable!(),
                };

                match class {
                    ImageClass::Sampled { .. } => write!(self.out, "texelFetch(")?,
                    ImageClass::Storage(_) => write!(self.out, "imageLoad(")?,
                    ImageClass::Depth => todo!(),
                }

                self.write_expr(image, ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(coordinate, ctx)?;

                match class {
                    ImageClass::Sampled { .. } => {
                        write!(self.out, ", ")?;
                        self.write_expr(index.unwrap(), ctx)?;
                        write!(self.out, ")")?
                    }
                    ImageClass::Storage(_) => write!(self.out, ")")?,
                    ImageClass::Depth => todo!(),
                }
            }
            Expression::Unary { op, expr } => {
                write!(
                    self.out,
                    "({} ",
                    match op {
                        UnaryOperator::Negate => "-",
                        UnaryOperator::Not => match *ctx.typifier.get(expr, &self.module.types) {
                            TypeInner::Scalar {
                                kind: ScalarKind::Sint,
                                ..
                            } => "~",
                            TypeInner::Scalar {
                                kind: ScalarKind::Uint,
                                ..
                            } => "~",
                            TypeInner::Scalar {
                                kind: ScalarKind::Bool,
                                ..
                            } => "!",
                            ref other =>
                                return Err(Error::Custom(format!(
                                    "Cannot apply not to type {:?}",
                                    other
                                ))),
                        },
                    }
                )?;

                self.write_expr(expr, ctx)?;

                write!(self.out, ")")?
            }
            Expression::Binary { op, left, right } => {
                write!(self.out, "(")?;
                self.write_expr(left, ctx)?;

                write!(
                    self.out,
                    " {} ",
                    match op {
                        BinaryOperator::Add => "+",
                        BinaryOperator::Subtract => "-",
                        BinaryOperator::Multiply => "*",
                        BinaryOperator::Divide => "/",
                        BinaryOperator::Modulo => "%",
                        BinaryOperator::Equal => "==",
                        BinaryOperator::NotEqual => "!=",
                        BinaryOperator::Less => "<",
                        BinaryOperator::LessEqual => "<=",
                        BinaryOperator::Greater => ">",
                        BinaryOperator::GreaterEqual => ">=",
                        BinaryOperator::And => "&",
                        BinaryOperator::ExclusiveOr => "^",
                        BinaryOperator::InclusiveOr => "|",
                        BinaryOperator::LogicalAnd => "&&",
                        BinaryOperator::LogicalOr => "||",
                        BinaryOperator::ShiftLeft => "<<",
                        BinaryOperator::ShiftRight => ">>",
                    }
                )?;

                self.write_expr(right, ctx)?;

                write!(self.out, ")")?
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                write!(self.out, "(")?;
                self.write_expr(condition, ctx)?;
                write!(self.out, " ? ")?;
                self.write_expr(accept, ctx)?;
                write!(self.out, " : ")?;
                self.write_expr(reject, ctx)?;
                write!(self.out, ")")?
            }
            Expression::Intrinsic { fun, argument } => {
                write!(
                    self.out,
                    "{}(",
                    match fun {
                        IntrinsicFunction::IsFinite => "!isinf",
                        IntrinsicFunction::IsInf => "isinf",
                        IntrinsicFunction::IsNan => "isnan",
                        IntrinsicFunction::IsNormal => "!isnan",
                        IntrinsicFunction::All => "all",
                        IntrinsicFunction::Any => "any",
                    }
                )?;

                self.write_expr(argument, ctx)?;

                write!(self.out, ")")?
            }
            Expression::Transpose(matrix) => {
                write!(self.out, "transpose(")?;
                self.write_expr(matrix, ctx)?;
                write!(self.out, ")")?
            }
            Expression::DotProduct(left, right) => {
                write!(self.out, "dot(")?;
                self.write_expr(left, ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(right, ctx)?;
                write!(self.out, ")")?
            }
            Expression::CrossProduct(left, right) => {
                write!(self.out, "cross(")?;
                self.write_expr(left, ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(right, ctx)?;
                write!(self.out, ")")?
            }
            Expression::As {
                expr,
                kind,
                convert,
            } => {
                if convert {
                    self.write_type(ctx.typifier.get_handle(expr).unwrap(), None)?;
                } else {
                    let source_kind = match *ctx.typifier.get(expr, &self.module.types) {
                        TypeInner::Scalar {
                            kind: source_kind, ..
                        } => source_kind,
                        TypeInner::Vector {
                            kind: source_kind, ..
                        } => source_kind,
                        _ => unreachable!(),
                    };

                    write!(
                        self.out,
                        "{}",
                        match (source_kind, kind) {
                            (ScalarKind::Float, ScalarKind::Sint) => "floatBitsToInt",
                            (ScalarKind::Float, ScalarKind::Uint) => "floatBitsToUInt",
                            (ScalarKind::Sint, ScalarKind::Float) => "intBitsToFloat",
                            (ScalarKind::Uint, ScalarKind::Float) => "uintBitsToFloat",
                            _ => {
                                return Err(Error::Custom(format!(
                                    "Cannot bitcast {:?} to {:?}",
                                    source_kind, kind
                                )));
                            }
                        }
                    )?;
                }

                write!(self.out, "(")?;
                self.write_expr(expr, ctx)?;
                write!(self.out, ")")?
            }
            Expression::Derivative { axis, expr } => {
                write!(
                    self.out,
                    "{}(",
                    match axis {
                        DerivativeAxis::X => "dFdx",
                        DerivativeAxis::Y => "dFdy",
                        DerivativeAxis::Width => "fwidth",
                    }
                )?;
                self.write_expr(expr, ctx)?;
                write!(self.out, ")")?
            }
            Expression::Call {
                origin: FunctionOrigin::Local(ref function),
                ref arguments,
            } => {
                write!(self.out, "{}(", &self.names[&NameKey::Function(*function)])?;
                self.write_slice(arguments, |this, _, arg| this.write_expr(*arg, ctx))?;
                write!(self.out, ")")?
            }
            Expression::Call {
                origin: crate::FunctionOrigin::External(ref name),
                ref arguments,
            } => match name.as_str() {
                "cos" | "normalize" | "sin" | "length" | "abs" | "floor" | "inverse"
                | "distance" | "dot" | "min" | "max" | "reflect" | "pow" | "step" | "cross"
                | "fclamp" | "clamp" | "mix" | "smoothstep" => {
                    let name = match name.as_str() {
                        "fclamp" => "clamp",
                        name => name,
                    };

                    write!(self.out, "{}(", name)?;
                    self.write_slice(arguments, |this, _, arg| this.write_expr(*arg, ctx))?;
                    write!(self.out, ")")?
                }
                "atan2" => {
                    write!(self.out, "atan(")?;
                    self.write_expr(arguments[1], ctx)?;
                    write!(self.out, ", ")?;
                    self.write_expr(arguments[0], ctx)?;
                    write!(self.out, ")")?
                }
                other => {
                    return Err(Error::Custom(format!(
                        "Unsupported function call {}",
                        other
                    )))
                }
            },
            Expression::ArrayLength(expr) => {
                write!(self.out, "uint(")?;
                self.write_expr(expr, ctx)?;
                write!(self.out, ".length())")?
            }
        }

        Ok(())
    }

    fn write_constant(&mut self, constant: &Constant) -> Result<(), Error> {
        match constant.inner {
            ConstantInner::Sint(int) => write!(self.out, "{}", int)?,
            ConstantInner::Uint(int) => write!(self.out, "{}u", int)?,
            ConstantInner::Float(float) => write!(self.out, "{:?}", float)?,
            ConstantInner::Bool(boolean) => write!(self.out, "{}", boolean)?,
            ConstantInner::Composite(ref components) => {
                self.write_type(constant.ty, None)?;
                write!(self.out, "(")?;
                self.write_slice(components, |this, _, arg| {
                    this.write_constant(&this.module.constants[*arg])
                })?;
                write!(self.out, ")")?
            }
        }

        Ok(())
    }
}

struct ScalarString<'a> {
    prefix: &'a str,
    full: &'a str,
}

fn glsl_scalar(kind: ScalarKind, width: crate::Bytes) -> Result<ScalarString<'static>, Error> {
    Ok(match kind {
        ScalarKind::Sint => ScalarString {
            prefix: "i",
            full: "int",
        },
        ScalarKind::Uint => ScalarString {
            prefix: "u",
            full: "uint",
        },
        ScalarKind::Float => match width {
            4 => ScalarString {
                prefix: "",
                full: "float",
            },
            8 => ScalarString {
                prefix: "d",
                full: "double",
            },
            _ => {
                return Err(Error::Custom(format!(
                    "Cannot build float of width {}",
                    width
                )))
            }
        },
        ScalarKind::Bool => ScalarString {
            prefix: "b",
            full: "bool",
        },
    })
}

fn glsl_built_in(built_in: BuiltIn) -> &'static str {
    match built_in {
        BuiltIn::Position => "gl_Position",
        BuiltIn::GlobalInvocationId => "gl_GlobalInvocationID",
        BuiltIn::BaseInstance => "gl_BaseInstance",
        BuiltIn::BaseVertex => "gl_BaseVertex",
        BuiltIn::ClipDistance => "gl_ClipDistance",
        BuiltIn::InstanceIndex => "gl_InstanceIndex",
        BuiltIn::VertexIndex => "gl_VertexIndex",
        BuiltIn::PointSize => "gl_PointSize",
        BuiltIn::FragCoord => "gl_FragCoord",
        BuiltIn::FrontFacing => "gl_FrontFacing",
        BuiltIn::SampleIndex => "gl_SampleID",
        BuiltIn::FragDepth => "gl_FragDepth",
        BuiltIn::LocalInvocationId => "gl_LocalInvocationID",
        BuiltIn::LocalInvocationIndex => "gl_LocalInvocationIndex",
        BuiltIn::WorkGroupId => "gl_WorkGroupID",
    }
}

fn glsl_storage_class(class: StorageClass) -> &'static str {
    match class {
        StorageClass::Function => "",
        StorageClass::Input => "in",
        StorageClass::Output => "out",
        StorageClass::Private => "",
        StorageClass::Storage => "buffer",
        StorageClass::Uniform => "uniform",
        StorageClass::Handle => "uniform",
        StorageClass::WorkGroup => "shared",
        StorageClass::PushConstant => "",
    }
}

fn glsl_interpolation(interpolation: Interpolation) -> Result<&'static str, Error> {
    Ok(match interpolation {
        Interpolation::Perspective => "smooth",
        Interpolation::Linear => "noperspective",
        Interpolation::Flat => "flat",
        Interpolation::Centroid => "centroid",
        Interpolation::Sample => "sample",
        Interpolation::Patch => {
            return Err(Error::Custom(
                "patch interpolation qualifier not supported".to_string(),
            ))
        }
    })
}

struct ImageDimension(crate::ImageDimension);
impl fmt::Display for ImageDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self.0 {
                crate::ImageDimension::D1 => "1D",
                crate::ImageDimension::D2 => "2D",
                crate::ImageDimension::D3 => "3D",
                crate::ImageDimension::Cube => "Cube",
            }
        )
    }
}

fn glsl_storage_format(format: StorageFormat) -> &'static str {
    match format {
        StorageFormat::R8Unorm => "r8",
        StorageFormat::R8Snorm => "r8_snorm",
        StorageFormat::R8Uint => "r8ui",
        StorageFormat::R8Sint => "r8i",
        StorageFormat::R16Uint => "r16ui",
        StorageFormat::R16Sint => "r16i",
        StorageFormat::R16Float => "r16f",
        StorageFormat::Rg8Unorm => "rg8",
        StorageFormat::Rg8Snorm => "rg8_snorm",
        StorageFormat::Rg8Uint => "rg8ui",
        StorageFormat::Rg8Sint => "rg8i",
        StorageFormat::R32Uint => "r32ui",
        StorageFormat::R32Sint => "r32i",
        StorageFormat::R32Float => "r32f",
        StorageFormat::Rg16Uint => "rg16ui",
        StorageFormat::Rg16Sint => "rg16i",
        StorageFormat::Rg16Float => "rg16f",
        StorageFormat::Rgba8Unorm => "rgba8ui",
        StorageFormat::Rgba8Snorm => "rgba8_snorm",
        StorageFormat::Rgba8Uint => "rgba8ui",
        StorageFormat::Rgba8Sint => "rgba8i",
        StorageFormat::Rgb10a2Unorm => "rgb10_a2ui",
        StorageFormat::Rg11b10Float => "r11f_g11f_b10f",
        StorageFormat::Rg32Uint => "rg32ui",
        StorageFormat::Rg32Sint => "rg32i",
        StorageFormat::Rg32Float => "rg32f",
        StorageFormat::Rgba16Uint => "rgba16ui",
        StorageFormat::Rgba16Sint => "rgba16i",
        StorageFormat::Rgba16Float => "rgba16f",
        StorageFormat::Rgba32Uint => "rgba32ui",
        StorageFormat::Rgba32Sint => "rgba32i",
        StorageFormat::Rgba32Float => "rgba32f",
    }
}

struct TextureMappingVisitor<'a> {
    names: &'a FastHashMap<NameKey, String>,
    expressions: &'a Arena<Expression>,
    map: &'a mut FastHashMap<String, TextureMapping>,
    error: Option<Error>,
}

impl<'a> Visitor for TextureMappingVisitor<'a> {
    fn visit_expr(&mut self, expr: &crate::Expression) {
        match expr {
            Expression::ImageSample { image, sampler, .. } => {
                let tex_handle = match self.expressions[*image] {
                    Expression::GlobalVariable(global) => global,
                    _ => unreachable!(),
                };
                let tex_name = self.names[&NameKey::GlobalVariable(tex_handle)].clone();

                let sampler_handle = match self.expressions[*sampler] {
                    Expression::GlobalVariable(global) => global,
                    _ => unreachable!(),
                };

                let mapping = self.map.entry(tex_name).or_insert(TextureMapping {
                    texture: tex_handle,
                    sampler: Some(sampler_handle),
                });

                if mapping.sampler != Some(sampler_handle) {
                    self.error = Some(Error::Custom(String::from(
                        "Cannot use texture with two different samplers",
                    )));
                }
            }
            Expression::ImageLoad { image, .. } => {
                let tex_handle = match self.expressions[*image] {
                    Expression::GlobalVariable(global) => global,
                    _ => unreachable!(),
                };
                let tex_name = self.names[&NameKey::GlobalVariable(tex_handle)].clone();

                let mapping = self.map.entry(tex_name).or_insert(TextureMapping {
                    texture: tex_handle,
                    sampler: None,
                });

                if mapping.sampler != None {
                    self.error = Some(Error::Custom(String::from(
                        "Cannot use texture with two different samplers",
                    )));
                }
            }
            _ => {}
        }
    }
}
