use crate::{
    proc::{Interface, ResolveContext, ResolveError, Typifier, Visitor},
    Arena, ArraySize, BinaryOperator, BuiltIn, ConservativeDepth, Constant, ConstantInner,
    DerivativeAxis, Expression, FastHashMap, Function, FunctionOrigin, GlobalVariable, Handle,
    ImageClass, Interpolation, IntrinsicFunction, LocalVariable, MemberOrigin, Module, ScalarKind,
    ShaderStage, Statement, StorageAccess, StorageClass, StorageFormat, StructMember, Type,
    TypeInner, UnaryOperator,
};
use std::{
    borrow::Cow,
    cmp::Ordering,
    fmt::{self, Error as FmtError, Write as FmtWrite},
    io::{Error as IoError, Write},
};

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

const SUPPORTED_CORE_VERSIONS: &[u16] = &[330, 400, 410, 420, 430, 440, 450];
const SUPPORTED_ES_VERSIONS: &[u16] = &[300, 310];

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
    pub fn write(&self, version: Version, out: &mut impl Write) -> Result<(), Error> {
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

pub fn write<'a>(
    module: &'a Module,
    out: &mut impl Write,
    options: Options,
) -> Result<FastHashMap<String, TextureMapping>, Error> {
    let (version, es) = match options.version {
        Version::Desktop(v) => (v, false),
        Version::Embedded(v) => (v, true),
    };

    if (!es && !SUPPORTED_CORE_VERSIONS.contains(&version))
        || (es && !SUPPORTED_ES_VERSIONS.contains(&version))
    {
        return Err(Error::Custom(format!(
            "Version not supported {}",
            options.version
        )));
    }

    let mut manager = FeaturesManager::new();
    let mut buf = Vec::new();

    writeln!(out, "#version {}\n", options.version)?;

    if es {
        writeln!(out, "precision highp float;\n")?;
    }

    let entry_point = module
        .entry_points
        .get(&options.entry_point)
        .ok_or_else(|| Error::Custom(String::from("Entry point not found")))?;
    let func = &entry_point.function;
    let stage = options.entry_point.0;

    if let Some(depth_test) = entry_point.early_depth_test {
        manager.request(Features::IMAGE_LOAD_STORE);
        writeln!(&mut buf, "layout(early_fragment_tests) in;\n")?;

        if let Some(conservative) = depth_test.conservative {
            manager.request(Features::CONSERVATIVE_DEPTH);

            writeln!(
                &mut buf,
                "layout (depth_{}) out float gl_FragDepth;\n",
                match conservative {
                    ConservativeDepth::GreaterEqual => "greater",
                    ConservativeDepth::LessEqual => "less",
                    ConservativeDepth::Unchanged => "unchanged",
                }
            )?;
        }
    }

    if let ShaderStage::Compute = stage {
        manager.request(Features::COMPUTE_SHADER)
    }

    let mut structs = FastHashMap::default();
    let mut built_structs = FastHashMap::default();

    // Do a first pass to collect names
    for (handle, ty) in module.types.iter() {
        match ty.inner {
            TypeInner::Struct { .. } => {
                let name = ty
                    .name
                    .clone()
                    .filter(|ident| is_valid_ident(ident))
                    .unwrap_or_else(|| format!("struct_{}", handle.index()));

                structs.insert(handle, name);
            }
            _ => continue,
        }
    }

    for ((_, global), usage) in module.global_variables.iter().zip(func.global_usage.iter()) {
        if usage.is_empty() {
            continue;
        }

        let block = match global.class {
            StorageClass::Storage | StorageClass::Uniform => true,
            _ => false,
        };

        match module.types[global.ty].inner {
            TypeInner::Struct { .. } if block => {
                built_structs.insert(global.ty, ());
            }
            _ => {}
        }
    }

    // Do a second pass to build the structs
    for (handle, ty) in module.types.iter() {
        match ty.inner {
            TypeInner::Struct { ref members } => {
                write_struct(
                    handle,
                    members,
                    module,
                    &structs,
                    &mut buf,
                    &mut built_structs,
                    &mut manager,
                )?;
            }
            _ => continue,
        }
    }

    writeln!(&mut buf)?;

    let mut functions = FastHashMap::default();

    for (handle, func) in module.functions.iter() {
        let name = func
            .name
            .clone()
            .filter(|ident| is_valid_ident(ident))
            .unwrap_or_else(|| format!("function_{}", handle.index()));

        writeln!(
            &mut buf,
            "{} {}({});",
            func.return_type
                .map(|ty| write_type(
                    ty,
                    &module.types,
                    &module.constants,
                    &structs,
                    None,
                    &mut manager
                ))
                .transpose()?
                .as_deref()
                .unwrap_or("void"),
            name,
            func.arguments
                .iter()
                .map(|arg| write_type(
                    arg.ty,
                    &module.types,
                    &module.constants,
                    &structs,
                    None,
                    &mut manager
                ))
                .collect::<Result<Vec<_>, _>>()?
                .join(","),
        )?;

        functions.insert(handle, name);
    }

    writeln!(&mut buf)?;

    let texture_mappings = collect_texture_mapping(
        functions
            .keys()
            .map(|handle| &module.functions[*handle])
            .chain(std::iter::once(func)),
    )?;
    let mut mappings_map = FastHashMap::default();
    let mut globals_lookup = FastHashMap::default();

    for ((handle, global), _) in module
        .global_variables
        .iter()
        .zip(func.global_usage.iter())
        .filter(|(_, usage)| !usage.is_empty())
    {
        match module.types[global.ty].inner {
            TypeInner::Image {
                dim,
                arrayed,
                class,
            } => {
                let mapping = if let Some(map) = texture_mappings.get_key_value(&handle) {
                    map
                } else {
                    log::warn!(
                        "Couldn't find a mapping for {:?}, handle {:?}",
                        global,
                        handle
                    );
                    continue;
                };

                if let TypeInner::Image {
                    class: ImageClass::Storage(storage_format),
                    ..
                } = module.types[global.ty].inner
                {
                    write!(
                        &mut buf,
                        "layout({}) ",
                        write_format_glsl(storage_format, &mut manager)
                    )?;
                }

                if global.storage_access == StorageAccess::LOAD {
                    write!(&mut buf, "readonly ")?;
                } else if global.storage_access == StorageAccess::STORE {
                    write!(&mut buf, "writeonly ")?;
                }

                let name = if let Some(ref binding) = global.binding {
                    match binding {
                        crate::Binding::Location(location) => format!("location_{}", location),
                        crate::Binding::Resource { group, binding } => {
                            format!("set_{}_binding_{}", group, binding)
                        }
                        crate::Binding::BuiltIn(_) => unreachable!(),
                    }
                } else {
                    global
                        .name
                        .clone()
                        .filter(|ident| is_valid_ident(ident))
                        .unwrap_or_else(|| format!("global_{}", handle.index()))
                };

                writeln!(
                    &mut buf,
                    "uniform {} {};",
                    write_image_type(dim, arrayed, class, &mut manager)?,
                    name
                )?;

                mappings_map.insert(
                    name.clone(),
                    TextureMapping {
                        texture: *mapping.0,
                        sampler: *mapping.1,
                    },
                );
                globals_lookup.insert(handle, name);
            }
            TypeInner::Sampler { .. } => {
                let name = global
                    .name
                    .clone()
                    .filter(|ident| is_valid_ident(ident))
                    .unwrap_or_else(|| format!("global_{}", handle.index()));

                globals_lookup.insert(handle, name);
            }
            _ => continue,
        }
    }

    for ((handle, global), _) in module
        .global_variables
        .iter()
        .zip(func.global_usage.iter())
        .filter(|(_, usage)| !usage.is_empty())
    {
        match module.types[global.ty].inner {
            TypeInner::Image { .. } | TypeInner::Sampler { .. } => continue,
            _ => {}
        }

        if let Some(crate::Binding::BuiltIn(built_in)) = global.binding {
            let semantic = match built_in {
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
            };

            globals_lookup.insert(handle, String::from(semantic));
            continue;
        }

        let name = if let Some(ref binding) = global.binding {
            let prefix = match global.class {
                StorageClass::Function => "fn",
                StorageClass::Input => "in",
                StorageClass::Output => "out",
                StorageClass::Private => "priv",
                StorageClass::Storage => "buffer",
                StorageClass::Uniform => "uniform",
                StorageClass::Handle => "handle",
                StorageClass::WorkGroup => "wg",
                StorageClass::PushConstant => "pc",
            };

            match binding {
                crate::Binding::Location(location) => format!("{}_location_{}", prefix, location),
                crate::Binding::Resource { group, binding } => {
                    format!("{}_set_{}_binding_{}", prefix, group, binding)
                }
                crate::Binding::BuiltIn(_) => unreachable!(),
            }
        } else {
            global
                .name
                .clone()
                .filter(|ident| is_valid_ident(ident))
                .unwrap_or_else(|| format!("global_{}", handle.index()))
        };

        if let TypeInner::Struct { .. } = module.types[global.ty].inner {
            if built_structs.get(&global.ty).is_none() {
                globals_lookup.insert(handle, name);
                continue;
            }
        }

        if global.storage_access == StorageAccess::LOAD {
            write!(&mut buf, "readonly ")?;
        } else if global.storage_access == StorageAccess::STORE {
            write!(&mut buf, "writeonly ")?;
        }

        if let Some(interpolation) = global.interpolation {
            match (stage, global.class) {
                (ShaderStage::Fragment, StorageClass::Input)
                | (ShaderStage::Vertex, StorageClass::Output) => {
                    write!(&mut buf, "{} ", write_interpolation(interpolation)?)?;
                }
                _ => {}
            };
        }

        let block = match global.class {
            StorageClass::Storage | StorageClass::Uniform => {
                Some(format!("global_block_{}", handle.index()))
            }
            _ => None,
        };

        writeln!(
            &mut buf,
            "{}{} {};",
            write_storage_class(global.class, &mut manager)?,
            write_type(
                global.ty,
                &module.types,
                &module.constants,
                &structs,
                block,
                &mut manager
            )?,
            name
        )?;

        globals_lookup.insert(handle, name);
    }

    writeln!(&mut buf)?;
    let mut typifier = Typifier::new();

    let mut write_function = |func: &Function, name: &str| -> Result<(), Error> {
        typifier.resolve_all(
            &func.expressions,
            &module.types,
            &ResolveContext {
                constants: &module.constants,
                global_vars: &module.global_variables,
                local_vars: &func.local_variables,
                functions: &module.functions,
                arguments: &func.arguments,
            },
        )?;

        let args: FastHashMap<_, _> = func
            .arguments
            .iter()
            .enumerate()
            .map(|(pos, arg)| {
                let name = arg
                    .name
                    .clone()
                    .filter(|ident| is_valid_ident(ident))
                    .unwrap_or_else(|| format!("arg_{}", pos + 1));
                (pos as u32, name)
            })
            .collect();

        writeln!(
            &mut buf,
            "{} {}({}) {{",
            func.return_type
                .map(|ty| write_type(
                    ty,
                    &module.types,
                    &module.constants,
                    &structs,
                    None,
                    &mut manager
                ))
                .transpose()?
                .as_deref()
                .unwrap_or("void"),
            name,
            func.arguments
                .iter()
                .enumerate()
                .map::<Result<_, Error>, _>(|(pos, arg)| {
                    let ty = write_type(
                        arg.ty,
                        &module.types,
                        &module.constants,
                        &structs,
                        None,
                        &mut manager,
                    )?;
                    Ok(format!("{} {}", ty, args[&(pos as u32)]))
                })
                .collect::<Result<Vec<_>, _>>()?
                .join(","),
        )?;

        let locals: FastHashMap<_, _> = func
            .local_variables
            .iter()
            .map(|(handle, local)| {
                (
                    handle,
                    local
                        .name
                        .clone()
                        .filter(|ident| is_valid_ident(ident))
                        .unwrap_or_else(|| format!("local_{}", handle.index())),
                )
            })
            .collect();

        let mut builder = StatementBuilder {
            functions: &functions,
            globals: &globals_lookup,
            locals_lookup: &locals,
            structs: &structs,
            args: &args,
            expressions: &func.expressions,
            typifier: &typifier,
        };

        for (handle, name) in locals.iter() {
            let var = &func.local_variables[*handle];
            write!(
                &mut buf,
                "\t{} {}",
                write_type(
                    var.ty,
                    &module.types,
                    &module.constants,
                    &structs,
                    None,
                    &mut manager
                )?,
                name
            )?;
            if let Some(init) = var.init {
                write!(
                    &mut buf,
                    " = {}",
                    write_constant(&module.constants[init], module, &mut builder, &mut manager)?
                )?;
            }
            writeln!(&mut buf, ";")?;
        }

        writeln!(&mut buf)?;

        for sta in func.body.iter() {
            writeln!(
                &mut buf,
                "{}",
                write_statement(sta, module, &mut builder, &mut manager, 1)?
            )?;
        }

        writeln!(&mut buf, "}}")?;

        Ok(())
    };

    for (handle, name) in functions.iter() {
        let func = &module.functions[*handle];
        write_function(func, name)?;
    }

    write_function(func, "main")?;

    writeln!(out)?;

    manager.write(options.version, out)?;
    out.write_all(&buf)?;

    Ok(mappings_map)
}

struct StatementBuilder<'a> {
    functions: &'a FastHashMap<Handle<Function>, String>,
    globals: &'a FastHashMap<Handle<GlobalVariable>, String>,
    locals_lookup: &'a FastHashMap<Handle<LocalVariable>, String>,
    structs: &'a FastHashMap<Handle<Type>, String>,
    args: &'a FastHashMap<u32, String>,
    expressions: &'a Arena<Expression>,
    typifier: &'a Typifier,
}

fn write_statement<'a, 'b>(
    sta: &Statement,
    module: &'a Module,
    builder: &'b mut StatementBuilder<'a>,
    manager: &mut FeaturesManager,
    indent: usize,
) -> Result<String, Error> {
    Ok(match sta {
        Statement::Block(block) => block
            .iter()
            .map(|sta| write_statement(sta, module, builder, manager, indent))
            .collect::<Result<Vec<_>, _>>()?
            .join("\n"),
        Statement::If {
            condition,
            accept,
            reject,
        } => {
            let mut out = String::new();

            writeln!(
                &mut out,
                "{}if({}) {{",
                "\t".repeat(indent),
                write_expression(&builder.expressions[*condition], module, builder, manager)?
            )?;

            for sta in accept {
                writeln!(
                    &mut out,
                    "{}",
                    write_statement(sta, module, builder, manager, indent + 1)?
                )?;
            }

            if !reject.is_empty() {
                writeln!(&mut out, "{}}} else {{", "\t".repeat(indent),)?;
                for sta in reject {
                    writeln!(
                        &mut out,
                        "{}",
                        write_statement(sta, module, builder, manager, indent + 1)?
                    )?;
                }
            }

            write!(&mut out, "{}}}", "\t".repeat(indent),)?;

            out
        }
        Statement::Switch {
            selector,
            cases,
            default,
        } => {
            let mut out = String::new();

            writeln!(
                &mut out,
                "{}switch({}) {{",
                "\t".repeat(indent),
                write_expression(&builder.expressions[*selector], module, builder, manager)?
            )?;

            for (label, (block, fallthrough)) in cases {
                writeln!(&mut out, "{}case {}:", "\t".repeat(indent + 1), label)?;

                for sta in block {
                    writeln!(
                        &mut out,
                        "{}",
                        write_statement(sta, module, builder, manager, indent + 2)?
                    )?;
                }

                if fallthrough.is_none() {
                    writeln!(&mut out, "{}break;", "\t".repeat(indent + 2),)?;
                }
            }

            if !default.is_empty() {
                writeln!(&mut out, "{}default:", "\t".repeat(indent + 1),)?;

                for sta in default {
                    writeln!(
                        &mut out,
                        "{}",
                        write_statement(sta, module, builder, manager, indent + 2)?
                    )?;
                }
            }

            write!(&mut out, "{}}}", "\t".repeat(indent),)?;

            out
        }
        Statement::Loop { body, continuing } => {
            let mut out = String::new();

            writeln!(&mut out, "{}while(true) {{", "\t".repeat(indent),)?;

            for sta in body.iter().chain(continuing.iter()) {
                writeln!(
                    &mut out,
                    "{}",
                    write_statement(sta, module, builder, manager, indent + 1)?
                )?;
            }

            write!(&mut out, "{}}}", "\t".repeat(indent),)?;

            out
        }
        Statement::Break => format!("{}break;", "\t".repeat(indent),),
        Statement::Continue => format!("{}continue;", "\t".repeat(indent),),
        Statement::Return { value } => format!(
            "{}{}",
            "\t".repeat(indent),
            if let Some(expr) = value {
                format!(
                    "return {};",
                    write_expression(&builder.expressions[*expr], module, builder, manager)?
                )
            } else {
                String::from("return;")
            }
        ),
        Statement::Kill => format!("{}discard;", "\t".repeat(indent)),
        Statement::Store { pointer, value } => format!(
            "{}{} = {};",
            "\t".repeat(indent),
            write_expression(&builder.expressions[*pointer], module, builder, manager)?,
            write_expression(&builder.expressions[*value], module, builder, manager)?
        ),
    })
}

fn write_expression<'a, 'b>(
    expr: &Expression,
    module: &'a Module,
    builder: &'b mut StatementBuilder<'a>,
    manager: &mut FeaturesManager,
) -> Result<Cow<'a, str>, Error> {
    Ok(match *expr {
        Expression::Access { base, index } => {
            let base_expr = write_expression(&builder.expressions[base], module, builder, manager)?;
            Cow::Owned(format!(
                "{}[{}]",
                base_expr,
                write_expression(&builder.expressions[index], module, builder, manager)?
            ))
        }
        Expression::AccessIndex { base, index } => {
            let base_expr = write_expression(&builder.expressions[base], module, builder, manager)?;

            match *builder.typifier.get(base, &module.types) {
                TypeInner::Vector { .. } => Cow::Owned(format!("{}[{}]", base_expr, index)),
                TypeInner::Matrix { .. } | TypeInner::Array { .. } => {
                    Cow::Owned(format!("{}[{}]", base_expr, index))
                }
                TypeInner::Struct { ref members } => {
                    if let MemberOrigin::BuiltIn(builtin) = members[index as usize].origin {
                        Cow::Borrowed(builtin_to_glsl(builtin))
                    } else {
                        Cow::Owned(format!(
                            "{}.{}",
                            base_expr,
                            members[index as usize]
                                .name
                                .as_ref()
                                .filter(|s| is_valid_ident(s))
                                .unwrap_or(&format!("_{}", index))
                        ))
                    }
                }
                ref other => return Err(Error::Custom(format!("Cannot index {:?}", other))),
            }
        }
        Expression::Constant(constant) => Cow::Owned(write_constant(
            &module.constants[constant],
            module,
            builder,
            manager,
        )?),
        Expression::Compose { ty, ref components } => {
            let constructor = match module.types[ty].inner {
                TypeInner::Vector { size, kind, width } => format!(
                    "{}vec{}",
                    map_scalar(kind, width, manager)?.prefix,
                    size as u8,
                ),
                TypeInner::Matrix {
                    columns,
                    rows,
                    width,
                } => format!(
                    "{}mat{}x{}",
                    map_scalar(crate::ScalarKind::Float, width, manager)?.prefix,
                    columns as u8,
                    rows as u8,
                ),
                TypeInner::Array { .. } => write_type(
                    ty,
                    &module.types,
                    &module.constants,
                    builder.structs,
                    None,
                    manager,
                )?
                .into_owned(),
                TypeInner::Struct { .. } => builder.structs.get(&ty).unwrap().clone(),
                _ => {
                    return Err(Error::Custom(format!(
                        "Cannot compose type {}",
                        write_type(
                            ty,
                            &module.types,
                            &module.constants,
                            builder.structs,
                            None,
                            manager
                        )?
                    )))
                }
            };

            Cow::Owned(format!(
                "{}({})",
                constructor,
                components
                    .iter()
                    .map::<Result<_, Error>, _>(|arg| Ok(write_expression(
                        &builder.expressions[*arg],
                        module,
                        builder,
                        manager,
                    )?))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(","),
            ))
        }
        Expression::FunctionArgument(pos) => Cow::Borrowed(builder.args.get(&pos).unwrap()),
        Expression::GlobalVariable(handle) => Cow::Borrowed(builder.globals.get(&handle).unwrap()),
        Expression::LocalVariable(handle) => {
            Cow::Borrowed(builder.locals_lookup.get(&handle).unwrap())
        }
        Expression::Load { pointer } => {
            write_expression(&builder.expressions[pointer], module, builder, manager)?
        }
        Expression::ImageSample {
            image,
            sampler,
            coordinate,
            level,
            depth_ref,
        } => {
            let image_expr =
                write_expression(&builder.expressions[image], module, builder, manager)?;
            write_expression(&builder.expressions[sampler], module, builder, manager)?;
            let coordinate_expr =
                write_expression(&builder.expressions[coordinate], module, builder, manager)?;

            let size = match *builder.typifier.get(coordinate, &module.types) {
                TypeInner::Vector { size, .. } => size,
                ref other => {
                    return Err(Error::Custom(format!(
                        "Cannot sample with coordinates of type {:?}",
                        other
                    )))
                }
            };

            let coordinate_expr = if let Some(depth_ref) = depth_ref {
                Cow::Owned(format!(
                    "vec{}({},{})",
                    size as u8 + 1,
                    coordinate_expr,
                    write_expression(&builder.expressions[depth_ref], module, builder, manager)?
                ))
            } else {
                coordinate_expr
            };

            //TODO: handle MS
            Cow::Owned(match level {
                crate::SampleLevel::Auto => format!("texture({},{})", image_expr, coordinate_expr),
                crate::SampleLevel::Zero => {
                    format!("textureLod({},{},0)", image_expr, coordinate_expr)
                }
                crate::SampleLevel::Exact(expr) => {
                    let level_expr =
                        write_expression(&builder.expressions[expr], module, builder, manager)?;
                    format!(
                        "textureLod({}, {}, {})",
                        image_expr, coordinate_expr, level_expr
                    )
                }
                crate::SampleLevel::Bias(bias) => {
                    let bias_expr =
                        write_expression(&builder.expressions[bias], module, builder, manager)?;
                    format!("texture({},{},{})", image_expr, coordinate_expr, bias_expr)
                }
            })
        }
        Expression::ImageLoad {
            image,
            coordinate,
            index,
        } => {
            let image_expr =
                write_expression(&builder.expressions[image], module, builder, manager)?;
            let coordinate_expr =
                write_expression(&builder.expressions[coordinate], module, builder, manager)?;

            let (dim, arrayed, class) = match *builder.typifier.get(image, &module.types) {
                TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => (dim, arrayed, class),
                ref other => return Err(Error::Custom(format!("Cannot load {:?}", other))),
            };

            Cow::Owned(match class {
                ImageClass::Sampled { kind, multi } => {
                    //TODO: fix this
                    let sampler_constructor = format!(
                        "{}sampler{}{}{}({})",
                        map_scalar(kind, 4, manager)?.prefix,
                        ImageDimension(dim),
                        if multi { "MS" } else { "" },
                        if arrayed { "Array" } else { "" },
                        image_expr,
                    );

                    let index_expr = write_expression(
                        &builder.expressions[index.unwrap()],
                        module,
                        builder,
                        manager,
                    )?;
                    format!(
                        "texelFetch({},{},{})",
                        sampler_constructor, coordinate_expr, index_expr
                    )
                }
                ImageClass::Storage(_) => format!("imageLoad({},{})", image_expr, coordinate_expr),
                ImageClass::Depth => todo!(),
            })
        }
        Expression::Unary { op, expr } => {
            let base_expr = write_expression(&builder.expressions[expr], module, builder, manager)?;

            Cow::Owned(format!(
                "({} {})",
                match op {
                    UnaryOperator::Negate => "-",
                    UnaryOperator::Not => match *builder.typifier.get(expr, &module.types) {
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
                },
                base_expr
            ))
        }
        Expression::Binary { op, left, right } => {
            let left_expr = write_expression(&builder.expressions[left], module, builder, manager)?;
            let right_expr =
                write_expression(&builder.expressions[right], module, builder, manager)?;

            let op_str = match op {
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
            };

            Cow::Owned(format!("({} {} {})", left_expr, op_str, right_expr))
        }
        Expression::Select {
            condition,
            accept,
            reject,
        } => {
            let cond_expr =
                write_expression(&builder.expressions[condition], module, builder, manager)?;
            let accept_expr =
                write_expression(&builder.expressions[accept], module, builder, manager)?;
            let reject_expr =
                write_expression(&builder.expressions[reject], module, builder, manager)?;
            Cow::Owned(format!(
                "({} ? {} : {})",
                cond_expr, accept_expr, reject_expr
            ))
        }
        Expression::Intrinsic { fun, argument } => {
            let expr = write_expression(&builder.expressions[argument], module, builder, manager)?;

            Cow::Owned(format!(
                "{:?}({})",
                match fun {
                    IntrinsicFunction::IsFinite => "!isinf",
                    IntrinsicFunction::IsInf => "isinf",
                    IntrinsicFunction::IsNan => "isnan",
                    IntrinsicFunction::IsNormal => "!isnan",
                    IntrinsicFunction::All => "all",
                    IntrinsicFunction::Any => "any",
                },
                expr
            ))
        }
        Expression::Transpose(matrix) => {
            let matrix_expr =
                write_expression(&builder.expressions[matrix], module, builder, manager)?;
            Cow::Owned(format!("transpose({})", matrix_expr))
        }
        Expression::DotProduct(left, right) => {
            let left_expr = write_expression(&builder.expressions[left], module, builder, manager)?;
            let right_expr =
                write_expression(&builder.expressions[right], module, builder, manager)?;
            Cow::Owned(format!("dot({},{})", left_expr, right_expr))
        }
        Expression::CrossProduct(left, right) => {
            let left_expr = write_expression(&builder.expressions[left], module, builder, manager)?;
            let right_expr =
                write_expression(&builder.expressions[right], module, builder, manager)?;
            Cow::Owned(format!("cross({},{})", left_expr, right_expr))
        }
        Expression::As {
            expr,
            kind,
            convert,
        } => {
            let value_expr =
                write_expression(&builder.expressions[expr], module, builder, manager)?;

            let (source_kind, ty_expr) = match *builder.typifier.get(expr, &module.types) {
                TypeInner::Scalar {
                    width,
                    kind: source_kind,
                } => (
                    source_kind,
                    Cow::Borrowed(map_scalar(kind, width, manager)?.full),
                ),
                TypeInner::Vector {
                    width,
                    kind: source_kind,
                    size,
                } => (
                    source_kind,
                    Cow::Owned(format!(
                        "{}vec{}",
                        map_scalar(kind, width, manager)?.prefix,
                        size as u32,
                    )),
                ),
                _ => return Err(Error::Custom(format!("Cannot convert {}", value_expr))),
            };

            let op = if convert {
                ty_expr
            } else {
                Cow::Borrowed(match (source_kind, kind) {
                    (ScalarKind::Float, ScalarKind::Sint) => "floatBitsToInt",
                    (ScalarKind::Float, ScalarKind::Uint) => "floatBitsToUInt",
                    (ScalarKind::Sint, ScalarKind::Float) => "intBitsToFloat",
                    (ScalarKind::Uint, ScalarKind::Float) => "uintBitsToFloat",
                    _ => {
                        return Err(Error::Custom(format!(
                            "Cannot bitcast {:?} to {:?}",
                            source_kind, kind
                        )))
                    }
                })
            };

            Cow::Owned(format!("{}({})", op, value_expr))
        }
        Expression::Derivative { axis, expr } => {
            let expr = write_expression(&builder.expressions[expr], module, builder, manager)?;

            Cow::Owned(format!(
                "{}({})",
                match axis {
                    DerivativeAxis::X => "dFdx",
                    DerivativeAxis::Y => "dFdy",
                    DerivativeAxis::Width => "fwidth",
                },
                expr
            ))
        }
        Expression::Call {
            origin: FunctionOrigin::Local(ref function),
            ref arguments,
        } => Cow::Owned(format!(
            "{}({})",
            builder.functions.get(function).unwrap(),
            arguments
                .iter()
                .map::<Result<_, Error>, _>(|arg| write_expression(
                    &builder.expressions[*arg],
                    module,
                    builder,
                    manager,
                ))
                .collect::<Result<Vec<_>, _>>()?
                .join(","),
        )),
        Expression::Call {
            origin: crate::FunctionOrigin::External(ref name),
            ref arguments,
        } => match name.as_str() {
            "cos" | "normalize" | "sin" | "length" | "abs" | "floor" | "inverse" => {
                let expr =
                    write_expression(&builder.expressions[arguments[0]], module, builder, manager)?;

                Cow::Owned(format!("{}({})", name, expr))
            }
            "fclamp" | "clamp" | "mix" | "smoothstep" => {
                let x =
                    write_expression(&builder.expressions[arguments[0]], module, builder, manager)?;
                let y =
                    write_expression(&builder.expressions[arguments[1]], module, builder, manager)?;
                let a =
                    write_expression(&builder.expressions[arguments[2]], module, builder, manager)?;

                let name = match name.as_str() {
                    "fclamp" => "clamp",
                    name => name,
                };

                Cow::Owned(format!("{}({}, {}, {})", name, x, y, a))
            }
            "atan2" => {
                let x =
                    write_expression(&builder.expressions[arguments[0]], module, builder, manager)?;
                let y =
                    write_expression(&builder.expressions[arguments[1]], module, builder, manager)?;

                Cow::Owned(format!("atan({}, {})", y, x))
            }
            "distance" | "dot" | "min" | "max" | "reflect" | "pow" | "step" | "cross" => {
                let x =
                    write_expression(&builder.expressions[arguments[0]], module, builder, manager)?;
                let y =
                    write_expression(&builder.expressions[arguments[1]], module, builder, manager)?;

                Cow::Owned(format!("{}({}, {})", name, x, y))
            }
            other => {
                return Err(Error::Custom(format!(
                    "Unsupported function call {}",
                    other
                )))
            }
        },
        Expression::ArrayLength(expr) => {
            let base = write_expression(&builder.expressions[expr], module, builder, manager)?;
            Cow::Owned(format!("uint({}.length())", base))
        }
    })
}

fn write_constant(
    constant: &Constant,
    module: &Module,
    builder: &mut StatementBuilder<'_>,
    manager: &mut FeaturesManager,
) -> Result<String, Error> {
    Ok(match constant.inner {
        ConstantInner::Sint(int) => int.to_string(),
        ConstantInner::Uint(int) => format!("{}u", int),
        ConstantInner::Float(float) => format!("{:?}", float),
        ConstantInner::Bool(boolean) => boolean.to_string(),
        ConstantInner::Composite(ref components) => format!(
            "{}({})",
            match module.types[constant.ty].inner {
                TypeInner::Vector { size, .. } => Cow::Owned(format!("vec{}", size as u8,)),
                TypeInner::Matrix { columns, rows, .. } =>
                    Cow::Owned(format!("mat{}x{}", columns as u8, rows as u8,)),
                TypeInner::Struct { .. } =>
                    Cow::<str>::Borrowed(builder.structs.get(&constant.ty).unwrap()),
                TypeInner::Array { .. } => write_type(
                    constant.ty,
                    &module.types,
                    &module.constants,
                    builder.structs,
                    None,
                    manager
                )?,
                _ =>
                    return Err(Error::Custom(format!(
                        "Cannot build constant of type {}",
                        write_type(
                            constant.ty,
                            &module.types,
                            &module.constants,
                            builder.structs,
                            None,
                            manager
                        )?
                    ))),
            },
            components
                .iter()
                .map(|component| write_constant(
                    &module.constants[*component],
                    module,
                    builder,
                    manager
                ))
                .collect::<Result<Vec<_>, _>>()?
                .join(","),
        ),
    })
}

struct ScalarString<'a> {
    prefix: &'a str,
    full: &'a str,
}

fn map_scalar(
    kind: ScalarKind,
    width: crate::Bytes,
    manager: &mut FeaturesManager,
) -> Result<ScalarString<'static>, Error> {
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
            8 => {
                manager.request(Features::DOUBLE_TYPE);

                ScalarString {
                    prefix: "d",
                    full: "double",
                }
            }
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

fn write_type<'a>(
    ty: Handle<Type>,
    types: &Arena<Type>,
    constants: &Arena<Constant>,
    structs: &'a FastHashMap<Handle<Type>, String>,
    block: Option<String>,
    manager: &mut FeaturesManager,
) -> Result<Cow<'a, str>, Error> {
    Ok(match types[ty].inner {
        TypeInner::Scalar { kind, width } => Cow::Borrowed(map_scalar(kind, width, manager)?.full),
        TypeInner::Vector { size, kind, width } => Cow::Owned(format!(
            "{}vec{}",
            map_scalar(kind, width, manager)?.prefix,
            size as u8
        )),
        TypeInner::Matrix {
            columns,
            rows,
            width,
        } => {
            if width == 8 {
                manager.request(Features::DOUBLE_TYPE);
            }

            Cow::Owned(format!(
                "{}mat{}x{}",
                map_scalar(crate::ScalarKind::Float, width, manager)?.prefix,
                columns as u8,
                rows as u8
            ))
        }
        TypeInner::Pointer { base, .. } => {
            write_type(base, types, constants, structs, None, manager)?
        }
        TypeInner::Array { base, size, .. } => {
            if let TypeInner::Array { .. } = types[base].inner {
                manager.request(Features::ARRAY_OF_ARRAYS)
            }

            Cow::Owned(format!(
                "{}[{}]",
                write_type(base, types, constants, structs, None, manager)?,
                write_array_size(size, constants)?
            ))
        }
        TypeInner::Struct { ref members } => {
            if let Some(name) = block {
                let mut out = String::new();
                writeln!(&mut out, "{} {{", name)?;

                for (idx, member) in members.iter().enumerate() {
                    writeln!(
                        &mut out,
                        "\t{} {};",
                        write_type(member.ty, types, constants, structs, None, manager)?,
                        member
                            .name
                            .clone()
                            .filter(|s| is_valid_ident(s))
                            .unwrap_or_else(|| format!("_{}", idx))
                    )?;
                }

                write!(&mut out, "}}")?;

                Cow::Owned(out)
            } else {
                Cow::Borrowed(structs.get(&ty).unwrap())
            }
        }
        _ => unreachable!(),
    })
}

fn write_image_type(
    dim: crate::ImageDimension,
    arrayed: bool,
    class: ImageClass,
    manager: &mut FeaturesManager,
) -> Result<String, Error> {
    if arrayed && dim == crate::ImageDimension::Cube {
        manager.request(Features::CUBE_TEXTURES_ARRAY)
    } else if dim == crate::ImageDimension::D1 {
        manager.request(Features::TEXTURE_1D)
    }

    let (base, kind, ms, comparison) = match class {
        ImageClass::Sampled { kind, multi: true } => {
            manager.request(Features::MULTISAMPLED_TEXTURES);
            if arrayed {
                manager.request(Features::MULTISAMPLED_TEXTURE_ARRAYS);
            }

            ("sampler", kind, "MS", "")
        }
        ImageClass::Sampled { kind, multi: false } => ("sampler", kind, "", ""),
        ImageClass::Depth => ("sampler", crate::ScalarKind::Float, "", "Shadow"),
        ImageClass::Storage(format) => ("image", format.into(), "", ""),
    };

    Ok(format!(
        "{}{}{}{}{}{}",
        map_scalar(kind, 4, manager)?.prefix,
        base,
        ImageDimension(dim),
        ms,
        if arrayed { "Array" } else { "" },
        comparison
    ))
}

fn write_storage_class(
    class: StorageClass,
    manager: &mut FeaturesManager,
) -> Result<&'static str, Error> {
    Ok(match class {
        StorageClass::Function => "",
        StorageClass::Input => "in ",
        StorageClass::Output => "out ",
        StorageClass::Private => "",
        StorageClass::Storage => {
            manager.request(Features::BUFFER_STORAGE);
            "buffer "
        }
        StorageClass::Uniform => "uniform ",
        StorageClass::Handle => "uniform ",
        StorageClass::WorkGroup => {
            manager.request(Features::COMPUTE_SHADER);
            "shared "
        }
        StorageClass::PushConstant => {
            manager.request(Features::PUSH_CONSTANT);
            ""
        }
    })
}

fn write_interpolation(interpolation: Interpolation) -> Result<&'static str, Error> {
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

fn write_array_size(size: ArraySize, constants: &Arena<Constant>) -> Result<String, Error> {
    Ok(match size {
        ArraySize::Constant(const_handle) => match constants[const_handle].inner {
            ConstantInner::Uint(size) => size.to_string(),
            _ => unreachable!(),
        },
        ArraySize::Dynamic => String::from(""),
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

fn write_struct(
    handle: Handle<Type>,
    members: &[StructMember],
    module: &Module,
    structs: &FastHashMap<Handle<Type>, String>,
    out: &mut impl Write,
    built_structs: &mut FastHashMap<Handle<Type>, ()>,
    manager: &mut FeaturesManager,
) -> Result<bool, Error> {
    if built_structs.get(&handle).is_some() {
        return Ok(true);
    }

    let mut tmp = String::new();

    let name = structs.get(&handle).unwrap();
    let mut fields = 0;

    writeln!(&mut tmp, "struct {} {{", name)?;
    for (idx, member) in members.iter().enumerate() {
        if let MemberOrigin::BuiltIn(_) = member.origin {
            continue;
        }

        if let TypeInner::Struct { ref members } = module.types[member.ty].inner {
            if !write_struct(
                member.ty,
                members,
                module,
                structs,
                out,
                built_structs,
                manager,
            )? {
                continue;
            }
        }

        writeln!(
            &mut tmp,
            "\t{} {};",
            write_type(
                member.ty,
                &module.types,
                &module.constants,
                &structs,
                None,
                manager
            )?,
            member
                .name
                .clone()
                .filter(|s| is_valid_ident(s))
                .unwrap_or_else(|| format!("_{}", idx))
        )?;

        fields += 1;
    }
    writeln!(&mut tmp, "}};")?;

    if fields != 0 {
        built_structs.insert(handle, ());
        writeln!(out, "{}", tmp)?;
    }

    Ok(fields != 0)
}

fn is_valid_ident(ident: &str) -> bool {
    ident.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && ident.contains(|c: char| c.is_ascii_alphanumeric() || c == '_')
        && !ident.starts_with("gl_")
        && ident != "main"
}

fn builtin_to_glsl(builtin: BuiltIn) -> &'static str {
    match builtin {
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

fn write_format_glsl(format: StorageFormat, manager: &mut FeaturesManager) -> &'static str {
    match format {
        StorageFormat::R8Unorm => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "r8"
        }
        StorageFormat::R8Snorm => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "r8_snorm"
        }
        StorageFormat::R8Uint => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "r8ui"
        }
        StorageFormat::R8Sint => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "r8i"
        }
        StorageFormat::R16Uint => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "r16ui"
        }
        StorageFormat::R16Sint => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "r16i"
        }
        StorageFormat::R16Float => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "r16f"
        }
        StorageFormat::Rg8Unorm => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "rg8"
        }
        StorageFormat::Rg8Snorm => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "rg8_snorm"
        }
        StorageFormat::Rg8Uint => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "rg8ui"
        }
        StorageFormat::Rg8Sint => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "rg8i"
        }
        StorageFormat::R32Uint => "r32ui",
        StorageFormat::R32Sint => "r32i",
        StorageFormat::R32Float => "r32f",
        StorageFormat::Rg16Uint => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "rg16ui"
        }
        StorageFormat::Rg16Sint => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "rg16i"
        }
        StorageFormat::Rg16Float => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "rg16f"
        }
        StorageFormat::Rgba8Unorm => "rgba8ui",
        StorageFormat::Rgba8Snorm => "rgba8_snorm",
        StorageFormat::Rgba8Uint => "rgba8ui",
        StorageFormat::Rgba8Sint => "rgba8i",
        StorageFormat::Rgb10a2Unorm => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "rgb10_a2ui"
        }
        StorageFormat::Rg11b10Float => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "r11f_g11f_b10f"
        }
        StorageFormat::Rg32Uint => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "rg32ui"
        }
        StorageFormat::Rg32Sint => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "rg32i"
        }
        StorageFormat::Rg32Float => {
            manager.request(Features::FULL_IMAGE_FORMATS);
            "rg32f"
        }
        StorageFormat::Rgba16Uint => "rgba16ui",
        StorageFormat::Rgba16Sint => "rgba16i",
        StorageFormat::Rgba16Float => "rgba16f",
        StorageFormat::Rgba32Uint => "rgba32ui",
        StorageFormat::Rgba32Sint => "rgba32i",
        StorageFormat::Rgba32Float => "rgba32f",
    }
}

struct TextureMappingVisitor<'a> {
    expressions: &'a Arena<Expression>,
    map: &'a mut FastHashMap<Handle<GlobalVariable>, Option<Handle<GlobalVariable>>>,
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

                let sampler_handle = match self.expressions[*sampler] {
                    Expression::GlobalVariable(global) => global,
                    _ => unreachable!(),
                };

                let sampler = self.map.entry(tex_handle).or_insert(Some(sampler_handle));

                if *sampler != Some(sampler_handle) {
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

                let sampler = self.map.entry(tex_handle).or_insert(None);

                if *sampler != None {
                    self.error = Some(Error::Custom(String::from(
                        "Cannot use texture with two different samplers",
                    )));
                }
            }
            _ => {}
        }
    }
}

fn collect_texture_mapping<'a>(
    functions: impl Iterator<Item = &'a Function>,
) -> Result<FastHashMap<Handle<GlobalVariable>, Option<Handle<GlobalVariable>>>, Error> {
    let mut mappings = FastHashMap::default();

    for func in functions {
        let mut interface = Interface {
            expressions: &func.expressions,
            local_variables: &func.local_variables,
            visitor: TextureMappingVisitor {
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
