use crate::{
    proc::{Interface, ResolveContext, ResolveError, Typifier, Visitor},
    Arena, ArraySize, BinaryOperator, BuiltIn, Constant, ConstantInner, DerivativeAxis, Expression,
    FastHashMap, Function, FunctionOrigin, GlobalVariable, Handle, ImageClass, Interpolation,
    IntrinsicFunction, LocalVariable, MemberOrigin, Module, ScalarKind, ShaderStage, Statement,
    StorageAccess, StorageClass, StorageFormat, StructMember, Type, TypeInner, UnaryOperator,
};
use std::{
    borrow::Cow,
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

const SUPPORTED_CORE_VERSIONS: &[u16] = &[450, 460];
const SUPPORTED_ES_VERSIONS: &[u16] = &[300, 310];

bitflags::bitflags! {
    struct SupportedFeatures: u32 {
        const BUFFER_STORAGE = 1;
        const SHARED_STORAGE = 1 << 1;
        const DOUBLE_TYPE = 1 << 2;
        const NON_FLOAT_MATRICES = 1 << 3;
        const MULTISAMPLED_TEXTURES = 1 << 4;
        const MULTISAMPLED_TEXTURE_ARRAYS = 1 << 5;
        const NON_2D_TEXTURE_ARRAYS = 1 << 6;
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

    writeln!(out, "#version {}\n", options.version)?;

    if es {
        writeln!(out, "precision highp float;\n")?;
    }

    let mut counter = 0;
    let mut names = FastHashMap::default();

    let mut namer = |name: Option<&'a String>| {
        if let Some(name) = name {
            if !is_valid_ident(name) || names.get(name.as_str()).is_some() {
                counter += 1;
                while names.get(format!("_{}", counter).as_str()).is_some() {
                    counter += 1;
                }
                format!("_{}", counter)
            } else {
                names.insert(name.as_str(), ());
                name.clone()
            }
        } else {
            counter += 1;
            while names.get(format!("_{}", counter).as_str()).is_some() {
                counter += 1;
            }
            format!("_{}", counter)
        }
    };

    let entry_point = module
        .entry_points
        .get(&options.entry_point)
        .ok_or_else(|| Error::Custom(String::from("Entry point not found")))?;
    let func = &entry_point.function;
    let stage = options.entry_point.0;

    if let ShaderStage::Compute = stage {
        if (es && version < 310) || (!es && version < 430) {
            return Err(Error::Custom(format!(
                "Version {} doesn't support compute shaders",
                options.version
            )));
        }

        if !es && version < 460 {
            writeln!(out, "#extension ARB_compute_shader : require")?;
        }
    }

    let mut features = SupportedFeatures::empty();

    if !es && version > 440 {
        features |= SupportedFeatures::DOUBLE_TYPE;
        features |= SupportedFeatures::NON_FLOAT_MATRICES;
        features |= SupportedFeatures::MULTISAMPLED_TEXTURE_ARRAYS;
        features |= SupportedFeatures::NON_2D_TEXTURE_ARRAYS;
    }

    if !es || version > 300 {
        features |= SupportedFeatures::BUFFER_STORAGE;
        features |= SupportedFeatures::SHARED_STORAGE;
        features |= SupportedFeatures::MULTISAMPLED_TEXTURES;
    }

    let mut structs = FastHashMap::default();
    let mut built_structs = FastHashMap::default();

    // Do a first pass to collect names
    for (handle, ty) in module.types.iter() {
        match ty.inner {
            TypeInner::Struct { .. } => {
                let name = namer(ty.name.as_ref());

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
            StorageClass::Input
            | StorageClass::Output
            | StorageClass::StorageBuffer
            | StorageClass::Uniform => true,
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
                    out,
                    &mut built_structs,
                    features,
                )?;
            }
            _ => continue,
        }
    }

    writeln!(out)?;

    let mut functions = FastHashMap::default();

    for (handle, func) in module.functions.iter() {
        let name = namer(func.name.as_ref());

        writeln!(
            out,
            "{} {}({});",
            func.return_type
                .map(|ty| write_type(ty, &module.types, &structs, None, features))
                .transpose()?
                .as_deref()
                .unwrap_or("void"),
            name,
            func.parameter_types
                .iter()
                .map(|ty| write_type(*ty, &module.types, &structs, None, features))
                .collect::<Result<Vec<_>, _>>()?
                .join(","),
        )?;

        functions.insert(handle, name);
    }

    writeln!(out)?;

    let texture_mappings = collect_texture_mapping(module, &functions)?;
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
                kind,
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
                    write!(out, "layout({}) ", write_format_glsl(storage_format))?;
                }

                if global.storage_access == StorageAccess::LOAD {
                    write!(out, "readonly ")?;
                } else if global.storage_access == StorageAccess::STORE {
                    write!(out, "writeonly ")?;
                }

                let name = namer(global.name.as_ref());

                let comparison = if let Some(sampler) = mapping.1 {
                    if let TypeInner::Sampler { comparison } =
                        module.types[module.global_variables[*sampler].ty].inner
                    {
                        comparison
                    } else {
                        false
                    }
                } else {
                    unreachable!()
                };

                writeln!(
                    out,
                    "uniform {} {};",
                    write_image_type(kind, dim, arrayed, class, comparison, features)?,
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
                let name = namer(global.name.as_ref());

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

        let name = global
            .name
            .clone()
            .ok_or_else(|| Error::Custom(String::from("Global names must be specified in es")))?;

        if let Some(ref binding) = global.binding {
            // Only vulkan glsl supports set/binding on the layout
            if let crate::Binding::Location(location) = binding {
                write!(out, "layout(location = {}) ", location)?;
            }

            write!(out, ") ")?;
        }

        if global.storage_access == StorageAccess::LOAD {
            write!(out, "readonly ")?;
        } else if global.storage_access == StorageAccess::STORE {
            write!(out, "writeonly ")?;
        }

        if let Some(interpolation) = global.interpolation {
            match (stage, global.class) {
                (ShaderStage::Fragment, StorageClass::Input)
                | (ShaderStage::Vertex, StorageClass::Output) => {
                    write!(out, "{} ", write_interpolation(interpolation)?)?;
                }
                _ => {}
            };
        }

        let block = match global.class {
            StorageClass::Input
            | StorageClass::Output
            | StorageClass::StorageBuffer
            | StorageClass::Uniform => Some(namer(None)),
            _ => None,
        };

        writeln!(
            out,
            "{}{} {};",
            write_storage_class(global.class, features)?,
            write_type(global.ty, &module.types, &structs, block, features)?,
            name
        )?;

        globals_lookup.insert(handle, name);
    }

    writeln!(out)?;
    let mut typifier = Typifier::new();

    for (handle, name) in functions.iter() {
        let func = &module.functions[*handle];
        typifier.resolve_all(
            &func.expressions,
            &module.types,
            &ResolveContext {
                constants: &module.constants,
                global_vars: &module.global_variables,
                local_vars: &func.local_variables,
                functions: &module.functions,
                parameter_types: &func.parameter_types,
            },
        )?;

        let args: FastHashMap<_, _> = func
            .parameter_types
            .iter()
            .enumerate()
            .map(|(pos, _ty)| (pos as u32, namer(None)))
            .collect();

        writeln!(
            out,
            "{} {}({}) {{",
            func.return_type
                .map(|ty| write_type(ty, &module.types, &structs, None, features))
                .transpose()?
                .as_deref()
                .unwrap_or("void"),
            name,
            func.parameter_types
                .iter()
                .zip(args.values())
                .map::<Result<_, Error>, _>(|(ty, name)| {
                    let ty = write_type(*ty, &module.types, &structs, None, features)?;

                    Ok(format!("{} {}", ty, name))
                })
                .collect::<Result<Vec<_>, _>>()?
                .join(","),
        )?;

        let locals: FastHashMap<_, _> = func
            .local_variables
            .iter()
            .map(|(handle, local)| (handle, namer(local.name.as_ref())))
            .collect();

        for (handle, name) in locals.iter() {
            writeln!(
                out,
                "\t{} {};",
                write_type(
                    func.local_variables[*handle].ty,
                    &module.types,
                    &structs,
                    None,
                    features
                )?,
                name
            )?;
        }

        writeln!(out)?;

        let builder = StatementBuilder {
            functions: &functions,
            globals: &globals_lookup,
            locals_lookup: &locals,
            structs: &structs,
            args: &args,
            expressions: &func.expressions,
            features,
            typifier: &typifier,
        };

        for sta in func.body.iter() {
            writeln!(out, "{}", write_statement(sta, module, &builder, 1)?)?;
        }

        writeln!(out, "}}")?;
    }

    Ok(mappings_map)
}

struct StatementBuilder<'a> {
    functions: &'a FastHashMap<Handle<Function>, String>,
    globals: &'a FastHashMap<Handle<GlobalVariable>, String>,
    locals_lookup: &'a FastHashMap<Handle<LocalVariable>, String>,
    structs: &'a FastHashMap<Handle<Type>, String>,
    args: &'a FastHashMap<u32, String>,
    expressions: &'a Arena<Expression>,
    features: SupportedFeatures,
    typifier: &'a Typifier,
}

fn write_statement<'a, 'b>(
    sta: &Statement,
    module: &'a Module,
    builder: &'b StatementBuilder<'a>,
    indent: usize,
) -> Result<String, Error> {
    Ok(match sta {
        Statement::Empty => String::new(),
        Statement::Block(block) => block
            .iter()
            .map(|sta| write_statement(sta, module, builder, indent))
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
                write_expression(&builder.expressions[*condition], module, builder)?
            )?;

            for sta in accept {
                writeln!(
                    &mut out,
                    "{}",
                    write_statement(sta, module, builder, indent + 1)?
                )?;
            }

            if !reject.is_empty() {
                writeln!(&mut out, "{}}} else {{", "\t".repeat(indent),)?;
                for sta in reject {
                    writeln!(
                        &mut out,
                        "{}",
                        write_statement(sta, module, builder, indent + 1)?
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
                write_expression(&builder.expressions[*selector], module, builder)?
            )?;

            for (label, (block, fallthrough)) in cases {
                writeln!(&mut out, "{}case {}:", "\t".repeat(indent + 1), label)?;

                for sta in block {
                    writeln!(
                        &mut out,
                        "{}",
                        write_statement(sta, module, builder, indent + 2)?
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
                        write_statement(sta, module, builder, indent + 2)?
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
                    write_statement(sta, module, builder, indent + 1)?
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
                    write_expression(&builder.expressions[*expr], module, builder)?
                )
            } else {
                String::from("return;")
            }
        ),
        Statement::Kill => format!("{}discard;", "\t".repeat(indent)),
        Statement::Store { pointer, value } => format!(
            "{}{} = {};",
            "\t".repeat(indent),
            write_expression(&builder.expressions[*pointer], module, builder)?,
            write_expression(&builder.expressions[*value], module, builder)?
        ),
    })
}

fn write_expression<'a, 'b>(
    expr: &Expression,
    module: &'a Module,
    builder: &'b StatementBuilder<'a>,
) -> Result<Cow<'a, str>, Error> {
    Ok(match *expr {
        Expression::Access { base, index } => {
            let base_expr = write_expression(&builder.expressions[base], module, builder)?;
            Cow::Owned(format!(
                "{}[{}]",
                base_expr,
                write_expression(&builder.expressions[index], module, builder)?
            ))
        }
        Expression::AccessIndex { base, index } => {
            let base_expr = write_expression(&builder.expressions[base], module, builder)?;

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
            builder.features,
        )?),
        Expression::Compose { ty, ref components } => {
            let constructor = match module.types[ty].inner {
                TypeInner::Vector { size, kind, width } => format!(
                    "{}vec{}",
                    map_scalar(kind, width, builder.features)?.prefix,
                    size as u8,
                ),
                TypeInner::Matrix {
                    columns,
                    rows,
                    kind,
                    width,
                } => format!(
                    "{}mat{}x{}",
                    map_scalar(kind, width, builder.features)?.prefix,
                    columns as u8,
                    rows as u8,
                ),
                TypeInner::Array { .. } => {
                    write_type(ty, &module.types, builder.structs, None, builder.features)?
                        .into_owned()
                }
                TypeInner::Struct { .. } => builder.structs.get(&ty).unwrap().clone(),
                _ => {
                    return Err(Error::Custom(format!(
                        "Cannot compose type {}",
                        write_type(ty, &module.types, builder.structs, None, builder.features)?
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
                        builder
                    )?))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(","),
            ))
        }
        Expression::FunctionParameter(pos) => Cow::Borrowed(builder.args.get(&pos).unwrap()),
        Expression::GlobalVariable(handle) => Cow::Borrowed(builder.globals.get(&handle).unwrap()),
        Expression::LocalVariable(handle) => {
            Cow::Borrowed(builder.locals_lookup.get(&handle).unwrap())
        }
        Expression::Load { pointer } => {
            write_expression(&builder.expressions[pointer], module, builder)?
        }
        Expression::ImageSample {
            image,
            sampler,
            coordinate,
            level,
            depth_ref,
        } => {
            let image_expr = write_expression(&builder.expressions[image], module, builder)?;
            write_expression(&builder.expressions[sampler], module, builder)?;
            let coordinate_expr =
                write_expression(&builder.expressions[coordinate], module, builder)?;

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
                    write_expression(&builder.expressions[depth_ref], module, builder)?
                ))
            } else {
                coordinate_expr
            };

            //TODO: handle MS
            Cow::Owned(match level {
                crate::SampleLevel::Auto => format!("texture({},{})", image_expr, coordinate_expr),
                crate::SampleLevel::Exact(expr) => {
                    let level_expr = write_expression(&builder.expressions[expr], module, builder)?;
                    format!(
                        "textureLod({}, {}, {})",
                        image_expr, coordinate_expr, level_expr
                    )
                }
                crate::SampleLevel::Bias(bias) => {
                    let bias_expr = write_expression(&builder.expressions[bias], module, builder)?;
                    format!("texture({},{},{})", image_expr, coordinate_expr, bias_expr)
                }
            })
        }
        Expression::ImageLoad {
            image,
            coordinate,
            index,
        } => {
            let image_expr = write_expression(&builder.expressions[image], module, builder)?;
            let coordinate_expr =
                write_expression(&builder.expressions[coordinate], module, builder)?;

            let (kind, dim, arrayed, class) = match *builder.typifier.get(image, &module.types) {
                TypeInner::Image {
                    kind,
                    dim,
                    arrayed,
                    class,
                } => (kind, dim, arrayed, class),
                ref other => return Err(Error::Custom(format!("Cannot load {:?}", other))),
            };

            Cow::Owned(match class {
                ImageClass::Sampled | ImageClass::Multisampled => {
                    let ms = match class {
                        crate::ImageClass::Multisampled => true,
                        _ => false,
                    };

                    //TODO: fix this
                    let sampler_constructor = format!(
                        "{}sampler{}{}{}({})",
                        map_scalar(kind, 4, builder.features)?.prefix,
                        ImageDimension(dim),
                        if ms { "MS" } else { "" },
                        if arrayed { "Array" } else { "" },
                        image_expr,
                    );

                    if !ms {
                        format!("texelFetch({},{})", sampler_constructor, coordinate_expr)
                    } else {
                        let index_expr =
                            write_expression(&builder.expressions[index], module, builder)?;

                        format!(
                            "texelFetch({},{},{})",
                            sampler_constructor, coordinate_expr, index_expr
                        )
                    }
                }
                ImageClass::Storage(_) => format!("imageLoad({},{})", image_expr, coordinate_expr),
                ImageClass::Depth => todo!(),
            })
        }
        Expression::Unary { op, expr } => {
            let base_expr = write_expression(&builder.expressions[expr], module, builder)?;

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
            let left_expr = write_expression(&builder.expressions[left], module, builder)?;
            let right_expr = write_expression(&builder.expressions[right], module, builder)?;

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
                BinaryOperator::ShiftLeftLogical => "<<",
                BinaryOperator::ShiftRightLogical => todo!(),
                BinaryOperator::ShiftRightArithmetic => ">>",
            };

            Cow::Owned(format!("({} {} {})", left_expr, op_str, right_expr))
        }
        Expression::Intrinsic { fun, argument } => {
            let expr = write_expression(&builder.expressions[argument], module, builder)?;

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
            let matrix_expr = write_expression(&builder.expressions[matrix], module, builder)?;
            Cow::Owned(format!("transpose({})", matrix_expr))
        }
        Expression::DotProduct(left, right) => {
            let left_expr = write_expression(&builder.expressions[left], module, builder)?;
            let right_expr = write_expression(&builder.expressions[right], module, builder)?;
            Cow::Owned(format!("dot({},{})", left_expr, right_expr))
        }
        Expression::CrossProduct(left, right) => {
            let left_expr = write_expression(&builder.expressions[left], module, builder)?;
            let right_expr = write_expression(&builder.expressions[right], module, builder)?;
            Cow::Owned(format!("cross({},{})", left_expr, right_expr))
        }
        Expression::As {
            expr,
            kind,
            convert,
        } => {
            let value_expr = write_expression(&builder.expressions[expr], module, builder)?;

            let (source_kind, ty_expr) = match *builder.typifier.get(expr, &module.types) {
                TypeInner::Scalar { width, kind } => (
                    kind,
                    Cow::Borrowed(map_scalar(kind, width, builder.features)?.full),
                ),
                TypeInner::Vector { width, kind, size } => (
                    kind,
                    Cow::Owned(format!(
                        "{}vec{}",
                        map_scalar(kind, width, builder.features)?.prefix,
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
            let expr = write_expression(&builder.expressions[expr], module, builder)?;

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
            ref origin,
            ref arguments,
        } => {
            match *origin {
                FunctionOrigin::Local(_) => {}
                FunctionOrigin::External(_) => {
                    write_expression(&builder.expressions[arguments[0]], module, builder)?;
                }
            };

            Cow::Owned(format!(
                "{}({})",
                match *origin {
                    FunctionOrigin::External(ref name) => name,
                    FunctionOrigin::Local(handle) => builder.functions.get(&handle).unwrap(),
                },
                arguments
                    .iter()
                    .map(|arg| Ok(write_expression(
                        &builder.expressions[*arg],
                        module,
                        builder
                    )?))
                    .collect::<Result<Vec<_>, Error>>()?
                    .join(","),
            ))
        }
        Expression::ArrayLength(expr) => {
            let base = write_expression(&builder.expressions[expr], module, builder)?;
            Cow::Owned(format!("uint({}.length())", base))
        }
    })
}

fn write_constant(
    constant: &Constant,
    module: &Module,
    builder: &StatementBuilder<'_>,
    features: SupportedFeatures,
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
                TypeInner::Array { .. } =>
                    write_type(constant.ty, &module.types, builder.structs, None, features)?,
                _ =>
                    return Err(Error::Custom(format!(
                        "Cannot build constant of type {}",
                        write_type(constant.ty, &module.types, builder.structs, None, features)?
                    ))),
            },
            components
                .iter()
                .map(|component| write_constant(
                    &module.constants[*component],
                    module,
                    builder,
                    features
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
    features: SupportedFeatures,
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
            8 if features.contains(SupportedFeatures::DOUBLE_TYPE) => ScalarString {
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

fn write_type<'a>(
    ty: Handle<Type>,
    types: &Arena<Type>,
    structs: &'a FastHashMap<Handle<Type>, String>,
    block: Option<String>,
    features: SupportedFeatures,
) -> Result<Cow<'a, str>, Error> {
    Ok(match types[ty].inner {
        TypeInner::Scalar { kind, width } => Cow::Borrowed(map_scalar(kind, width, features)?.full),
        TypeInner::Vector { size, kind, width } => Cow::Owned(format!(
            "{}vec{}",
            map_scalar(kind, width, features)?.prefix,
            size as u8
        )),
        TypeInner::Matrix {
            columns,
            rows,
            kind,
            width,
        } => Cow::Owned(format!(
            "{}mat{}x{}",
            if (width == 4 && kind == ScalarKind::Float)
                || features.contains(SupportedFeatures::NON_FLOAT_MATRICES)
            {
                map_scalar(kind, width, features)?.prefix
            } else {
                return Err(Error::Custom(format!(
                    "Cannot build matrix of base type {:?}",
                    kind
                )));
            },
            columns as u8,
            rows as u8
        )),
        TypeInner::Pointer { base, .. } => write_type(base, types, structs, None, features)?,
        TypeInner::Array { base, size, .. } => Cow::Owned(format!(
            "{}[{}]",
            write_type(base, types, structs, None, features)?,
            write_array_size(size)?
        )),
        TypeInner::Struct { ref members } => {
            if let Some(name) = block {
                let mut out = String::new();
                writeln!(&mut out, "{} {{", name)?;

                for (idx, member) in members.iter().enumerate() {
                    writeln!(
                        &mut out,
                        "\t{} {};",
                        write_type(member.ty, types, structs, None, features)?,
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
    kind: ScalarKind,
    dim: crate::ImageDimension,
    arrayed: bool,
    class: ImageClass,
    comparison: bool,
    features: SupportedFeatures,
) -> Result<String, Error> {
    if arrayed
        && dim != crate::ImageDimension::D2
        && !features.contains(SupportedFeatures::NON_2D_TEXTURE_ARRAYS)
    {
        return Err(Error::Custom(String::from(
            "Arrayed non 2d images aren't supported",
        )));
    }

    Ok(format!(
        "{}{}{}{}{}",
        map_scalar(kind, 4, features)?.prefix,
        match class {
            ImageClass::Storage(_) => "image",
            _ => "sampler",
        },
        ImageDimension(dim),
        write_image_flags(arrayed, class, features)?,
        if comparison { "Shadow" } else { "" }
    ))
}

fn write_storage_class(
    class: StorageClass,
    features: SupportedFeatures,
) -> Result<&'static str, Error> {
    Ok(match class {
        StorageClass::Constant => "",
        StorageClass::Function => "",
        StorageClass::Input => "in ",
        StorageClass::Output => "out ",
        StorageClass::Private => "",
        StorageClass::StorageBuffer => {
            if features.contains(SupportedFeatures::BUFFER_STORAGE) {
                "buffer "
            } else {
                return Err(Error::Custom(String::from(
                    "buffer storage class isn't supported in glsl es",
                )));
            }
        }
        StorageClass::Uniform => "uniform ",
        StorageClass::WorkGroup => {
            if features.contains(SupportedFeatures::SHARED_STORAGE) {
                "shared "
            } else {
                return Err(Error::Custom(String::from(
                    "workgroup storage class isn't supported in glsl es",
                )));
            }
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

fn write_array_size(size: ArraySize) -> Result<String, Error> {
    Ok(match size {
        ArraySize::Static(size) => size.to_string(),
        ArraySize::Dynamic => String::from(""),
    })
}

fn write_image_flags(
    arrayed: bool,
    class: ImageClass,
    features: SupportedFeatures,
) -> Result<String, Error> {
    let ms = match class {
        ImageClass::Multisampled => {
            if !features.contains(SupportedFeatures::MULTISAMPLED_TEXTURES) {
                return Err(Error::Custom(String::from(
                    "Multi sampled textures aren't supported",
                )));
            }
            if arrayed && !features.contains(SupportedFeatures::MULTISAMPLED_TEXTURE_ARRAYS) {
                return Err(Error::Custom(String::from(
                    "Multi sampled texture arrays aren't supported",
                )));
            }
            "MS"
        }
        _ => "",
    };

    let array = if arrayed { "Array" } else { "" };

    Ok(format!("{}{}", ms, array))
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
    features: SupportedFeatures,
) -> Result<(), Error> {
    if built_structs.get(&handle).is_some() {
        return Ok(());
    }

    let mut tmp = String::new();

    let name = structs.get(&handle).unwrap();

    writeln!(&mut tmp, "struct {} {{", name)?;
    for (idx, member) in members.iter().enumerate() {
        if let MemberOrigin::BuiltIn(_) = member.origin {
            continue;
        }

        if let TypeInner::Struct { ref members } = module.types[member.ty].inner {
            write_struct(
                member.ty,
                members,
                module,
                structs,
                out,
                built_structs,
                features,
            )?;
        }

        writeln!(
            &mut tmp,
            "\t{} {};",
            write_type(member.ty, &module.types, &structs, None, features)?,
            member
                .name
                .clone()
                .filter(|s| is_valid_ident(s))
                .unwrap_or_else(|| format!("_{}", idx))
        )?;
    }
    writeln!(&mut tmp, "}};")?;

    built_structs.insert(handle, ());

    writeln!(out, "{}", tmp)?;

    Ok(())
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

fn write_format_glsl(format: StorageFormat) -> &'static str {
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

fn collect_texture_mapping(
    module: &Module,
    functions: &FastHashMap<Handle<Function>, String>,
) -> Result<FastHashMap<Handle<GlobalVariable>, Option<Handle<GlobalVariable>>>, Error> {
    let mut mappings = FastHashMap::default();

    for function in functions.keys() {
        let func = &module.functions[*function];

        let mut interface = Interface {
            expressions: &func.expressions,
            visitor: TextureMappingVisitor {
                expressions: &func.expressions,
                map: &mut mappings,
                error: None,
            },
        };
        interface.traverse(&func.body);
    }

    Ok(mappings)
}
