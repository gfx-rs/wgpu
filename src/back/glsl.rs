use crate::{
    Arena, ArraySize, BinaryOperator, BuiltIn, Constant, ConstantInner, DerivativeAxis, Expression,
    FastHashMap, Function, FunctionOrigin, GlobalVariable, Handle, ImageFlags, IntrinsicFunction,
    LocalVariable, Module, ScalarKind, Statement, StorageClass, Type, TypeInner, UnaryOperator,
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

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::FormatError(err) => write!(f, "Formatting error {}", err),
            Error::IoError(err) => write!(f, "Io error: {}", err),
            Error::Custom(err) => write!(f, "{}", err),
        }
    }
}

pub fn write(module: &Module, out: &mut impl Write) -> Result<(), Error> {
    writeln!(out, "#version 450 core")?;

    let mut counter = 0;
    let mut names = FastHashMap::default();

    let mut namer = |name: Option<&String>| {
        if let Some(name) = name {
            names.insert(name.clone(), ());
            name.clone()
        } else {
            counter += 1;
            while names.get(&format!("_{}", counter)).is_some() {
                counter += 1;
            }
            format!("_{}", counter)
        }
    };

    let mut structs = FastHashMap::default();

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

    // Do a second pass to build the structs
    // TODO: glsl is order dependent so we need to build structs in order
    for (handle, ty) in module.types.iter() {
        match ty.inner {
            TypeInner::Struct { ref members } => {
                let name = structs.get(&handle).unwrap();

                writeln!(out, "struct {} {{", name)?;
                for (idx, member) in members.iter().enumerate() {
                    writeln!(
                        out,
                        "   {} {};",
                        write_type(member.ty, &module.types, &structs)?,
                        member.name.clone().unwrap_or_else(|| idx.to_string())
                    )?;
                }
                writeln!(out, "}};")?;
            }
            _ => continue,
        }
    }

    let mut globals_lookup = FastHashMap::default();

    for (handle, global) in module.global_variables.iter() {
        if let Some(crate::Binding::BuiltIn(built_in)) = global.binding {
            let semantic = match built_in {
                BuiltIn::Position => "gl_position",
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

        if let Some(ref binding) = global.binding {
            write!(out, "layout({}) ", Binding(binding.clone()))?;
        }

        let name = namer(global.name.as_ref());

        writeln!(
            out,
            "{}{} {};",
            write_storage_class(global.class)?,
            write_type(global.ty, &module.types, &structs)?,
            name
        )?;

        globals_lookup.insert(handle, name);
    }

    let mut functions = FastHashMap::default();

    // Do a first pass to collect names
    for (handle, func) in module.functions.iter() {
        functions.insert(handle, namer(func.name.as_ref()));
    }

    // TODO: glsl is order dependent so we need to build functions in order
    for (handle, func) in module.functions.iter() {
        let name = functions.get(&handle).unwrap();

        writeln!(
            out,
            "{} {}({}) {{",
            func.return_type
                .map_or(Ok(String::from("void")), |ty| write_type(
                    ty,
                    &module.types,
                    &structs
                ))?,
            name,
            func.parameter_types
                .iter()
                .map(|ty| write_type(*ty, &module.types, &structs))
                .collect::<Result<Vec<_>, _>>()?
                .join(","),
        )?;

        let locals: FastHashMap<_, _> = func
            .local_variables
            .iter()
            .map(|(handle, local)| (handle, namer(local.name.as_ref())))
            .collect();

        let mut builder = StatementBuilder {
            functions: &functions,
            globals: &globals_lookup,
            locals_lookup: &locals,
            structs: &structs,
            args: &func
                .parameter_types
                .iter()
                .enumerate()
                .map(|(pos, ty)| (pos as u32, (namer(None), *ty)))
                .collect(),
            expressions: &func.expressions,
            locals: &func.local_variables,
        };

        for (handle, name) in locals.iter() {
            let ty = write_type(func.local_variables[*handle].ty, &module.types, &structs)?;
            let init = func.local_variables[*handle].init;
            if let Some(init) = init {
                writeln!(
                    out,
                    "{} {} = {init};",
                    ty,
                    name,
                    init = write_expression(&func.expressions[init], &module, &mut builder)?.0,
                )?;
            } else {
                writeln!(out, "{} {};", ty, name,)?;
            }
        }

        for sta in func.body.iter() {
            writeln!(out, "{}", write_statement(sta, module, &mut builder)?)?;
        }

        writeln!(out, "}}")?;
    }

    Ok(())
}

struct Binding(crate::Binding);
impl fmt::Display for Binding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            crate::Binding::BuiltIn(_) => write!(f, ""), // Ignore because they are variables with a predefined name
            crate::Binding::Location(location) => write!(f, "location={}", location),
            crate::Binding::Descriptor { set, binding } => {
                write!(f, "set={},binding={}", set, binding)
            }
        }
    }
}

struct StatementBuilder<'a> {
    pub functions: &'a FastHashMap<Handle<Function>, String>,
    pub globals: &'a FastHashMap<Handle<GlobalVariable>, String>,
    pub locals_lookup: &'a FastHashMap<Handle<LocalVariable>, String>,
    pub structs: &'a FastHashMap<Handle<Type>, String>,
    pub args: &'a FastHashMap<u32, (String, Handle<Type>)>,
    pub expressions: &'a Arena<Expression>,
    pub locals: &'a Arena<LocalVariable>,
}

fn write_statement(
    sta: &Statement,
    module: &Module,
    builder: &mut StatementBuilder<'_>,
) -> Result<String, Error> {
    Ok(match sta {
        Statement::Empty => String::new(),
        Statement::Block(block) => block
            .iter()
            .map(|sta| write_statement(sta, module, builder))
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
                "if({}) {{",
                write_expression(&builder.expressions[*condition], module, builder)?.0
            )?;
            for sta in accept {
                writeln!(&mut out, "{}", write_statement(sta, module, builder)?)?;
            }
            writeln!(&mut out, "}} else {{")?;
            for sta in reject {
                writeln!(&mut out, "{}", write_statement(sta, module, builder)?)?;
            }
            write!(&mut out, "}}")?;

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
                "switch({}) {{",
                write_expression(&builder.expressions[*selector], module, builder)?.0
            )?;

            for (label, (block, fallthrough)) in cases {
                writeln!(&mut out, "   case {}:", label)?;

                for sta in block {
                    writeln!(&mut out, "      {}", write_statement(sta, module, builder)?)?;
                }

                if fallthrough.is_some() {
                    writeln!(&mut out, "      break;")?;
                }
            }

            writeln!(&mut out, "   default:")?;

            for sta in default {
                writeln!(&mut out, "      {}", write_statement(sta, module, builder)?)?;
            }

            write!(&mut out, "}}")?;

            out
        }
        Statement::Loop { body, continuing } => {
            let mut out = String::new();

            writeln!(&mut out, "while(true) {{",)?;

            for sta in body.iter().chain(continuing.iter()) {
                writeln!(&mut out, "    {}", write_statement(sta, module, builder)?)?;
            }

            write!(&mut out, "}}")?;

            out
        }
        Statement::Break => String::from("break;"),
        Statement::Continue => String::from("continue;"),
        Statement::Return { value } => format!(
            "return  {};",
            value.map_or::<Result<_, Error>, _>(Ok(String::from("")), |expr| Ok(
                write_expression(&builder.expressions[expr], module, builder)?.0
            ))?
        ),
        Statement::Kill => String::from("discard;"),
        Statement::Store { pointer, value } => format!(
            "{} = {};",
            write_expression(&builder.expressions[*pointer], module, builder)?.0,
            write_expression(&builder.expressions[*value], module, builder)?.0
        ),
    })
}

fn write_expression<'a>(
    expr: &Expression,
    module: &'a Module,
    builder: &mut StatementBuilder<'_>,
) -> Result<(String, Cow<'a, TypeInner>), Error> {
    Ok(match expr {
        Expression::Access { base, index } => {
            let (base_expr, ty) = write_expression(&builder.expressions[*base], module, builder)?;

            let inner = match ty.as_ref() {
                TypeInner::Vector { kind, width, .. } | TypeInner::Matrix { kind, width, .. } => {
                    Cow::Owned(TypeInner::Scalar {
                        kind: *kind,
                        width: *width,
                    })
                }
                TypeInner::Array { base, .. } => Cow::Borrowed(&module.types[*base].inner),
                _ => return Err(Error::Custom(format!("Cannot dynamically index {:?}", ty))),
            };

            (
                format!(
                    "{}[{}]",
                    base_expr,
                    write_expression(&builder.expressions[*index], module, builder)?.0
                ),
                inner,
            )
        }
        Expression::AccessIndex { base, index } => {
            let (base_expr, ty) = write_expression(&builder.expressions[*base], module, builder)?;

            match ty.as_ref() {
                TypeInner::Vector { kind, width, .. } | TypeInner::Matrix { kind, width, .. } => (
                    format!("{}[{}]", base_expr, index),
                    Cow::Owned(TypeInner::Scalar {
                        kind: *kind,
                        width: *width,
                    }),
                ),
                TypeInner::Array { base, .. } => (
                    format!("{}[{}]", base_expr, index),
                    Cow::Borrowed(&module.types[*base].inner),
                ),
                TypeInner::Struct { members } => (
                    format!(
                        "{}.{}",
                        base_expr,
                        members[*index as usize]
                            .name
                            .as_ref()
                            .unwrap_or(&index.to_string())
                    ),
                    Cow::Borrowed(&module.types[members[*index as usize].ty].inner),
                ),
                _ => return Err(Error::Custom(format!("Cannot index {:?}", ty))),
            }
        }
        Expression::Constant(constant) => (
            write_constant(&module.constants[*constant], module, builder)?,
            Cow::Borrowed(&module.types[module.constants[*constant].ty].inner),
        ),
        Expression::Compose { ty, components } => {
            let constructor = match module.types[*ty].inner {
                TypeInner::Vector { size, kind, width } => format!(
                    "{}vec{}",
                    match kind {
                        ScalarKind::Sint => "i",
                        ScalarKind::Uint => "u",
                        ScalarKind::Float => match width {
                            4 => "",
                            8 => "d",
                            _ =>
                                return Err(Error::Custom(format!(
                                    "Cannot build float of width {}",
                                    width
                                ))),
                        },
                        ScalarKind::Bool => "b",
                    },
                    size as u8,
                ),
                TypeInner::Matrix {
                    columns,
                    rows,
                    kind,
                    width,
                } => format!(
                    "{}mat{}x{}",
                    match kind {
                        ScalarKind::Sint => "i",
                        ScalarKind::Uint => "u",
                        ScalarKind::Float => match width {
                            4 => "",
                            8 => "d",
                            _ =>
                                return Err(Error::Custom(format!(
                                    "Cannot build float of width {}",
                                    width
                                ))),
                        },
                        ScalarKind::Bool => "b",
                    },
                    columns as u8,
                    rows as u8,
                ),
                TypeInner::Array { .. } => write_type(*ty, &module.types, builder.structs)?,
                TypeInner::Struct { .. } => builder.structs.get(ty).unwrap().clone(),
                _ => {
                    return Err(Error::Custom(format!(
                        "Cannot compose type {}",
                        write_type(*ty, &module.types, builder.structs)?
                    )))
                }
            };

            (
                format!(
                    "{}({})",
                    constructor,
                    components
                        .iter()
                        .map::<Result<_, Error>, _>(|arg| Ok(write_expression(
                            &builder.expressions[*arg],
                            module,
                            builder
                        )?
                        .0))
                        .collect::<Result<Vec<_>, _>>()?
                        .join(","),
                ),
                Cow::Borrowed(&module.types[*ty].inner),
            )
        }
        Expression::FunctionParameter(pos) => {
            let (arg, ty) = builder.args.get(&pos).unwrap().clone();

            (arg, Cow::Borrowed(&module.types[ty].inner))
        }
        Expression::GlobalVariable(handle) => (
            builder.globals.get(&handle).unwrap().clone(),
            Cow::Borrowed(&module.types[module.global_variables[*handle].ty].inner),
        ),
        Expression::LocalVariable(handle) => (
            builder.locals_lookup.get(&handle).unwrap().clone(),
            Cow::Borrowed(&module.types[builder.locals[*handle].ty].inner),
        ),
        Expression::Load { pointer } => {
            write_expression(&builder.expressions[*pointer], module, builder)?
        }
        Expression::ImageSample {
            image,
            sampler,
            coordinate,
            depth_ref,
        } => {
            let (image_expr, image_ty) =
                write_expression(&builder.expressions[*image], module, builder)?;
            let (sampler_expr, sampler_ty) =
                write_expression(&builder.expressions[*sampler], module, builder)?;
            let (coordinate_expr, coordinate_ty) =
                write_expression(&builder.expressions[*coordinate], module, builder)?;

            let (kind, dim, arrayed, ms, width) = match image_ty.as_ref() {
                TypeInner::Image { base, dim, flags } => match module.types[*base].inner {
                    TypeInner::Scalar { kind, width } => (
                        kind,
                        *dim,
                        flags.contains(ImageFlags::ARRAYED),
                        flags.contains(ImageFlags::MULTISAMPLED),
                        width,
                    ),
                    _ => {
                        return Err(Error::Custom(format!(
                            "Cannot build image of {}",
                            write_type(*base, &module.types, builder.structs)?
                        )))
                    }
                },
                TypeInner::DepthImage { dim, arrayed } => {
                    (ScalarKind::Float, *dim, *arrayed, false, 4)
                }
                _ => return Err(Error::Custom(format!("Cannot sample {:?}", image_ty))),
            };

            let shadow = match sampler_ty.as_ref() {
                TypeInner::Sampler { comparison } => *comparison,
                _ => {
                    return Err(Error::Custom(format!(
                        "Cannot have a sampler of {:?}",
                        sampler_ty
                    )))
                }
            };

            let size = match coordinate_ty.as_ref() {
                TypeInner::Vector { size, .. } => *size,
                _ => {
                    return Err(Error::Custom(format!(
                        "Cannot sample with coordinates of type {:?}",
                        coordinate_ty
                    )))
                }
            };

            let sampler_constructor = format!(
                "{}sampler{}{}{}{}({},{})",
                match kind {
                    ScalarKind::Sint => "i",
                    ScalarKind::Uint => "u",
                    ScalarKind::Float => "",
                    _ => return Err(Error::Custom(String::from("Cannot build image of bools",))),
                },
                ImageDimension(dim),
                if ms { "MS" } else { "" },
                if arrayed { "Array" } else { "" },
                if shadow { "Shadow" } else { "" },
                image_expr,
                sampler_expr
            );

            let coordinate = if let Some(depth_ref) = depth_ref {
                format!(
                    "vec{}({},{})",
                    size as u8 + 1,
                    coordinate_expr,
                    write_expression(&builder.expressions[*depth_ref], module, builder)?.0
                )
            } else {
                coordinate_expr
            };

            let expr = if !ms {
                format!("texture({},{})", sampler_constructor, coordinate)
            } else {
                todo!()
            };

            let ty = if shadow {
                Cow::Owned(TypeInner::Scalar { kind, width })
            } else {
                Cow::Owned(TypeInner::Vector { kind, width, size })
            };

            (expr, ty)
        }
        Expression::Unary { op, expr } => {
            let (expr, ty) = write_expression(&builder.expressions[*expr], module, builder)?;

            (
                format!(
                    "({} {})",
                    match op {
                        UnaryOperator::Negate => "-",
                        UnaryOperator::Not => "~",
                    },
                    expr
                ),
                ty,
            )
        }
        Expression::Binary { op, left, right } => {
            let (left_expr, left_ty) =
                write_expression(&builder.expressions[*left], module, builder)?;
            let (right_expr, right_ty) =
                write_expression(&builder.expressions[*right], module, builder)?;

            let op = match op {
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

            let ty = match (left_ty.as_ref(), right_ty.as_ref()) {
                (TypeInner::Scalar { .. }, TypeInner::Scalar { .. }) => left_ty,
                (TypeInner::Scalar { .. }, TypeInner::Vector { .. }) => right_ty,
                (TypeInner::Scalar { .. }, TypeInner::Matrix { .. }) => right_ty,
                (TypeInner::Vector { .. }, TypeInner::Scalar { .. }) => left_ty,
                (TypeInner::Vector { .. }, TypeInner::Vector { .. }) => left_ty,
                (TypeInner::Vector { .. }, TypeInner::Matrix { .. }) => left_ty,
                (TypeInner::Matrix { .. }, TypeInner::Scalar { .. }) => left_ty,
                (TypeInner::Matrix { .. }, TypeInner::Vector { .. }) => right_ty,
                (TypeInner::Matrix { .. }, TypeInner::Matrix { .. }) => left_ty,
                _ => {
                    return Err(Error::Custom(format!(
                        "Cannot apply {} to {} and {}",
                        op, left_expr, right_expr
                    )))
                }
            };

            (format!("({} {} {})", left_expr, op, right_expr), ty)
        }
        Expression::Intrinsic { fun, argument } => {
            let (expr, ty) = write_expression(&builder.expressions[*argument], module, builder)?;

            (
                format!(
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
                ),
                ty,
            )
        }
        Expression::DotProduct(left, right) => {
            let (left_expr, left_ty) =
                write_expression(&builder.expressions[*left], module, builder)?;
            let (right_expr, _) = write_expression(&builder.expressions[*right], module, builder)?;

            let ty = match left_ty.as_ref() {
                TypeInner::Vector { kind, width, .. } => Cow::Owned(TypeInner::Scalar {
                    kind: *kind,
                    width: *width,
                }),
                _ => {
                    return Err(Error::Custom(format!(
                        "Cannot apply dot product to {}",
                        left_expr
                    )))
                }
            };

            (format!("dot({},{})", left_expr, right_expr), ty)
        }
        Expression::CrossProduct(left, right) => {
            let (left_expr, left_ty) =
                write_expression(&builder.expressions[*left], module, builder)?;
            let (right_expr, _) = write_expression(&builder.expressions[*right], module, builder)?;

            (format!("cross({},{})", left_expr, right_expr), left_ty)
        }
        Expression::Derivative { axis, expr } => {
            let (expr, ty) = write_expression(&builder.expressions[*expr], module, builder)?;

            (
                format!(
                    "{}({})",
                    match axis {
                        DerivativeAxis::X => "dFdx",
                        DerivativeAxis::Y => "dFdy",
                        DerivativeAxis::Width => "fwidth",
                    },
                    expr
                ),
                ty,
            )
        }
        Expression::Call { origin, arguments } => {
            let ty = match origin {
                FunctionOrigin::Local(function) => module.functions[*function]
                    .return_type
                    .map(|ty| Cow::Borrowed(&module.types[ty].inner))
                    .unwrap_or(Cow::Owned(
                        TypeInner::Sampler { comparison: false }, /*Dummy type*/
                    )),
                FunctionOrigin::External(_) => {
                    write_expression(&builder.expressions[arguments[0]], module, builder)?.1
                }
            };

            (
                format!(
                    "{}({})",
                    match origin {
                        FunctionOrigin::External(name) => name,
                        FunctionOrigin::Local(handle) => builder.functions.get(&handle).unwrap(),
                    },
                    arguments
                        .iter()
                        .map::<Result<_, Error>, _>(|arg| Ok(write_expression(
                            &builder.expressions[*arg],
                            module,
                            builder
                        )?
                        .0))
                        .collect::<Result<Vec<_>, _>>()?
                        .join(","),
                ),
                ty,
            )
        }
    })
}

fn write_constant(
    constant: &Constant,
    module: &Module,
    builder: &StatementBuilder<'_>,
) -> Result<String, Error> {
    Ok(match constant.inner {
        ConstantInner::Sint(int) => int.to_string(),
        ConstantInner::Uint(int) => int.to_string(),
        ConstantInner::Float(float) => format!("{:?}", float),
        ConstantInner::Bool(boolean) => boolean.to_string(),
        ConstantInner::Composite(ref components) => format!(
            "{}({})",
            match module.types[constant.ty].inner {
                TypeInner::Vector { size, .. } => format!("vec{}", size as u8,),
                TypeInner::Matrix { columns, rows, .. } =>
                    format!("mat{}x{}", columns as u8, rows as u8,),
                TypeInner::Struct { .. } => builder.structs.get(&constant.ty).unwrap().clone(),
                TypeInner::Array { .. } => write_type(constant.ty, &module.types, builder.structs)?,
                _ =>
                    return Err(Error::Custom(format!(
                        "Cannot build constant of type {}",
                        write_type(constant.ty, &module.types, builder.structs)?
                    ))),
            },
            components
                .iter()
                .map(|component| write_constant(&module.constants[*component], module, builder))
                .collect::<Result<Vec<_>, _>>()?
                .join(","),
        ),
    })
}

fn write_type<'a>(
    ty: Handle<Type>,
    types: &'a Arena<Type>,
    structs: &'a FastHashMap<Handle<Type>, String>,
) -> Result<String, Error> {
    Ok(match types[ty].inner {
        TypeInner::Scalar { kind, width } => match kind {
            ScalarKind::Sint => String::from("int"),
            ScalarKind::Uint => String::from("uint"),
            ScalarKind::Float => match width {
                4 => String::from("float"),
                8 => String::from("double"),
                _ => {
                    return Err(Error::Custom(format!(
                        "Cannot build float of width {}",
                        width
                    )))
                }
            },
            ScalarKind::Bool => String::from("bool"),
        },
        TypeInner::Vector { size, kind, width } => format!(
            "{}vec{}",
            match kind {
                ScalarKind::Sint => "i",
                ScalarKind::Uint => "u",
                ScalarKind::Float => match width {
                    4 => "",
                    8 => "d",
                    _ =>
                        return Err(Error::Custom(format!(
                            "Cannot build float of width {}",
                            width
                        ))),
                },
                ScalarKind::Bool => "b",
            },
            size as u8
        ),
        TypeInner::Matrix {
            columns,
            rows,
            kind,
            width,
        } => format!(
            "{}mat{}x{}",
            match kind {
                ScalarKind::Sint => "i",
                ScalarKind::Uint => "u",
                ScalarKind::Float => match width {
                    4 => "",
                    8 => "d",
                    _ =>
                        return Err(Error::Custom(format!(
                            "Cannot build float of width {}",
                            width
                        ))),
                },
                ScalarKind::Bool => "b",
            },
            columns as u8,
            rows as u8
        ),
        TypeInner::Pointer { base, .. } => write_type(base, types, structs)?,
        TypeInner::Array { base, size, .. } => format!(
            "{}[{}]",
            write_type(base, types, structs)?,
            write_array_size(size)?
        ),
        TypeInner::Struct { .. } => structs.get(&ty).unwrap().clone(),
        TypeInner::Image { base, dim, flags } => format!(
            "{}texture{}{}",
            match types[base].inner {
                TypeInner::Scalar { kind, .. } => match kind {
                    ScalarKind::Sint => "i",
                    ScalarKind::Uint => "u",
                    ScalarKind::Float => "",
                    _ =>
                        return Err(Error::Custom(String::from(
                            "Cannot build image of booleans",
                        ))),
                },
                _ =>
                    return Err(Error::Custom(format!(
                        "Cannot build image of type {}",
                        write_type(base, types, structs)?
                    ))),
            },
            ImageDimension(dim),
            write_image_flags(flags)?
        ),
        TypeInner::DepthImage { dim, arrayed } => format!(
            "texture{}{}",
            ImageDimension(dim),
            if arrayed { "Array" } else { "" }
        ),
        TypeInner::Sampler { comparison } => String::from(if comparison {
            "sampler"
        } else {
            "samplerShadow"
        }),
    })
}

fn write_storage_class(class: StorageClass) -> Result<String, Error> {
    Ok(String::from(match class {
        StorageClass::Constant => "const ",
        StorageClass::Function => "",
        StorageClass::Input => "in ",
        StorageClass::Output => "out ",
        StorageClass::Private => "",
        StorageClass::StorageBuffer => "buffer ",
        StorageClass::Uniform => "uniform ",
        StorageClass::WorkGroup => "shared ",
    }))
}

fn write_array_size(size: ArraySize) -> Result<String, Error> {
    Ok(match size {
        ArraySize::Static(size) => size.to_string(),
        ArraySize::Dynamic => String::from(""),
    })
}

fn write_image_flags(flags: ImageFlags) -> Result<String, Error> {
    let mut out = String::new();

    if flags.contains(ImageFlags::MULTISAMPLED) {
        write!(out, "MS")?;
    }

    if flags.contains(ImageFlags::ARRAYED) {
        write!(out, "Array")?;
    }

    Ok(out)
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
