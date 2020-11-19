use super::{keywords::RESERVED, Error, LocationMode, Options, ResolvedBinding};
use crate::{
    arena::Handle,
    proc::{EntryPointIndex, NameKey, Namer, ResolveContext, Typifier},
    FastHashMap,
};
use std::{
    fmt::{Display, Error as FmtError, Formatter},
    io::Write,
};

struct Level(usize);
impl Level {
    fn next(&self) -> Self {
        Level(self.0 + 1)
    }
}
impl Display for Level {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), FmtError> {
        (0..self.0).map(|_| formatter.write_str("\t")).collect()
    }
}

struct TypedGlobalVariable<'a> {
    module: &'a crate::Module,
    names: &'a FastHashMap<NameKey, String>,
    handle: Handle<crate::GlobalVariable>,
    usage: crate::GlobalUse,
}

impl<'a> TypedGlobalVariable<'a> {
    fn try_fmt<W: Write>(&self, out: &mut W) -> Result<(), Error> {
        let var = &self.module.global_variables[self.handle];
        let name = &self.names[&NameKey::GlobalVariable(self.handle)];
        let ty = &self.module.types[var.ty];
        let ty_name = &self.names[&NameKey::Type(var.ty)];

        let (space_qualifier, reference) = match ty.inner {
            crate::TypeInner::Struct { .. } => match var.class {
                crate::StorageClass::Uniform | crate::StorageClass::Storage => {
                    let space = if self.usage.contains(crate::GlobalUse::STORE) {
                        "device "
                    } else {
                        "constant "
                    };
                    (space, "&")
                }
                _ => ("", ""),
            },
            _ => ("", ""),
        };
        Ok(write!(
            out,
            "{}{}{} {}",
            space_qualifier, ty_name, reference, name
        )?)
    }
}

pub struct Writer<W> {
    out: W,
    names: FastHashMap<NameKey, String>,
    typifier: Typifier,
}

fn scalar_kind_string(kind: crate::ScalarKind) -> &'static str {
    match kind {
        crate::ScalarKind::Float => "float",
        crate::ScalarKind::Sint => "int",
        crate::ScalarKind::Uint => "uint",
        crate::ScalarKind::Bool => "bool",
    }
}

fn vector_size_string(size: crate::VectorSize) -> &'static str {
    match size {
        crate::VectorSize::Bi => "2",
        crate::VectorSize::Tri => "3",
        crate::VectorSize::Quad => "4",
    }
}

const OUTPUT_STRUCT_NAME: &str = "output";
const LOCATION_INPUT_STRUCT_NAME: &str = "input";
const COMPONENTS: &[char] = &['x', 'y', 'z', 'w'];

fn separate(is_last: bool) -> &'static str {
    if is_last {
        ""
    } else {
        ","
    }
}

enum FunctionOrigin {
    Handle(Handle<crate::Function>),
    EntryPoint(EntryPointIndex),
}

struct ExpressionContext<'a> {
    function: &'a crate::Function,
    origin: FunctionOrigin,
    module: &'a crate::Module,
}

impl<W: Write> Writer<W> {
    /// Creates a new `Writer` instance.
    pub fn new(out: W) -> Self {
        Writer {
            out,
            names: FastHashMap::default(),
            typifier: Typifier::new(),
        }
    }

    /// Finishes writing and returns the output.
    pub fn finish(self) -> W {
        self.out
    }

    fn put_call(
        &mut self,
        name: &str,
        parameters: &[Handle<crate::Expression>],
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        if !name.is_empty() {
            write!(self.out, "metal::{}", name)?;
        }
        write!(self.out, "(")?;
        for (i, &handle) in parameters.iter().enumerate() {
            if i != 0 {
                write!(self.out, ", ")?;
            }
            self.put_expression(handle, context)?;
        }
        write!(self.out, ")")?;
        Ok(())
    }

    fn put_expression(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        let expression = &context.function.expressions[expr_handle];
        log::trace!("expression {:?} = {:?}", expr_handle, expression);
        match *expression {
            crate::Expression::Access { base, index } => {
                self.put_expression(base, context)?;
                write!(self.out, "[")?;
                self.put_expression(index, context)?;
                write!(self.out, "]")?;
            }
            crate::Expression::AccessIndex { base, index } => {
                self.put_expression(base, context)?;
                let resolved = self.typifier.get(base, &context.module.types);
                match *resolved {
                    crate::TypeInner::Struct { .. } => {
                        let base_ty = self.typifier.get_handle(base).unwrap();
                        let name = &self.names[&NameKey::StructMember(base_ty, index)];
                        write!(self.out, ".{}", name)?;
                    }
                    crate::TypeInner::Matrix { .. } | crate::TypeInner::Vector { .. } => {
                        write!(self.out, ".{}", COMPONENTS[index as usize])?;
                    }
                    crate::TypeInner::Array { .. } => {
                        write!(self.out, "[{}]", index)?;
                    }
                    _ => {
                        // unexpected indexing, should fail validation
                    }
                }
            }
            crate::Expression::Constant(handle) => self.put_constant(handle, context.module)?,
            crate::Expression::Compose { ty, ref components } => {
                let inner = &context.module.types[ty].inner;
                match *inner {
                    crate::TypeInner::Vector { size, kind, .. } => {
                        write!(
                            self.out,
                            "{}{}",
                            scalar_kind_string(kind),
                            vector_size_string(size)
                        )?;
                        self.put_call("", components, context)?;
                    }
                    crate::TypeInner::Scalar { width: 4, kind } if components.len() == 1 => {
                        write!(self.out, "{}", scalar_kind_string(kind),)?;
                        self.put_call("", components, context)?;
                    }
                    _ => return Err(Error::UnsupportedCompose(ty)),
                }
            }
            crate::Expression::FunctionArgument(index) => {
                let fun_handle = match context.origin {
                    FunctionOrigin::Handle(handle) => handle,
                    FunctionOrigin::EntryPoint(_) => unreachable!(),
                };
                let name = &self.names[&NameKey::FunctionArgument(fun_handle, index)];
                write!(self.out, "{}", name)?;
            }
            crate::Expression::GlobalVariable(handle) => {
                let var = &context.module.global_variables[handle];
                match var.class {
                    crate::StorageClass::Output => {
                        if let crate::TypeInner::Struct { .. } = context.module.types[var.ty].inner
                        {
                            return Ok(());
                        }
                        write!(self.out, "{}.", OUTPUT_STRUCT_NAME)?;
                    }
                    crate::StorageClass::Input => {
                        if let Some(crate::Binding::Location(_)) = var.binding {
                            write!(self.out, "{}.", LOCATION_INPUT_STRUCT_NAME)?;
                        }
                    }
                    _ => {}
                }
                let name = &self.names[&NameKey::GlobalVariable(handle)];
                write!(self.out, "{}", name)?;
            }
            crate::Expression::LocalVariable(handle) => {
                let name_key = match context.origin {
                    FunctionOrigin::Handle(fun_handle) => {
                        NameKey::FunctionLocal(fun_handle, handle)
                    }
                    FunctionOrigin::EntryPoint(ep_index) => {
                        NameKey::EntryPointLocal(ep_index, handle)
                    }
                };
                let name = &self.names[&name_key];
                write!(self.out, "{}", name)?;
            }
            crate::Expression::Load { pointer } => {
                //write!(self.out, "*")?;
                self.put_expression(pointer, context)?;
            }
            crate::Expression::ImageSample {
                image,
                sampler,
                coordinate,
                level,
                depth_ref,
            } => {
                let op = match depth_ref {
                    Some(_) => "sample_compare",
                    None => "sample",
                };
                //TODO: handle arrayed images
                self.put_expression(image, context)?;
                write!(self.out, ".{}(", op)?;
                self.put_expression(sampler, context)?;
                write!(self.out, ", ")?;
                self.put_expression(coordinate, context)?;
                if let Some(dref) = depth_ref {
                    write!(self.out, ", ")?;
                    self.put_expression(dref, context)?;
                }
                match level {
                    crate::SampleLevel::Auto => {}
                    crate::SampleLevel::Zero => {
                        write!(self.out, ", level(0)")?;
                    }
                    crate::SampleLevel::Exact(h) => {
                        write!(self.out, ", level(")?;
                        self.put_expression(h, context)?;
                        write!(self.out, ")")?;
                    }
                    crate::SampleLevel::Bias(h) => {
                        write!(self.out, ", bias(")?;
                        self.put_expression(h, context)?;
                        write!(self.out, ")")?;
                    }
                }
                write!(self.out, ")")?;
            }
            crate::Expression::ImageLoad {
                image,
                coordinate,
                index,
            } => {
                //TODO: handle arrayed images
                self.put_expression(image, context)?;
                write!(self.out, ".read(")?;
                self.put_expression(coordinate, context)?;
                if let Some(index) = index {
                    write!(self.out, ", ")?;
                    self.put_expression(index, context)?;
                }
                write!(self.out, ")")?;
            }
            crate::Expression::Unary { op, expr } => {
                let op_str = match op {
                    crate::UnaryOperator::Negate => "-",
                    crate::UnaryOperator::Not => "!",
                };
                write!(self.out, "{}", op_str)?;
                self.put_expression(expr, context)?;
            }
            crate::Expression::Binary { op, left, right } => {
                let op_str = match op {
                    crate::BinaryOperator::Add => "+",
                    crate::BinaryOperator::Subtract => "-",
                    crate::BinaryOperator::Multiply => "*",
                    crate::BinaryOperator::Divide => "/",
                    crate::BinaryOperator::Modulo => "%",
                    crate::BinaryOperator::Equal => "==",
                    crate::BinaryOperator::NotEqual => "!=",
                    crate::BinaryOperator::Less => "<",
                    crate::BinaryOperator::LessEqual => "<=",
                    crate::BinaryOperator::Greater => "==",
                    crate::BinaryOperator::GreaterEqual => ">=",
                    crate::BinaryOperator::And => "&",
                    _ => return Err(Error::UnsupportedBinaryOp(op)),
                };
                let kind = self
                    .typifier
                    .get(left, &context.module.types)
                    .scalar_kind()
                    .ok_or(Error::UnsupportedBinaryOp(op))?;
                if op == crate::BinaryOperator::Modulo && kind == crate::ScalarKind::Float {
                    write!(self.out, "fmod(")?;
                    self.put_expression(left, context)?;
                    write!(self.out, ", ")?;
                    self.put_expression(right, context)?;
                    write!(self.out, ")")?;
                } else {
                    //write!(self.out, "(")?;
                    self.put_expression(left, context)?;
                    write!(self.out, " {} ", op_str)?;
                    self.put_expression(right, context)?;
                    //write!(self.out, ")")?;
                }
            }
            crate::Expression::Select {
                condition,
                accept,
                reject,
            } => {
                write!(self.out, "(")?;
                self.put_expression(condition, context)?;
                write!(self.out, " ? ")?;
                self.put_expression(accept, context)?;
                write!(self.out, " : ")?;
                self.put_expression(reject, context)?;
                write!(self.out, ")")?;
            }
            crate::Expression::Intrinsic { fun, argument } => {
                let op = match fun {
                    crate::IntrinsicFunction::Any => "any",
                    crate::IntrinsicFunction::All => "all",
                    crate::IntrinsicFunction::IsNan => "",
                    crate::IntrinsicFunction::IsInf => "",
                    crate::IntrinsicFunction::IsFinite => "",
                    crate::IntrinsicFunction::IsNormal => "",
                };
                self.put_call(op, &[argument], context)?;
            }
            crate::Expression::Transpose(expr) => {
                self.put_call("transpose", &[expr], context)?;
            }
            crate::Expression::DotProduct(a, b) => {
                self.put_call("dot", &[a, b], context)?;
            }
            crate::Expression::CrossProduct(a, b) => {
                self.put_call("cross", &[a, b], context)?;
            }
            crate::Expression::As {
                expr,
                kind,
                convert,
            } => {
                let scalar = scalar_kind_string(kind);
                let size = match *self.typifier.get(expr, &context.module.types) {
                    crate::TypeInner::Scalar { .. } => "",
                    crate::TypeInner::Vector { size, .. } => vector_size_string(size),
                    _ => return Err(Error::Validation),
                };
                let op = if convert { "static_cast" } else { "as_type" };
                write!(self.out, "{}<{}{}>(", op, scalar, size)?;
                self.put_expression(expr, context)?;
                write!(self.out, ")")?;
            }
            crate::Expression::Derivative { axis, expr } => {
                let op = match axis {
                    crate::DerivativeAxis::X => "dfdx",
                    crate::DerivativeAxis::Y => "dfdy",
                    crate::DerivativeAxis::Width => "fwidth",
                };
                self.put_call(op, &[expr], context)?;
            }
            crate::Expression::Call {
                origin: crate::FunctionOrigin::Local(handle),
                ref arguments,
            } => {
                let name = &self.names[&NameKey::Function(handle)];
                write!(self.out, "{}", name)?;
                self.put_call("", arguments, context)?;
            }
            crate::Expression::Call {
                origin: crate::FunctionOrigin::External(ref name),
                ref arguments,
            } => match name.as_str() {
                "atan2" | "cos" | "distance" | "length" | "mix" | "normalize" | "sin" => {
                    self.put_call(name, arguments, context)?;
                }
                "fclamp" => {
                    self.put_call("clamp", arguments, context)?;
                }
                other => return Err(Error::UnsupportedCall(other.to_owned())),
            },
            crate::Expression::ArrayLength(expr) => match *self
                .typifier
                .get(expr, &context.module.types)
            {
                crate::TypeInner::Array {
                    size: crate::ArraySize::Constant(const_handle),
                    ..
                } => {
                    self.put_constant(const_handle, context.module)?;
                }
                crate::TypeInner::Array { .. } => return Err(Error::UnsupportedDynamicArrayLength),
                _ => return Err(Error::Validation),
            },
        }
        Ok(())
    }

    fn put_constant(
        &mut self,
        handle: Handle<crate::Constant>,
        module: &crate::Module,
    ) -> Result<(), Error> {
        let constant = &module.constants[handle];
        match constant.inner {
            crate::ConstantInner::Sint(value) => {
                write!(self.out, "{}", value)?;
            }
            crate::ConstantInner::Uint(value) => {
                write!(self.out, "{}", value)?;
            }
            crate::ConstantInner::Float(value) => {
                write!(self.out, "{}", value)?;
                if value.fract() == 0.0 {
                    write!(self.out, ".0")?;
                }
            }
            crate::ConstantInner::Bool(value) => {
                write!(self.out, "{}", value)?;
            }
            crate::ConstantInner::Composite(ref constituents) => {
                let ty_name = &self.names[&NameKey::Type(constant.ty)];
                write!(self.out, "{}(", ty_name)?;
                for (i, &handle) in constituents.iter().enumerate() {
                    if i != 0 {
                        write!(self.out, ", ")?;
                    }
                    self.put_constant(handle, module)?;
                }
                write!(self.out, ")")?;
            }
        }
        Ok(())
    }

    fn put_block(
        &mut self,
        level: Level,
        statements: &[crate::Statement],
        context: &ExpressionContext,
        return_value: Option<&str>,
    ) -> Result<(), Error> {
        for statement in statements {
            log::trace!("statement[{}] {:?}", level.0, statement);
            match *statement {
                crate::Statement::Block(ref block) => {
                    if !block.is_empty() {
                        writeln!(self.out, "{}{{", level)?;
                        self.put_block(level.next(), block, context, return_value)?;
                        writeln!(self.out, "{}}}", level)?;
                    }
                }
                crate::Statement::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    write!(self.out, "{}if (", level)?;
                    self.put_expression(condition, context)?;
                    writeln!(self.out, ") {{")?;
                    self.put_block(level.next(), accept, context, return_value)?;
                    if !reject.is_empty() {
                        writeln!(self.out, "{}}} else {{", level)?;
                        self.put_block(level.next(), reject, context, return_value)?;
                    }
                    writeln!(self.out, "{}}}", level)?;
                }
                crate::Statement::Switch {
                    selector,
                    ref cases,
                    ref default,
                } => {
                    write!(self.out, "{}switch(", level)?;
                    self.put_expression(selector, context)?;
                    writeln!(self.out, ") {{")?;
                    let lcase = level.next();
                    for (&value, &(ref block, ref fall_through)) in cases.iter() {
                        writeln!(self.out, "{}case {}: {{", lcase, value)?;
                        self.put_block(lcase.next(), block, context, return_value)?;
                        if fall_through.is_none() {
                            writeln!(self.out, "{}break;", lcase.next())?;
                        }
                        writeln!(self.out, "{}}}", lcase)?;
                    }
                    writeln!(self.out, "{}default: {{", lcase)?;
                    self.put_block(lcase.next(), default, context, return_value)?;
                    writeln!(self.out, "{}}}", lcase)?;
                    writeln!(self.out, "{}}}", level)?;
                }
                crate::Statement::Loop {
                    ref body,
                    ref continuing,
                } => {
                    writeln!(self.out, "{}while(true) {{", level)?;
                    self.put_block(level.next(), body, context, return_value)?;
                    if !continuing.is_empty() {
                        //TODO
                    }
                    writeln!(self.out, "{}}}", level)?;
                }
                crate::Statement::Break => {
                    writeln!(self.out, "{}break;", level)?;
                }
                crate::Statement::Continue => {
                    writeln!(self.out, "{}continue;", level)?;
                }
                crate::Statement::Return {
                    value: Some(expr_handle),
                } => {
                    write!(self.out, "{}return ", level)?;
                    self.put_expression(expr_handle, context)?;
                    writeln!(self.out, ";")?;
                }
                crate::Statement::Return { value: None } => {
                    if let Some(string) = return_value {
                        writeln!(self.out, "{}return {};", level, string)?;
                    }
                }
                crate::Statement::Kill => {
                    writeln!(self.out, "{}discard_fragment();", level)?;
                }
                crate::Statement::Store { pointer, value } => {
                    //write!(self.out, "\t*")?;
                    write!(self.out, "{}", level)?;
                    self.put_expression(pointer, context)?;
                    write!(self.out, " = ")?;
                    self.put_expression(value, context)?;
                    writeln!(self.out, ";")?;
                }
            }
        }
        Ok(())
    }

    pub fn write(&mut self, module: &crate::Module, options: &Options) -> Result<(), Error> {
        self.names.clear();
        Namer::process(module, RESERVED, &mut self.names);

        writeln!(self.out, "#include <metal_stdlib>")?;
        writeln!(self.out, "#include <simd/simd.h>")?;

        writeln!(self.out)?;
        self.write_type_defs(module)?;

        writeln!(self.out)?;
        self.write_functions(module, options)?;

        Ok(())
    }

    fn write_type_defs(&mut self, module: &crate::Module) -> Result<(), Error> {
        for (handle, ty) in module.types.iter() {
            let name = &self.names[&NameKey::Type(handle)];
            match ty.inner {
                crate::TypeInner::Scalar { kind, .. } => {
                    write!(self.out, "typedef {} {}", scalar_kind_string(kind), name)?;
                }
                crate::TypeInner::Vector { size, kind, .. } => {
                    write!(
                        self.out,
                        "typedef {}{} {}",
                        scalar_kind_string(kind),
                        vector_size_string(size),
                        name
                    )?;
                }
                crate::TypeInner::Matrix { columns, rows, .. } => {
                    write!(
                        self.out,
                        "typedef {}{}x{} {}",
                        scalar_kind_string(crate::ScalarKind::Float),
                        vector_size_string(columns),
                        vector_size_string(rows),
                        name
                    )?;
                }
                crate::TypeInner::Pointer { base, class } => {
                    use crate::StorageClass as Sc;
                    let base_name = &self.names[&NameKey::Type(base)];
                    let class_name = match class {
                        Sc::Input | Sc::Output => continue,
                        Sc::Uniform => "constant",
                        Sc::Storage => "device",
                        Sc::Handle
                        | Sc::Private
                        | Sc::Function
                        | Sc::WorkGroup
                        | Sc::PushConstant => "",
                    };
                    write!(self.out, "typedef {} {} *{}", class_name, base_name, name)?;
                }
                crate::TypeInner::Array {
                    base,
                    size,
                    stride: _,
                } => {
                    let base_name = &self.names[&NameKey::Type(base)];
                    write!(self.out, "typedef {} {}[", base_name, name)?;
                    match size {
                        crate::ArraySize::Constant(const_handle) => {
                            self.put_constant(const_handle, module)?;
                            write!(self.out, "]")?;
                        }
                        crate::ArraySize::Dynamic => write!(self.out, "1]")?,
                    }
                }
                crate::TypeInner::Struct { ref members } => {
                    writeln!(self.out, "struct {} {{", name)?;
                    for (index, member) in members.iter().enumerate() {
                        let member_name = &self.names[&NameKey::StructMember(handle, index as u32)];
                        let base_name = &self.names[&NameKey::Type(member.ty)];
                        write!(self.out, "\t{} {}", base_name, member_name)?;
                        match member.origin {
                            crate::MemberOrigin::Empty => {}
                            crate::MemberOrigin::BuiltIn(built_in) => {
                                ResolvedBinding::BuiltIn(built_in)
                                    .try_fmt_decorated(&mut self.out, "")?;
                            }
                            crate::MemberOrigin::Offset(_) => {
                                //TODO
                            }
                        }
                        writeln!(self.out, ";")?;
                    }
                    write!(self.out, "}}")?;
                }
                crate::TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => {
                    let dim_str = match dim {
                        crate::ImageDimension::D1 => "1d",
                        crate::ImageDimension::D2 => "2d",
                        crate::ImageDimension::D3 => "3d",
                        crate::ImageDimension::Cube => "Cube",
                    };
                    let (texture_str, msaa_str, kind, access) = match class {
                        crate::ImageClass::Sampled { kind, multi } => {
                            ("texture", if multi { "_ms" } else { "" }, kind, "sample")
                        }
                        crate::ImageClass::Depth => {
                            ("depth", "", crate::ScalarKind::Float, "sample")
                        }
                        crate::ImageClass::Storage(format) => {
                            let (_, global) = module
                                .global_variables
                                .iter()
                                .find(|(_, var)| var.ty == handle)
                                .expect("Unable to find a global variable using the image type");
                            let access = if global
                                .storage_access
                                .contains(crate::StorageAccess::LOAD | crate::StorageAccess::STORE)
                            {
                                "read_write"
                            } else if global.storage_access.contains(crate::StorageAccess::STORE) {
                                "write"
                            } else if global.storage_access.contains(crate::StorageAccess::LOAD) {
                                "read"
                            } else {
                                return Err(Error::InvalidImageAccess(global.storage_access));
                            };
                            ("texture", "", format.into(), access)
                        }
                    };
                    let base_name = scalar_kind_string(kind);
                    let array_str = if arrayed { "_array" } else { "" };
                    write!(
                        self.out,
                        "typedef {}{}{}{}<{}, access::{}> {}",
                        texture_str, dim_str, msaa_str, array_str, base_name, access, name
                    )?;
                }
                crate::TypeInner::Sampler { comparison: _ } => {
                    write!(self.out, "typedef sampler {}", name)?;
                }
            }
            writeln!(self.out, ";")?;
        }
        Ok(())
    }

    fn write_functions(&mut self, module: &crate::Module, options: &Options) -> Result<(), Error> {
        for (fun_handle, fun) in module.functions.iter() {
            self.typifier.resolve_all(
                &fun.expressions,
                &module.types,
                &ResolveContext {
                    constants: &module.constants,
                    global_vars: &module.global_variables,
                    local_vars: &fun.local_variables,
                    functions: &module.functions,
                    arguments: &fun.arguments,
                },
            )?;

            let fun_name = &self.names[&NameKey::Function(fun_handle)];
            let result_type_name = match fun.return_type {
                Some(ret_ty) => &self.names[&NameKey::Type(ret_ty)],
                None => "void",
            };
            writeln!(self.out, "{} {}(", result_type_name, fun_name)?;

            for (index, arg) in fun.arguments.iter().enumerate() {
                let name = &self.names[&NameKey::FunctionArgument(fun_handle, index as u32)];
                let param_type_name = &self.names[&NameKey::Type(arg.ty)];
                let separator = separate(index + 1 == fun.arguments.len());
                writeln!(self.out, "\t{} {}{}", param_type_name, name, separator)?;
            }
            writeln!(self.out, ") {{")?;

            for (local_handle, local) in fun.local_variables.iter() {
                let ty_name = &self.names[&NameKey::Type(local.ty)];
                let local_name = &self.names[&NameKey::FunctionLocal(fun_handle, local_handle)];
                write!(self.out, "\t{} {}", ty_name, local_name)?;
                if let Some(value) = local.init {
                    write!(self.out, " = ")?;
                    self.put_constant(value, module)?;
                }
                writeln!(self.out, ";")?;
            }

            let context = ExpressionContext {
                function: fun,
                origin: FunctionOrigin::Handle(fun_handle),
                module,
            };
            self.put_block(Level(1), &fun.body, &context, None)?;
            writeln!(self.out, "}}")?;
        }

        for (ep_index, (&(stage, _), ep)) in module.entry_points.iter().enumerate() {
            let fun = &ep.function;
            self.typifier.resolve_all(
                &fun.expressions,
                &module.types,
                &ResolveContext {
                    constants: &module.constants,
                    global_vars: &module.global_variables,
                    local_vars: &fun.local_variables,
                    functions: &module.functions,
                    arguments: &fun.arguments,
                },
            )?;

            // find the entry point(s) and inputs/outputs
            let mut last_used_global = None;
            for ((handle, var), &usage) in module.global_variables.iter().zip(&fun.global_usage) {
                match var.class {
                    crate::StorageClass::Input => {
                        if let Some(crate::Binding::Location(_)) = var.binding {
                            continue;
                        }
                    }
                    crate::StorageClass::Output => continue,
                    _ => {}
                }
                if !usage.is_empty() {
                    last_used_global = Some(handle);
                }
            }

            let fun_name = &self.names[&NameKey::EntryPoint(ep_index as _)];
            let output_name = format!("{}Output", fun_name);
            let location_input_name = format!("{}Input", fun_name);

            let (em_str, in_mode, out_mode) = match stage {
                crate::ShaderStage::Vertex => (
                    "vertex",
                    LocationMode::VertexInput,
                    LocationMode::Intermediate,
                ),
                crate::ShaderStage::Fragment { .. } => (
                    "fragment",
                    LocationMode::Intermediate,
                    LocationMode::FragmentOutput,
                ),
                crate::ShaderStage::Compute { .. } => {
                    ("kernel", LocationMode::Uniform, LocationMode::Uniform)
                }
            };

            let return_value = match stage {
                crate::ShaderStage::Vertex | crate::ShaderStage::Fragment => {
                    // make dedicated input/output structs
                    writeln!(self.out, "struct {} {{", location_input_name)?;

                    for ((handle, var), &usage) in
                        module.global_variables.iter().zip(&fun.global_usage)
                    {
                        if var.class != crate::StorageClass::Input
                            || !usage.contains(crate::GlobalUse::LOAD)
                        {
                            continue;
                        }
                        // if it's a struct, lift all the built-in contents up to the root
                        if let crate::TypeInner::Struct { ref members } = module.types[var.ty].inner
                        {
                            for (index, member) in members.iter().enumerate() {
                                if let crate::MemberOrigin::BuiltIn(built_in) = member.origin {
                                    let name =
                                        &self.names[&NameKey::StructMember(var.ty, index as u32)];
                                    let ty_name = &self.names[&NameKey::Type(member.ty)];
                                    write!(self.out, "\t{} {}", ty_name, name)?;
                                    ResolvedBinding::BuiltIn(built_in)
                                        .try_fmt_decorated(&mut self.out, ";\n")?;
                                }
                            }
                        } else if let Some(ref binding @ crate::Binding::Location(_)) = var.binding
                        {
                            let tyvar = TypedGlobalVariable {
                                module,
                                names: &self.names,
                                handle,
                                usage: crate::GlobalUse::empty(),
                            };
                            let resolved = options.resolve_binding(stage, binding, in_mode)?;

                            write!(self.out, "\t")?;
                            tyvar.try_fmt(&mut self.out)?;
                            resolved.try_fmt_decorated(&mut self.out, ";\n")?;
                        }
                    }
                    writeln!(self.out, "}};")?;

                    writeln!(self.out, "struct {} {{", output_name)?;
                    for ((handle, var), &usage) in
                        module.global_variables.iter().zip(&fun.global_usage)
                    {
                        if var.class != crate::StorageClass::Output
                            || !usage.contains(crate::GlobalUse::STORE)
                        {
                            continue;
                        }
                        // if it's a struct, lift all the built-in contents up to the root
                        if let crate::TypeInner::Struct { ref members } = module.types[var.ty].inner
                        {
                            for (index, member) in members.iter().enumerate() {
                                let name =
                                    &self.names[&NameKey::StructMember(var.ty, index as u32)];
                                let ty_name = &self.names[&NameKey::Type(member.ty)];
                                match member.origin {
                                    crate::MemberOrigin::Empty => {}
                                    crate::MemberOrigin::BuiltIn(built_in) => {
                                        write!(self.out, "\t{} {}", ty_name, name)?;
                                        ResolvedBinding::BuiltIn(built_in)
                                            .try_fmt_decorated(&mut self.out, ";\n")?;
                                    }
                                    crate::MemberOrigin::Offset(_) => {
                                        //TODO
                                    }
                                }
                            }
                        } else {
                            let tyvar = TypedGlobalVariable {
                                module,
                                names: &self.names,
                                handle,
                                usage: crate::GlobalUse::empty(),
                            };
                            write!(self.out, "\t")?;
                            tyvar.try_fmt(&mut self.out)?;
                            if let Some(ref binding) = var.binding {
                                let resolved = options.resolve_binding(stage, binding, out_mode)?;
                                resolved.try_fmt_decorated(&mut self.out, "")?;
                            }
                            writeln!(self.out, ";")?;
                        }
                    }
                    writeln!(self.out, "}};")?;

                    writeln!(self.out, "{} {} {}(", em_str, output_name, fun_name)?;
                    let separator = separate(last_used_global.is_none());
                    writeln!(
                        self.out,
                        "\t{} {} [[stage_in]]{}",
                        location_input_name, LOCATION_INPUT_STRUCT_NAME, separator
                    )?;

                    Some(OUTPUT_STRUCT_NAME)
                }
                crate::ShaderStage::Compute => {
                    writeln!(self.out, "{} void {}(", em_str, fun_name)?;
                    None
                }
            };

            for ((handle, var), &usage) in module.global_variables.iter().zip(&fun.global_usage) {
                if usage.is_empty() || var.class == crate::StorageClass::Output {
                    continue;
                }
                if var.class == crate::StorageClass::Input {
                    if let Some(crate::Binding::Location(_)) = var.binding {
                        // location inputs are put into a separate struct
                        continue;
                    }
                }
                let loc_mode = match (stage, var.class) {
                    (crate::ShaderStage::Vertex, crate::StorageClass::Input) => {
                        LocationMode::VertexInput
                    }
                    (crate::ShaderStage::Vertex, crate::StorageClass::Output)
                    | (crate::ShaderStage::Fragment { .. }, crate::StorageClass::Input) => {
                        LocationMode::Intermediate
                    }
                    (crate::ShaderStage::Fragment { .. }, crate::StorageClass::Output) => {
                        LocationMode::FragmentOutput
                    }
                    _ => LocationMode::Uniform,
                };
                let resolved =
                    options.resolve_binding(stage, var.binding.as_ref().unwrap(), loc_mode)?;
                let tyvar = TypedGlobalVariable {
                    module,
                    names: &self.names,
                    handle,
                    usage,
                };
                let separator = separate(last_used_global == Some(handle));
                write!(self.out, "\t")?;
                tyvar.try_fmt(&mut self.out)?;
                resolved.try_fmt_decorated(&mut self.out, separator)?;
                if let Some(value) = var.init {
                    write!(self.out, " = ")?;
                    self.put_constant(value, module)?;
                }
                writeln!(self.out)?;
            }
            writeln!(self.out, ") {{")?;

            match stage {
                crate::ShaderStage::Vertex | crate::ShaderStage::Fragment => {
                    writeln!(self.out, "\t{} {};", output_name, OUTPUT_STRUCT_NAME)?;
                }
                crate::ShaderStage::Compute => {}
            }
            for (local_handle, local) in fun.local_variables.iter() {
                let name = &self.names[&NameKey::EntryPointLocal(ep_index as _, local_handle)];
                let ty_name = &self.names[&NameKey::Type(local.ty)];
                write!(self.out, "\t{} {}", ty_name, name)?;
                if let Some(value) = local.init {
                    write!(self.out, " = ")?;
                    self.put_constant(value, module)?;
                }
                writeln!(self.out, ";")?;
            }

            let context = ExpressionContext {
                function: fun,
                origin: FunctionOrigin::EntryPoint(ep_index as _),
                module,
            };
            self.put_block(Level(1), &fun.body, &context, return_value)?;
            writeln!(self.out, "}}")?;
        }

        Ok(())
    }
}
