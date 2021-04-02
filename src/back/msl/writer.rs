use super::{keywords::RESERVED, Error, LocationMode, Options, TranslationInfo};
use crate::{
    arena::Handle,
    proc::{EntryPointIndex, NameKey, Namer, TypeResolution},
    valid::{FunctionInfo, GlobalUse, ModuleInfo},
    FastHashMap,
};
use bit_set::BitSet;
use std::{
    fmt::{Display, Error as FmtError, Formatter, Write},
    iter,
};

const NAMESPACE: &str = "metal";
const INDENT: &str = "    ";
const BAKE_PREFIX: &str = "_expr";

#[derive(Clone)]
struct Level(usize);
impl Level {
    fn next(&self) -> Self {
        Level(self.0 + 1)
    }
}
impl Display for Level {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), FmtError> {
        (0..self.0).try_for_each(|_| formatter.write_str(INDENT))
    }
}

struct TypedGlobalVariable<'a> {
    module: &'a crate::Module,
    names: &'a FastHashMap<NameKey, String>,
    handle: Handle<crate::GlobalVariable>,
    usage: GlobalUse,
    reference: bool,
}

impl<'a> TypedGlobalVariable<'a> {
    fn try_fmt<W: Write>(&self, out: &mut W) -> Result<(), Error> {
        let var = &self.module.global_variables[self.handle];
        let name = &self.names[&NameKey::GlobalVariable(self.handle)];
        let ty_name = &self.names[&NameKey::Type(var.ty)];

        let (space, access, reference) = match var.class.get_name(self.usage) {
            Some(space) if self.reference => {
                let access = match var.class {
                    crate::StorageClass::Private | crate::StorageClass::WorkGroup
                        if !self.usage.contains(GlobalUse::WRITE) =>
                    {
                        "const"
                    }
                    _ => "",
                };
                (space, access, "&")
            }
            _ => ("", "", ""),
        };

        Ok(write!(
            out,
            "{}{}{}{}{}{} {}",
            space,
            if space.is_empty() { "" } else { " " },
            ty_name,
            if access.is_empty() { "" } else { " " },
            access,
            reference,
            name,
        )?)
    }
}

pub struct Writer<W> {
    out: W,
    names: FastHashMap<NameKey, String>,
    named_expressions: BitSet,
    namer: Namer,
    #[cfg(test)]
    put_expression_stack_pointers: crate::FastHashSet<*const ()>,
    #[cfg(test)]
    put_block_stack_pointers: crate::FastHashSet<*const ()>,
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

const COMPONENTS: &[char] = &['x', 'y', 'z', 'w'];

fn separate(need_separator: bool) -> &'static str {
    if need_separator {
        ","
    } else {
        ""
    }
}

impl crate::StorageClass {
    /// Returns true for storage classes, for which the global
    /// variables are passed in function arguments.
    /// These arguments need to be passed through any functions
    /// called from the entry point.
    fn needs_pass_through(&self) -> bool {
        match *self {
            crate::StorageClass::Uniform
            | crate::StorageClass::Storage
            | crate::StorageClass::Private
            | crate::StorageClass::Handle => true,
            _ => false,
        }
    }

    fn get_name(&self, global_use: GlobalUse) -> Option<&'static str> {
        match *self {
            Self::Handle => None,
            Self::Uniform | Self::PushConstant => Some("constant"),
            //TODO: should still be "constant" for read-only buffers
            Self::Storage => Some(if global_use.contains(GlobalUse::WRITE) {
                "device"
            } else {
                "constant"
            }),
            Self::Private | Self::Function => Some("thread"),
            Self::WorkGroup => Some("threadgroup"),
        }
    }
}

enum FunctionOrigin {
    Handle(Handle<crate::Function>),
    EntryPoint(EntryPointIndex),
}

struct ExpressionContext<'a> {
    function: &'a crate::Function,
    origin: FunctionOrigin,
    info: &'a FunctionInfo,
    module: &'a crate::Module,
    options: &'a Options,
}

impl<'a> ExpressionContext<'a> {
    fn resolve_type(&self, handle: Handle<crate::Expression>) -> &'a crate::TypeInner {
        self.info[handle].ty.inner_with(&self.module.types)
    }
}

struct StatementContext<'a> {
    expression: ExpressionContext<'a>,
    mod_info: &'a ModuleInfo,
    result_struct: Option<&'a str>,
}

impl<W: Write> Writer<W> {
    /// Creates a new `Writer` instance.
    pub fn new(out: W) -> Self {
        Writer {
            out,
            names: FastHashMap::default(),
            named_expressions: BitSet::new(),
            namer: Namer::default(),
            #[cfg(test)]
            put_expression_stack_pointers: Default::default(),
            #[cfg(test)]
            put_block_stack_pointers: Default::default(),
        }
    }

    /// Finishes writing and returns the output.
    pub fn finish(self) -> W {
        self.out
    }

    fn put_call_parameters(
        &mut self,
        parameters: impl Iterator<Item = Handle<crate::Expression>>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        write!(self.out, "(")?;
        for (i, handle) in parameters.enumerate() {
            if i != 0 {
                write!(self.out, ", ")?;
            }
            self.put_expression(handle, context, true)?;
        }
        write!(self.out, ")")?;
        Ok(())
    }

    fn put_image_query(
        &mut self,
        image: Handle<crate::Expression>,
        query: &str,
        level: Option<Handle<crate::Expression>>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        self.put_expression(image, context, false)?;
        write!(self.out, ".get_{}(", query)?;
        if let Some(expr) = level {
            self.put_expression(expr, context, true)?;
        }
        write!(self.out, ")")?;
        Ok(())
    }

    fn put_image_size_query(
        &mut self,
        image: Handle<crate::Expression>,
        level: Option<Handle<crate::Expression>>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        //Note: MSL only has separate width/height/depth queries,
        // so compose the result of them.
        let dim = match *context.resolve_type(image) {
            crate::TypeInner::Image { dim, .. } => dim,
            ref other => unreachable!("Unexpected type {:?}", other),
        };
        match dim {
            crate::ImageDimension::D1 => {
                write!(self.out, "int(")?;
                self.put_image_query(image, "width", level, context)?;
                write!(self.out, ")")?;
            }
            crate::ImageDimension::D2 => {
                write!(self.out, "int2(")?;
                self.put_image_query(image, "width", level, context)?;
                write!(self.out, ", ")?;
                self.put_image_query(image, "height", level, context)?;
                write!(self.out, ")")?;
            }
            crate::ImageDimension::D3 => {
                write!(self.out, "int3(")?;
                self.put_image_query(image, "width", level, context)?;
                write!(self.out, ", ")?;
                self.put_image_query(image, "height", level, context)?;
                write!(self.out, ", ")?;
                self.put_image_query(image, "depth", level, context)?;
                write!(self.out, ")")?;
            }
            crate::ImageDimension::Cube => {
                write!(self.out, "int(")?;
                self.put_image_query(image, "width", level, context)?;
                write!(self.out, ").xxx")?;
            }
        }
        Ok(())
    }

    fn put_storage_image_coordinate(
        &mut self,
        expr: Handle<crate::Expression>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        // coordinates in IR are int, but Metal expects uint
        let size_str = match *context.info[expr].ty.inner_with(&context.module.types) {
            crate::TypeInner::Scalar { .. } => "",
            crate::TypeInner::Vector { size, .. } => vector_size_string(size),
            _ => return Err(Error::Validation),
        };
        write!(self.out, "{}::uint{}(", NAMESPACE, size_str)?;
        self.put_expression(expr, context, true)?;
        write!(self.out, ")")?;
        Ok(())
    }

    fn put_initialization_component(
        &mut self,
        component: Handle<crate::Expression>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        // we can't initialize the array members just like other members,
        // we have to unwrap them one level deeper...
        let component_res = &context.info[component].ty;
        if let crate::TypeInner::Array {
            size: crate::ArraySize::Constant(const_handle),
            ..
        } = *component_res.inner_with(&context.module.types)
        {
            //HACK: we are forcefully duplicating the expression here,
            // it would be nice to find a more C++ idiomatic solution for initializing array members
            let size = context.module.constants[const_handle]
                .to_array_length()
                .unwrap();
            write!(self.out, "{{")?;
            for j in 0..size {
                if j != 0 {
                    write!(self.out, ",")?;
                }
                self.put_expression(component, context, false)?;
                write!(self.out, "[{}]", j)?;
            }
            write!(self.out, "}}")?;
        } else {
            self.put_expression(component, context, true)?;
        }
        Ok(())
    }

    fn put_expression(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        context: &ExpressionContext,
        is_scoped: bool,
    ) -> Result<(), Error> {
        // Add to the set in order to track the stack size.
        #[cfg(test)]
        #[allow(trivial_casts)]
        self.put_expression_stack_pointers
            .insert(&expr_handle as *const _ as *const ());

        if self.named_expressions.contains(expr_handle.index()) {
            write!(self.out, "{}{}", BAKE_PREFIX, expr_handle.index())?;
            return Ok(());
        }

        let expression = &context.function.expressions[expr_handle];
        log::trace!("expression {:?} = {:?}", expr_handle, expression);
        match *expression {
            crate::Expression::Access { base, index } => {
                self.put_expression(base, context, false)?;
                write!(self.out, "[")?;
                self.put_expression(index, context, true)?;
                write!(self.out, "]")?;
            }
            crate::Expression::AccessIndex { base, index } => {
                self.put_expression(base, context, false)?;
                let base_res = &context.info[base].ty;
                let mut resolved = base_res.inner_with(&context.module.types);
                let base_ty_handle = match *resolved {
                    crate::TypeInner::Pointer { base, class: _ } => {
                        resolved = &context.module.types[base].inner;
                        Some(base)
                    }
                    _ => base_res.handle(),
                };
                match *resolved {
                    crate::TypeInner::Struct { .. } => {
                        let base_ty = base_ty_handle.unwrap();
                        let name = &self.names[&NameKey::StructMember(base_ty, index)];
                        write!(self.out, ".{}", name)?;
                    }
                    crate::TypeInner::ValuePointer { .. } | crate::TypeInner::Vector { .. } => {
                        write!(self.out, ".{}", COMPONENTS[index as usize])?;
                    }
                    crate::TypeInner::Matrix { .. } => {
                        write!(self.out, "[{}]", index)?;
                    }
                    crate::TypeInner::Array { .. } => {
                        write!(self.out, "[{}]", index)?;
                    }
                    _ => {
                        // unexpected indexing, should fail validation
                    }
                }
            }
            crate::Expression::Constant(handle) => {
                let handle_name = &self.names[&NameKey::Constant(handle)];
                write!(self.out, "{}", handle_name)?;
            }
            crate::Expression::Compose { ty, ref components } => {
                let inner = &context.module.types[ty].inner;
                match *inner {
                    crate::TypeInner::Scalar { width: 4, kind } if components.len() == 1 => {
                        write!(self.out, "{}", scalar_kind_string(kind))?;
                        self.put_call_parameters(components.iter().cloned(), context)?;
                    }
                    crate::TypeInner::Vector { size, kind, .. } => {
                        write!(
                            self.out,
                            "{}::{}{}",
                            NAMESPACE,
                            scalar_kind_string(kind),
                            vector_size_string(size)
                        )?;
                        self.put_call_parameters(components.iter().cloned(), context)?;
                    }
                    crate::TypeInner::Matrix { columns, rows, .. } => {
                        let kind = crate::ScalarKind::Float;
                        write!(
                            self.out,
                            "{}::{}{}x{}",
                            NAMESPACE,
                            scalar_kind_string(kind),
                            vector_size_string(columns),
                            vector_size_string(rows)
                        )?;
                        self.put_call_parameters(components.iter().cloned(), context)?;
                    }
                    crate::TypeInner::Array { .. } | crate::TypeInner::Struct { .. } => {
                        write!(self.out, "{} {{", &self.names[&NameKey::Type(ty)])?;
                        for (i, &component) in components.iter().enumerate() {
                            if i != 0 {
                                write!(self.out, ", ")?;
                            }
                            self.put_initialization_component(component, context)?;
                        }
                        write!(self.out, "}}")?;
                    }
                    _ => return Err(Error::UnsupportedCompose(ty)),
                }
            }
            crate::Expression::FunctionArgument(index) => {
                let name_key = match context.origin {
                    FunctionOrigin::Handle(handle) => NameKey::FunctionArgument(handle, index),
                    FunctionOrigin::EntryPoint(ep_index) => {
                        NameKey::EntryPointArgument(ep_index, index)
                    }
                };
                let name = &self.names[&name_key];
                write!(self.out, "{}", name)?;
            }
            crate::Expression::GlobalVariable(handle) => {
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
                self.put_expression(pointer, context, is_scoped)?;
            }
            crate::Expression::ImageSample {
                image,
                sampler,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => {
                let op = match depth_ref {
                    Some(_) => "sample_compare",
                    None => "sample",
                };
                self.put_expression(image, context, false)?;
                write!(self.out, ".{}(", op)?;
                self.put_expression(sampler, context, true)?;
                write!(self.out, ", ")?;
                self.put_expression(coordinate, context, true)?;
                if let Some(expr) = array_index {
                    write!(self.out, ", ")?;
                    self.put_expression(expr, context, true)?;
                }
                if let Some(dref) = depth_ref {
                    write!(self.out, ", ")?;
                    self.put_expression(dref, context, true)?;
                }
                match level {
                    crate::SampleLevel::Auto => {}
                    crate::SampleLevel::Zero => {
                        //TODO: do we support Zero on `Sampled` image classes?
                    }
                    crate::SampleLevel::Exact(h) => {
                        write!(self.out, ", {}::level(", NAMESPACE)?;
                        self.put_expression(h, context, true)?;
                        write!(self.out, ")")?;
                    }
                    crate::SampleLevel::Bias(h) => {
                        write!(self.out, ", {}::bias(", NAMESPACE)?;
                        self.put_expression(h, context, true)?;
                        write!(self.out, ")")?;
                    }
                    crate::SampleLevel::Gradient { x, y } => {
                        write!(self.out, ", {}::gradient(", NAMESPACE)?;
                        self.put_expression(x, context, true)?;
                        write!(self.out, ", ")?;
                        self.put_expression(y, context, true)?;
                        write!(self.out, ")")?;
                    }
                }
                if let Some(constant) = offset {
                    let offset_str = &self.names[&NameKey::Constant(constant)];
                    write!(self.out, ", {}", offset_str)?;
                }
                write!(self.out, ")")?;
            }
            crate::Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                index,
            } => {
                self.put_expression(image, context, false)?;
                write!(self.out, ".read(")?;
                self.put_storage_image_coordinate(coordinate, context)?;
                if let Some(expr) = array_index {
                    write!(self.out, ", ")?;
                    self.put_expression(expr, context, true)?;
                }
                if let Some(index) = index {
                    write!(self.out, ", ")?;
                    self.put_expression(index, context, true)?;
                }
                write!(self.out, ")")?;
            }
            //Note: for all the queries, the signed integers are expected,
            // so a conversion is needed.
            crate::Expression::ImageQuery { image, query } => match query {
                crate::ImageQuery::Size { level } => {
                    self.put_image_size_query(image, level, context)?;
                }
                crate::ImageQuery::NumLevels => {
                    write!(self.out, "int(")?;
                    self.put_expression(image, context, false)?;
                    write!(self.out, ".get_num_mip_levels())")?;
                }
                crate::ImageQuery::NumLayers => {
                    write!(self.out, "int(")?;
                    self.put_expression(image, context, false)?;
                    write!(self.out, ".get_array_size())")?;
                }
                crate::ImageQuery::NumSamples => {
                    write!(self.out, "int(")?;
                    self.put_expression(image, context, false)?;
                    write!(self.out, ".get_num_samples())")?;
                }
            },
            crate::Expression::Unary { op, expr } => {
                let op_str = match op {
                    crate::UnaryOperator::Negate => "-",
                    crate::UnaryOperator::Not => "!",
                };
                write!(self.out, "{}", op_str)?;
                self.put_expression(expr, context, false)?;
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
                    crate::BinaryOperator::Greater => ">",
                    crate::BinaryOperator::GreaterEqual => ">=",
                    crate::BinaryOperator::And => "&",
                    crate::BinaryOperator::ExclusiveOr => "^",
                    crate::BinaryOperator::InclusiveOr => "|",
                    crate::BinaryOperator::LogicalAnd => "&&",
                    crate::BinaryOperator::LogicalOr => "||",
                    crate::BinaryOperator::ShiftLeft => "<<",
                    crate::BinaryOperator::ShiftRight => ">>",
                };
                let kind = context
                    .resolve_type(left)
                    .scalar_kind()
                    .ok_or(Error::UnsupportedBinaryOp(op))?;
                if op == crate::BinaryOperator::Modulo && kind == crate::ScalarKind::Float {
                    write!(self.out, "{}::fmod(", NAMESPACE)?;
                    self.put_expression(left, context, true)?;
                    write!(self.out, ", ")?;
                    self.put_expression(right, context, true)?;
                    write!(self.out, ")")?;
                } else {
                    if !is_scoped {
                        write!(self.out, "(")?;
                    }
                    self.put_expression(left, context, false)?;
                    write!(self.out, " {} ", op_str)?;
                    self.put_expression(right, context, false)?;
                    if !is_scoped {
                        write!(self.out, ")")?;
                    }
                }
            }
            crate::Expression::Select {
                condition,
                accept,
                reject,
            } => match *context.resolve_type(condition) {
                crate::TypeInner::Scalar {
                    kind: crate::ScalarKind::Bool,
                    ..
                } => {
                    if !is_scoped {
                        write!(self.out, "(")?;
                    }
                    self.put_expression(condition, context, false)?;
                    write!(self.out, " ? ")?;
                    self.put_expression(accept, context, false)?;
                    write!(self.out, " : ")?;
                    self.put_expression(reject, context, false)?;
                    if !is_scoped {
                        write!(self.out, ")")?;
                    }
                }
                crate::TypeInner::Vector {
                    kind: crate::ScalarKind::Bool,
                    ..
                } => {
                    write!(self.out, "{}::select(", NAMESPACE)?;
                    self.put_expression(accept, context, true)?;
                    write!(self.out, ", ")?;
                    self.put_expression(reject, context, true)?;
                    write!(self.out, ", ")?;
                    self.put_expression(condition, context, true)?;
                    write!(self.out, ")")?;
                }
                _ => return Err(Error::Validation),
            },
            crate::Expression::Derivative { axis, expr } => {
                let op = match axis {
                    crate::DerivativeAxis::X => "dfdx",
                    crate::DerivativeAxis::Y => "dfdy",
                    crate::DerivativeAxis::Width => "fwidth",
                };
                write!(self.out, "{}::{}", NAMESPACE, op)?;
                self.put_call_parameters(iter::once(expr), context)?;
            }
            crate::Expression::Relational { fun, argument } => {
                let op = match fun {
                    crate::RelationalFunction::Any => "any",
                    crate::RelationalFunction::All => "all",
                    crate::RelationalFunction::IsNan => "isnan",
                    crate::RelationalFunction::IsInf => "isinf",
                    crate::RelationalFunction::IsFinite => "isfinite",
                    crate::RelationalFunction::IsNormal => "isnormal",
                };
                write!(self.out, "{}::{}", NAMESPACE, op)?;
                self.put_call_parameters(iter::once(argument), context)?;
            }
            crate::Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
            } => {
                use crate::MathFunction as Mf;

                let fun_name = match fun {
                    // comparison
                    Mf::Abs => "abs",
                    Mf::Min => "min",
                    Mf::Max => "max",
                    Mf::Clamp => "clamp",
                    // trigonometry
                    Mf::Cos => "cos",
                    Mf::Cosh => "cosh",
                    Mf::Sin => "sin",
                    Mf::Sinh => "sinh",
                    Mf::Tan => "tan",
                    Mf::Tanh => "tanh",
                    Mf::Acos => "acos",
                    Mf::Asin => "asin",
                    Mf::Atan => "atan",
                    Mf::Atan2 => "atan2",
                    // decomposition
                    Mf::Ceil => "ceil",
                    Mf::Floor => "floor",
                    Mf::Round => "round",
                    Mf::Fract => "fract",
                    Mf::Trunc => "trunc",
                    Mf::Modf => "modf",
                    Mf::Frexp => "frexp",
                    Mf::Ldexp => "ldexp",
                    // exponent
                    Mf::Exp => "exp",
                    Mf::Exp2 => "exp2",
                    Mf::Log => "log",
                    Mf::Log2 => "log2",
                    Mf::Pow => "pow",
                    // geometry
                    Mf::Dot => "dot",
                    Mf::Outer => return Err(Error::UnsupportedCall(format!("{:?}", fun))),
                    Mf::Cross => "cross",
                    Mf::Distance => "distance",
                    Mf::Length => "length",
                    Mf::Normalize => "normalize",
                    Mf::FaceForward => "faceforward",
                    Mf::Reflect => "reflect",
                    // computational
                    Mf::Sign => "sign",
                    Mf::Fma => "fma",
                    Mf::Mix => "mix",
                    Mf::Step => "step",
                    Mf::SmoothStep => "smoothstep",
                    Mf::Sqrt => "sqrt",
                    Mf::InverseSqrt => "rsqrt",
                    Mf::Inverse => return Err(Error::UnsupportedCall(format!("{:?}", fun))),
                    Mf::Transpose => "transpose",
                    Mf::Determinant => "determinant",
                    // bits
                    Mf::CountOneBits => "popcount",
                    Mf::ReverseBits => "reverse_bits",
                };

                write!(self.out, "{}::{}", NAMESPACE, fun_name)?;
                self.put_call_parameters(iter::once(arg).chain(arg1).chain(arg2), context)?;
            }
            crate::Expression::As {
                expr,
                kind,
                convert,
            } => {
                let scalar = scalar_kind_string(kind);
                let size = match *context.resolve_type(expr) {
                    crate::TypeInner::Scalar { .. } => "",
                    crate::TypeInner::Vector { size, .. } => vector_size_string(size),
                    _ => return Err(Error::Validation),
                };
                let op = if convert { "static_cast" } else { "as_type" };
                write!(self.out, "{}<{}{}>(", op, scalar, size)?;
                self.put_expression(expr, context, true)?;
                write!(self.out, ")")?;
            }
            // has to be a named expression
            crate::Expression::Call(_) => unreachable!(),
            crate::Expression::ArrayLength(expr) => match *context.resolve_type(expr) {
                crate::TypeInner::Array {
                    size: crate::ArraySize::Constant(const_handle),
                    ..
                } => {
                    let size_str = &self.names[&NameKey::Constant(const_handle)];
                    write!(self.out, "{}", size_str)?;
                }
                crate::TypeInner::Array { .. } => {
                    return Err(Error::FeatureNotImplemented(
                        "dynamic array size".to_string(),
                    ))
                }
                _ => return Err(Error::Validation),
            },
        }
        Ok(())
    }

    fn put_return_value(
        &mut self,
        level: Level,
        expr_handle: Handle<crate::Expression>,
        result_struct: Option<&str>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        match result_struct {
            Some(struct_name) => {
                let result_ty = context.function.result.as_ref().unwrap().ty;
                match context.module.types[result_ty].inner {
                    crate::TypeInner::Struct {
                        block: _,
                        ref members,
                    } => {
                        let tmp = "_tmp";
                        write!(self.out, "{}const auto {} = ", level, tmp)?;
                        self.put_expression(expr_handle, context, true)?;
                        writeln!(self.out, ";")?;
                        write!(self.out, "{}return {} {{", level, struct_name)?;
                        let mut is_first = true;
                        for (index, member) in members.iter().enumerate() {
                            if !context.options.allow_point_size
                                && member.binding
                                    == Some(crate::Binding::BuiltIn(crate::BuiltIn::PointSize))
                            {
                                continue;
                            }
                            let comma = if is_first { "" } else { "," };
                            is_first = false;
                            let name = &self.names[&NameKey::StructMember(result_ty, index as u32)];
                            // logic similar to `put_initialization_component`
                            if let crate::TypeInner::Array {
                                size: crate::ArraySize::Constant(const_handle),
                                ..
                            } = context.module.types[member.ty].inner
                            {
                                let size = context.module.constants[const_handle]
                                    .to_array_length()
                                    .unwrap();
                                write!(self.out, "{} {{", comma)?;
                                for j in 0..size {
                                    if j != 0 {
                                        write!(self.out, ",")?;
                                    }
                                    write!(self.out, "{}.{}[{}]", tmp, name, j)?;
                                }
                                write!(self.out, "}}")?;
                            } else {
                                write!(self.out, "{} {}.{}", comma, tmp, name)?;
                            }
                        }
                    }
                    _ => {
                        write!(self.out, "{}return {} {{ ", level, struct_name)?;
                        self.put_expression(expr_handle, context, true)?;
                    }
                }
                write!(self.out, " }}")?;
            }
            None => {
                write!(self.out, "{}return ", level)?;
                self.put_expression(expr_handle, context, true)?;
            }
        }
        writeln!(self.out, ";")?;
        Ok(())
    }

    fn start_baking_expression(
        &mut self,
        handle: Handle<crate::Expression>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        match context.info[handle].ty {
            TypeResolution::Handle(ty_handle) => {
                let ty_name = &self.names[&NameKey::Type(ty_handle)];
                write!(self.out, "{}", ty_name)?;
            }
            TypeResolution::Value(crate::TypeInner::Scalar { kind, .. }) => {
                write!(self.out, "{}", scalar_kind_string(kind))?;
            }
            TypeResolution::Value(crate::TypeInner::Vector { size, kind, .. }) => {
                write!(
                    self.out,
                    "{}::{}{}",
                    NAMESPACE,
                    scalar_kind_string(kind),
                    vector_size_string(size)
                )?;
            }
            TypeResolution::Value(ref other) => {
                log::error!("Type {:?} isn't a known local", other);
                return Err(Error::FeatureNotImplemented("weird local type".to_string()));
            }
        }

        //TODO: figure out the naming scheme that wouldn't collide with user names.
        write!(self.out, " {}{} = ", BAKE_PREFIX, handle.index())?;
        Ok(())
    }

    fn put_block(
        &mut self,
        level: Level,
        statements: &[crate::Statement],
        context: &StatementContext,
    ) -> Result<(), Error> {
        // Add to the set in order to track the stack size.
        #[cfg(test)]
        #[allow(trivial_casts)]
        self.put_block_stack_pointers
            .insert(&level as *const _ as *const ());

        for statement in statements {
            log::trace!("statement[{}] {:?}", level.0, statement);
            match *statement {
                crate::Statement::Emit(ref range) => {
                    for handle in range.clone() {
                        let min_ref_count =
                            context.expression.function.expressions[handle].bake_ref_count();
                        if min_ref_count <= context.expression.info[handle].ref_count {
                            write!(self.out, "{}", level)?;
                            self.start_baking_expression(handle, &context.expression)?;
                            self.put_expression(handle, &context.expression, true)?;
                            writeln!(self.out, ";")?;
                            self.named_expressions.insert(handle.index());
                        }
                    }
                }
                crate::Statement::Block(ref block) => {
                    if !block.is_empty() {
                        writeln!(self.out, "{}{{", level)?;
                        self.put_block(level.next(), block, context)?;
                        writeln!(self.out, "{}}}", level)?;
                    }
                }
                crate::Statement::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    write!(self.out, "{}if (", level)?;
                    self.put_expression(condition, &context.expression, true)?;
                    writeln!(self.out, ") {{")?;
                    self.put_block(level.next(), accept, context)?;
                    if !reject.is_empty() {
                        writeln!(self.out, "{}}} else {{", level)?;
                        self.put_block(level.next(), reject, context)?;
                    }
                    writeln!(self.out, "{}}}", level)?;
                }
                crate::Statement::Switch {
                    selector,
                    ref cases,
                    ref default,
                } => {
                    write!(self.out, "{}switch(", level)?;
                    self.put_expression(selector, &context.expression, true)?;
                    writeln!(self.out, ") {{")?;
                    let lcase = level.next();
                    for case in cases.iter() {
                        writeln!(self.out, "{}case {}: {{", lcase, case.value)?;
                        self.put_block(lcase.next(), &case.body, context)?;
                        if case.fall_through {
                            writeln!(self.out, "{}break;", lcase.next())?;
                        }
                        writeln!(self.out, "{}}}", lcase)?;
                    }
                    writeln!(self.out, "{}default: {{", lcase)?;
                    self.put_block(lcase.next(), default, context)?;
                    writeln!(self.out, "{}}}", lcase)?;
                    writeln!(self.out, "{}}}", level)?;
                }
                crate::Statement::Loop {
                    ref body,
                    ref continuing,
                } => {
                    if !continuing.is_empty() {
                        let gate_name = self.namer.call("loop_init");
                        writeln!(self.out, "{}bool {} = true;", level, gate_name)?;
                        writeln!(self.out, "{}while(true) {{", level)?;
                        let lif = level.next();
                        writeln!(self.out, "{}if (!{}) {{", lif, gate_name)?;
                        self.put_block(lif.next(), continuing, context)?;
                        writeln!(self.out, "{}}}", lif)?;
                        writeln!(self.out, "{}{} = false;", lif, gate_name)?;
                    } else {
                        writeln!(self.out, "{}while(true) {{", level)?;
                    }
                    self.put_block(level.next(), body, context)?;
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
                    self.put_return_value(
                        level.clone(),
                        expr_handle,
                        context.result_struct,
                        &context.expression,
                    )?;
                }
                crate::Statement::Return { value: None } => {
                    writeln!(self.out, "{}return;", level)?;
                }
                crate::Statement::Kill => {
                    writeln!(self.out, "{}{}::discard_fragment();", level, NAMESPACE)?;
                }
                crate::Statement::Store { pointer, value } => {
                    // we can't assign fixed-size arrays
                    let pointer_info = &context.expression.info[pointer];
                    let array_size =
                        match *pointer_info.ty.inner_with(&context.expression.module.types) {
                            crate::TypeInner::Pointer { base, .. } => {
                                match context.expression.module.types[base].inner {
                                    crate::TypeInner::Array {
                                        size: crate::ArraySize::Constant(ch),
                                        ..
                                    } => Some(ch),
                                    _ => None,
                                }
                            }
                            _ => None,
                        };
                    match array_size {
                        Some(const_handle) => {
                            let size = context.expression.module.constants[const_handle]
                                .to_array_length()
                                .unwrap();
                            write!(self.out, "{}for(int _i=0; _i<{}; ++_i) ", level, size)?;
                            self.put_expression(pointer, &context.expression, true)?;
                            write!(self.out, "[_i] = ")?;
                            self.put_expression(value, &context.expression, true)?;
                            writeln!(self.out, "[_i];")?;
                        }
                        None => {
                            write!(self.out, "{}", level)?;
                            self.put_expression(pointer, &context.expression, true)?;
                            write!(self.out, " = ")?;
                            self.put_expression(value, &context.expression, true)?;
                            writeln!(self.out, ";")?;
                        }
                    }
                }
                crate::Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => {
                    write!(self.out, "{}", level)?;
                    self.put_expression(image, &context.expression, false)?;
                    write!(self.out, ".write(")?;
                    self.put_expression(value, &context.expression, true)?;
                    write!(self.out, ", ")?;
                    self.put_storage_image_coordinate(coordinate, &context.expression)?;
                    if let Some(expr) = array_index {
                        write!(self.out, ", ")?;
                        self.put_expression(expr, &context.expression, true)?;
                    }
                    writeln!(self.out, ");")?;
                }
                crate::Statement::Call {
                    function,
                    ref arguments,
                    result,
                } => {
                    write!(self.out, "{}", level)?;
                    if let Some(expr) = result {
                        self.start_baking_expression(expr, &context.expression)?;
                        self.named_expressions.insert(expr.index());
                    }
                    let fun_name = &self.names[&NameKey::Function(function)];
                    write!(self.out, "{}(", fun_name)?;
                    // first, write down the actual arguments
                    for (i, &handle) in arguments.iter().enumerate() {
                        if i != 0 {
                            write!(self.out, ", ")?;
                        }
                        self.put_expression(handle, &context.expression, true)?;
                    }
                    // follow-up with any global resources used
                    let mut separate = !arguments.is_empty();
                    let fun_info = &context.mod_info[function];
                    for (handle, var) in context.expression.module.global_variables.iter() {
                        if !fun_info[handle].is_empty() && var.class.needs_pass_through() {
                            let name = &self.names[&NameKey::GlobalVariable(handle)];
                            if separate {
                                write!(self.out, ", ")?;
                            } else {
                                separate = true;
                            }
                            write!(self.out, "{}", name)?;
                        }
                    }
                    // done
                    writeln!(self.out, ");")?;
                }
            }
        }

        // un-emit expressions
        //TODO: take care of loop/continuing?
        for statement in statements {
            if let crate::Statement::Emit(ref range) = *statement {
                for handle in range.clone() {
                    self.named_expressions.remove(handle.index());
                }
            }
        }
        Ok(())
    }

    pub fn write(
        &mut self,
        module: &crate::Module,
        info: &ModuleInfo,
        options: &Options,
    ) -> Result<TranslationInfo, Error> {
        self.names.clear();
        self.namer.reset(module, RESERVED, &mut self.names);

        writeln!(self.out, "#include <metal_stdlib>")?;
        writeln!(self.out, "#include <simd/simd.h>")?;
        writeln!(self.out)?;

        self.write_scalar_constants(module)?;
        self.write_type_defs(module)?;
        self.write_composite_constants(module)?;
        self.write_functions(module, info, options)
    }

    fn write_type_defs(&mut self, module: &crate::Module) -> Result<(), Error> {
        for (handle, ty) in module.types.iter() {
            let name = &self.names[&NameKey::Type(handle)];
            let global_use = GlobalUse::all(); //TODO
            match ty.inner {
                // work around Metal toolchain bug with `uint` typedef
                crate::TypeInner::Scalar {
                    kind: crate::ScalarKind::Uint,
                    ..
                } => {
                    writeln!(self.out, "typedef metal::uint {};", name)?;
                }
                crate::TypeInner::Scalar { kind, .. } => {
                    writeln!(self.out, "typedef {} {};", scalar_kind_string(kind), name)?;
                }
                crate::TypeInner::Vector { size, kind, .. } => {
                    writeln!(
                        self.out,
                        "typedef {}::{}{} {};",
                        NAMESPACE,
                        scalar_kind_string(kind),
                        vector_size_string(size),
                        name
                    )?;
                }
                crate::TypeInner::Matrix { columns, rows, .. } => {
                    writeln!(
                        self.out,
                        "typedef {}::{}{}x{} {};",
                        NAMESPACE,
                        scalar_kind_string(crate::ScalarKind::Float),
                        vector_size_string(columns),
                        vector_size_string(rows),
                        name
                    )?;
                }
                crate::TypeInner::Pointer { base, class } => {
                    let base_name = &self.names[&NameKey::Type(base)];
                    let class_name = match class.get_name(global_use) {
                        Some(name) => name,
                        None => continue,
                    };
                    writeln!(self.out, "typedef {} {} *{};", class_name, base_name, name)?;
                }
                crate::TypeInner::ValuePointer {
                    size: None,
                    kind,
                    width: _,
                    class,
                } => {
                    let class_name = match class.get_name(global_use) {
                        Some(name) => name,
                        None => continue,
                    };
                    writeln!(
                        self.out,
                        "typedef {} {} *{};",
                        class_name,
                        scalar_kind_string(kind),
                        name
                    )?;
                }
                crate::TypeInner::ValuePointer {
                    size: Some(size),
                    kind,
                    width: _,
                    class,
                } => {
                    let class_name = match class.get_name(global_use) {
                        Some(name) => name,
                        None => continue,
                    };
                    writeln!(
                        self.out,
                        "typedef {} {}::{}{} {};",
                        class_name,
                        NAMESPACE,
                        scalar_kind_string(kind),
                        vector_size_string(size),
                        name
                    )?;
                }
                crate::TypeInner::Array {
                    base,
                    size,
                    stride: _,
                } => {
                    let base_name = &self.names[&NameKey::Type(base)];
                    let size_str = match size {
                        crate::ArraySize::Constant(const_handle) => {
                            &self.names[&NameKey::Constant(const_handle)]
                        }
                        crate::ArraySize::Dynamic => "1",
                    };
                    writeln!(self.out, "typedef {} {}[{}];", base_name, name, size_str)?;
                }
                crate::TypeInner::Struct {
                    block: _,
                    ref members,
                } => {
                    writeln!(self.out, "struct {} {{", name)?;
                    for (index, member) in members.iter().enumerate() {
                        let member_name = &self.names[&NameKey::StructMember(handle, index as u32)];
                        let base_name = &self.names[&NameKey::Type(member.ty)];
                        writeln!(self.out, "{}{} {};", INDENT, base_name, member_name)?;
                    }
                    writeln!(self.out, "}};")?;
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
                        crate::ImageDimension::Cube => "cube",
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
                                .find(|&(_, ref var)| var.ty == handle)
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
                                return Err(Error::Validation);
                            };
                            ("texture", "", format.into(), access)
                        }
                    };
                    let base_name = scalar_kind_string(kind);
                    let array_str = if arrayed { "_array" } else { "" };
                    writeln!(
                        self.out,
                        "typedef {}::{}{}{}{}<{}, {}::access::{}> {};",
                        NAMESPACE,
                        texture_str,
                        dim_str,
                        msaa_str,
                        array_str,
                        base_name,
                        NAMESPACE,
                        access,
                        name
                    )?;
                }
                crate::TypeInner::Sampler { comparison: _ } => {
                    writeln!(self.out, "typedef {}::sampler {};", NAMESPACE, name)?;
                }
            }
        }
        Ok(())
    }

    fn write_scalar_constants(&mut self, module: &crate::Module) -> Result<(), Error> {
        for (handle, constant) in module.constants.iter() {
            match constant.inner {
                crate::ConstantInner::Scalar {
                    width: _,
                    ref value,
                } => {
                    let name = &self.names[&NameKey::Constant(handle)];
                    write!(self.out, "constexpr constant ")?;
                    match *value {
                        crate::ScalarValue::Sint(value) => {
                            write!(self.out, "int {} = {}", name, value)?;
                        }
                        crate::ScalarValue::Uint(value) => {
                            write!(self.out, "unsigned {} = {}u", name, value)?;
                        }
                        crate::ScalarValue::Float(value) => {
                            write!(self.out, "float {} = {}", name, value)?;
                            if value.fract() == 0.0 {
                                write!(self.out, ".0")?;
                            }
                        }
                        crate::ScalarValue::Bool(value) => {
                            write!(self.out, "bool {} = {}", name, value)?;
                        }
                    }
                    writeln!(self.out, ";")?;
                }
                crate::ConstantInner::Composite { .. } => {}
            }
        }
        Ok(())
    }

    fn write_composite_constants(&mut self, module: &crate::Module) -> Result<(), Error> {
        for (handle, constant) in module.constants.iter() {
            match constant.inner {
                crate::ConstantInner::Scalar { .. } => {}
                crate::ConstantInner::Composite { ty, ref components } => {
                    let name = &self.names[&NameKey::Constant(handle)];
                    let ty_name = &self.names[&NameKey::Type(ty)];
                    write!(self.out, "constexpr constant {} {} = {{", ty_name, name,)?;
                    for (i, &sub_handle) in components.iter().enumerate() {
                        let separator = if i != 0 { ", " } else { "" };
                        let sub_name = &self.names[&NameKey::Constant(sub_handle)];
                        write!(self.out, "{}{}", separator, sub_name)?;
                    }
                    writeln!(self.out, "}};")?;
                }
            }
        }
        Ok(())
    }

    // Returns the array of mapped entry point names.
    fn write_functions(
        &mut self,
        module: &crate::Module,
        mod_info: &ModuleInfo,
        options: &Options,
    ) -> Result<TranslationInfo, Error> {
        let mut pass_through_globals = Vec::new();
        for (fun_handle, fun) in module.functions.iter() {
            let fun_info = &mod_info[fun_handle];
            pass_through_globals.clear();
            for (handle, var) in module.global_variables.iter() {
                if !fun_info[handle].is_empty() && var.class.needs_pass_through() {
                    pass_through_globals.push(handle);
                }
            }

            let fun_name = &self.names[&NameKey::Function(fun_handle)];
            let result_type_name = match fun.result {
                Some(ref result) => &self.names[&NameKey::Type(result.ty)],
                None => "void",
            };
            writeln!(self.out, "{} {}(", result_type_name, fun_name)?;

            for (index, arg) in fun.arguments.iter().enumerate() {
                let name = &self.names[&NameKey::FunctionArgument(fun_handle, index as u32)];
                let param_type_name = &self.names[&NameKey::Type(arg.ty)];
                let separator =
                    separate(!pass_through_globals.is_empty() || index + 1 != fun.arguments.len());
                writeln!(
                    self.out,
                    "{}{} {}{}",
                    INDENT, param_type_name, name, separator
                )?;
            }
            for (index, &handle) in pass_through_globals.iter().enumerate() {
                let tyvar = TypedGlobalVariable {
                    module,
                    names: &self.names,
                    handle,
                    usage: fun_info[handle],
                    reference: true,
                };
                let separator = separate(index + 1 != pass_through_globals.len());
                write!(self.out, "{}", INDENT)?;
                tyvar.try_fmt(&mut self.out)?;
                writeln!(self.out, "{}", separator)?;
            }
            writeln!(self.out, ") {{")?;

            for (local_handle, local) in fun.local_variables.iter() {
                let ty_name = &self.names[&NameKey::Type(local.ty)];
                let local_name = &self.names[&NameKey::FunctionLocal(fun_handle, local_handle)];
                write!(self.out, "{}{} {}", INDENT, ty_name, local_name)?;
                if let Some(value) = local.init {
                    let value_str = &self.names[&NameKey::Constant(value)];
                    write!(self.out, " = {}", value_str)?;
                }
                writeln!(self.out, ";")?;
            }

            let context = StatementContext {
                expression: ExpressionContext {
                    function: fun,
                    origin: FunctionOrigin::Handle(fun_handle),
                    info: fun_info,
                    module,
                    options,
                },
                mod_info,
                result_struct: None,
            };
            self.named_expressions.clear();
            self.put_block(Level(1), &fun.body, &context)?;
            writeln!(self.out, "}}")?;
            writeln!(self.out)?;
        }

        let mut info = TranslationInfo {
            entry_point_names: Vec::with_capacity(module.entry_points.len()),
        };
        for (ep_index, ep) in module.entry_points.iter().enumerate() {
            let fun = &ep.function;
            let fun_info = mod_info.get_entry_point(ep_index);
            // skip this entry point if any global bindings are missing
            if !options.fake_missing_bindings {
                if let Some(err) = module
                    .global_variables
                    .iter()
                    .find_map(|(var_handle, var)| {
                        if !fun_info[var_handle].is_empty() {
                            if let Some(ref br) = var.binding {
                                if let Err(e) = options.resolve_global_binding(ep.stage, br) {
                                    return Some(e);
                                }
                            }
                        }
                        None
                    })
                {
                    info.entry_point_names.push(Err(err));
                    continue;
                }
            }

            let fun_name = &self.names[&NameKey::EntryPoint(ep_index as _)];
            info.entry_point_names.push(Ok(fun_name.clone()));

            let stage_out_name = format!("{}Output", fun_name);
            let stage_in_name = format!("{}Input", fun_name);

            let (em_str, in_mode, out_mode) = match ep.stage {
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

            let mut argument_members = Vec::new();
            for (arg_index, arg) in fun.arguments.iter().enumerate() {
                match module.types[arg.ty].inner {
                    crate::TypeInner::Struct {
                        block: _,
                        ref members,
                    } => {
                        for (member_index, member) in members.iter().enumerate() {
                            argument_members.push((
                                NameKey::StructMember(arg.ty, member_index as u32),
                                member.ty,
                                member.binding.as_ref(),
                            ))
                        }
                    }
                    _ => argument_members.push((
                        NameKey::EntryPointArgument(ep_index as _, arg_index as u32),
                        arg.ty,
                        arg.binding.as_ref(),
                    )),
                }
            }
            let varyings_member_name = self.namer.call("varyings");
            let mut varying_count = 0;
            if !argument_members.is_empty() {
                writeln!(self.out, "struct {} {{", stage_in_name)?;
                for &(ref name_key, ty, binding) in argument_members.iter() {
                    let binding = match binding {
                        Some(ref binding @ &crate::Binding::Location(..)) => binding,
                        _ => continue,
                    };
                    varying_count += 1;
                    let name = &self.names[&name_key];
                    let type_name = &self.names[&NameKey::Type(ty)];
                    let resolved = options.resolve_local_binding(binding, in_mode)?;
                    write!(self.out, "{}{} {}", INDENT, type_name, name)?;
                    resolved.try_fmt_decorated(&mut self.out, "")?;
                    writeln!(self.out, ";")?;
                }
                writeln!(self.out, "}};")?;
            }

            let result_member_name = self.namer.call("member");
            let result_type_name = match fun.result {
                Some(ref result) => {
                    let mut result_members = Vec::new();
                    if let crate::TypeInner::Struct {
                        block: _,
                        ref members,
                    } = module.types[result.ty].inner
                    {
                        for (member_index, member) in members.iter().enumerate() {
                            result_members.push((
                                &self.names[&NameKey::StructMember(result.ty, member_index as u32)],
                                member.ty,
                                member.binding.as_ref(),
                            ));
                        }
                    } else {
                        result_members.push((
                            &result_member_name,
                            result.ty,
                            result.binding.as_ref(),
                        ));
                    }
                    writeln!(self.out, "struct {} {{", stage_out_name)?;
                    for (name, ty, binding) in result_members {
                        let type_name = &self.names[&NameKey::Type(ty)];
                        let binding = binding.ok_or(Error::Validation)?;
                        if !options.allow_point_size
                            && *binding == crate::Binding::BuiltIn(crate::BuiltIn::PointSize)
                        {
                            continue;
                        }
                        let resolved = options.resolve_local_binding(binding, out_mode)?;
                        write!(self.out, "{}{} {}", INDENT, type_name, name)?;
                        resolved.try_fmt_decorated(&mut self.out, "")?;
                        writeln!(self.out, ";")?;
                    }
                    writeln!(self.out, "}};")?;
                    &stage_out_name
                }
                None => "void",
            };
            writeln!(self.out, "{} {} {}(", em_str, result_type_name, fun_name)?;

            let mut is_first_argument = true;
            if varying_count != 0 {
                writeln!(
                    self.out,
                    "  {} {} [[stage_in]]",
                    stage_in_name, varyings_member_name
                )?;
                is_first_argument = false;
            }
            for &(ref name_key, ty, binding) in argument_members.iter() {
                let binding = match binding {
                    Some(ref binding @ &crate::Binding::BuiltIn(..)) => binding,
                    _ => continue,
                };
                let name = &self.names[&name_key];
                let type_name = &self.names[&NameKey::Type(ty)];
                let resolved = options.resolve_local_binding(binding, in_mode)?;
                let separator = if is_first_argument {
                    is_first_argument = false;
                    ' '
                } else {
                    ','
                };
                write!(self.out, "{} {} {}", separator, type_name, name)?;
                resolved.try_fmt_decorated(&mut self.out, "\n")?;
            }
            for (handle, var) in module.global_variables.iter() {
                let usage = fun_info[handle];
                if usage.is_empty() || var.class == crate::StorageClass::Private {
                    continue;
                }
                let tyvar = TypedGlobalVariable {
                    module,
                    names: &self.names,
                    handle,
                    usage,
                    reference: true,
                };
                let separator = if is_first_argument {
                    is_first_argument = false;
                    ' '
                } else {
                    ','
                };
                write!(self.out, "{} ", separator)?;
                tyvar.try_fmt(&mut self.out)?;
                if let Some(ref binding) = var.binding {
                    let resolved = options.resolve_global_binding(ep.stage, binding).unwrap();
                    resolved.try_fmt_decorated(&mut self.out, "")?;
                }
                if let Some(value) = var.init {
                    let value_str = &self.names[&NameKey::Constant(value)];
                    write!(self.out, " = {}", value_str)?;
                }
                writeln!(self.out)?;
            }

            // end of the entry point argument list
            writeln!(self.out, ") {{")?;

            // Metal doesn't support private mutable variables outside of functions,
            // so we put them here, just like the locals.
            for (handle, var) in module.global_variables.iter() {
                let usage = fun_info[handle];
                if usage.is_empty() || var.class != crate::StorageClass::Private {
                    continue;
                }
                let tyvar = TypedGlobalVariable {
                    module,
                    names: &self.names,
                    handle,
                    usage,
                    reference: false,
                };
                write!(self.out, "{}", INDENT)?;
                tyvar.try_fmt(&mut self.out)?;
                let value_str = match var.init {
                    Some(value) => &self.names[&NameKey::Constant(value)],
                    None => "{}",
                };
                writeln!(self.out, " = {};", value_str)?;
            }

            // Now refactor the inputs in a way that the rest of the code expects
            for (arg_index, arg) in fun.arguments.iter().enumerate() {
                let arg_name =
                    &self.names[&NameKey::EntryPointArgument(ep_index as _, arg_index as u32)];
                match module.types[arg.ty].inner {
                    crate::TypeInner::Struct {
                        block: _,
                        ref members,
                    } => {
                        let struct_name = &self.names[&NameKey::Type(arg.ty)];
                        write!(
                            self.out,
                            "{}const {} {} = {{ ",
                            INDENT, struct_name, arg_name
                        )?;
                        for (member_index, member) in members.iter().enumerate() {
                            let name =
                                &self.names[&NameKey::StructMember(arg.ty, member_index as u32)];
                            if member_index != 0 {
                                write!(self.out, ", ")?;
                            }
                            if let Some(crate::Binding::Location(..)) = member.binding {
                                write!(self.out, "{}.", varyings_member_name)?;
                            }
                            write!(self.out, "{}", name)?;
                        }
                        writeln!(self.out, " }};")?;
                    }
                    _ => {
                        if let Some(crate::Binding::Location(..)) = arg.binding {
                            writeln!(
                                self.out,
                                "{}const auto {} = {}.{};",
                                INDENT, arg_name, varyings_member_name, arg_name
                            )?;
                        }
                    }
                }
            }

            // Finally, declare all the local variables that we need
            //TODO: we can postpone this till the relevant expressions are emitted
            for (local_handle, local) in fun.local_variables.iter() {
                let name = &self.names[&NameKey::EntryPointLocal(ep_index as _, local_handle)];
                let ty_name = &self.names[&NameKey::Type(local.ty)];
                write!(self.out, "{}{} {}", INDENT, ty_name, name)?;
                if let Some(value) = local.init {
                    let value_str = &self.names[&NameKey::Constant(value)];
                    write!(self.out, " = {}", value_str)?;
                }
                writeln!(self.out, ";")?;
            }

            let context = StatementContext {
                expression: ExpressionContext {
                    function: fun,
                    origin: FunctionOrigin::EntryPoint(ep_index as _),
                    info: fun_info,
                    module,
                    options,
                },
                mod_info,
                result_struct: Some(&stage_out_name),
            };
            self.named_expressions.clear();
            self.put_block(Level(1), &fun.body, &context)?;
            writeln!(self.out, "}}")?;
            if ep_index + 1 != module.entry_points.len() {
                writeln!(self.out)?;
            }
        }

        Ok(info)
    }
}

#[test]
fn test_stack_size() {
    use crate::valid::ValidationFlags;
    // create a module with at least one expression nested
    let mut module = crate::Module::default();
    let constant = module.constants.append(crate::Constant {
        name: None,
        specialization: None,
        inner: crate::ConstantInner::Scalar {
            value: crate::ScalarValue::Float(1.0),
            width: 4,
        },
    });
    let mut fun = crate::Function::default();
    let const_expr = fun
        .expressions
        .append(crate::Expression::Constant(constant));
    let nested_expr = fun.expressions.append(crate::Expression::Unary {
        op: crate::UnaryOperator::Negate,
        expr: const_expr,
    });
    fun.body
        .push(crate::Statement::Emit(fun.expressions.range_from(1)));
    fun.body.push(crate::Statement::If {
        condition: nested_expr,
        accept: Vec::new(),
        reject: Vec::new(),
    });
    let _ = module.functions.append(fun);
    // analyse the module
    let info = crate::valid::Validator::new(ValidationFlags::empty())
        .validate(&module)
        .unwrap();
    // process the module
    let mut writer = Writer::new(String::new());
    writer.write(&module, &info, &Default::default()).unwrap();

    {
        // check expression stack
        let mut addresses = !0usize..0usize;
        for pointer in writer.put_expression_stack_pointers {
            addresses.start = addresses.start.min(pointer as usize);
            addresses.end = addresses.end.max(pointer as usize);
        }
        let stack_size = addresses.end - addresses.start;
        // check the size (in debug only)
        // last observed macOS value: 18768
        if stack_size < 18000 || stack_size > 20000 {
            panic!("`put_expression` stack size {} has changed!", stack_size);
        }
    }

    {
        // check block stack
        let mut addresses = !0usize..0usize;
        for pointer in writer.put_block_stack_pointers {
            addresses.start = addresses.start.min(pointer as usize);
            addresses.end = addresses.end.max(pointer as usize);
        }
        let stack_size = addresses.end - addresses.start;
        // check the size (in debug only)
        // last observed macOS value: 12736
        if stack_size < 12000 || stack_size > 13500 {
            panic!("`put_block` stack size {} has changed!", stack_size);
        }
    }
}
