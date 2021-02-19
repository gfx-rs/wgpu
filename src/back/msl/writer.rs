use super::{keywords::RESERVED, Error, LocationMode, Options, TranslationInfo};
use crate::{
    arena::Handle,
    proc::{
        analyzer::{Analysis, FunctionInfo, GlobalUse},
        EntryPointIndex, Interface, NameKey, Namer, ResolveContext, Typifier, Visitor,
    },
    FastHashMap,
};
use bit_set::BitSet;
use std::{
    fmt::{Display, Error as FmtError, Formatter},
    io::Write,
    iter, mem,
};

const NAMESPACE: &str = "metal";
const INDENT: &str = "    ";

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
                    let space = if self.usage.contains(GlobalUse::WRITE) {
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
    named_expressions: BitSet,
    visit_mask: BitSet,
    typifier: Typifier,
    namer: Namer,
    temp_bake_handles: Vec<Handle<crate::Expression>>,
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

impl crate::StorageClass {
    /// Returns true for storage classes, for which the global
    /// variables are passed in function arguments.
    /// These arguments need to be passed through any functions
    /// called from the entry point.
    fn needs_pass_through(&self) -> bool {
        match *self {
            crate::StorageClass::Input
            | crate::StorageClass::Uniform
            | crate::StorageClass::Storage
            | crate::StorageClass::Handle => true,
            _ => false,
        }
    }
}

struct BakeExpressionVisitor<'a> {
    named_expressions: &'a mut BitSet,
    bake_handles: &'a mut Vec<Handle<crate::Expression>>,
    fun_info: &'a FunctionInfo,
    exclude: Option<Handle<crate::Expression>>,
}
impl Visitor for BakeExpressionVisitor<'_> {
    fn visit_expr(&mut self, handle: Handle<crate::Expression>, expr: &crate::Expression) {
        use crate::Expression as E;
        // filter out the expressions that don't need to bake
        let min_ref_count = match *expr {
            // The following expressions can be inlined nicely.
            E::AccessIndex { .. }
            | E::Constant(_)
            | E::FunctionArgument(_)
            | E::GlobalVariable(_)
            | E::LocalVariable(_) => !0,
            // Image sampling and function calling are nice to isolate
            // into separate statements even when done only once.
            E::ImageSample { .. } | E::ImageLoad { .. } | E::Call { .. } => 1,
            // Bake only expressions referenced more than once.
            _ => 2,
        };

        let modifier = if self.exclude == Some(handle) { 1 } else { 0 };
        if self.fun_info[handle].ref_count - modifier >= min_ref_count
            && self.named_expressions.insert(handle.index())
        {
            self.bake_handles.push(handle);
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
    module: &'a crate::Module,
    analysis: &'a Analysis,
}

struct StatementContext<'a> {
    expression: ExpressionContext<'a>,
    fun_info: &'a FunctionInfo,
    return_value: Option<&'a str>,
}

impl<W: Write> Writer<W> {
    /// Creates a new `Writer` instance.
    pub fn new(out: W) -> Self {
        Writer {
            out,
            names: FastHashMap::default(),
            named_expressions: BitSet::new(),
            visit_mask: BitSet::new(),
            typifier: Typifier::new(),
            namer: Namer::default(),
            temp_bake_handles: Vec::new(),
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
            self.put_expression(handle, context)?;
        }
        write!(self.out, ")")?;
        Ok(())
    }

    fn put_local_call(
        &mut self,
        fun_handle: Handle<crate::Function>,
        parameters: &[Handle<crate::Expression>],
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        let fun_name = &self.names[&NameKey::Function(fun_handle)];
        write!(self.out, "{}(", fun_name)?;
        // first, write down the actual arguments
        for (i, &handle) in parameters.iter().enumerate() {
            if i != 0 {
                write!(self.out, ", ")?;
            }
            self.put_expression(handle, context)?;
        }
        // follow-up with any global resources used
        let mut separate = !parameters.is_empty();
        let fun_info = &context.analysis[fun_handle];
        for (handle, var) in context.module.global_variables.iter() {
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
        self.put_expression(image, context)?;
        write!(self.out, ".get_{}(", query)?;
        if let Some(expr) = level {
            self.put_expression(expr, context)?;
        }
        write!(self.out, ")")?;
        Ok(())
    }

    fn put_expression(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        context: &ExpressionContext,
    ) -> Result<(), Error> {
        if self.named_expressions.contains(expr_handle.index()) {
            write!(self.out, "expr{}", expr_handle.index())?;
            return Ok(());
        }

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
                    crate::TypeInner::Vector { .. } => {
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
                array_index,
                offset,
                level,
                depth_ref,
            } => {
                let op = match depth_ref {
                    Some(_) => "sample_compare",
                    None => "sample",
                };
                self.put_expression(image, context)?;
                write!(self.out, ".{}(", op)?;
                self.put_expression(sampler, context)?;
                write!(self.out, ", ")?;
                self.put_expression(coordinate, context)?;
                if let Some(expr) = array_index {
                    write!(self.out, ", ")?;
                    self.put_expression(expr, context)?;
                }
                if let Some(dref) = depth_ref {
                    write!(self.out, ", ")?;
                    self.put_expression(dref, context)?;
                }
                match level {
                    crate::SampleLevel::Auto => {}
                    crate::SampleLevel::Zero => {
                        //TODO: do we support Zero on `Sampled` image classes?
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
                    crate::SampleLevel::Gradient { x, y } => {
                        write!(self.out, ", gradient(")?;
                        self.put_expression(x, context)?;
                        write!(self.out, ", ")?;
                        self.put_expression(y, context)?;
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
                self.put_expression(image, context)?;
                write!(self.out, ".read(")?;
                self.put_expression(coordinate, context)?;
                if let Some(expr) = array_index {
                    write!(self.out, ", ")?;
                    self.put_expression(expr, context)?;
                }
                if let Some(index) = index {
                    write!(self.out, ", ")?;
                    self.put_expression(index, context)?;
                }
                write!(self.out, ")")?;
            }
            //Note: for all the queries, the signed integers are expected,
            // so a conversion is needed.
            crate::Expression::ImageQuery { image, query } => match query {
                crate::ImageQuery::Size { level } => {
                    //Note: MSL only has separate width/height/depth queries,
                    // so compose the result of them.
                    let dim = match *self.typifier.get(image, &context.module.types) {
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
                }
                crate::ImageQuery::NumLevels => {
                    write!(self.out, "int(")?;
                    self.put_expression(image, context)?;
                    write!(self.out, ".get_num_mip_levels())")?;
                }
                crate::ImageQuery::NumLayers => {
                    write!(self.out, "int(")?;
                    self.put_expression(image, context)?;
                    write!(self.out, ".get_array_size())")?;
                }
                crate::ImageQuery::NumSamples => {
                    write!(self.out, "int(")?;
                    self.put_expression(image, context)?;
                    write!(self.out, ".get_num_samples())")?;
                }
            },
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
                    write!(self.out, "(")?;
                    self.put_expression(left, context)?;
                    write!(self.out, " {} ", op_str)?;
                    self.put_expression(right, context)?;
                    write!(self.out, ")")?;
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
            crate::Expression::Call {
                function,
                ref arguments,
            } => {
                self.put_local_call(function, arguments, context)?;
            }
            crate::Expression::ArrayLength(expr) => {
                match *self.typifier.get(expr, &context.module.types) {
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
                }
            }
        }
        Ok(())
    }

    // Write down any required intermediate results
    fn prepare_expression(
        &mut self,
        level: Level,
        root_handle: Handle<crate::Expression>,
        context: &StatementContext,
        exclude_root: bool,
    ) -> Result<(), Error> {
        // set up the search
        self.visit_mask.clear();
        let mut interface = Interface {
            expressions: &context.expression.function.expressions,
            local_variables: &context.expression.function.local_variables,
            visitor: BakeExpressionVisitor {
                named_expressions: &mut self.named_expressions,
                bake_handles: &mut self.temp_bake_handles,
                fun_info: context.fun_info,
                exclude: if exclude_root {
                    Some(root_handle)
                } else {
                    None
                },
            },
            mask: &mut self.visit_mask,
        };
        // populate the bake handles
        interface.traverse_expr(root_handle);
        // bake
        let mut temp_bake_handles = mem::replace(&mut self.temp_bake_handles, Vec::new());
        for handle in temp_bake_handles.drain(..).rev() {
            write!(self.out, "{}", level)?;
            match self.typifier.get_handle(handle) {
                Ok(ty_handle) => {
                    let ty_name = &self.names[&NameKey::Type(ty_handle)];
                    write!(self.out, "{}", ty_name)?;
                }
                Err(&crate::TypeInner::Scalar { kind, .. }) => {
                    write!(self.out, "{}", scalar_kind_string(kind))?;
                }
                Err(&crate::TypeInner::Vector { size, kind, .. }) => {
                    write!(
                        self.out,
                        "{}::{}{}",
                        NAMESPACE,
                        scalar_kind_string(kind),
                        vector_size_string(size)
                    )?;
                }
                Err(other) => {
                    log::error!("Type {:?} isn't a known local", other);
                    return Err(Error::FeatureNotImplemented("weird local type".to_string()));
                }
            }

            //TODO: figure out the naming scheme that wouldn't collide with user names.
            write!(self.out, " expr{} = ", handle.index())?;
            // Make sure to temporarily unblock the expression before writing it down.
            self.named_expressions.remove(handle.index());
            self.put_expression(handle, &context.expression)?;
            self.named_expressions.insert(handle.index());
            writeln!(self.out, ";")?;
        }

        self.temp_bake_handles = temp_bake_handles;
        Ok(())
    }

    fn put_block(
        &mut self,
        level: Level,
        statements: &[crate::Statement],
        context: &StatementContext,
    ) -> Result<(), Error> {
        for statement in statements {
            log::trace!("statement[{}] {:?}", level.0, statement);
            match *statement {
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
                    self.prepare_expression(level.clone(), condition, context, false)?;
                    write!(self.out, "{}if (", level)?;
                    self.put_expression(condition, &context.expression)?;
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
                    self.prepare_expression(level.clone(), selector, context, false)?;
                    write!(self.out, "{}switch(", level)?;
                    self.put_expression(selector, &context.expression)?;
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
                    self.prepare_expression(level.clone(), expr_handle, context, true)?;
                    write!(self.out, "{}return ", level)?;
                    self.put_expression(expr_handle, &context.expression)?;
                    writeln!(self.out, ";")?;
                }
                crate::Statement::Return { value: None } => {
                    writeln!(
                        self.out,
                        "{}return {};",
                        level,
                        context.return_value.unwrap_or_default(),
                    )?;
                }
                crate::Statement::Kill => {
                    writeln!(self.out, "{}discard_fragment();", level)?;
                }
                crate::Statement::Store { pointer, value } => {
                    self.prepare_expression(level.clone(), value, context, true)?;
                    write!(self.out, "{}", level)?;
                    self.put_expression(pointer, &context.expression)?;
                    write!(self.out, " = ")?;
                    self.put_expression(value, &context.expression)?;
                    writeln!(self.out, ";")?;
                }
                crate::Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => {
                    self.put_expression(image, &context.expression)?;
                    write!(self.out, ".write(")?;
                    self.put_expression(value, &context.expression)?;
                    write!(self.out, ", ")?;
                    self.put_expression(coordinate, &context.expression)?;
                    if let Some(expr) = array_index {
                        write!(self.out, ", ")?;
                        self.put_expression(expr, &context.expression)?;
                    }
                    write!(self.out, ");")?;
                }
                crate::Statement::Call {
                    function,
                    ref arguments,
                } => {
                    for &arg in arguments {
                        self.prepare_expression(level.clone(), arg, context, false)?;
                    }
                    self.put_local_call(function, arguments, &context.expression)?;
                    writeln!(self.out, ";")?;
                }
            }
        }
        Ok(())
    }

    pub fn write(
        &mut self,
        module: &crate::Module,
        analysis: &Analysis,
        options: &Options,
    ) -> Result<TranslationInfo, Error> {
        self.names.clear();
        self.namer.reset(module, RESERVED, &mut self.names);

        writeln!(self.out, "#include <metal_stdlib>")?;
        writeln!(self.out, "#include <simd/simd.h>")?;
        writeln!(self.out)?;

        self.write_type_defs(module)?;
        self.write_constants(module)?;
        self.write_functions(module, analysis, options)
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
                        "typedef {}::{}{} {}",
                        NAMESPACE,
                        scalar_kind_string(kind),
                        vector_size_string(size),
                        name
                    )?;
                }
                crate::TypeInner::Matrix { columns, rows, .. } => {
                    write!(
                        self.out,
                        "typedef {}::{}{}x{} {}",
                        NAMESPACE,
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
                    let size_str = match size {
                        crate::ArraySize::Constant(const_handle) => {
                            &self.names[&NameKey::Constant(const_handle)]
                        }
                        crate::ArraySize::Dynamic => "1",
                    };
                    write!(self.out, "typedef {} {}[{}]", base_name, name, size_str)?;
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
                                return Err(Error::Validation);
                            };
                            ("texture", "", format.into(), access)
                        }
                    };
                    let base_name = scalar_kind_string(kind);
                    let array_str = if arrayed { "_array" } else { "" };
                    write!(
                        self.out,
                        "typedef {}::{}{}{}{}<{}, {}::access::{}> {}",
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
                    write!(self.out, "typedef {}::sampler {}", NAMESPACE, name)?;
                }
            }
            writeln!(self.out, ";")?;
            writeln!(self.out)?;
        }
        Ok(())
    }

    fn write_constants(&mut self, module: &crate::Module) -> Result<(), Error> {
        for (handle, constant) in module.constants.iter() {
            write!(self.out, "constexpr constant ")?;
            let name = &self.names[&NameKey::Constant(handle)];
            match constant.inner {
                crate::ConstantInner::Scalar {
                    width: _,
                    ref value,
                } => match *value {
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
                },
                crate::ConstantInner::Composite { ty, ref components } => {
                    let ty_name = &self.names[&NameKey::Type(ty)];
                    write!(self.out, "{} {} = {}(", ty_name, name, ty_name)?;
                    for (i, &sub_handle) in components.iter().enumerate() {
                        let separator = if i != 0 { ", " } else { "" };
                        let sub_name = &self.names[&NameKey::Constant(sub_handle)];
                        write!(self.out, "{}{}", separator, sub_name)?;
                    }
                    write!(self.out, ")")?;
                }
            }
            writeln!(self.out, ";")?;
        }
        Ok(())
    }

    // Returns the array of mapped entry point names.
    fn write_functions(
        &mut self,
        module: &crate::Module,
        analysis: &Analysis,
        options: &Options,
    ) -> Result<TranslationInfo, Error> {
        let mut pass_through_globals = Vec::new();
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

            let fun_info = &analysis[fun_handle];
            pass_through_globals.clear();
            for (handle, var) in module.global_variables.iter() {
                if !fun_info[handle].is_empty() && var.class.needs_pass_through() {
                    pass_through_globals.push(handle);
                }
            }

            let fun_name = &self.names[&NameKey::Function(fun_handle)];
            let result_type_name = match fun.return_type {
                Some(ret_ty) => &self.names[&NameKey::Type(ret_ty)],
                None => "void",
            };
            writeln!(self.out, "{} {}(", result_type_name, fun_name)?;

            for (index, arg) in fun.arguments.iter().enumerate() {
                let name = &self.names[&NameKey::FunctionArgument(fun_handle, index as u32)];
                let param_type_name = &self.names[&NameKey::Type(arg.ty)];
                let separator =
                    separate(pass_through_globals.is_empty() && index + 1 == fun.arguments.len());
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
                };
                let separator = separate(index + 1 == pass_through_globals.len());
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
                    module,
                    analysis,
                },
                fun_info,
                return_value: None,
            };
            self.named_expressions.clear();
            self.put_block(Level(1), &fun.body, &context)?;
            writeln!(self.out, "}}")?;
            writeln!(self.out)?;
        }

        let mut info = TranslationInfo {
            entry_point_names: Vec::with_capacity(module.entry_points.len()),
        };
        for (ep_index, (&(stage, ref ep_name), ep)) in module.entry_points.iter().enumerate() {
            let fun = &ep.function;
            let fun_info = analysis.get_entry_point(stage, ep_name);
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
            for (handle, var) in module.global_variables.iter() {
                match var.class {
                    crate::StorageClass::Input => {
                        if let Some(crate::Binding::Location(_)) = var.binding {
                            continue;
                        }
                    }
                    crate::StorageClass::Output => continue,
                    _ => {}
                }
                if !fun_info[handle].is_empty() {
                    last_used_global = Some(handle);
                }
            }

            let fun_name = &self.names[&NameKey::EntryPoint(ep_index as _)];
            info.entry_point_names.push(fun_name.clone());
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
                    for (handle, var) in module.global_variables.iter() {
                        if var.class != crate::StorageClass::Input
                            || !fun_info[handle].contains(GlobalUse::READ)
                        {
                            continue;
                        }
                        if let Some(crate::Binding::BuiltIn(_)) = var.binding {
                            // MSL disallows built-ins in input structs
                            continue;
                        }
                        let tyvar = TypedGlobalVariable {
                            module,
                            names: &self.names,
                            handle,
                            usage: GlobalUse::empty(),
                        };
                        write!(self.out, "{}", INDENT)?;
                        tyvar.try_fmt(&mut self.out)?;
                        let resolved = options.resolve_binding(stage, var, in_mode)?;
                        resolved.try_fmt_decorated(&mut self.out, ";")?;
                        writeln!(self.out)?;
                    }
                    writeln!(self.out, "}};")?;
                    writeln!(self.out)?;

                    writeln!(self.out, "struct {} {{", output_name)?;
                    for (handle, var) in module.global_variables.iter() {
                        if var.class != crate::StorageClass::Output
                            || !fun_info[handle].contains(GlobalUse::WRITE)
                        {
                            continue;
                        }

                        let tyvar = TypedGlobalVariable {
                            module,
                            names: &self.names,
                            handle,
                            usage: GlobalUse::empty(),
                        };
                        write!(self.out, "{}", INDENT)?;
                        tyvar.try_fmt(&mut self.out)?;
                        let resolved = options.resolve_binding(stage, var, out_mode)?;
                        resolved.try_fmt_decorated(&mut self.out, ";")?;
                        writeln!(self.out)?;
                    }
                    writeln!(self.out, "}};")?;
                    writeln!(self.out)?;

                    writeln!(self.out, "{} {} {}(", em_str, output_name, fun_name)?;
                    let separator = separate(last_used_global.is_none());
                    writeln!(
                        self.out,
                        "{}{} {} [[stage_in]]{}",
                        INDENT, location_input_name, LOCATION_INPUT_STRUCT_NAME, separator
                    )?;

                    Some(OUTPUT_STRUCT_NAME)
                }
                crate::ShaderStage::Compute => {
                    writeln!(self.out, "{} void {}(", em_str, fun_name)?;
                    None
                }
            };

            for (handle, var) in module.global_variables.iter() {
                let usage = fun_info[handle];
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
                let tyvar = TypedGlobalVariable {
                    module,
                    names: &self.names,
                    handle,
                    usage,
                };
                let separator = separate(last_used_global == Some(handle));
                write!(self.out, "{}", INDENT)?;
                tyvar.try_fmt(&mut self.out)?;
                if var.binding.is_some() {
                    let resolved = options.resolve_binding(stage, var, loc_mode)?;
                    resolved.try_fmt_decorated(&mut self.out, separator)?;
                }
                if let Some(value) = var.init {
                    let value_str = &self.names[&NameKey::Constant(value)];
                    write!(self.out, " = {}", value_str)?;
                }
                writeln!(self.out)?;
            }
            writeln!(self.out, ") {{")?;

            match stage {
                crate::ShaderStage::Vertex | crate::ShaderStage::Fragment => {
                    writeln!(
                        self.out,
                        "{}{} {};",
                        INDENT, output_name, OUTPUT_STRUCT_NAME
                    )?;
                }
                crate::ShaderStage::Compute => {}
            }
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
                    module,
                    analysis,
                },
                fun_info,
                return_value,
            };
            self.named_expressions.clear();
            self.put_block(Level(1), &fun.body, &context)?;
            writeln!(self.out, "}}")?;
            let is_last = ep_index == module.entry_points.len() - 1;
            if !is_last {
                writeln!(self.out)?;
            }
        }

        Ok(info)
    }
}
