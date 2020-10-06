/*! Metal Shading Language (MSL) backend

## Binding model

Metal's bindings are flat per resource. Since there isn't an obvious mapping
from SPIR-V's descriptor sets, we require a separate mapping provided in the options.
This mapping may have one or more resource end points for each descriptor set + index
pair.

## Outputs

In Metal, built-in shader outputs can not be nested into structures within
the output struct. If there is a structure in the outputs, and it contains any built-ins,
we move them up to the root output structure that we define ourselves.
!*/

use crate::{
    arena::Handle,
    proc::{ResolveContext, ResolveError, Typifier},
    FastHashMap,
};
use std::fmt::{Display, Error as FmtError, Formatter, Write};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BindTarget {
    pub buffer: Option<u8>,
    pub texture: Option<u8>,
    pub sampler: Option<u8>,
    pub mutable: bool,
}

#[derive(Clone, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub struct BindSource {
    pub group: u32,
    pub binding: u32,
}

pub type BindingMap = FastHashMap<BindSource, BindTarget>;

enum ResolvedBinding {
    BuiltIn(crate::BuiltIn),
    Attribute(u32),
    Color(u32),
    User { prefix: &'static str, index: u32 },
    Resource(BindTarget),
}

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

// Note: some of these should be removed in favor of proper IR validation.

#[derive(Debug)]
pub enum Error {
    Format(FmtError),
    Type(ResolveError),
    UnexpectedLocation,
    MissingBinding(Handle<crate::GlobalVariable>),
    MissingBindTarget(BindSource),
    InvalidImageAccess(crate::StorageAccess),
    MutabilityViolation(Handle<crate::GlobalVariable>),
    BadName(String),
    UnexpectedGlobalType(Handle<crate::Type>),
    UnimplementedBindTarget(BindTarget),
    UnsupportedCompose(Handle<crate::Type>),
    UnsupportedBinaryOp(crate::BinaryOperator),
    UnexpectedSampleLevel(crate::SampleLevel),
    UnsupportedCall(String),
    UnsupportedDynamicArrayLength,
    UnableToReturnValue(Handle<crate::Expression>),
    AccessIndexExceedsStaticLength(u32, u32),
    /// The source IR is not valid.
    Validation,
}

impl From<FmtError> for Error {
    fn from(e: FmtError) -> Self {
        Error::Format(e)
    }
}

impl From<ResolveError> for Error {
    fn from(e: ResolveError) -> Self {
        Error::Type(e)
    }
}

#[derive(Clone, Copy, Debug)]
enum LocationMode {
    VertexInput,
    FragmentOutput,
    Intermediate,
    Uniform,
}

#[derive(Debug, Clone, Copy)]
pub struct Options<'a> {
    pub binding_map: &'a BindingMap,
}

impl Options<'_> {
    fn resolve_binding(
        self,
        binding: &crate::Binding,
        mode: LocationMode,
    ) -> Result<ResolvedBinding, Error> {
        match *binding {
            crate::Binding::BuiltIn(built_in) => Ok(ResolvedBinding::BuiltIn(built_in)),
            crate::Binding::Location(index) => match mode {
                LocationMode::VertexInput => Ok(ResolvedBinding::Attribute(index)),
                LocationMode::FragmentOutput => Ok(ResolvedBinding::Color(index)),
                LocationMode::Intermediate => Ok(ResolvedBinding::User {
                    prefix: "loc",
                    index,
                }),
                LocationMode::Uniform => Err(Error::UnexpectedLocation),
            },
            crate::Binding::Resource { group, binding } => {
                let source = BindSource { group, binding };
                self.binding_map
                    .get(&source)
                    .cloned()
                    .map(ResolvedBinding::Resource)
                    .ok_or(Error::MissingBindTarget(source))
            }
        }
    }
}

trait Indexed {
    const CLASS: &'static str;
    const PREFIX: bool = false;
    fn id(&self) -> usize;
}

impl Indexed for Handle<crate::Type> {
    const CLASS: &'static str = "Type";
    fn id(&self) -> usize {
        self.index()
    }
}
impl Indexed for Handle<crate::GlobalVariable> {
    const CLASS: &'static str = "global";
    fn id(&self) -> usize {
        self.index()
    }
}
impl Indexed for Handle<crate::LocalVariable> {
    const CLASS: &'static str = "local";
    fn id(&self) -> usize {
        self.index()
    }
}
impl Indexed for Handle<crate::Function> {
    const CLASS: &'static str = "function";
    fn id(&self) -> usize {
        self.index()
    }
}

struct MemberIndex(usize);
impl Indexed for MemberIndex {
    const CLASS: &'static str = "field";
    fn id(&self) -> usize {
        self.0
    }
}
struct ParameterIndex(usize);
impl Indexed for ParameterIndex {
    const CLASS: &'static str = "param";
    fn id(&self) -> usize {
        self.0
    }
}

enum NameSource<'a> {
    Custom { name: &'a str, prefix: bool },
    Index(usize),
}

const RESERVED_NAMES: &[&str] = &["main"];

struct Name<'a> {
    class: &'static str,
    source: NameSource<'a>,
}
impl Display for Name<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self.source {
            NameSource::Custom {
                name,
                prefix: false,
            } if RESERVED_NAMES.contains(&name) => write!(formatter, "{}_", name),
            NameSource::Custom {
                name,
                prefix: false,
            } => formatter.write_str(name),
            NameSource::Custom { name, prefix: true } => {
                let (head, tail) = name.split_at(1);
                write!(formatter, "{}{}{}", self.class, head.to_uppercase(), tail)
            }
            NameSource::Index(index) => write!(formatter, "{}{}", self.class, index),
        }
    }
}
impl<I: Indexed> From<I> for Name<'_> {
    fn from(index: I) -> Self {
        Name {
            class: I::CLASS,
            source: NameSource::Index(index.id()),
        }
    }
}

trait AsName {
    fn or_index<I: Indexed>(&self, index: I) -> Name;
}
impl AsName for Option<String> {
    fn or_index<I: Indexed>(&self, index: I) -> Name {
        Name {
            class: I::CLASS,
            source: match *self {
                Some(ref name) if !name.is_empty() => NameSource::Custom {
                    name,
                    prefix: I::PREFIX,
                },
                _ => NameSource::Index(index.id()),
            },
        }
    }
}

struct TypedGlobalVariable<'a> {
    module: &'a crate::Module,
    handle: Handle<crate::GlobalVariable>,
    usage: crate::GlobalUse,
}

impl<'a> TypedGlobalVariable<'a> {
    fn try_fmt<W: Write>(&self, formatter: &mut W) -> Result<(), Error> {
        let var = &self.module.global_variables[self.handle];
        let name = var.name.or_index(self.handle);
        let ty = &self.module.types[var.ty];
        let ty_name = ty.name.or_index(var.ty);

        let (space_qualifier, reference) = match ty.inner {
            crate::TypeInner::Struct { .. } => match var.class {
                crate::StorageClass::Constant
                | crate::StorageClass::Uniform
                | crate::StorageClass::StorageBuffer => {
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
            formatter,
            "{}{}{} {}",
            space_qualifier, ty_name, reference, name
        )?)
    }
}

impl ResolvedBinding {
    fn try_fmt<W: Write>(&self, formatter: &mut W) -> Result<(), Error> {
        match *self {
            ResolvedBinding::BuiltIn(built_in) => {
                use crate::BuiltIn as Bi;
                let name = match built_in {
                    // vertex
                    Bi::BaseInstance => "base_instance",
                    Bi::BaseVertex => "base_vertex",
                    Bi::ClipDistance => "clip_distance",
                    Bi::InstanceIndex => "instance_id",
                    Bi::PointSize => "point_size",
                    Bi::Position => "position",
                    Bi::VertexIndex => "vertex_id",
                    // fragment
                    Bi::FragCoord => "position",
                    Bi::FragDepth => "depth(any)",
                    Bi::FrontFacing => "front_facing",
                    Bi::SampleIndex => "sample_id",
                    // compute
                    Bi::GlobalInvocationId => "thread_position_in_grid",
                    Bi::LocalInvocationId => "thread_position_in_threadgroup",
                    Bi::LocalInvocationIndex => "thread_index_in_threadgroup",
                    Bi::WorkGroupId => "threadgroup_position_in_grid",
                };
                Ok(formatter.write_str(name)?)
            }
            ResolvedBinding::Attribute(index) => Ok(write!(formatter, "attribute({})", index)?),
            ResolvedBinding::Color(index) => Ok(write!(formatter, "color({})", index)?),
            ResolvedBinding::User { prefix, index } => {
                Ok(write!(formatter, "user({}{})", prefix, index)?)
            }
            ResolvedBinding::Resource(ref target) => {
                if let Some(id) = target.buffer {
                    Ok(write!(formatter, "buffer({})", id)?)
                } else if let Some(id) = target.texture {
                    Ok(write!(formatter, "texture({})", id)?)
                } else if let Some(id) = target.sampler {
                    Ok(write!(formatter, "sampler({})", id)?)
                } else {
                    Err(Error::UnimplementedBindTarget(target.clone()))
                }
            }
        }
    }

    fn try_fmt_decorated<W: Write>(
        &self,
        formatter: &mut W,
        terminator: &str,
    ) -> Result<(), Error> {
        formatter.write_str(" [[")?;
        self.try_fmt(formatter)?;
        formatter.write_str("]]")?;
        formatter.write_str(terminator)?;
        Ok(())
    }
}

pub struct Writer<W> {
    out: W,
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

impl<W: Write> Writer<W> {
    fn put_call(
        &mut self,
        name: &str,
        parameters: &[Handle<crate::Expression>],
        function: &crate::Function,
        module: &crate::Module,
    ) -> Result<(), Error> {
        write!(self.out, "{}(", name)?;
        for (i, &handle) in parameters.iter().enumerate() {
            if i != 0 {
                write!(self.out, ", ")?;
            }
            self.put_expression(handle, function, module)?;
        }
        write!(self.out, ")")?;
        Ok(())
    }

    fn put_expression(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        function: &crate::Function,
        module: &crate::Module,
    ) -> Result<(), Error> {
        let expression = &function.expressions[expr_handle];
        log::trace!("expression {:?} = {:?}", expr_handle, expression);
        match *expression {
            crate::Expression::Access { base, index } => {
                self.put_expression(base, function, module)?;
                self.out.write_str("[")?;
                self.put_expression(index, function, module)?;
                self.out.write_str("]")?;
            }
            crate::Expression::AccessIndex { base, index } => {
                self.put_expression(base, function, module)?;
                match *self.typifier.get(base, &module.types) {
                    crate::TypeInner::Struct { ref members } => {
                        let member = &members[index as usize];
                        let name = member.name.or_index(MemberIndex(index as usize));
                        write!(self.out, ".{}", name)?;
                    }
                    crate::TypeInner::Matrix { rows: size, .. }
                    | crate::TypeInner::Vector { size, .. } => {
                        if index >= size as u32 {
                            return Err(Error::AccessIndexExceedsStaticLength(index, size as u32));
                        }
                        write!(self.out, ".{}", COMPONENTS[index as usize])?;
                    }
                    crate::TypeInner::Array { size, .. } => {
                        if let crate::ArraySize::Static(length) = size {
                            if index >= length {
                                return Err(Error::AccessIndexExceedsStaticLength(index, length));
                            }
                        }
                        write!(self.out, "[{}]", index)?;
                    }
                    _ => {
                        // unexpected indexing, should fail validation
                    }
                }
            }
            crate::Expression::Constant(handle) => self.put_constant(handle, module)?,
            crate::Expression::Compose { ty, ref components } => {
                let inner = &module.types[ty].inner;
                match *inner {
                    crate::TypeInner::Vector { size, kind, .. } => {
                        write!(
                            self.out,
                            "{}{}",
                            scalar_kind_string(kind),
                            vector_size_string(size)
                        )?;
                        self.put_call("", components, function, module)?;
                    }
                    crate::TypeInner::Scalar { width: 4, kind } if components.len() == 1 => {
                        self.put_call(scalar_kind_string(kind), components, function, module)?;
                    }
                    _ => return Err(Error::UnsupportedCompose(ty)),
                }
            }
            crate::Expression::FunctionParameter(index) => {
                let name = Name::from(ParameterIndex(index as usize));
                write!(self.out, "{}", name)?;
            }
            crate::Expression::GlobalVariable(handle) => {
                let var = &module.global_variables[handle];
                match var.class {
                    crate::StorageClass::Output => {
                        if let crate::TypeInner::Struct { .. } = module.types[var.ty].inner {
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
                let name = var.name.or_index(handle);
                write!(self.out, "{}", name)?;
            }
            crate::Expression::LocalVariable(handle) => {
                let var = &function.local_variables[handle];
                let name = var.name.or_index(handle);
                write!(self.out, "{}", name)?;
            }
            crate::Expression::Load { pointer } => {
                //write!(self.out, "*")?;
                self.put_expression(pointer, function, module)?;
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
                self.put_expression(image, function, module)?;
                write!(self.out, ".{}(", op)?;
                self.put_expression(sampler, function, module)?;
                write!(self.out, ", ")?;
                self.put_expression(coordinate, function, module)?;
                if let Some(dref) = depth_ref {
                    write!(self.out, ", ")?;
                    self.put_expression(dref, function, module)?;
                }
                match level {
                    crate::SampleLevel::Auto => {}
                    crate::SampleLevel::Zero => {
                        write!(self.out, ", level(0)")?;
                    }
                    crate::SampleLevel::Exact(h) => {
                        write!(self.out, ", level(")?;
                        self.put_expression(h, function, module)?;
                        write!(self.out, ")")?;
                    }
                    crate::SampleLevel::Bias(h) => {
                        write!(self.out, ", bias(")?;
                        self.put_expression(h, function, module)?;
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
                self.put_expression(image, function, module)?;
                write!(self.out, ".read(")?;
                self.put_expression(coordinate, function, module)?;
                if let Some(index) = index {
                    write!(self.out, ", ")?;
                    self.put_expression(index, function, module)?;
                }
                write!(self.out, ")")?;
            }
            crate::Expression::Unary { op, expr } => {
                let op_str = match op {
                    crate::UnaryOperator::Negate => "-",
                    crate::UnaryOperator::Not => "!",
                };
                write!(self.out, "{}", op_str)?;
                self.put_expression(expr, function, module)?;
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
                    other => return Err(Error::UnsupportedBinaryOp(other)),
                };
                //write!(self.out, "(")?;
                self.put_expression(left, function, module)?;
                write!(self.out, " {} ", op_str)?;
                self.put_expression(right, function, module)?;
                //write!(self.out, ")")?;
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
                self.put_call(op, &[argument], function, module)?;
            }
            crate::Expression::Transpose(expr) => {
                self.put_call("transpose", &[expr], function, module)?;
            }
            crate::Expression::DotProduct(a, b) => {
                self.put_call("dot", &[a, b], function, module)?;
            }
            crate::Expression::CrossProduct(a, b) => {
                self.put_call("cross", &[a, b], function, module)?;
            }
            crate::Expression::As {
                expr,
                kind,
                convert,
            } => {
                let scalar = scalar_kind_string(kind);
                let size = match *self.typifier.get(expr, &module.types) {
                    crate::TypeInner::Scalar { .. } => "",
                    crate::TypeInner::Vector { size, .. } => vector_size_string(size),
                    _ => return Err(Error::Validation),
                };
                let op = if convert { "static_cast" } else { "as_type" };
                write!(self.out, "{}<{}{}>(", op, scalar, size)?;
                self.put_expression(expr, function, module)?;
                write!(self.out, ")")?;
            }
            crate::Expression::Derivative { axis, expr } => {
                let op = match axis {
                    crate::DerivativeAxis::X => "dfdx",
                    crate::DerivativeAxis::Y => "dfdy",
                    crate::DerivativeAxis::Width => "fwidth",
                };
                self.put_call(op, &[expr], function, module)?;
            }
            crate::Expression::Call {
                origin: crate::FunctionOrigin::Local(handle),
                ref arguments,
            } => {
                let name = module.functions[handle].name.or_index(handle);
                write!(self.out, "{}", name)?;
                self.put_call("", arguments, function, module)?;
            }
            crate::Expression::Call {
                origin: crate::FunctionOrigin::External(ref name),
                ref arguments,
            } => match name.as_str() {
                "atan2" | "cos" | "distance" | "length" | "mix" | "normalize" | "sin" => {
                    self.put_call(name, arguments, function, module)?;
                }
                "fclamp" => {
                    self.put_call("clamp", arguments, function, module)?;
                }
                other => return Err(Error::UnsupportedCall(other.to_owned())),
            },
            crate::Expression::ArrayLength(expr) => {
                let size = match *self.typifier.get(expr, &module.types) {
                    crate::TypeInner::Array {
                        size: crate::ArraySize::Static(size),
                        ..
                    } => size,
                    crate::TypeInner::Array { .. } => {
                        return Err(Error::UnsupportedDynamicArrayLength)
                    }
                    _ => return Err(Error::Validation),
                };
                write!(self.out, "{}", size)?;
            }
        }
        Ok(())
    }

    fn put_constant(
        &mut self,
        handle: Handle<crate::Constant>,
        module: &crate::Module,
    ) -> Result<(), Error> {
        let constant = &module.constants[handle];
        let ty = &module.types[constant.ty];

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
                    self.out.write_str(".0")?;
                }
            }
            crate::ConstantInner::Bool(value) => {
                write!(self.out, "{}", value)?;
            }
            crate::ConstantInner::Composite(ref constituents) => {
                let ty_name = ty.name.or_index(constant.ty);
                write!(self.out, "{}(", ty_name)?;
                for (i, handle) in constituents.iter().enumerate() {
                    if i != 0 {
                        write!(self.out, ", ")?;
                    }
                    self.put_constant(*handle, module)?;
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
        function: &crate::Function,
        module: &crate::Module,
    ) -> Result<(), Error> {
        for statement in statements {
            log::trace!("statement[{}] {:?}", level.0, statement);
            match *statement {
                crate::Statement::Block(ref block) => {
                    if !block.is_empty() {
                        writeln!(self.out, "{}{{", level)?;
                        self.put_block(level.next(), block, function, module)?;
                        writeln!(self.out, "{}}}", level)?;
                    }
                }
                crate::Statement::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    write!(self.out, "{}if (", level)?;
                    self.put_expression(condition, function, module)?;
                    writeln!(self.out, ") {{")?;
                    self.put_block(level.next(), accept, function, module)?;
                    if !reject.is_empty() {
                        writeln!(self.out, "{}}} else {{", level)?;
                        self.put_block(level.next(), reject, function, module)?;
                    }
                    writeln!(self.out, "{}}}", level)?;
                }
                crate::Statement::Switch {
                    selector,
                    ref cases,
                    ref default,
                } => {
                    write!(self.out, "{}switch(", level)?;
                    self.put_expression(selector, function, module)?;
                    writeln!(self.out, ") {{")?;
                    let lcase = level.next();
                    for (&value, &(ref block, ref fall_through)) in cases.iter() {
                        writeln!(self.out, "{}case {}: {{", lcase, value)?;
                        self.put_block(lcase.next(), block, function, module)?;
                        if fall_through.is_none() {
                            writeln!(self.out, "{}break;", lcase.next())?;
                        }
                        writeln!(self.out, "{}}}", lcase)?;
                    }
                    writeln!(self.out, "{}default: {{", lcase)?;
                    self.put_block(lcase.next(), default, function, module)?;
                    writeln!(self.out, "{}}}", lcase)?;
                    writeln!(self.out, "{}}}", level)?;
                }
                crate::Statement::Loop {
                    ref body,
                    ref continuing,
                } => {
                    writeln!(self.out, "{}while(true) {{", level)?;
                    self.put_block(level.next(), body, function, module)?;
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
                crate::Statement::Return { value } => {
                    write!(self.out, "{}return ", level)?;
                    match value {
                        None => self.out.write_str(OUTPUT_STRUCT_NAME)?,
                        Some(expr_handle) => {
                            self.put_expression(expr_handle, function, module)?;
                        }
                    }
                    writeln!(self.out, ";")?;
                }
                crate::Statement::Kill => {
                    writeln!(self.out, "{}discard_fragment();", level)?;
                }
                crate::Statement::Store { pointer, value } => {
                    //write!(self.out, "\t*")?;
                    write!(self.out, "{}", level)?;
                    self.put_expression(pointer, function, module)?;
                    write!(self.out, " = ")?;
                    self.put_expression(value, function, module)?;
                    writeln!(self.out, ";")?;
                }
            }
        }
        Ok(())
    }

    pub fn write(&mut self, module: &crate::Module, options: Options) -> Result<(), Error> {
        writeln!(self.out, "#include <metal_stdlib>")?;
        writeln!(self.out, "#include <simd/simd.h>")?;
        writeln!(self.out, "using namespace metal;")?;

        writeln!(self.out)?;
        self.write_type_defs(module)?;

        writeln!(self.out)?;
        self.write_functions(module, options)?;

        Ok(())
    }

    fn write_type_defs<'a>(&mut self, module: &'a crate::Module) -> Result<(), Error> {
        for (handle, ty) in module.types.iter() {
            let name = ty.name.or_index(handle);
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
                    let base_name = module.types[base].name.or_index(base);
                    let class_name = match class {
                        Sc::Input | Sc::Output => continue,
                        Sc::Constant | Sc::Uniform => "constant",
                        Sc::StorageBuffer => "device",
                        Sc::Private | Sc::Function | Sc::WorkGroup => "",
                    };
                    write!(self.out, "typedef {} {} *{}", class_name, base_name, name)?;
                }
                crate::TypeInner::Array {
                    base,
                    size,
                    stride: _,
                } => {
                    let base_name = module.types[base].name.or_index(base);
                    let resolved_size = match size {
                        crate::ArraySize::Static(length) => length,
                        crate::ArraySize::Dynamic => 1,
                    };
                    write!(
                        self.out,
                        "typedef {} {}[{}]",
                        base_name, name, resolved_size
                    )?;
                }
                crate::TypeInner::Struct { ref members } => {
                    writeln!(self.out, "struct {} {{", name)?;
                    for (index, member) in members.iter().enumerate() {
                        let name = member.name.or_index(MemberIndex(index));
                        let base_name = module.types[member.ty].name.or_index(member.ty);
                        write!(self.out, "\t{} {}", base_name, name)?;
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

    fn write_functions(&mut self, module: &crate::Module, options: Options) -> Result<(), Error> {
        for (fun_handle, fun) in module.functions.iter() {
            self.typifier.resolve_all(
                &fun.expressions,
                &module.types,
                &ResolveContext {
                    constants: &module.constants,
                    global_vars: &module.global_variables,
                    local_vars: &fun.local_variables,
                    functions: &module.functions,
                    parameter_types: &fun.parameter_types,
                },
            )?;

            let fun_name = fun.name.or_index(fun_handle);
            let result_type_handle = fun.return_type.unwrap();
            let result_type_name = module.types[result_type_handle]
                .name
                .or_index(result_type_handle);
            writeln!(self.out, "{} {}(", result_type_name, fun_name)?;

            for (index, &ty) in fun.parameter_types.iter().enumerate() {
                let name = Name::from(ParameterIndex(index));
                let member_type_name = module.types[ty].name.or_index(ty);
                let separator = separate(index + 1 == fun.parameter_types.len());
                writeln!(self.out, "\t{} {}{}", member_type_name, name, separator)?;
            }
            writeln!(self.out, ") {{")?;

            for (local_handle, local) in fun.local_variables.iter() {
                let ty_name = module.types[local.ty].name.or_index(local.ty);
                write!(
                    self.out,
                    "\t{} {}",
                    ty_name,
                    local.name.or_index(local_handle)
                )?;
                if let Some(value) = local.init {
                    write!(self.out, " = ")?;
                    self.put_expression(value, fun, module)?;
                }
                writeln!(self.out, ";")?;
            }
            self.put_block(Level(1), &fun.body, fun, module)?;
            writeln!(self.out, "}}")?;
        }

        for (&(stage, ref ep_name), ep) in module.entry_points.iter() {
            let fun = &ep.function;
            self.typifier.resolve_all(
                &fun.expressions,
                &module.types,
                &ResolveContext {
                    constants: &module.constants,
                    global_vars: &module.global_variables,
                    local_vars: &fun.local_variables,
                    functions: &module.functions,
                    parameter_types: &fun.parameter_types,
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

            let fun_name = format!("{}{:?}", ep_name, stage);
            let output_name = Name {
                class: "Output",
                source: NameSource::Custom {
                    name: &fun_name,
                    prefix: true,
                },
            };

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
            let location_input_name = Name {
                class: "Input",
                source: NameSource::Custom {
                    name: &fun_name,
                    prefix: true,
                },
            };

            match stage {
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
                        let ty_handle = var.ty;
                        if let crate::TypeInner::Struct { ref members } =
                            module.types[ty_handle].inner
                        {
                            for (index, member) in members.iter().enumerate() {
                                if let crate::MemberOrigin::BuiltIn(built_in) = member.origin {
                                    let name = member.name.or_index(MemberIndex(index));
                                    let ty_name = module.types[member.ty].name.or_index(member.ty);
                                    write!(self.out, "\t{} {}", ty_name, name)?;
                                    ResolvedBinding::BuiltIn(built_in)
                                        .try_fmt_decorated(&mut self.out, ";\n")?;
                                }
                            }
                        } else if let Some(ref binding @ crate::Binding::Location(_)) = var.binding
                        {
                            let tyvar = TypedGlobalVariable {
                                module,
                                handle,
                                usage: crate::GlobalUse::empty(),
                            };
                            let resolved = options.resolve_binding(binding, in_mode)?;

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
                        let ty_handle = var.ty;
                        if let crate::TypeInner::Struct { ref members } =
                            module.types[ty_handle].inner
                        {
                            for (index, member) in members.iter().enumerate() {
                                let name = member.name.or_index(MemberIndex(index));
                                let ty_name = module.types[member.ty].name.or_index(member.ty);
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
                                handle,
                                usage: crate::GlobalUse::empty(),
                            };
                            write!(self.out, "\t")?;
                            tyvar.try_fmt(&mut self.out)?;
                            if let Some(ref binding) = var.binding {
                                let resolved = options.resolve_binding(binding, out_mode)?;
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
                }
                crate::ShaderStage::Compute => {
                    writeln!(self.out, "{} void {}(", em_str, fun_name)?;
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
                let resolved = options.resolve_binding(var.binding.as_ref().unwrap(), loc_mode)?;
                let tyvar = TypedGlobalVariable {
                    module,
                    handle,
                    usage,
                };
                let separator = separate(last_used_global == Some(handle));
                write!(self.out, "\t")?;
                tyvar.try_fmt(&mut self.out)?;
                resolved.try_fmt_decorated(&mut self.out, separator)?;
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
                let ty_name = module.types[local.ty].name.or_index(local.ty);
                write!(
                    self.out,
                    "\t{} {}",
                    ty_name,
                    local.name.or_index(local_handle)
                )?;
                if let Some(value) = local.init {
                    write!(self.out, " = ")?;
                    self.put_expression(value, fun, module)?;
                }
                writeln!(self.out, ";")?;
            }
            self.put_block(Level(1), &fun.body, fun, module)?;
            writeln!(self.out, "}}")?;
        }

        Ok(())
    }
}

pub fn write_string(module: &crate::Module, options: Options) -> Result<String, Error> {
    let mut w = Writer {
        out: String::new(),
        typifier: Typifier::new(),
    };
    w.write(module, options)?;
    Ok(w.out)
}
