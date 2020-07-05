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

use std::fmt::{Display, Error as FmtError, Formatter, Write};

use crate::{arena::Handle, FastHashMap};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BindTarget {
    pub buffer: Option<u8>,
    pub texture: Option<u8>,
    pub sampler: Option<u8>,
    pub mutable: bool,
}

#[derive(Clone, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub struct BindSource {
    pub set: u32,
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
    UnexpectedLocation,
    MixedExecutionModels(crate::Handle<crate::Function>),
    MissingBinding(crate::Handle<crate::GlobalVariable>),
    MissingBindTarget(BindSource),
    InvalidImageFlags(crate::ImageFlags),
    MutabilityViolation(crate::Handle<crate::GlobalVariable>),
    BadName(String),
    UnexpectedGlobalType(crate::Handle<crate::Type>),
    UnimplementedBindTarget(BindTarget),
    UnexpectedIndexing(crate::TypeInner),
    UnsupportedCompose(crate::Handle<crate::Type>),
    UnexpectedLoadPointer(crate::TypeInner),
    UnsupportedBinaryOp(crate::BinaryOperator),
    UnableToInferBinaryOpOutput(crate::TypeInner, crate::BinaryOperator, crate::TypeInner),
    UnexpectedImageType(crate::TypeInner),
    UnexpectedDistanceArgument(crate::TypeInner),
    UnsupportedCall(String),
    UnsupportedExpression(crate::Expression),
    UnableToReturnValue(crate::Handle<crate::Expression>),
    UnsupportedStatement(crate::Statement),
    AccessIndexExceedsStaticLength(u32, u32),
}

impl From<FmtError> for Error {
    fn from(e: FmtError) -> Self {
        Error::Format(e)
    }
}

#[derive(Clone, Copy, Debug)]
enum LocationMode {
    VertexInput,
    FragmentOutput,
    Intermediate,
    Uniform,
}

fn dim_str(dim: crate::ImageDimension) -> &'static str {
    match dim {
        crate::ImageDimension::D1 => "1d",
        crate::ImageDimension::D2 => "2d",
        crate::ImageDimension::D3 => "3d",
        crate::ImageDimension::Cube => "Cube",
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Options<'a> {
    pub binding_map: &'a BindingMap,
}

impl Options<'_> {
    fn resolve_binding(
        &self,
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
            crate::Binding::Descriptor { set, binding } => {
                let source = BindSource { set, binding };
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

impl Indexed for crate::Handle<crate::Type> {
    const CLASS: &'static str = "Type";
    fn id(&self) -> usize {
        self.index()
    }
}
impl Indexed for crate::Handle<crate::GlobalVariable> {
    const CLASS: &'static str = "global";
    fn id(&self) -> usize {
        self.index()
    }
}
impl Indexed for crate::Handle<crate::LocalVariable> {
    const CLASS: &'static str = "local";
    fn id(&self) -> usize {
        self.index()
    }
}
impl Indexed for crate::Handle<crate::Function> {
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
struct InputStructIndex(crate::Handle<crate::Function>);
impl Indexed for InputStructIndex {
    const CLASS: &'static str = "Input";
    const PREFIX: bool = true;
    fn id(&self) -> usize {
        self.0.index()
    }
}
struct OutputStructIndex(crate::Handle<crate::Function>);
impl Indexed for OutputStructIndex {
    const CLASS: &'static str = "Output";
    const PREFIX: bool = true;
    fn id(&self) -> usize {
        self.0.index()
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
    handle: crate::Handle<crate::GlobalVariable>,
    usage: crate::GlobalUse,
}

impl<'a> TypedGlobalVariable<'a> {
    fn try_fmt<W: Write>(&self, formatter: &mut W) -> Result<(), Error> {
        let var = &self.module.global_variables[self.handle];
        let name = var.name.or_index(self.handle);
        let (space_qualifier, reference) = match var.class {
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
        };
        let ty_name = self.module.types[var.ty].name.or_index(var.ty);
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

#[derive(Debug)]
enum MaybeOwned<'a, T: 'a> {
    Borrowed(&'a T),
    Owned(T),
}

impl<T> MaybeOwned<'_, T> {
    fn borrow(&self) -> &T {
        match *self {
            MaybeOwned::Borrowed(inner) => inner,
            MaybeOwned::Owned(ref inner) => inner,
        }
    }
}

impl crate::Module {
    fn borrow_type(&self, handle: Handle<crate::Type>) -> MaybeOwned<crate::TypeInner> {
        MaybeOwned::Borrowed(&self.types[handle].inner)
    }
}

impl<W: Write> Writer<W> {
    fn put_expression<'a>(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        function: &crate::Function,
        module: &'a crate::Module,
    ) -> Result<MaybeOwned<'a, crate::TypeInner>, Error> {
        let expression = &function.expressions[expr_handle];
        log::trace!("expression {:?} = {:?}", expr_handle, expression);
        match *expression {
            crate::Expression::Access { base, index } => {
                match *self.put_expression(base, function, module)?.borrow() {
                    crate::TypeInner::Array { base, .. } => {
                        //TODO: add size check
                        self.out.write_str("[")?;
                        self.put_expression(index, function, module)?;
                        self.out.write_str("]")?;
                        Ok(module.borrow_type(base))
                    }
                    ref other => Err(Error::UnexpectedIndexing(other.clone())),
                }
            }
            crate::Expression::AccessIndex { base, index } => {
                match *self.put_expression(base, function, module)?.borrow() {
                    crate::TypeInner::Struct { ref members } => {
                        let member = &members[index as usize];
                        let name = member.name.or_index(MemberIndex(index as usize));
                        write!(self.out, ".{}", name)?;
                        Ok(module.borrow_type(member.ty))
                    }
                    crate::TypeInner::Matrix {
                        rows, kind, width, ..
                    } => {
                        write!(self.out, ".{}", COMPONENTS[index as usize])?;
                        Ok(MaybeOwned::Owned(crate::TypeInner::Vector {
                            size: rows,
                            kind,
                            width,
                        }))
                    }
                    crate::TypeInner::Vector { kind, width, .. } => {
                        write!(self.out, ".{}", COMPONENTS[index as usize])?;
                        Ok(MaybeOwned::Owned(crate::TypeInner::Scalar { kind, width }))
                    }
                    crate::TypeInner::Array {
                        base,
                        size,
                        stride: _,
                    } => {
                        if let crate::ArraySize::Static(length) = size {
                            return Err(Error::AccessIndexExceedsStaticLength(index, length));
                        }
                        write!(self.out, "[{}]", index)?;
                        Ok(module.borrow_type(base))
                    }
                    ref other => Err(Error::UnexpectedIndexing(other.clone())),
                }
            }
            crate::Expression::Constant(handle) => self.put_constant(handle, module),
            crate::Expression::Compose { ty, ref components } => {
                let inner = &module.types[ty].inner;
                match *inner {
                    crate::TypeInner::Vector { size, kind, .. } => {
                        write!(
                            self.out,
                            "{}{}(",
                            scalar_kind_string(kind),
                            vector_size_string(size)
                        )?;
                        for (i, &handle) in components.iter().enumerate() {
                            if i != 0 {
                                write!(self.out, ",")?;
                            }
                            self.put_expression(handle, function, module)?;
                        }
                        write!(self.out, ")")?;
                    }
                    _ => return Err(Error::UnsupportedCompose(ty)),
                }
                Ok(MaybeOwned::Borrowed(inner))
            }
            crate::Expression::GlobalVariable(handle) => {
                let var = &module.global_variables[handle];
                let inner = &module.types[var.ty].inner;
                match var.class {
                    crate::StorageClass::Output => {
                        if let crate::TypeInner::Struct { .. } = *inner {
                            return Ok(MaybeOwned::Borrowed(inner));
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
                Ok(MaybeOwned::Borrowed(inner))
            }
            crate::Expression::LocalVariable(handle) => {
                let var = &function.local_variables[handle];
                let inner = &module.types[var.ty].inner;
                let name = var.name.or_index(handle);
                write!(self.out, "{}", name)?;
                Ok(MaybeOwned::Borrowed(inner))
            }
            crate::Expression::Load { pointer } => {
                //write!(self.out, "*")?;
                match *self.put_expression(pointer, function, module)?.borrow() {
                    crate::TypeInner::Pointer { base, .. } => Ok(module.borrow_type(base)),
                    ref other => Err(Error::UnexpectedLoadPointer(other.clone())),
                }
            }
            crate::Expression::Unary { op, expr } => {
                let op_str = match op {
                    crate::UnaryOperator::Negate => "-",
                    crate::UnaryOperator::Not => "!",
                };
                write!(self.out, "{}", op_str)?;
                self.put_expression(expr, function, module)
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
                    other => return Err(Error::UnsupportedBinaryOp(other)),
                };
                //write!(self.out, "(")?;
                let ty_left = self.put_expression(left, function, module)?;
                write!(self.out, " {} ", op_str)?;
                let ty_right = self.put_expression(right, function, module)?;
                //write!(self.out, ")")?;

                Ok(if op_str.len() == 1 {
                    match (ty_left.borrow(), ty_right.borrow()) {
                        (
                            &crate::TypeInner::Scalar { kind, width },
                            &crate::TypeInner::Scalar { .. },
                        ) => MaybeOwned::Owned(crate::TypeInner::Scalar { kind, width }),
                        (
                            &crate::TypeInner::Scalar { .. },
                            &crate::TypeInner::Vector { size, kind, width },
                        )
                        | (
                            &crate::TypeInner::Vector { size, kind, width },
                            &crate::TypeInner::Scalar { .. },
                        )
                        | (
                            &crate::TypeInner::Vector { size, kind, width },
                            &crate::TypeInner::Vector { .. },
                        ) => MaybeOwned::Owned(crate::TypeInner::Vector { size, kind, width }),
                        (other_left, other_right) => {
                            return Err(Error::UnableToInferBinaryOpOutput(
                                other_left.clone(),
                                op,
                                other_right.clone(),
                            ))
                        }
                    }
                } else {
                    MaybeOwned::Owned(crate::TypeInner::Scalar {
                        kind: crate::ScalarKind::Bool,
                        width: 1,
                    })
                })
            }
            crate::Expression::ImageSample {
                image,
                sampler,
                coordinate,
                depth_ref: None,
            } => {
                let ty_image = self.put_expression(image, function, module)?;
                write!(self.out, ".sample(")?;
                self.put_expression(sampler, function, module)?;
                write!(self.out, ", ")?;
                self.put_expression(coordinate, function, module)?;
                write!(self.out, ")")?;
                match *ty_image.borrow() {
                    crate::TypeInner::Image { base, .. } => Ok(module.borrow_type(base)),
                    ref other => Err(Error::UnexpectedImageType(other.clone())),
                }
            }
            crate::Expression::ImageSample {
                image,
                sampler,
                coordinate,
                depth_ref: Some(dref),
            } => {
                self.put_expression(image, function, module)?;
                write!(self.out, ".sample_compare(")?;
                self.put_expression(sampler, function, module)?;
                write!(self.out, ", ")?;
                self.put_expression(coordinate, function, module)?;
                write!(self.out, ", ")?;
                self.put_expression(dref, function, module)?;
                write!(self.out, ")")?;
                Ok(MaybeOwned::Owned(crate::TypeInner::Scalar {
                    kind: crate::ScalarKind::Float,
                    width: 4,
                }))
            }
            crate::Expression::Call {
                ref name,
                ref arguments,
            } => match name.as_str() {
                "cos" | "normalize" | "sin" => {
                    write!(self.out, "{}(", name)?;
                    let result = self.put_expression(arguments[0], function, module)?;
                    write!(self.out, ")")?;
                    Ok(result)
                }
                "fclamp" => {
                    write!(self.out, "clamp(")?;
                    let result = self.put_expression(arguments[0], function, module)?;
                    write!(self.out, ", ")?;
                    self.put_expression(arguments[1], function, module)?;
                    write!(self.out, ", ")?;
                    self.put_expression(arguments[2], function, module)?;
                    write!(self.out, ")")?;
                    Ok(result)
                }
                "atan2" => {
                    write!(self.out, "{}(", name)?;
                    let result = self.put_expression(arguments[0], function, module)?;
                    write!(self.out, ", ")?;
                    self.put_expression(arguments[1], function, module)?;
                    write!(self.out, ")")?;
                    Ok(result)
                }
                "distance" => {
                    write!(self.out, "distance(")?;
                    let result = match *self
                        .put_expression(arguments[0], function, module)?
                        .borrow()
                    {
                        crate::TypeInner::Vector { kind, width, .. } => {
                            crate::TypeInner::Scalar { kind, width }
                        }
                        ref other => return Err(Error::UnexpectedDistanceArgument(other.clone())),
                    };
                    write!(self.out, ", ")?;
                    self.put_expression(arguments[1], function, module)?;
                    write!(self.out, ")")?;
                    Ok(MaybeOwned::Owned(result))
                }
                "length" => {
                    write!(self.out, "length(")?;
                    let result = match *self
                        .put_expression(arguments[0], function, module)?
                        .borrow()
                    {
                        crate::TypeInner::Vector { kind, width, .. } => {
                            crate::TypeInner::Scalar { kind, width }
                        }
                        ref other => return Err(Error::UnexpectedDistanceArgument(other.clone())),
                    };
                    write!(self.out, ")")?;
                    Ok(MaybeOwned::Owned(result))
                }

                other => Err(Error::UnsupportedCall(other.to_owned())),
            },
            ref other => Err(Error::UnsupportedExpression(other.clone())),
        }
    }

    fn put_constant<'a>(
        &mut self,
        handle: Handle<crate::Constant>,
        module: &'a crate::Module,
    ) -> Result<MaybeOwned<'a, crate::TypeInner>, Error> {
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

        Ok(MaybeOwned::Borrowed(&ty.inner))
    }

    fn put_statement<'a>(
        &mut self,
        level: Level,
        statement: &crate::Statement,
        function: &crate::Function,
        has_output: bool,
        module: &'a crate::Module,
    ) -> Result<(), Error> {
        log::trace!("statement[{}] {:?}", level.0, statement);
        match *statement {
            crate::Statement::Empty => {}
            crate::Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                write!(self.out, "{}if (", level)?;
                self.put_expression(condition, function, module)?;
                writeln!(self.out, ") {{")?;
                for s in accept {
                    self.put_statement(level.next(), s, function, has_output, module)?;
                }
                if !reject.is_empty() {
                    writeln!(self.out, "{}}} else {{", level)?;
                    for s in reject {
                        self.put_statement(level.next(), s, function, has_output, module)?;
                    }
                }
                writeln!(self.out, "{}}}", level)?;
            }
            crate::Statement::Loop {
                ref body,
                ref continuing,
            } => {
                writeln!(self.out, "{}while(true) {{", level)?;
                for s in body {
                    self.put_statement(level.next(), s, function, has_output, module)?;
                }
                if !continuing.is_empty() {
                    //TODO
                }
                writeln!(self.out, "{}}}", level)?;
            }
            crate::Statement::Store { pointer, value } => {
                //write!(self.out, "\t*")?;
                write!(self.out, "{}", level)?;
                self.put_expression(pointer, function, module)?;
                write!(self.out, " = ")?;
                self.put_expression(value, function, module)?;
                writeln!(self.out, ";")?;
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
                    None if has_output => self.out.write_str(OUTPUT_STRUCT_NAME)?,
                    None => {}
                    Some(expr_handle) if has_output => {
                        return Err(Error::UnableToReturnValue(expr_handle));
                    }
                    Some(expr_handle) => {
                        self.put_expression(expr_handle, function, module)?;
                    }
                }
                writeln!(self.out, ";")?;
            }
            ref other => return Err(Error::UnsupportedStatement(other.clone())),
        };
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

    fn write_type_defs(&mut self, module: &crate::Module) -> Result<(), Error> {
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
                crate::TypeInner::Matrix {
                    columns,
                    rows,
                    kind,
                    ..
                } => {
                    write!(
                        self.out,
                        "typedef {}{}x{} {}",
                        scalar_kind_string(kind),
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
                crate::TypeInner::Image { base, dim, flags } => {
                    let base_name = module.types[base].name.or_index(base);
                    let dim_str = dim_str(dim);
                    let msaa_str = if flags.contains(crate::ImageFlags::MULTISAMPLED) {
                        "_ms"
                    } else {
                        ""
                    };
                    let array_str = if flags.contains(crate::ImageFlags::ARRAYED) {
                        "_array"
                    } else {
                        ""
                    };
                    let access = if flags.contains(crate::ImageFlags::SAMPLED) {
                        if flags.intersects(crate::ImageFlags::CAN_STORE) {
                            return Err(Error::InvalidImageFlags(flags));
                        }
                        "sample"
                    } else if flags
                        .contains(crate::ImageFlags::CAN_LOAD | crate::ImageFlags::CAN_STORE)
                    {
                        "read_write"
                    } else if flags.contains(crate::ImageFlags::CAN_STORE) {
                        "write"
                    } else if flags.contains(crate::ImageFlags::CAN_LOAD) {
                        "read"
                    } else {
                        return Err(Error::InvalidImageFlags(flags));
                    };
                    write!(
                        self.out,
                        "typedef texture{}{}{}<{}, access::{}> {}",
                        dim_str, msaa_str, array_str, base_name, access, name
                    )?;
                }
                crate::TypeInner::DepthImage { dim, arrayed } => {
                    let dim_str = dim_str(dim);
                    let array_str = if arrayed { "_array" } else { "" };
                    write!(
                        self.out,
                        "typedef depth{}{}<float, access::sample> {}",
                        dim_str, array_str, name
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
            let fun_name = fun.name.or_index(fun_handle);
            // find the entry point(s) and inputs/outputs
            let mut shader_stage = None;
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
            for ep in module.entry_points.iter() {
                if ep.function == fun_handle {
                    if shader_stage.is_some() {
                        if shader_stage != Some(ep.stage) {
                            return Err(Error::MixedExecutionModels(fun_handle));
                        }
                    } else {
                        shader_stage = Some(ep.stage);
                    }
                }
            }
            let output_name = fun.name.or_index(OutputStructIndex(fun_handle));

            // make dedicated input/output structs
            if let Some(stage) = shader_stage {
                assert_eq!(fun.return_type, None);
                let (em_str, in_mode, out_mode) = match stage {
                    crate::ShaderStage::Vertex => (
                        "vertex",
                        LocationMode::VertexInput,
                        LocationMode::Intermediate,
                    ),
                    crate::ShaderStage::Fragment => (
                        "fragment",
                        LocationMode::Intermediate,
                        LocationMode::FragmentOutput,
                    ),
                    crate::ShaderStage::Compute => {
                        ("kernel", LocationMode::Uniform, LocationMode::Uniform)
                    }
                };
                let location_input_name = fun.name.or_index(InputStructIndex(fun_handle));

                if stage != crate::ShaderStage::Compute {
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
                } else {
                    writeln!(self.out, "{} void {}(", em_str, fun_name)?;
                }

                for ((handle, var), &usage) in module.global_variables.iter().zip(&fun.global_usage)
                {
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
                        | (crate::ShaderStage::Fragment, crate::StorageClass::Input) => {
                            LocationMode::Intermediate
                        }
                        (crate::ShaderStage::Fragment, crate::StorageClass::Output) => {
                            LocationMode::FragmentOutput
                        }
                        _ => LocationMode::Uniform,
                    };
                    let resolved =
                        options.resolve_binding(var.binding.as_ref().unwrap(), loc_mode)?;
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
            } else {
                let result_type_name = match fun.return_type {
                    Some(type_id) => module.types[type_id].name.or_index(type_id),
                    None => Name {
                        class: "",
                        source: NameSource::Custom {
                            name: "void",
                            prefix: false,
                        },
                    },
                };
                writeln!(self.out, "{} {}(", result_type_name, fun_name)?;
                for (index, &ty) in fun.parameter_types.iter().enumerate() {
                    let name = Name::from(ParameterIndex(index));
                    let member_type_name = module.types[ty].name.or_index(ty);
                    let separator = separate(
                        index + 1 == fun.parameter_types.len() && last_used_global.is_none(),
                    );
                    writeln!(self.out, "\t{} {}{}", member_type_name, name, separator)?;
                }
            }
            writeln!(self.out, ") {{")?;

            // write down function body
            let has_output = match shader_stage {
                Some(crate::ShaderStage::Vertex) | Some(crate::ShaderStage::Fragment) => {
                    writeln!(self.out, "\t{} {};", output_name, OUTPUT_STRUCT_NAME)?;
                    true
                }
                Some(crate::ShaderStage::Compute) | None => false,
            };
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
            for statement in fun.body.iter() {
                self.put_statement(Level(1), statement, fun, has_output, module)?;
            }
            writeln!(self.out, "}}")?;
        }

        Ok(())
    }
}

pub fn write_string(module: &crate::Module, options: Options) -> Result<String, Error> {
    let mut w = Writer { out: String::new() };
    w.write(module, options)?;
    Ok(w.out)
}
