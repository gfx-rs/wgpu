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

use std::{
    fmt::{
        Display, Error as FmtError, Formatter, Write,
    },
};

use crate::{
    storage::Token,
    FastHashMap, FastHashSet
};

#[derive(Clone, Debug, PartialEq)]
pub struct BindTarget {
    pub buffer: Option<u8>,
    pub texture: Option<u8>,
    pub sampler: Option<u8>,
}

#[derive(Clone, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub struct BindSource {
    pub set: spirv::Word,
    pub binding: spirv::Word,
}

pub type BindingMap = FastHashMap<BindSource, BindTarget>;

enum ResolvedBinding {
    BuiltIn(spirv::BuiltIn),
    Attribute(spirv::Word),
    Color(spirv::Word),
    User { prefix: &'static str, index: spirv::Word },
    Resource(BindTarget),
}

#[derive(Debug)]
pub enum Error {
    Format(FmtError),
    UnsupportedExecutionModel(spirv::ExecutionModel),
    UnexpectedLocation,
    MixedExecutionModels(crate::Token<crate::Function>),
    MissingBinding(crate::Token<crate::GlobalVariable>),
    MissingBindTarget(BindSource),
    BadName(String),
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

pub struct Options<'a> {
    pub binding_map: &'a BindingMap,
}

impl Options<'_> {
    fn resolve_binding(&self, binding: &crate::Binding, mode: LocationMode) -> Result<ResolvedBinding, Error> {
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

impl Indexed for crate::Token<crate::Type> {
    const CLASS: &'static str = "Type";
    fn id(&self) -> usize { self.index() }
}
impl Indexed for crate::Token<crate::GlobalVariable> {
    const CLASS: &'static str = "global";
    fn id(&self) -> usize { self.index() }
}
impl Indexed for crate::Token<crate::Function> {
    const CLASS: &'static str = "function";
    fn id(&self) -> usize { self.index() }
}

struct MemberIndex(usize);
impl Indexed for MemberIndex {
    const CLASS: &'static str = "field";
    fn id(&self) -> usize { self.0 }
}
struct ParameterIndex(usize);
impl Indexed for ParameterIndex {
    const CLASS: &'static str = "param";
    fn id(&self) -> usize { self.0 }
}
struct InputStructIndex(crate::Token<crate::Function>);
impl Indexed for InputStructIndex {
    const CLASS: &'static str = "Input";
    const PREFIX: bool = true;
    fn id(&self) -> usize { self.0.index() }
}
struct OutputStructIndex(crate::Token<crate::Function>);
impl Indexed for OutputStructIndex {
    const CLASS: &'static str = "Output";
    const PREFIX: bool = true;
    fn id(&self) -> usize { self.0.index() }
}

enum NameSource<'a> {
    Custom { name: &'a str, prefix: bool },
    Index(usize),
}

const RESERVED_NAMES: &[&str] = &[
    "main",
];

struct Name<'a> {
    class: &'static str,
    source: NameSource<'a>,
}
impl Display for Name<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self.source {
            NameSource::Custom { name, prefix: false } if RESERVED_NAMES.contains(&name) => {
                write!(formatter, "{}_", name)
            }
            NameSource::Custom { name, prefix: false } => formatter.write_str(name),
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
                Some(ref name) if !name.is_empty() => NameSource::Custom { name, prefix: I::PREFIX },
                _ => NameSource::Index(index.id()),
            },
        }
    }
}

struct Starred<T>(T);
impl<T: Display> Display for Starred<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(formatter, "*{}", self.0)
    }
}

struct TypedGlobalVariable<'a> {
    module: &'a crate::Module,
    token: crate::Token<crate::GlobalVariable>,
}
impl Display for TypedGlobalVariable<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), FmtError> {
        let var = &self.module.global_variables[self.token];
        let name = var.name.or_index(self.token);
        let ty = &self.module.types[var.ty];
        match ty.inner {
            crate::TypeInner::Pointer { base, class }  => {
                let ty_token = match class {
                    spirv::StorageClass::Input |
                    spirv::StorageClass::Output |
                    spirv::StorageClass::UniformConstant => base,
                    _ => var.ty
                };
                let ty_name = self.module.types[ty_token].name.or_index(ty_token);
                write!(formatter, "{} {}", ty_name, name)
            }
            _ => panic!("Unexpected global type {:?}", var.ty),
        }
    }
}

impl Display for ResolvedBinding {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), FmtError> {
        match *self {
            ResolvedBinding::BuiltIn(built_in) => {
                let name = match built_in {
                    spirv::BuiltIn::ClipDistance => "clip_distance",
                    spirv::BuiltIn::PointSize => "point_size",
                    spirv::BuiltIn::Position => "position",
                    _ => panic!("Built in {:?} is not implemented", built_in),
                };
                formatter.write_str(name)
            }
            ResolvedBinding::Attribute(index) => {
                write!(formatter, "attribute({})", index)
            }
            ResolvedBinding::Color(index) => {
                write!(formatter, "color({})", index)
            }
            ResolvedBinding::User { prefix, index } => {
                write!(formatter, "user({}{})", prefix, index)
            }
            ResolvedBinding::Resource(ref target) => {
                if let Some(id) = target.buffer {
                    write!(formatter, "buffer({})", id)
                } else if let Some(id) = target.texture {
                    write!(formatter, "texture({})", id)
                } else if let Some(id) = target.sampler {
                    write!(formatter, "sampler({})", id)
                } else {
                    unimplemented!()
                }
            }
        }
    }
}

pub struct Writer<W> {
    out: W,
}

fn scalar_kind_string(kind: crate::ScalarKind) -> &'static str {
    match kind {
        crate::ScalarKind::Float => "float",
        crate::ScalarKind::Sint => "signed int",
        crate::ScalarKind::Uint => "unsigned int",
    }
}

fn vector_size_string(size: crate::VectorSize) -> &'static str {
    match size {
        crate::VectorSize::Bi => "2",
        crate::VectorSize::Tri => "3",
        crate::VectorSize::Quad => "4",
    }
}

const NAME_INPUT: &str = "input";
const NAME_OUTPUT: &str = "output";
const COMPONENTS: &[char] = &['x', 'y', 'z', 'w'];

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
    fn borrow_type(&self, token: Token<crate::Type>) -> MaybeOwned<crate::TypeInner> {
        MaybeOwned::Borrowed(&self.types[token].inner)
    }
}

impl<W: Write> Writer<W> {
    fn put_expression<'a>(
        &mut self,
        expr_token: crate::Token<crate::Expression>,
        expressions: &crate::Storage<crate::Expression>,
        module: &'a crate::Module,
    ) -> Result<MaybeOwned<'a, crate::TypeInner>, Error> {
        let expression = &expressions[expr_token];
        log::trace!("expression {:?}", expression);
        match *expression {
            crate::Expression::AccessIndex { base, index } => {
                match *self.put_expression(base, expressions, module)?.borrow() {
                    crate::TypeInner::Struct { ref members } => {
                        let member = &members[index as usize];
                        let name = member.name.or_index(MemberIndex(index as usize));
                        write!(self.out, ".{}", name)?;
                        Ok(module.borrow_type(member.ty))
                    }
                    crate::TypeInner::Matrix { rows, kind, width, .. } => {
                        write!(self.out, ".{}", COMPONENTS[index as usize])?;
                        Ok(MaybeOwned::Owned(crate::TypeInner::Vector { size: rows, kind, width }))
                    }
                    crate::TypeInner::Vector { kind, width, .. } => {
                        write!(self.out, ".{}", COMPONENTS[index as usize])?;
                        Ok(MaybeOwned::Owned(crate::TypeInner::Scalar { kind, width }))
                    }
                    crate::TypeInner::Array { base, size } => {
                        assert!(index < size);
                        write!(self.out, "[{}]", index)?;
                        Ok(module.borrow_type(base))
                    }
                    ref other => panic!("Unexpected indexing of {:?}", other),
                }
            }
            crate::Expression::Constant(token) => {
                let kind = match module.constants[token].inner {
                    crate::ConstantInner::Sint(value) => {
                        write!(self.out, "{}", value)?;
                        crate::ScalarKind::Sint
                    }
                    crate::ConstantInner::Uint(value) => {
                        write!(self.out, "{}", value)?;
                        crate::ScalarKind::Uint
                    }
                    crate::ConstantInner::Float(value) => {
                        write!(self.out, "{}", value)?;
                        crate::ScalarKind::Float
                    }
                };
                let width = 32; //TODO: not sure how to get that...
                Ok(MaybeOwned::Owned(crate::TypeInner::Scalar { kind, width }))
            }
            crate::Expression::Compose { ty, ref components } => {
                let inner = &module.types[ty].inner;
                match *inner {
                    crate::TypeInner::Vector { size, kind, .. } => {
                        write!(self.out, "{}{}(", scalar_kind_string(kind), vector_size_string(size))?;
                        for (i, &token) in components.iter().enumerate() {
                            if i != 0 {
                                write!(self.out, ",")?;
                            }
                            self.put_expression(token, expressions, module)?;
                        }
                        write!(self.out, ")")?;
                    }
                    _ => panic!("Unsupported compose {:?}", ty),
                }
                Ok(MaybeOwned::Borrowed(inner))
            }
            crate::Expression::GlobalVariable(token) => {
                let var = &module.global_variables[token];
                let inner = &module.types[var.ty].inner;
                match var.class {
                    spirv::StorageClass::Output => {
                        self.out.write_str(NAME_OUTPUT)?;
                        if let crate::TypeInner::Pointer { base, .. } = *inner {
                            let base_inner = &module.types[base].inner;
                            if let crate::TypeInner::Struct { .. } = *base_inner {
                                return Ok(MaybeOwned::Borrowed(base_inner));
                            }
                        }
                        self.out.write_str(".")?;
                    }
                    spirv::StorageClass::Input => {
                        write!(self.out, "{}.", NAME_INPUT)?;
                    }
                    _ => ()
                }
                let name = var.name.or_index(token);
                write!(self.out, "{}", name)?;
                Ok(MaybeOwned::Borrowed(inner))
            }
            crate::Expression::Load { pointer } => {
                //write!(self.out, "*")?;
                match *self.put_expression(pointer, expressions, module)?.borrow() {
                    crate::TypeInner::Pointer { base, .. } => {
                        Ok(module.borrow_type(base))
                    }
                    ref other => panic!("Unexpected load pointer {:?}", other),
                }
            }
            crate::Expression::Mul(left, right) => {
                write!(self.out, "(")?;
                let ty_left = self.put_expression(left, expressions, module)?;
                write!(self.out, " * ")?;
                let ty_right = self.put_expression(right, expressions, module)?;
                write!(self.out, ")")?;
                Ok(match (ty_left.borrow(), ty_right.borrow()) {
                    (&crate::TypeInner::Vector { size, kind, width }, &crate::TypeInner::Scalar { .. }) =>
                        MaybeOwned::Owned(crate::TypeInner::Vector { size, kind, width }),
                    other => panic!("Unable to infer Mul for {:?}", other),
                })
            }
            crate::Expression::ImageSample { image, sampler, coordinate } => {
                let ty_image = self.put_expression(image, expressions, module)?;
                write!(self.out, ".sample(")?;
                self.put_expression(sampler, expressions, module)?;
                write!(self.out, ", ")?;
                self.put_expression(coordinate, expressions, module)?;
                write!(self.out, ")")?;
                match *ty_image.borrow() {
                    crate::TypeInner::Image { base, .. } => Ok(module.borrow_type(base)),
                    ref other => panic!("Unexpected image type {:?}", other),
                }
            }
            ref other => panic!("Unsupported {:?}", other),
        }
    }

    pub fn write(&mut self, module: &crate::Module, options: Options) -> Result<(), Error> {
        writeln!(self.out, "#include <metal_stdlib>")?;
        writeln!(self.out, "#include <simd/simd.h>")?;
        writeln!(self.out, "using namespace metal;")?;

        // write down complex types
        writeln!(self.out, "")?;
        for (token, ty) in module.types.iter() {
            write!(self.out, "typedef ")?;
            let name = ty.name.or_index(token);
            match ty.inner {
                crate::TypeInner::Void => {
                    write!(self.out, "void {}", name)?;
                },
                crate::TypeInner::Scalar { kind, .. } => {
                    write!(self.out, "{} {}", scalar_kind_string(kind), name)?;
                },
                crate::TypeInner::Vector { size, kind, .. } => {
                    write!(self.out, "{}{} {}", scalar_kind_string(kind), vector_size_string(size), name)?;
                },
                crate::TypeInner::Matrix { columns, rows, kind, .. } => {
                    write!(self.out, "{}{}x{} {}", scalar_kind_string(kind), vector_size_string(columns), vector_size_string(rows), name)?;
                }
                crate::TypeInner::Pointer { base, class } => {
                    let base_name = module.types[base].name.or_index(base);
                    let class_name = match class {
                        spirv::StorageClass::Input |
                        spirv::StorageClass::Output => continue,
                        spirv::StorageClass::UniformConstant => "constant",
                        other => {
                            log::warn!("Unexpected pointer class {:?}", other);
                            ""
                        }
                    };
                    write!(self.out, "{} {} *{}", class_name, base_name, name)?;
                }
                crate::TypeInner::Array { base, size } => {
                    let base_name = module.types[base].name.or_index(base);
                    write!(self.out, "{} {}[{}]", base_name, name, size)?;
                }
                crate::TypeInner::Struct { ref members } => {
                    writeln!(self.out, "struct {} {{", name)?;
                    for (index, member) in members.iter().enumerate() {
                        let name = member.name.or_index(MemberIndex(index));
                        let base_name = module.types[member.ty].name.or_index(member.ty);
                        write!(self.out, "\t{} {}", base_name, name)?;
                        if let Some(ref binding) = member.binding {
                            let resolved = options.resolve_binding(binding, LocationMode::Intermediate)?;
                            write!(self.out, " [[{}]]", resolved)?;
                        }
                        writeln!(self.out, ";")?;
                    }
                    writeln!(self.out, "}};")?;
                }
                crate::TypeInner::Image { base, dim, flags } => {
                    let base_name = module.types[base].name.or_index(base);
                    let dim = match dim {
                        spirv::Dim::Dim1D => "1d",
                        spirv::Dim::Dim2D => "2d",
                        spirv::Dim::Dim3D => "3d",
                        spirv::Dim::DimCube => "Cube",
                        _ => panic!("Unsupported dim {:?}", dim),
                    };
                    let access = if flags.contains(crate::ImageFlags::READABLE | crate::ImageFlags::WRITABLE) {
                        "read_write"
                    } else if flags.contains(crate::ImageFlags::WRITABLE) {
                        "write"
                    } else {
                        assert!(flags.contains(crate::ImageFlags::READABLE));
                        "read"
                    };
                    write!(self.out, "texture{}<{}, access::{}> {}", dim, base_name, access, name)?;
                }
                crate::TypeInner::Sampler => {
                    write!(self.out, "sampler {}", name)?;
                }
            }
            writeln!(self.out, ";")?;
        }

        // write down functions
        let mut uniforms_used = FastHashSet::default();
        writeln!(self.out, "")?;
        for (fun_token, fun) in module.functions.iter() {
            let fun_name = fun.name.or_index(fun_token);
            // find the entry point(s) and inputs/outputs
            let mut exec_model = None;
            let mut var_inputs = FastHashSet::default();
            let mut var_outputs = FastHashSet::default();
            for ep in module.entry_points.iter() {
                if ep.function == fun_token {
                    var_inputs.extend(ep.inputs.iter().cloned());
                    var_outputs.extend(ep.outputs.iter().cloned());
                    if exec_model.is_some() {
                        if exec_model != Some(ep.exec_model) {
                            return Err(Error::MixedExecutionModels(fun_token));
                        }
                    } else {
                        exec_model = Some(ep.exec_model);
                    }
                }
            }
            let input_name = fun.name.or_index(InputStructIndex(fun_token));
            let output_name = fun.name.or_index(OutputStructIndex(fun_token));
            // make dedicated input/output structs
            if let Some(em) = exec_model {
                writeln!(self.out, "struct {} {{", input_name)?;
                let (em_str, in_mode, out_mode) = match em {
                    spirv::ExecutionModel::Vertex => ("vertex", LocationMode::VertexInput, LocationMode::Intermediate),
                    spirv::ExecutionModel::Fragment => ("fragment", LocationMode::Intermediate, LocationMode::FragmentOutput),
                    spirv::ExecutionModel::GLCompute => ("compute", LocationMode::Uniform, LocationMode::Uniform),
                    _ => return Err(Error::UnsupportedExecutionModel(em)),
                };
                for &token in var_inputs.iter() {
                    let var = &module.global_variables[token];
                    let tyvar = TypedGlobalVariable { module, token };
                    write!(self.out, "\t{}", tyvar)?;
                    if let Some(ref binding) = var.binding {
                        let resolved = options.resolve_binding(binding, in_mode)?;
                        write!(self.out, " [[{}]]", resolved)?;
                    }
                    writeln!(self.out, ";")?;
                }
                writeln!(self.out, "}};")?;
                writeln!(self.out, "struct {} {{", output_name)?;
                for &token in var_outputs.iter() {
                    let var = &module.global_variables[token];
                    // if it's a struct, lift all the built-in contents up to the root
                    if let crate::TypeInner::Pointer { base, .. } = module.types[var.ty].inner {
                        if let crate::TypeInner::Struct { ref members } = module.types[base].inner {
                            for (index, member) in members.iter().enumerate() {
                                let name = member.name.or_index(MemberIndex(index));
                                let ty_name = module.types[member.ty].name.or_index(member.ty);
                                let binding = member.binding
                                    .as_ref()
                                    .ok_or(Error::MissingBinding(token))?;
                                let resolved = options.resolve_binding(binding, out_mode)?;
                                writeln!(self.out, "\t{} {} [[{}]];", ty_name, name, resolved)?;
                            }
                            continue
                        }
                    }
                    let tyvar = TypedGlobalVariable { module, token };
                    write!(self.out, "\t{}", tyvar)?;
                    if let Some(ref binding) = var.binding {
                        let resolved = options.resolve_binding(binding, out_mode)?;
                        write!(self.out, " [[{}]]", resolved)?;
                    }
                    writeln!(self.out, ";")?;
                }
                writeln!(self.out, "}};")?;
                writeln!(self.out, "{} {} {}(", em_str, output_name, fun_name)?;
                writeln!(self.out, "\t{} {} [[stage_in]],", input_name, NAME_INPUT)?;
            } else {
                let result_type_name = module.types[fun.return_type].name.or_index(fun.return_type);
                writeln!(self.out, "{} {}(", result_type_name, fun_name)?;
                for (index, &ty) in fun.parameter_types.iter().enumerate() {
                    let name = Name::from(ParameterIndex(index));
                    let member_type_name = module.types[ty].name.or_index(ty);
                    writeln!(self.out, "\t{} {},", member_type_name, name)?;
                }
            }
            for (_, expr) in fun.expressions.iter() {
                if let crate::Expression::GlobalVariable(token) = *expr {
                    let var = &module.global_variables[token];
                    if var.class == spirv::StorageClass::UniformConstant && !uniforms_used.contains(&token) {
                        uniforms_used.insert(token);
                        let binding = var.binding
                            .as_ref()
                            .ok_or(Error::MissingBinding(token))?;
                        let resolved = options.resolve_binding(binding, LocationMode::Uniform)?;
                        let var = TypedGlobalVariable { module, token };
                        writeln!(self.out, "\t{} [[{}]],", var, resolved)?;
                    }
                }
            }
            // add an extra parameter to make Metal happy about the comma
            match exec_model {
                Some(spirv::ExecutionModel::Vertex) => {
                    writeln!(self.out, "\tunsigned _dummy [[vertex_id]]")?;
                }
                Some(spirv::ExecutionModel::Fragment) => {
                    writeln!(self.out, "\tbool _dummy [[front_facing]]")?;
                }
                Some(spirv::ExecutionModel::GLCompute) => {
                    writeln!(self.out, "\tunsigned _dummy [[threads_per_grid]]")?;
                }
                _ => {
                    writeln!(self.out, "\tint _dummy")?;
                }
            }
            writeln!(self.out, ") {{")?;
            // write down function body
            writeln!(self.out, "\t{} {};", output_name, NAME_OUTPUT)?;
            for statement in fun.body.iter() {
                log::trace!("statement {:?}", statement);
                match *statement {
                    crate::Statement::Store { pointer, value } => {
                        //write!(self.out, "\t*")?;
                        write!(self.out, "\t")?;
                        self.put_expression(pointer, &fun.expressions, module)?;
                        write!(self.out, " = ")?;
                        self.put_expression(value, &fun.expressions, module)?;
                        writeln!(self.out, ";")?;
                    }
                    crate::Statement::Return { value: None } => {
                        writeln!(self.out, "\treturn {};", NAME_OUTPUT)?;
                    }
                    _ => panic!("Unsupported {:?}", statement),
                }
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
