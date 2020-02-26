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

use crate::{FastHashMap, FastHashSet};

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

impl Indexed for crate::Token<crate::PointerDeclaration> {
    const CLASS: &'static str = "Pointer";
    fn id(&self) -> usize { self.index() }
}
impl Indexed for crate::Token<crate::ArrayDeclaration> {
    const CLASS: &'static str = "Array";
    fn id(&self) -> usize { self.index() }
}
impl Indexed for crate::Token<crate::StructDeclaration> {
    const CLASS: &'static str = "Struct";
    fn id(&self) -> usize { self.index() }
}
impl Indexed for crate::Token<crate::ImageDeclaration> {
    const CLASS: &'static str = "Image";
    fn id(&self) -> usize { self.index() }
}
impl Indexed for crate::Token<crate::SamplerDeclaration> {
    const CLASS: &'static str = "Sampler";
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

struct TypedVar<'a, T>(&'a crate::Type, &'a T, &'a crate::ComplexTypes);
impl<T: Display> Display for TypedVar<'_, T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), FmtError> {
        match *self.0 {
            crate::Type::Void => {
                write!(formatter, "void {}", self.1)
            },
            crate::Type::Scalar { kind, .. } => {
                write!(formatter, "{} {}", scalar_kind_string(kind), self.1)
            },
            crate::Type::Vector { size, kind, .. } => {
                write!(formatter, "{}{} {}", scalar_kind_string(kind), vector_size_string(size), self.1)
            },
            crate::Type::Matrix { columns, rows, kind, .. } => {
                write!(formatter, "{}{}x{} {}", scalar_kind_string(kind), vector_size_string(columns), vector_size_string(rows), self.1)
            }
            crate::Type::Pointer(token) => {
                let decl = &self.2.pointers[token];
                write!(formatter, "{} {}", decl.name.or_index(token), self.1)
            }
            crate::Type::Array(token) => {
                let decl = &self.2.arrays[token];
                write!(formatter, "{} {}", decl.name.or_index(token), self.1)
            }
            crate::Type::Struct(token) => {
                let decl = &self.2.structs[token];
                write!(formatter, "{} {}", decl.name.or_index(token), self.1)
            }
            crate::Type::Image(token) => {
                let decl = &self.2.images[token];
                write!(formatter, "{} {}", decl.name.or_index(token), self.1)
            }
            crate::Type::Sampler(token) => {
                let decl = &self.2.samplers[token];
                write!(formatter, "{} {}", decl.name.or_index(token), self.1)
            }
        }
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
        match var.ty {
            crate::Type::Pointer(token) => {
                let decl = &self.module.complex_types.pointers[token];
                let ty = match decl.class {
                    spirv::StorageClass::Input |
                    spirv::StorageClass::Output |
                    spirv::StorageClass::UniformConstant => &decl.base,
                    _ => &var.ty
                };
                let tv = TypedVar(ty, &name, &self.module.complex_types);
                write!(formatter, "{}", tv)
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

impl<W: Write> Writer<W> {
    fn put_expression<'a>(
        &mut self,
        expr_token: crate::Token<crate::Expression>,
        expressions: &crate::Storage<crate::Expression>,
        module: &crate::Module,
    ) -> Result<crate::Type, Error> {
        let expression = &expressions[expr_token];
        log::trace!("expression {:?}", expression);
        match *expression {
            crate::Expression::AccessIndex { base, index } => {
                match self.put_expression(base, expressions, module)? {
                    crate::Type::Struct(token) => {
                        let member = &module.complex_types.structs[token].members[index as usize];
                        let name = member.name.or_index(MemberIndex(index as usize));
                        write!(self.out, ".{}", name)?;
                        Ok(member.ty.clone())
                    }
                    crate::Type::Matrix { rows, kind, width, .. } => {
                        write!(self.out, ".{}", COMPONENTS[index as usize])?;
                        Ok(crate::Type::Vector { size: rows, kind, width })
                    }
                    crate::Type::Vector { kind, width, .. } => {
                        write!(self.out, ".{}", COMPONENTS[index as usize])?;
                        Ok(crate::Type::Scalar { kind, width })
                    }
                    _ => {
                        write!(self.out, "[{}]", index)?;
                        Ok(crate::Type::Void) //TODO
                    }
                }
            }
            crate::Expression::Constant(ref constant) => {
                let kind = match *constant {
                    crate::Constant::Sint(value) => {
                        write!(self.out, "{}", value)?;
                        crate::ScalarKind::Sint
                    }
                    crate::Constant::Uint(value) => {
                        write!(self.out, "{}", value)?;
                        crate::ScalarKind::Uint
                    }
                    crate::Constant::Float(value) => {
                        write!(self.out, "{}", value)?;
                        crate::ScalarKind::Float
                    }
                };
                let width = 32; //TODO: not sure how to get that...
                Ok(crate::Type::Scalar { kind, width })
            }
            crate::Expression::Compose { ref ty, ref components } => {
                match *ty {
                    crate::Type::Vector { size, kind, .. } => {
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
                Ok(ty.clone())
            }
            crate::Expression::GlobalVariable(token) => {
                let var = &module.global_variables[token];
                match var.class {
                    spirv::StorageClass::Output => {
                        self.out.write_str(NAME_OUTPUT)?;
                        if let crate::Type::Pointer(pt) = var.ty {
                            if let crate::Type::Struct(_) = module.complex_types.pointers[pt].base {
                                return Ok(module.complex_types.pointers[pt].base.clone());
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
                Ok(var.ty.clone())
            }
            crate::Expression::Load { pointer } => {
                //write!(self.out, "*")?;
                match self.put_expression(pointer, expressions, module)? {
                    crate::Type::Pointer(token) => {
                        Ok(module.complex_types.pointers[token].base.clone())
                    }
                    other => panic!("Unexpected load pointer {:?}", other),
                }
            }
            crate::Expression::Mul(left, right) => {
                write!(self.out, "(")?;
                let ty_left = self.put_expression(left, expressions, module)?;
                write!(self.out, " * ")?;
                let ty_right = self.put_expression(right, expressions, module)?;
                write!(self.out, ")")?;
                Ok(match (ty_left, ty_right) {
                    (crate::Type::Vector { size, kind, width }, crate::Type::Scalar { .. }) =>
                        crate::Type::Vector { size, kind, width },
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
                match ty_image {
                    crate::Type::Image(token) => Ok(module.complex_types.images[token].ty.clone()),
                    other => panic!("Unexpected image type {:?}", other),
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
        for (token, decl) in module.complex_types.images.iter() {
            let name = decl.name.or_index(token);
            let dim = match decl.dim {
                spirv::Dim::Dim1D => "1d",
                spirv::Dim::Dim2D => "2d",
                spirv::Dim::Dim3D => "3d",
                spirv::Dim::DimCube => "Cube",
                _ => panic!("Unsupported dim {:?}", decl.dim),
            };
            let base = match decl.ty {
                crate::Type::Scalar { kind, .. } |
                crate::Type::Vector { kind, .. } => scalar_kind_string(kind),
                _ => panic!("Unsupported image type {:?}", decl.ty),
            };
            writeln!(self.out, "typedef texture{}<{}> {};", dim, base, name)?;
        }
        for (token, decl) in module.complex_types.samplers.iter() {
            let name = decl.name.or_index(token);
            writeln!(self.out, "typedef sampler {};", name)?;
        }
        for (token, decl) in module.complex_types.pointers.iter() {
            let name = Starred(decl.name.or_index(token));
            // Input and output are always provided as pointers,
            // but we are only interested in the contents.
            //TODO: consider resolving this at the root
            let class = match decl.class {
                spirv::StorageClass::Input |
                spirv::StorageClass::Output => continue,
                spirv::StorageClass::UniformConstant => "constant",
                other => {
                    log::warn!("Unexpected pointer class {:?}", other);
                    ""
                }
            };
            if let crate::Type::Struct(st) = decl.base {
                let var = module.complex_types.structs[st].name.or_index(st);
                writeln!(self.out, "struct {}; //forward decl", var)?;
            }
            let tv = TypedVar(&decl.base, &name, &module.complex_types);
            writeln!(self.out, "typedef {} {};", class, tv)?;
        }
        for (token, decl) in module.complex_types.arrays.iter() {
            let name = decl.name.or_index(token);
            let tv = TypedVar(&decl.base, &name, &module.complex_types);
            writeln!(self.out, "typedef {}[{}];", tv, decl.length)?;
        }
        for (token, decl) in module.complex_types.structs.iter() {
            writeln!(self.out, "struct {} {{", decl.name.or_index(token))?;
            for (index, member) in decl.members.iter().enumerate() {
                let name = member.name.or_index(MemberIndex(index));
                let tv = TypedVar(&member.ty, &name, &module.complex_types);
                write!(self.out, "\t{}", tv)?;
                if let Some(ref binding) = member.binding {
                    let resolved = options.resolve_binding(binding, LocationMode::Intermediate)?;
                    write!(self.out, " [[{}]]", resolved)?;
                }
                writeln!(self.out, ";")?;
            }
            writeln!(self.out, "}};")?;
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
                    if let crate::Type::Pointer(pt) = var.ty {
                        if let crate::Type::Struct(st) = module.complex_types.pointers[pt].base {
                            let sd = &module.complex_types.structs[st];
                            for (index, member) in sd.members.iter().enumerate() {
                                let name = member.name.or_index(MemberIndex(index));
                                let tv = TypedVar(&member.ty, &name, &module.complex_types);
                                let binding = member.binding
                                    .as_ref()
                                    .ok_or(Error::MissingBinding(token))?;
                                let resolved = options.resolve_binding(binding, out_mode)?;
                                writeln!(self.out, "\t{} [[{}]];", tv, resolved)?;
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
                let fun_tv = TypedVar(&fun.return_type, &fun_name, &module.complex_types);
                writeln!(self.out, "{}(", fun_tv)?;
                for (index, ty) in fun.parameter_types.iter().enumerate() {
                    let name = Name::from(ParameterIndex(index));
                    let tv = TypedVar(ty, &name, &module.complex_types);
                    writeln!(self.out, "\t{},", tv)?;
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
