use std::fmt::{Display, Error as FmtError, Formatter, Write};


pub struct Options {
}

#[derive(Debug)]
pub enum Error {
    Format(FmtError)
}

impl From<FmtError> for Error {
    fn from(e: FmtError) -> Self {
        Error::Format(e)
    }
}

trait Indexed {
    const CLASS: &'static str;
    fn id(&self) -> usize;
}

impl Indexed for crate::Token<crate::StructDeclaration> {
    const CLASS: &'static str = "Struct";
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


enum Name<'a, I> {
    Custom(&'a str),
    Index(I),
}

impl<I: Indexed> Display for Name<'_, I> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), FmtError> {
        match *self {
            Name::Custom(name) => formatter.write_str(name),
            Name::Index(ref index) => write!(formatter, "{}{}", I::CLASS, index.id()),
        }
    }
}

trait AsName {
    fn or_index<I>(&self, index: I) -> Name<I>;
}

impl AsName for Option<String> {
    fn or_index<I>(&self, index: I) -> Name<I> {
        match *self {
            Some(ref name) => Name::Custom(name),
            None => Name::Index(index),
        }
    }
}

struct TypedVar<'a, T>(&'a crate::Type, &'a T, &'a crate::Storage<crate::StructDeclaration>);

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
            crate::Type::Array { ref base, length } => {
                write!(formatter, "{}[{}]", TypedVar(base, self.1, self.2), length)
            }
            crate::Type::Pointer { ref base, .. } => {
                write!(formatter, "{}*{}", TypedVar(base, &"", self.2), self.1)
            }
            crate::Type::Struct(token) => {
                write!(formatter, "{} {}", self.2[token].name.or_index(token), self.1)
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

impl<W: Write> Writer<W> {
    pub fn write(&mut self, module: &crate::Module) -> Result<(), Error> {
        writeln!(self.out, "#include <metal_stdlib>")?;
        writeln!(self.out, "#include <simd/simd.h>")?;
        writeln!(self.out, "using namespace metal;")?;

        writeln!(self.out, "")?;
        for (token, decl) in module.struct_declarations.iter() {
            writeln!(self.out, "struct {} {{", decl.name.or_index(token))?;
            for (index, member) in decl.members.iter().enumerate() {
                let name = member.name.or_index(MemberIndex(index));
                let tv = TypedVar(&member.ty, &name, &module.struct_declarations);
                writeln!(self.out, "\t{};", tv)?;
            }
            writeln!(self.out, "}};")?;
        }

        let mut globals_used = Vec::new();
        writeln!(self.out, "")?;
        for (fun_token, fun) in module.functions.iter() {
            let fun_name = fun.name.or_index(fun_token);
            let fun_tv = TypedVar(&fun.return_type, &fun_name, &module.struct_declarations);
            writeln!(self.out, "{}(", fun_tv)?;
            for (index, ty) in fun.parameter_types.iter().enumerate() {
                let name = Name::Index(ParameterIndex(index));
                let tv = TypedVar(ty, &name, &module.struct_declarations);
                writeln!(self.out, "\t{},", tv)?;
            }
            for (_, expr) in fun.expressions.iter() {
                if let crate::Expression::GlobalVariable(token) = *expr {
                    if !globals_used.contains(&token) {
                        globals_used.push(token);
                        let var = &module.global_variables[token];
                        let name = var.name.or_index(token);
                        let tv = TypedVar(&var.ty, &name, &module.struct_declarations);
                        writeln!(self.out, "\t{},", tv)?;
                    }
                }
            }
            writeln!(self.out, ") {{")?;
            writeln!(self.out, "}}")?;
        }

        Ok(())
    }
}

pub fn write_string(module: &crate::Module, _options: &Options) -> Result<String, Error> {
    let mut w = Writer { out: String::new() };
    w.write(module)?;
    Ok(w.out)
}
