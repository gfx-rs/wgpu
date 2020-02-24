use std::fmt::{Error as FmtError, Write};


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
    fn put_type_pre(&mut self,  ty: &crate::Type) -> Result<(), FmtError> {
        match *ty {
            crate::Type::Void => {
                write!(self.out, "void")
            },
            crate::Type::Scalar { kind, .. } => {
                write!(self.out, "{}", scalar_kind_string(kind))
            },
            crate::Type::Vector { size, kind, .. } => {
                write!(self.out, "{}{}", scalar_kind_string(kind), vector_size_string(size))
            },
            crate::Type::Matrix { columns, rows, kind, .. } => {
                write!(self.out, "{}{}x{}", scalar_kind_string(kind), vector_size_string(columns), vector_size_string(rows))
            }
            crate::Type::Array { ref base, .. } => {
                self.put_type_pre(base)
            }
            _ => panic!("Unknown type {:?}", ty),
        }
    }

    fn put_type_post(&mut self,  ty: &crate::Type) -> Result<(), FmtError> {
        match *ty {
            crate::Type::Array { length, .. } => {
                write!(self.out, "[{}]", length)
            }
            _ => Ok(())
        }
    }

    fn put_struct_name(
        &mut self,
        token: crate::Token<crate::StructDeclaration>,
        module: &crate::Module,
    ) -> Result<(), FmtError> {
        match module.struct_declarations[token].name {
            Some(ref name) => write!(self.out, "{}", name),
            None => write!(self.out, "_Struct{}", token.index()),
        }
    }

    fn put_struct_member_name(
        &mut self,
        token: crate::Token<crate::StructDeclaration>,
        index: usize,
        module: &crate::Module,
    ) -> Result<(), FmtError> {
        match module.struct_declarations[token].members[index].name {
            Some(ref name) => write!(self.out, "{}", name),
            None => write!(self.out, "_field{}", index),
        }
    }

    pub fn write(&mut self, module: &crate::Module) -> Result<(), Error> {
        writeln!(self.out, "#include <metal_stdlib>")?;
        writeln!(self.out, "#include <simd/simd.h>")?;
        writeln!(self.out, "using namespace metal;")?;

        writeln!(self.out, "")?;
        for (token, decl) in module.struct_declarations.iter() {
            write!(self.out, "struct ")?;
            self.put_struct_name(token, module)?;
            writeln!(self.out, " {{")?;
            for (index, member) in decl.members.iter().enumerate() {
                write!(self.out, "\t")?;
                self.put_type_pre(&member.ty)?;
                write!(self.out, " ")?;
                self.put_struct_member_name(token, index, module)?;
                self.put_type_post(&member.ty)?;
                writeln!(self.out, ";")?;
            }
            writeln!(self.out, "}};")?;
        }

        Ok(())
    }
}

pub fn write_string(module: &crate::Module, _options: &Options) -> Result<String, Error> {
    let mut w = Writer { out: String::new() };
    w.write(module)?;
    Ok(w.out)
}
