use super::Error;
use crate::FastHashMap;
use crate::{
    back::wgsl::keywords::RESERVED, Function, Module, ShaderStage, StructLevel, TypeInner,
};
use crate::{
    proc::{NameKey, Namer},
    StructMember,
};
use std::io::Write;

const _INDENT: &str = "    ";

/// Shorthand result used internally by the backend
type BackendResult = Result<(), Error>;

enum Decoration {
    VertexStage,
    FragmentStage,
    ComputeStage { workgroup_size: [u32; 3] },
    Block,
}

pub struct Writer<W> {
    out: W,
    names: FastHashMap<NameKey, String>,
    namer: Namer,
}

impl<W: Write> Writer<W> {
    pub fn new(out: W) -> Self {
        Writer {
            out,
            names: FastHashMap::default(),
            namer: Namer::default(),
        }
    }

    pub fn write(&mut self, module: &Module) -> BackendResult {
        self.names.clear();
        self.namer.reset(module, RESERVED, &mut self.names);

        // Write all structs
        for (handle, ty) in module.types.iter() {
            if let TypeInner::Struct {
                level, ref members, ..
            } = ty.inner
            {
                let name = &self.names[&NameKey::Type(handle)].clone();
                let block = level == StructLevel::Root;
                self.write_struct(name, block, members)?;
                writeln!(self.out)?;
            }
        }

        for (_, ep) in module.entry_points.iter().enumerate() {
            let decoration = match ep.stage {
                ShaderStage::Vertex => Decoration::VertexStage,
                ShaderStage::Fragment => Decoration::FragmentStage,
                ShaderStage::Compute => Decoration::ComputeStage {
                    workgroup_size: ep.workgroup_size,
                },
            };

            self.write_decoration(decoration)?;
            // Add a newline after decoration
            writeln!(self.out)?;
            self.write_function(&ep.function)?;
        }

        // Add a newline at the end of file
        writeln!(self.out)?;

        Ok(())
    }

    /// Helper method used to write structs
    /// https://gpuweb.github.io/gpuweb/wgsl/#functions
    ///
    /// # Notes
    /// Ends in a newline
    fn write_function(&mut self, func: &Function) -> BackendResult {
        write!(self.out, "fn {}(", func.name.as_ref().unwrap())?; // TODO: unnamed function?
        write!(self.out, ")")?;

        write!(self.out, "{{")?;
        write!(self.out, "}}")?;
        Ok(())
    }

    /// Helper method to write a decoration
    ///
    /// # Notes
    /// Adds no leading or trailing whitespace
    fn write_decoration(&mut self, decoration: Decoration) -> BackendResult {
        write!(self.out, "[[")?;
        match decoration {
            Decoration::VertexStage => write!(self.out, "stage(vertex)")?,
            Decoration::FragmentStage => write!(self.out, "stage(fragment)")?,
            Decoration::ComputeStage { workgroup_size } => {
                write!(
                    self.out,
                    "{}",
                    format!(
                        "stage(compute), workgroup_size({}, {}, {})",
                        workgroup_size[0], workgroup_size[1], workgroup_size[2]
                    )
                )?;
            }
            Decoration::Block => {
                write!(self.out, "block")?;
            }
        };
        write!(self.out, "]]")?;

        Ok(())
    }

    /// Helper method used to write structs
    ///
    /// # Notes
    /// Ends in a newline
    fn write_struct(
        &mut self,
        name: &str,
        block: bool,
        _members: &[StructMember],
    ) -> BackendResult {
        if block {
            self.write_decoration(Decoration::Block)?;
            writeln!(self.out)?;
        }
        write!(self.out, "struct {} {{", name)?;
        write!(self.out, "}}")?;

        writeln!(self.out)?;

        Ok(())
    }

    pub fn finish(self) -> W {
        self.out
    }
}
