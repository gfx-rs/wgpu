/*!
Lowering of [`Immediate`] globals.

WGSL lays out immediates like `storage` buffers (GLSL's `std430`), but HLSL can only read
root constants through a `cbuffer`, which packs values like `std140`: every nested struct
starts on a new 16-byte register. Padding cannot fix this, since it only moves members later.

Instead, we declare a flat array of `uint4` registers, and assemble the typed value in a
`static` variable at the start of each entry point that uses it. All byte offsets are
compile-time constants, much like [`storage`] loads from `ByteAddressBuffer`s:

```ignore
struct Inner { a: u32, b: u32 }
struct Data { x: u32, inner: Inner, y: u32 }
var<immediate> data: Data;
```

```ignore
cbuffer data_block : register(b0) { uint4 data_raw[1]; };
static Data data;

void main() {
    data = ConstructData(
        asuint(data_raw[0].x),
        ConstructInner(asuint(data_raw[0].y), asuint(data_raw[0].z)),
        asuint(data_raw[0].w)
    );
    // ...
}
```

[`Immediate`]: crate::AddressSpace::Immediate
[`storage`]: super::storage
*/

use alloc::{format, string::String};
use core::fmt::Write;

use super::{help::WrappedConstructor, BackendResult, Error};
use crate::{
    back,
    proc::{Alignment, NameKey},
    Handle,
};

/// The number of 4-byte words in one register of the raw immediates `cbuffer`.
const WORDS_PER_REGISTER: u32 = 4;

/// Component names of a register, indexed by the word's position within it.
const COMPONENTS: [char; WORDS_PER_REGISTER as usize] = ['x', 'y', 'z', 'w'];

/// Returns an expression for the 4-byte word with index `word` in the raw immediates `raw`.
fn immediate_word(raw: &str, word: u32) -> String {
    let component = COMPONENTS[(word % WORDS_PER_REGISTER) as usize];
    format!("{raw}[{}].{component}", word / WORDS_PER_REGISTER)
}

impl<W: Write> super::Writer<'_, W> {
    /// Writes the declarations for an [`Immediate`] global.
    ///
    /// This declares the raw `cbuffer` that the root constants are bound to,
    /// and the `static` variable that holds the typed value.
    /// See the [module documentation](self) for background.
    ///
    /// [`Immediate`]: crate::AddressSpace::Immediate
    pub(super) fn write_global_immediate(
        &mut self,
        module: &crate::Module,
        handle: Handle<crate::GlobalVariable>,
        global: &crate::GlobalVariable,
    ) -> BackendResult {
        let target = self
            .options
            .immediates_target
            .as_ref()
            .expect("No bind target was defined for the immediates block");
        let (register, space) = (target.register, target.space);

        let word_count = module.types[global.ty]
            .inner
            .size(module.to_ctx())
            .div_ceil(4);
        let register_count = word_count.div_ceil(WORDS_PER_REGISTER).max(1);

        let name = self.names[&NameKey::GlobalVariable(handle)].clone();
        let block_name = self.namer.call(&format!("{name}_block"));
        let raw_name = self.namer.call(&format!("{name}_raw"));

        write!(self.out, "cbuffer {block_name} : register(b{register}")?;
        if space != 0 {
            write!(self.out, ", space{space}")?;
        }
        writeln!(self.out, ") {{ uint4 {raw_name}[{register_count}]; }};")?;

        write!(self.out, "static ")?;
        self.write_type(module, global.ty)?;
        writeln!(self.out, " {name};")?;

        self.immediate_raw_names.insert(handle, raw_name);

        Ok(())
    }

    /// Writes the wrapped constructor functions needed to assemble the
    /// [`Immediate`] globals' values out of their raw words.
    ///
    /// [`Immediate`]: crate::AddressSpace::Immediate
    pub(super) fn write_immediate_constructors(&mut self, module: &crate::Module) -> BackendResult {
        for (_, global) in module.global_variables.iter() {
            if global.space == crate::AddressSpace::Immediate {
                self.write_wrapped_constructors_for_type(module, global.ty)?;
            }
        }
        Ok(())
    }

    /// Writes the statements at the start of an entry point that assemble
    /// the value of each [`Immediate`] global that the entry point uses.
    ///
    /// [`Immediate`]: crate::AddressSpace::Immediate
    pub(super) fn write_immediates_initialization(
        &mut self,
        module: &crate::Module,
        info: &crate::valid::FunctionInfo,
    ) -> BackendResult {
        for (handle, global) in module.global_variables.iter() {
            if global.space != crate::AddressSpace::Immediate || info[handle].is_empty() {
                continue;
            }

            let name = self.names[&NameKey::GlobalVariable(handle)].clone();
            let raw = self.immediate_raw_names[&handle].clone();

            write!(self.out, "{}{name} = ", back::INDENT)?;
            self.write_immediate_load(module, &raw, global.ty, 0)?;
            writeln!(self.out, ";")?;
        }
        Ok(())
    }

    /// Writes an expression that loads a value of type `ty` from the raw
    /// immediate words in `raw`, starting `offset` bytes into the immediate data.
    fn write_immediate_load(
        &mut self,
        module: &crate::Module,
        raw: &str,
        ty: Handle<crate::Type>,
        offset: u32,
    ) -> BackendResult {
        match module.types[ty].inner {
            crate::TypeInner::Scalar(scalar) => self.write_immediate_scalar(raw, scalar, offset),
            crate::TypeInner::Vector { size, scalar } => {
                self.write_immediate_vector(raw, size, scalar, offset)
            }
            crate::TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => {
                write!(
                    self.out,
                    "{}{}x{}(",
                    scalar.to_hlsl_str()?,
                    columns as u8,
                    rows as u8,
                )?;

                // Note: Matrices containing vec3s, due to padding, act like they contain vec4s.
                let column_stride = Alignment::from(rows) * scalar.width as u32;
                for column in 0..columns as u32 {
                    if column != 0 {
                        write!(self.out, ", ")?;
                    }
                    self.write_immediate_vector(
                        raw,
                        rows,
                        scalar,
                        offset + column * column_stride,
                    )?;
                }
                write!(self.out, ")")?;
                Ok(())
            }
            crate::TypeInner::Struct { ref members, .. } => {
                let constructor = WrappedConstructor { ty };
                self.write_wrapped_constructor_function_name(module, constructor)?;
                write!(self.out, "(")?;
                for (i, member) in members.iter().enumerate() {
                    if i != 0 {
                        write!(self.out, ", ")?;
                    }
                    self.write_immediate_load(module, raw, member.ty, offset + member.offset)?;
                }
                write!(self.out, ")")?;
                Ok(())
            }
            ref other => Err(Error::Unimplemented(format!(
                "immediate data of type {other:?}"
            ))),
        }
    }

    /// Writes an expression that loads a scalar from the raw immediate words in `raw`.
    fn write_immediate_scalar(
        &mut self,
        raw: &str,
        scalar: crate::Scalar,
        offset: u32,
    ) -> BackendResult {
        use crate::ScalarKind as Kind;

        let word = offset / 4;
        match (scalar.kind, scalar.width) {
            (Kind::Float | Kind::Sint | Kind::Uint, 4) => {
                let cast = scalar.kind.to_hlsl_cast();
                write!(self.out, "{cast}({})", immediate_word(raw, word))?;
            }
            // 16-bit values share a word with their neighbor.
            (Kind::Float | Kind::Sint | Kind::Uint, 2) => {
                let bits = format!("{} >> {}u", immediate_word(raw, word), (offset % 4) * 8);
                if scalar.kind == Kind::Float {
                    write!(self.out, "asfloat16(uint16_t({bits}))")?;
                } else {
                    write!(self.out, "{}({bits})", scalar.to_hlsl_str()?)?;
                }
            }
            // 64-bit values occupy two words, with the low word first.
            (Kind::Float | Kind::Sint | Kind::Uint, 8) => {
                let low = immediate_word(raw, word);
                let high = immediate_word(raw, word + 1);
                let bits = format!("uint64_t({low}) | (uint64_t({high}) << 32u)");
                match scalar.kind {
                    Kind::Float => write!(self.out, "asdouble({low}, {high})")?,
                    Kind::Sint => write!(self.out, "int64_t({bits})")?,
                    _ => write!(self.out, "({bits})")?,
                }
            }
            _ => return Err(Error::UnsupportedScalar(scalar)),
        }
        Ok(())
    }

    /// Writes an expression that loads a vector from the raw immediate words in `raw`.
    ///
    /// The components are loaded one at a time, which works wherever they happen to be
    /// relative to register boundaries. The shader compiler merges them back into a
    /// single constant buffer load when they share a register.
    fn write_immediate_vector(
        &mut self,
        raw: &str,
        size: crate::VectorSize,
        scalar: crate::Scalar,
        offset: u32,
    ) -> BackendResult {
        let count = size as u32;

        write!(self.out, "{}{}(", scalar.to_hlsl_str()?, count)?;
        for component in 0..count {
            if component != 0 {
                write!(self.out, ", ")?;
            }
            self.write_immediate_scalar(raw, scalar, offset + component * scalar.width as u32)?;
        }
        write!(self.out, ")")?;
        Ok(())
    }
}
