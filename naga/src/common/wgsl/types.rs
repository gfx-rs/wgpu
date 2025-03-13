//! Code for formatting Naga IR types as WGSL source code.

use super::{address_space_str, ToWgsl, TryToWgsl};
use crate::common;
use crate::{Handle, TypeInner};

use core::fmt::Write;

/// A context for printing Naga IR types as WGSL.
///
/// This trait's default methods [`write_type`] and
/// [`write_type_inner`] do the work of formatting types as WGSL.
/// Implementors must provide the remaining methods, to customize
/// behavior for the context at hand.
///
/// For example, the WGSL backend would provide an implementation of
/// [`type_name`] that handles hygienic renaming, whereas the WGSL
/// front end would simply show the name that was given in the source.
///
/// [`write_type`]: TypeContext::write_type
/// [`write_type_inner`]: TypeContext::write_type_inner
/// [`type_name`]: TypeContext::type_name
pub trait TypeContext<W: Write> {
    /// Return the [`Type`] referred to by `handle`.
    ///
    /// [`Type`]: crate::Type
    fn lookup_type(&self, handle: Handle<crate::Type>) -> &crate::Type;

    /// Return the name to be used for the type referred to by
    /// `handle`.
    fn type_name(&self, handle: Handle<crate::Type>) -> &str;

    /// Write the WGSL form of `override` to `out`.
    fn write_override(&self, r#override: Handle<crate::Override>, out: &mut W)
        -> core::fmt::Result;

    /// Write the type `ty` as it would appear in a value's declaration.
    ///
    /// Write the type referred to by `ty` in `module` as it would appear in
    /// a `var`, `let`, etc. declaration, or in a function's argument list.
    fn write_type(&self, handle: Handle<crate::Type>, out: &mut W) -> core::fmt::Result {
        let ty = self.lookup_type(handle);
        match ty.inner {
            TypeInner::Struct { .. } => out.write_str(self.type_name(handle))?,
            ref other => self.write_type_inner(other, out)?,
        }

        Ok(())
    }

    /// Write the [`TypeInner`] `inner` as it would appear in a value's declaration.
    ///
    /// Write `inner` as it would appear in a `var`, `let`, etc.
    /// declaration, or in a function's argument list.
    ///
    /// Note that this cannot handle writing [`Struct`] types: those
    /// must be referred to by name, but the name isn't available in
    /// [`TypeInner`].
    ///
    /// [`Struct`]: TypeInner::Struct
    fn write_type_inner(&self, inner: &TypeInner, out: &mut W) -> core::fmt::Result {
        try_write_type_inner(self, inner, out)
    }
}

fn try_write_type_inner<C, W>(ctx: &C, inner: &TypeInner, out: &mut W) -> core::fmt::Result
where
    C: TypeContext<W> + ?Sized,
    W: Write,
{
    fn unwrap_to_wgsl<T: TryToWgsl + Copy + core::fmt::Debug>(value: T) -> &'static str {
        value.try_to_wgsl().unwrap_or_else(|| {
            unreachable!(
                "validation should have forbidden {}: {value:?}",
                T::DESCRIPTION
            );
        })
    }

    match *inner {
        TypeInner::Vector { size, scalar } => write!(
            out,
            "vec{}<{}>",
            common::vector_size_str(size),
            unwrap_to_wgsl(scalar),
        )?,
        TypeInner::Sampler { comparison: false } => {
            write!(out, "sampler")?;
        }
        TypeInner::Sampler { comparison: true } => {
            write!(out, "sampler_comparison")?;
        }
        TypeInner::Image {
            dim,
            arrayed,
            class,
        } => {
            // More about texture types: https://gpuweb.github.io/gpuweb/wgsl/#sampled-texture-type
            use crate::ImageClass as Ic;

            let dim_str = dim.to_wgsl();
            let arrayed_str = if arrayed { "_array" } else { "" };
            match class {
                Ic::Sampled { kind, multi } => {
                    let multisampled_str = if multi { "multisampled_" } else { "" };
                    let type_str = unwrap_to_wgsl(crate::Scalar { kind, width: 4 });
                    write!(
                        out,
                        "texture_{multisampled_str}{dim_str}{arrayed_str}<{type_str}>"
                    )?;
                }
                Ic::Depth { multi } => {
                    let multisampled_str = if multi { "multisampled_" } else { "" };
                    write!(
                        out,
                        "texture_depth_{multisampled_str}{dim_str}{arrayed_str}"
                    )?;
                }
                Ic::Storage { format, access } => {
                    let format_str = format.to_wgsl();
                    let access_str = if access.contains(crate::StorageAccess::ATOMIC) {
                        ",atomic"
                    } else if access
                        .contains(crate::StorageAccess::LOAD | crate::StorageAccess::STORE)
                    {
                        ",read_write"
                    } else if access.contains(crate::StorageAccess::LOAD) {
                        ",read"
                    } else {
                        ",write"
                    };
                    write!(
                        out,
                        "texture_storage_{dim_str}{arrayed_str}<{format_str}{access_str}>"
                    )?;
                }
            }
        }
        TypeInner::Scalar(scalar) => {
            write!(out, "{}", unwrap_to_wgsl(scalar))?;
        }
        TypeInner::Atomic(scalar) => {
            write!(out, "atomic<{}>", unwrap_to_wgsl(scalar))?;
        }
        TypeInner::Array {
            base,
            size,
            stride: _,
        } => {
            // More info https://gpuweb.github.io/gpuweb/wgsl/#array-types
            // array<A, 3> -- Constant array
            // array<A> -- Dynamic array
            write!(out, "array<")?;
            match size {
                crate::ArraySize::Constant(len) => {
                    ctx.write_type(base, out)?;
                    write!(out, ", {len}")?;
                }
                crate::ArraySize::Pending(r#override) => {
                    ctx.write_override(r#override, out)?;
                }
                crate::ArraySize::Dynamic => {
                    ctx.write_type(base, out)?;
                }
            }
            write!(out, ">")?;
        }
        TypeInner::BindingArray { base, size } => {
            // More info https://github.com/gpuweb/gpuweb/issues/2105
            write!(out, "binding_array<")?;
            match size {
                crate::ArraySize::Constant(len) => {
                    ctx.write_type(base, out)?;
                    write!(out, ", {len}")?;
                }
                crate::ArraySize::Pending(r#override) => {
                    ctx.write_override(r#override, out)?;
                }
                crate::ArraySize::Dynamic => {
                    ctx.write_type(base, out)?;
                }
            }
            write!(out, ">")?;
        }
        TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => {
            write!(
                out,
                "mat{}x{}<{}>",
                common::vector_size_str(columns),
                common::vector_size_str(rows),
                unwrap_to_wgsl(scalar)
            )?;
        }
        TypeInner::Pointer { base, space } => {
            let (address, maybe_access) = address_space_str(space);
            // Everything but `AddressSpace::Handle` gives us a `address` name, but
            // Naga IR never produces pointers to handles, so it doesn't matter much
            // how we write such a type. Just write it as the base type alone.
            if let Some(space) = address {
                write!(out, "ptr<{space}, ")?;
            }
            ctx.write_type(base, out)?;
            if address.is_some() {
                if let Some(access) = maybe_access {
                    write!(out, ", {access}")?;
                }
                write!(out, ">")?;
            }
        }
        TypeInner::ValuePointer {
            size: None,
            scalar,
            space,
        } => {
            let (address, maybe_access) = address_space_str(space);
            if let Some(space) = address {
                write!(out, "ptr<{}, {}", space, unwrap_to_wgsl(scalar))?;
                if let Some(access) = maybe_access {
                    write!(out, ", {access}")?;
                }
                write!(out, ">")?;
            } else {
                unreachable!("ValuePointer to AddressSpace::Handle {inner:?}");
            }
        }
        TypeInner::ValuePointer {
            size: Some(size),
            scalar,
            space,
        } => {
            let (address, maybe_access) = address_space_str(space);
            if let Some(space) = address {
                write!(
                    out,
                    "ptr<{}, vec{}<{}>",
                    space,
                    common::vector_size_str(size),
                    unwrap_to_wgsl(scalar)
                )?;
                if let Some(access) = maybe_access {
                    write!(out, ", {access}")?;
                }
                write!(out, ">")?;
            } else {
                unreachable!("ValuePointer to AddressSpace::Handle {inner:?}");
            }
            write!(out, ">")?;
        }
        TypeInner::AccelerationStructure { vertex_return } => {
            let caps = if vertex_return { "<vertex_return>" } else { "" };
            write!(out, "acceleration_structure{}", caps)?
        }
        TypeInner::Struct { .. } => {
            unreachable!("structs can only be referenced by name in WGSL");
        }
        TypeInner::RayQuery { vertex_return } => {
            let caps = if vertex_return { "<vertex_return>" } else { "" };
            write!(out, "ray_query{}", caps)?
        }
    }

    Ok(())
}
