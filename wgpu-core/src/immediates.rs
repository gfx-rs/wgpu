/// Returns the bitmask of slots covered by a `set_immediates(offset, size_bytes)` call.
pub(crate) fn slots_for_range(offset: u32, size_bytes: u32) -> u64 {
    // u128 upcast to avoid overflow panic on n = 64
    let bits_below = |n: u32| ((1u128 << n.min(64)) - 1) as u64;
    let lo = offset / 4;
    let hi = (offset + size_bytes).div_ceil(4);
    bits_below(hi) - bits_below(lo)
}

/// Computes a bitmask of which u32 immediate slots must be set before draw/dispatch.
/// Bit N is set if the u32 at byte N*4 must be written by `set_immediates`.
///
/// For structs, gaps between members are padding and those slots need not be set.
/// For scalars, vectors, and matrices, all slots in the span are required
/// (the spec only defines padding exemptions at the struct-member level).
pub(crate) fn slots_for_type(ty: &naga::TypeInner, gctx: naga::proc::GlobalCtx) -> u64 {
    match *ty {
        naga::TypeInner::Struct { ref members, .. } => {
            let mut mask: u64 = 0;
            for member in members {
                let member_size = gctx.types[member.ty].inner.size(gctx);
                mask |= slots_for_range(member.offset, member_size);
            }
            mask
        }
        _ => {
            let size = ty.size(gctx);
            slots_for_range(0, size)
        }
    }
}

/// Returns the `var<immediate>` type from a naga module, if any.
fn immediate_type(module: &naga::Module) -> Option<&naga::TypeInner> {
    module
        .global_variables
        .iter()
        .find(|(_, var)| var.space == naga::AddressSpace::Immediate)
        .map(|(_, var)| &module.types[var.ty].inner)
}

/// Returns the required immediate slot bitmask for a naga module.
/// Zero if the module has no `var<immediate>`.
pub(crate) fn slots_for_module(module: &naga::Module) -> u64 {
    immediate_type(module).map_or(0, |ty| slots_for_type(ty, module.to_ctx()))
}

/// Returns the byte size of the `var<immediate>` type in a naga module.
/// Zero if the module has no `var<immediate>`.
pub(crate) fn size_for_module(module: &naga::Module) -> u32 {
    immediate_type(module).map_or(0, |ty| ty.size(module.to_ctx()))
}

#[cfg(test)]
#[cfg(feature = "wgsl")]
mod tests {
    use super::slots_for_module;

    fn immediate_slots(wgsl: &str) -> u64 {
        slots_for_module(&naga::front::wgsl::parse_str(wgsl).unwrap())
    }

    #[test]
    fn non_struct() {
        assert_eq!(immediate_slots("var<immediate> im: vec4<f32>;"), 0b1111);
        assert_eq!(immediate_slots("var<immediate> im: mat4x4<f32>;"), 0xFFFF);
    }

    #[test]
    fn struct_with_padding() {
        assert_eq!(
            immediate_slots(
                "struct S { a: f32, b: vec4<f32> }
                 var<immediate> im: S;"
            ),
            0b1111_0001,
        );
    }

    #[test]
    fn struct_no_padding() {
        assert_eq!(
            immediate_slots(
                "struct S { a: f32, b: f32 }
                 var<immediate> im: S;"
            ),
            0b11,
        );
    }
}
