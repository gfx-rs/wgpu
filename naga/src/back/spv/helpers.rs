use crate::{Handle, UniqueArena};
use spirv::Word;

pub(super) fn bytes_to_words(bytes: &[u8]) -> Vec<Word> {
    bytes
        .chunks(4)
        .map(|chars| chars.iter().rev().fold(0u32, |u, c| (u << 8) | *c as u32))
        .collect()
}

pub(super) fn string_to_words(input: &str) -> Vec<Word> {
    let bytes = input.as_bytes();
    let mut words = bytes_to_words(bytes);

    if bytes.len() % 4 == 0 {
        // nul-termination
        words.push(0x0u32);
    }

    words
}

pub(super) const fn map_storage_class(space: crate::AddressSpace) -> spirv::StorageClass {
    match space {
        crate::AddressSpace::Handle => spirv::StorageClass::UniformConstant,
        crate::AddressSpace::Function => spirv::StorageClass::Function,
        crate::AddressSpace::Private => spirv::StorageClass::Private,
        crate::AddressSpace::Storage { .. } => spirv::StorageClass::StorageBuffer,
        crate::AddressSpace::Uniform => spirv::StorageClass::Uniform,
        crate::AddressSpace::WorkGroup => spirv::StorageClass::Workgroup,
        crate::AddressSpace::PushConstant => spirv::StorageClass::PushConstant,
    }
}

pub(super) fn contains_builtin(
    binding: Option<&crate::Binding>,
    ty: Handle<crate::Type>,
    arena: &UniqueArena<crate::Type>,
    built_in: crate::BuiltIn,
) -> bool {
    if let Some(&crate::Binding::BuiltIn(bi)) = binding {
        bi == built_in
    } else if let crate::TypeInner::Struct { ref members, .. } = arena[ty].inner {
        members
            .iter()
            .any(|member| contains_builtin(member.binding.as_ref(), member.ty, arena, built_in))
    } else {
        false // unreachable
    }
}

impl crate::AddressSpace {
    pub(super) const fn to_spirv_semantics_and_scope(
        self,
    ) -> (spirv::MemorySemantics, spirv::Scope) {
        match self {
            Self::Storage { .. } => (spirv::MemorySemantics::UNIFORM_MEMORY, spirv::Scope::Device),
            Self::WorkGroup => (
                spirv::MemorySemantics::WORKGROUP_MEMORY,
                spirv::Scope::Workgroup,
            ),
            _ => (spirv::MemorySemantics::empty(), spirv::Scope::Invocation),
        }
    }
}

/// Return true if the global requires a type decorated with `Block`.
///
/// Vulkan spec v1.3 §15.6.2, "Descriptor Set Interface", says:
///
/// > Variables identified with the `Uniform` storage class are used to
/// > access transparent buffer backed resources. Such variables must
/// > be:
/// >
/// > -   typed as `OpTypeStruct`, or an array of this type,
/// >
/// > -   identified with a `Block` or `BufferBlock` decoration, and
/// >
/// > -   laid out explicitly using the `Offset`, `ArrayStride`, and
/// >     `MatrixStride` decorations as specified in §15.6.4, "Offset
/// >     and Stride Assignment."
// See `back::spv::GlobalVariable::access_id` for details.
pub fn global_needs_wrapper(ir_module: &crate::Module, var: &crate::GlobalVariable) -> bool {
    match var.space {
        crate::AddressSpace::Uniform
        | crate::AddressSpace::Storage { .. }
        | crate::AddressSpace::PushConstant => {}
        _ => return false,
    };
    match ir_module.types[var.ty].inner {
        crate::TypeInner::Struct {
            ref members,
            span: _,
        } => match members.last() {
            Some(member) => match ir_module.types[member.ty].inner {
                // Structs with dynamically sized arrays can't be copied and can't be wrapped.
                crate::TypeInner::Array {
                    size: crate::ArraySize::Dynamic,
                    ..
                } => false,
                _ => true,
            },
            None => false,
        },
        crate::TypeInner::BindingArray { .. } => false,
        // if it's not a structure or a binding array, let's wrap it to be able to put "Block"
        _ => true,
    }
}
