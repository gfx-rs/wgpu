use crate::{Arena, Handle};
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

pub(super) fn map_storage_class(class: crate::StorageClass) -> spirv::StorageClass {
    match class {
        crate::StorageClass::Handle => spirv::StorageClass::UniformConstant,
        crate::StorageClass::Function => spirv::StorageClass::Function,
        crate::StorageClass::Private => spirv::StorageClass::Private,
        crate::StorageClass::Storage => spirv::StorageClass::StorageBuffer,
        crate::StorageClass::Uniform => spirv::StorageClass::Uniform,
        crate::StorageClass::WorkGroup => spirv::StorageClass::Workgroup,
        crate::StorageClass::PushConstant => spirv::StorageClass::PushConstant,
    }
}

pub(super) fn contains_builtin(
    binding: Option<&crate::Binding>,
    ty: Handle<crate::Type>,
    arena: &Arena<crate::Type>,
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
