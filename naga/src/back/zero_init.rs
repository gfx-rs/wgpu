use std::num::NonZeroU32;

use crate::{GlobalVariable, Handle, Module, Type};

#[derive(Debug)]
pub(crate) enum ZeroInitKind {
    LocalPlusIndex {
        // The amount local_invocation_index should be multiplied by
        // We could use this to implement a more cache-efficient zeroing for big arrays,
        // i.e. multiply each by 2, then add 0, 1, etc
        // so each thread handles items which are next to each other
        // multiple: Option<NonZeroU32>,
        /// The amount to be added to local_invocation_index
        index: Option<NonZeroU32>,
        /// The amount to
        if_less_than: Option<u32>,
    },
    NotArray,
}

/// A helper driver for implementing zero initialisation
///
/// This is needed because of https://github.com/gfx-rs/wgpu/issues/4592.
/// That is, the previously used behaviour had significant compilation time costs
pub(crate) fn zero_init<'a>(
    module: &'a Module,
    variables: impl Iterator<Item = (Handle<GlobalVariable>, &'a GlobalVariable)>,
    workgroup_length: u32,
) -> Vec<(Handle<GlobalVariable>, ZeroInitKind)> {
    if workgroup_length == 0 {
        // No need to zero initialise, as we won't get any values anyway
        return vec![];
    }
    let mut total_len = 0;
    let mut workgroup_variables = variables
        .map(|(handle, var)| {
            debug_assert_eq!(var.space, crate::AddressSpace::WorkGroup);
            let len = if workgroup_length == 1 {
                // Treat workgroups of size one as not an array, because otherwise we'd output loads of statements for them
                // TODO: Heuristics here?
                None
            } else {
                array_len(module, var.ty)
            };
            let item = len.map(|len| {
                let multiples = len / workgroup_length;
                let remainder = len % workgroup_length;
                (multiples, remainder)
            });

            total_len += item
                .map(|(multiples, remainder)| multiples + (remainder != 0) as u32)
                .unwrap_or(1);
            (handle, var, item)
        })
        .collect::<Vec<_>>();

    // Sort the biggest indices to the front, with the non-array items at the end
    workgroup_variables.sort_by_key(|(_, _, len)| std::cmp::Reverse(*len));
    let mut results = Vec::with_capacity(total_len as usize);

    for &(handle, _, len) in &workgroup_variables {
        if let Some((multiples, _)) = len {
            // Consider 6 items, with workgroup length of 2
            // multiples is 3
            // We output: +0, +2, +4, which give correct maximum index of five for thread index 1
            for i in 0..multiples {
                results.push((
                    handle,
                    ZeroInitKind::LocalPlusIndex {
                        index: NonZeroU32::new(i * workgroup_length),
                        if_less_than: None,
                    },
                ))
            }
        } else {
            break;
        }
    }

    for &(handle, _, len) in &workgroup_variables {
        if let Some((multiples, remainder)) = len {
            if remainder == 0 {
                continue;
            }

            // Consider 3 items, with workgroup length of 2
            // multiples is 1, remainder is 1
            // We output: +2, only if index is less than remainder (which is 1)
            // i.e. that gets max index 2 in thread 0
            results.push((
                handle,
                ZeroInitKind::LocalPlusIndex {
                    index: NonZeroU32::new(multiples * workgroup_length),
                    if_less_than: Some(remainder),
                },
            ))
        } else {
            results.push((handle, ZeroInitKind::NotArray));
            break;
        }
    }
    results
}

fn array_len(module: &Module, ty: Handle<Type>) -> Option<u32> {
    match &module.types[ty].inner {
        crate::TypeInner::Array {
            base: _base, size, ..
        } => match size {
            crate::ArraySize::Constant(e) => {
                // If e is small, and `base` is big, then we *could*
                // split up parts of the base
                // We choose not to do this, as it gets very complicated
                return Some(e.get());
            }
            crate::ArraySize::Dynamic => {
                log::error!("Arrays in the workgroup address space can't be dynamically sized");
            }
        },
        _ => (),
    }
    None
}
