use crate::Handle;
use bit_set::BitSet;

/// Remove all unused expressions from `module.const_expressions`.
///
/// Abstract values make extensive use of 64-bit literals, which the
/// validator should never see.
pub fn compact(module: &mut crate::Module) {
    // Trace which expressions in `module.const_expressions` are actually used.
    let used = trace(module);

    // Assuming unused expressions are squeezed out of the arena,
    // compute a map from pre-squeeze indices to post-squeeze
    // expression handles.
    let mut next = 0;
    let compacted: Vec<Option<Handle<crate::Expression>>> = module
        .const_expressions
        .iter()
        .map(|(handle, _)| {
            if used.contains(handle.index()) {
                let index = next;
                next += 1;
                std::num::NonZeroU32::new(index + 1).map(Handle::new)
            } else {
                None
            }
        })
        .collect();

    adjust(module, &compacted);
}

fn trace(module: &crate::Module) -> BitSet {
    let mut used = BitSet::new();

    // Note uses by global constants.
    for (_, constant) in module.constants.iter() {
        used.insert(constant.init.index());
    }

    // Note uses by global variable initializers.
    for (_, global) in module.global_variables.iter() {
        if let Some(init) = global.init {
            used.insert(init.index());
        }
    }

    // Note uses by functions' expressions.
    for (_, fun) in module.functions.iter() {
        trace_function(fun, &mut used);
    }

    // Note uses by entry points' expressions.
    for entry_point in &module.entry_points {
        trace_function(&entry_point.function, &mut used);
    }

    // Note transitive uses by other used constant expressions.
    //
    // Since we know that expressions only refer to other expressions
    // appearing before them in the arena, we can do this without a
    // stack by making a single pass from back to front.
    for (handle, expr) in module.const_expressions.iter().rev() {
        if used.contains(handle.index()) {
            match *expr {
                crate::Expression::Compose { ref components, .. } => {
                    for component in components {
                        used.insert(component.index());
                    }
                }
                crate::Expression::Splat { value, .. } => {
                    used.insert(value.index());
                }
                _ => {}
            }
        }
    }

    used
}

fn trace_function(function: &crate::Function, used: &mut BitSet) {
    for (_, expr) in function.expressions.iter() {
        match *expr {
            crate::Expression::ImageSample {
                offset: Some(offset),
                ..
            } => {
                used.insert(offset.index());
            }
            _ => {}
        }
    }
}

fn adjust(module: &mut crate::Module, compacted: &[Option<Handle<crate::Expression>>]) {
    // Remove unused expressions from the constant arena,
    // and adjust the handles in retained expressions.
    module.const_expressions.retain_mut(|handle, expr| {
        match compacted[handle.index()] {
            Some(_) => {
                // This expression is used, and thus its handles are worth adjusting.
                match *expr {
                    crate::Expression::Compose {
                        ref mut components, ..
                    } => {
                        for component in components {
                            *component = compacted[component.index()].unwrap();
                        }
                    }
                    crate::Expression::Splat { ref mut value, .. } => {
                        *value = compacted[value.index()].unwrap();
                    }
                    _ => {}
                }
                true
            }
            None => false,
        }
    });

    // Adjust uses by global constants.
    for (_, constant) in module.constants.iter_mut() {
        constant.init = compacted[constant.init.index()].unwrap();
    }

    // Adjust uses by global variable initializers.
    for (_, global) in module.global_variables.iter_mut() {
        if let Some(ref mut init) = global.init {
            *init = compacted[init.index()].unwrap();
        }
    }

    // Adjust uses by functions' expressions.
    for (_, fun) in module.functions.iter_mut() {
        adjust_function(fun, compacted);
    }

    // Adjust uses by entry points' expressions.
    for entry_point in &mut module.entry_points {
        adjust_function(&mut entry_point.function, compacted);
    }
}

fn adjust_function(
    function: &mut crate::Function,
    compacted: &[Option<Handle<crate::Expression>>],
) {
    for (_, expr) in function.expressions.iter_mut() {
        match *expr {
            crate::Expression::ImageSample {
                offset: Some(ref mut offset),
                ..
            } => {
                *offset = compacted[offset.index()].unwrap();
            }
            _ => {}
        }
    }
}
