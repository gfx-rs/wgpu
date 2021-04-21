pub use crate::{Arena, Handle};

impl crate::Module {
    /// Apply the usual default interpolation for vertex shader outputs and fragment shader inputs.
    ///
    /// For every [`Binding`] that is a vertex shader output or a fragment shader
    /// input, and that has an `interpolation` or `sampling` of `None`, assign a
    /// default interpolation and sampling as follows:
    ///
    /// - If the `Binding`'s type contains only 32-bit floating-point values or
    ///   vectors of such, default its interpolation to `Perspective` and its
    ///   sampling to `Center`.
    ///
    /// - Otherwise, mark its interpolation as `Flat`.
    ///
    /// When struct appear in input or output types, apply these rules to their
    /// leaves, since those are the things that actually get assigned locations.
    ///
    /// This function is a utility front ends may use to satisfy the Naga IR's
    /// requirement that all I/O `Binding`s from the vertex shader to the
    /// fragment shader must have non-`None` `interpolation` values. This
    /// requirement is meant to ensure that input languages' policies have been
    /// applied appropriately.
    ///
    /// All the shader languages Naga supports have similar rules:
    /// perspective-correct, center-sampled interpolation is the default for any
    /// binding that can vary, and everything else either defaults to flat, or
    /// requires an explicit flat qualifier/attribute/what-have-you.
    ///
    /// [`Binding`]: super::Binding
    pub fn apply_common_default_interpolation(&mut self) {
        use crate::{Binding, ScalarKind, Type, TypeInner};

        /// Choose a default interpolation for a function argument or result.
        ///
        /// `binding` refers to the `Binding` whose type is `ty`. If `ty` is a struct, then it's the
        /// bindings of the struct's members that we care about, and the binding of the struct
        /// itself is meaningless, so `binding` should be `None`.
        fn default_binding_or_struct(
            binding: &mut Option<Binding>,
            ty: Handle<Type>,
            types: &mut Arena<Type>,
        ) {
            match types.get_mut(ty).inner {
                // A struct. It's the individual members we care about, so recurse.
                TypeInner::Struct {
                    members: ref mut m, ..
                } => {
                    // To choose the right interpolations for `members`, we must consult other
                    // elements of `types`. But both `members` and the types it refers to are stored
                    // in `types`, and Rust won't let us mutate one element of the `Arena`'s `Vec`
                    // while reading others.
                    //
                    // So, temporarily swap the member list out its type, assign appropriate
                    // interpolations to its members, and then swap the list back in.
                    use std::mem::replace;
                    let mut members = replace(m, vec![]);

                    for member in &mut members {
                        default_binding_or_struct(&mut member.binding, member.ty, types);
                    }

                    // Swap the member list back in. It's essential that we call `types.get_mut`
                    // afresh here, rather than just using `m`: it's only because `m` was dead that
                    // we were able to pass `types` to the recursive call.
                    match types.get_mut(ty).inner {
                        TypeInner::Struct {
                            members: ref mut m, ..
                        } => replace(m, members),
                        _ => unreachable!("ty must be a struct"),
                    };
                }

                // Some interpolatable type.
                //
                // GLSL has 64-bit floats, but it won't interpolate them. WGSL and MSL only have
                // 32-bit floats. SPIR-V has 16- and 64-bit float capabilities, but Vulkan is vague
                // about what can and cannot be interpolated.
                TypeInner::Scalar {
                    kind: ScalarKind::Float,
                    width: 4,
                }
                | TypeInner::Vector {
                    kind: ScalarKind::Float,
                    width: 4,
                    ..
                } => {
                    // unwrap: all `EntryPoint` arguments or return values must either be structures
                    // or have a `Binding`.
                    let binding = binding.as_mut().unwrap();
                    if let Binding::Location {
                        ref mut interpolation,
                        ref mut sampling,
                        ..
                    } = *binding
                    {
                        if interpolation.is_none() {
                            *interpolation = Some(crate::Interpolation::Perspective);
                        }
                        if sampling.is_none() && *interpolation != Some(crate::Interpolation::Flat)
                        {
                            *sampling = Some(crate::Sampling::Center);
                        }
                    }
                }

                // Some type that can't be interpolated.
                _ => {
                    // unwrap: all `EntryPoint` arguments or return values must either be structures
                    // or have a `Binding`.
                    let binding = binding.as_mut().unwrap();
                    if let Binding::Location {
                        ref mut interpolation,
                        ref mut sampling,
                        ..
                    } = *binding
                    {
                        *interpolation = Some(crate::Interpolation::Flat);
                        *sampling = None;
                    }
                }
            }
        }

        for ep in &mut self.entry_points {
            let function = &mut ep.function;
            match ep.stage {
                crate::ShaderStage::Fragment => {
                    for arg in &mut function.arguments {
                        default_binding_or_struct(&mut arg.binding, arg.ty, &mut self.types);
                    }
                }
                crate::ShaderStage::Vertex => {
                    if let Some(result) = function.result.as_mut() {
                        default_binding_or_struct(&mut result.binding, result.ty, &mut self.types);
                    }
                }
                _ => (),
            }
        }
    }
}
