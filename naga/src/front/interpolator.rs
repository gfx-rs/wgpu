/*!
Interpolation defaults.
*/

impl crate::Binding {
    /// Apply default interpolation (if applicable) for `ty` to the binding.
    ///
    /// This function is a utility front ends may use to satisfy the Naga IR's
    /// requirement, meant to ensure that input languages' policies have been
    /// applied appropriately, that all I/O `Binding`s from the vertex shader to the
    /// fragment shader must have non-`None` `interpolation` values.
    ///
    /// All the shader languages Naga supports have similar rules:
    /// perspective-correct, center-sampled interpolation is the default for any
    /// binding that can vary, and an explicit flat qualifier/attribute/what-have-you is
    /// required for bindings that cannot.
    ///
    /// - If `binding` is not a [`Location`] binding, or if its [`interpolation`] is
    ///   already set, then this function makes no changes.
    ///
    /// - If `ty` is a floating-point scalar, vector, or matrix type, then
    ///   apply the default [`Perspective`] interpolation and [`Center`] sampling.
    ///
    /// - If `ty` is an integral scalar or vector, make no changes; if the interpolation was
    ///   `None`, it will remain so, and be rejected by the validator.
    ///
    /// For struct types, the bindings are defined on the members, so there is nothing to
    /// adjust on the struct itself.
    ///
    /// Other non-struct types are not permitted as user-defined IO values, and will be
    /// rejected by the validator.
    ///
    /// [`Binding`]: crate::Binding
    /// [`Location`]: crate::Binding::Location
    /// [`interpolation`]: crate::Binding::Location::interpolation
    /// [`Perspective`]: crate::Interpolation::Perspective
    /// [`Flat`]: crate::Interpolation::Flat
    /// [`Center`]: crate::Sampling::Center
    pub(crate) fn apply_default_interpolation(&mut self, ty: &crate::TypeInner) {
        let crate::Binding::Location {
            interpolation: ref mut interpolation @ None,
            ref mut sampling,
            ..
        } = *self
        else {
            return;
        };

        if let Some(crate::ScalarKind::Float) = ty.scalar_kind() {
            *interpolation = Some(crate::Interpolation::Perspective);
            *sampling = Some(crate::Sampling::Center);
        }
    }
}
