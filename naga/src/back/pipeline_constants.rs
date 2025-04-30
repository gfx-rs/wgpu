use alloc::{
    borrow::Cow,
    string::{String, ToString},
};
use core::mem;

use hashbrown::HashSet;
use thiserror::Error;

use super::PipelineConstants;
use crate::{
    arena::HandleVec,
    ir,
    proc::{ConstantEvaluator, ConstantEvaluatorError, Emitter, U32EvalError},
    valid::{
        Capabilities, FunctionInfo, ModuleInfo, UnresolvedOverrides, ValidationError,
        ValidationFlags, Validator,
    },
    Arena, Block, Constant, Expression, FastHashMap, Function, Handle, Literal, Module, Override,
    Range, Scalar, Span, Statement, TypeInner, WithSpan,
};

#[cfg(no_std)]
use num_traits::float::FloatCore as _;

#[derive(Error, Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub enum PipelineConstantError {
    #[error("Missing value for pipeline-overridable constant with identifier string: '{0}'")]
    MissingValue(String),
    #[error(
        "Source f64 value needs to be finite ({}) for number destinations",
        "NaNs and Inifinites are not allowed"
    )]
    SrcNeedsToBeFinite,
    #[error("Source f64 value doesn't fit in destination")]
    DstRangeTooSmall,
    #[error(transparent)]
    ConstantEvaluatorError(#[from] ConstantEvaluatorError),
    #[error(transparent)]
    ValidationError(#[from] WithSpan<ValidationError>),
    #[error("workgroup_size override isn't strictly positive")]
    NegativeWorkgroupSize,
    #[error("unable to evaluate workgroup_size override")]
    WorkgroupSizeOverrideEvaluationError,
}

// Returns the key to use for an override in `pipeline_constants`.
fn override_key(ov: &Override) -> Cow<'_, str> {
    if let Some(id) = ov.id {
        Cow::Owned(id.to_string())
    } else if let Some(ref name) = ov.name {
        Cow::Borrowed(name)
    } else {
        unreachable!()
    }
}

#[derive(Debug)]
pub struct ProcessOverridesOutput<'a> {
    pub module: Cow<'a, Module>,
    pub info: Cow<'a, ModuleInfo>,
    pub unresolved: UnresolvedOverrides,
}

/// Check the global usage in `fun_info` for any globals affected by unresolved
/// overrides.
///
/// If any is found, returns `Some`, otherwise returns `None`.
fn check_for_unresolved_global_use<'a>(
    globals: impl Iterator<Item = (Handle<ir::GlobalVariable>, &'a ir::GlobalVariable)>,
    unresolved: &UnresolvedOverrides,
    fun_info: &FunctionInfo,
) -> Option<Handle<Override>> {
    for (var_handle, _) in globals {
        match unresolved.global_variables.get(&var_handle) {
            Some(&o_handle) if !fun_info[var_handle].is_empty() => {
                return Some(o_handle);
            }
            _ => {}
        }
    }
    None
}

/// Replace overrides in `module` with constants.
///
/// If no changes are needed, this just returns `Cow::Borrowed`
/// references to `module` and `module_info`. Otherwise, it clones
/// `module`, updates it with evaluated override expressions, and returns
/// `Cow::Owned` values holding the simplified module and its validation
/// results.
///
/// If `entry_point` is specified, then any override referenced by
/// that entry point must be supplied, and other overrides are
/// optional. The returned module may still have override expressions,
/// but they should not be reachable from the specified entry point.
///
/// If `entry_point` is not specified, then all overrides must be specified.
///
/// This function completely rewrites both the [`global`] and function-local
/// arenas, replacing [`Expression::Override`] with [`Expression::Constant`].
/// It then updates expressions, statements, and initializers that refer to a
/// an updated expression.
///
/// The types arena is not updated. This means that the size of an array (in the
/// workgroup space, because this is the only place override-sized arrays are
/// permitted) may still require indirection through an override handle to the
/// initializer expression, which will be an evaluated constant. See
/// [#6787](https://github.com/gfx-rs/wgpu/pull/6787).
///
/// [`global`]: Module::global_expressions
pub fn process_overrides<'a>(
    module: &'a Module,
    module_info: &'a ModuleInfo,
    entry_point: Option<(ir::ShaderStage, &str)>,
    pipeline_constants: &PipelineConstants,
) -> Result<ProcessOverridesOutput<'a>, PipelineConstantError> {
    let mut unresolved = UnresolvedOverrides::default();

    if module.overrides.is_empty() {
        return Ok(ProcessOverridesOutput {
            module: Cow::Borrowed(module),
            info: Cow::Borrowed(module_info),
            unresolved,
        });
    }

    let mut module = module.clone();

    // A map from override handles to the handles of the constants
    // we've replaced them with.
    let mut override_map = HandleVec::with_capacity(module.overrides.len());

    // A map from `module`'s original global expression handles to
    // handles in the new, simplified global expression arena.
    let mut adjusted_global_expressions = HandleVec::with_capacity(module.global_expressions.len());

    // The set of constants whose initializer handles we've already
    // updated to refer to the newly built global expression arena.
    //
    // All constants in `module` must have their `init` handles
    // updated to point into the new, simplified global expression
    // arena. Some of these we can most easily handle as a side effect
    // during the simplification process, but we must handle the rest
    // in a final fixup pass, guided by `adjusted_global_expressions`. We
    // add their handles to this set, so that the final fixup step can
    // leave them alone.
    let mut adjusted_constant_initializers = HashSet::with_capacity(module.constants.len());

    let mut global_expression_kind_tracker = crate::proc::ExpressionKindTracker::new();
    let mut global_expressions_missing_overrides = FastHashMap::default();
    let mut layouter = crate::proc::Layouter::default();

    // An iterator through the original overrides table, consumed in
    // approximate tandem with the global expressions.
    let mut overrides = mem::take(&mut module.overrides);
    let mut override_iter = overrides.iter_mut_span();

    // Do two things in tandem:
    //
    // - Rebuild the global expression arena from scratch, replacing
    //   `Override` expressions in `module.global_expressions` that can
    //   now be evaluated with `Constant` expressions.
    //
    // - Build a new `Constant` in `module.constants` to take the
    //   place of each `Override`.
    //
    // Build a map from old global expression handles to their
    // fully-evaluated counterparts in `adjusted_global_expressions` as we
    // go.
    //
    // Why in tandem? Overrides refer to expressions, and expressions
    // refer to overrides, so we can't disentangle the two into
    // separate phases. However, we can take advantage of the fact
    // that the overrides and expressions must form a DAG, and work
    // our way from the leaves to the roots, replacing and evaluating
    // as we go.
    //
    // Although the two loops are nested, this is really two
    // alternating phases: we adjust and evaluate constant expressions
    // until we hit an `Override` expression, at which point we switch
    // to building `Constant`s for `Overrides` until we've handled the
    // one used by the expression. Then we switch back to processing
    // expressions. Because we know they form a DAG, we know the
    // `Override` expressions we encounter can only have initializers
    // referring to global expressions we've already simplified.
    for (old_h, expr, span) in module.global_expressions.drain() {
        let mut expr = match expr {
            Expression::Override(h) => {
                match override_map.get(h) {
                    Some(&Some(new_h)) => {
                        // Already evaluated.
                        Expression::Constant(new_h)
                    }
                    Some(&None) => {
                        // Already processed and could not evaluate. Leave
                        // expression unchanged.
                        expr
                    }
                    None => {
                        let mut result = None;
                        for entry in override_iter.by_ref() {
                            let stop = entry.0 == h;
                            result = process_override(
                                entry,
                                pipeline_constants,
                                &mut module,
                                &mut override_map,
                                &adjusted_global_expressions,
                                &mut adjusted_constant_initializers,
                                &mut global_expression_kind_tracker,
                            )?;
                            if stop {
                                break;
                            }
                        }
                        match result {
                            None => {
                                // Could not evaluate. Leave expression
                                // unchanged.
                                expr
                            }
                            Some(new_h) => Expression::Constant(new_h),
                        }
                    }
                }
            }
            Expression::Constant(c_h) => {
                if adjusted_constant_initializers.insert(c_h) {
                    let init = &mut module.constants[c_h].init;
                    *init = adjusted_global_expressions[*init];
                }
                expr
            }
            expr => expr,
        };
        // Attempt constant evaluation now that overrides referenced by this
        // expression may have been resolved. If they have not been resolved,
        // the expression will remain with `ExpressionKind::Override`.
        let mut evaluator = ConstantEvaluator::for_wgsl_module(
            &mut module,
            &mut global_expression_kind_tracker,
            &mut layouter,
            false,
        );
        adjust_expr(&adjusted_global_expressions, &mut expr);
        match evaluator.try_eval_and_append(expr, span) {
            Err((expr, ConstantEvaluatorError::Override(ov_h))) => {
                let h = module.global_expressions.append(expr, span);
                global_expression_kind_tracker.insert(h, crate::proc::ExpressionKind::Override);
                adjusted_global_expressions.insert(old_h, h);
                global_expressions_missing_overrides.insert(h, ov_h);
                log::trace!("global {:?} initializer missing override {:?}", h, ov_h);
            }
            Err((_, e)) => return Err(e.into()),
            Ok(h) => adjusted_global_expressions.insert(old_h, h),
        }
    }

    // Finish processing any overrides we didn't visit in the loop above.
    for entry in override_iter {
        match *entry.1 {
            Override { name: Some(_), .. } | Override { id: Some(_), .. } => {
                process_override(
                    entry,
                    pipeline_constants,
                    &mut module,
                    &mut override_map,
                    &adjusted_global_expressions,
                    &mut adjusted_constant_initializers,
                    &mut global_expression_kind_tracker,
                )?;
            }
            Override {
                init: Some(ref mut init),
                ..
            } => {
                // Anonymous override representing by an array size expression.
                // These are not handled through `process_override`, are not
                // replaced by a constant, and are not added to `override_map`.
                *init = adjusted_global_expressions[*init];
            }
            _ => {}
        }
    }

    // Update the initialization expression handles of all `Constant`s
    // and `GlobalVariable`s. Skip `Constant`s we'd already updated en
    // passant.
    for (_, c) in module
        .constants
        .iter_mut()
        .filter(|&(c_h, _)| !adjusted_constant_initializers.contains(&c_h))
    {
        c.init = adjusted_global_expressions[c.init];
    }

    // Identify which global variables are still unusable due to missing
    // overrides. Overrides can appear in the initializer, and in the
    // case of workgroup space arrays, in the array size.
    for (v_handle, v) in module.global_variables.iter_mut() {
        if let Some(ref mut init) = v.init {
            *init = adjusted_global_expressions[*init];
            if let Some(&o_handle) = global_expressions_missing_overrides.get(init) {
                log::trace!(
                    "global {:?} initializer missing override {:?}",
                    v.name,
                    overrides[o_handle].name
                );
                unresolved.global_variables.insert(v_handle, o_handle);
            }
        } else if let TypeInner::Array {
            size: crate::ArraySize::Pending(o_handle),
            ..
        } = module.types[v.ty].inner
        {
            let resolved = match override_map.get(o_handle) {
                Some(&Some(_)) => {
                    // Override was processed successfully.
                    true
                }
                Some(&None) => {
                    // Override could not be processed.
                    false
                }
                None => {
                    // Anonymous override for array size expression
                    // These are not added to override_map.
                    match overrides[o_handle].init {
                        Some(init) => global_expression_kind_tracker.is_const(init),
                        None => {
                            // This should not happen.
                            log::error!("anonymous override with no initializer?");
                            true
                        }
                    }
                }
            };
            if !resolved {
                log::trace!(
                    "array size of global {:?} missing override {:?}",
                    v.name,
                    overrides[o_handle].name
                );
                unresolved.global_variables.insert(v_handle, o_handle);
            }
        }
    }

    // Process functions, taking note of which ones require overrides that were
    // not specified. Like expressions, callees are guaranteed to appear before
    // their callers.
    let mut functions = mem::take(&mut module.functions);
    for (f_handle, function) in functions.iter_mut() {
        let result = if let Some(o_handle) = process_function(
            &mut module,
            &override_map,
            &unresolved.functions,
            &mut layouter,
            function,
        )? {
            log::trace!(
                "function {:?} missing override {:?}",
                function.name,
                overrides[o_handle].name
            );
            Some(o_handle)
        } else {
            check_for_unresolved_global_use(
                module.global_variables.iter(),
                &unresolved,
                &module_info[f_handle],
            )
        };
        if let Some(o_handle) = result {
            unresolved.functions.insert(f_handle, o_handle);
        }
    }
    module.functions = functions;

    // Process entry points
    let mut entry_points = mem::take(&mut module.entry_points);
    for (ep_index, ep) in entry_points.iter_mut().enumerate() {
        let result = if let Some(o_handle) = process_function(
            &mut module,
            &override_map,
            &unresolved.functions,
            &mut layouter,
            &mut ep.function,
        )? {
            log::trace!(
                "entry point {} missing override {:?}",
                ep.name,
                overrides[o_handle].name
            );
            Some(o_handle)
        } else if let Some(o_handle) =
            process_workgroup_size_override(&mut module, &adjusted_global_expressions, ep)?
        {
            Some(o_handle)
        } else {
            check_for_unresolved_global_use(
                module.global_variables.iter(),
                &unresolved,
                module_info.get_entry_point(ep_index),
            )
        };
        if let Some(o_handle) = result {
            // We found a missing override that is required by this entry point.
            // Decide whether that is an error.
            match entry_point {
                Some((tgt_stage, tgt_name)) if ep.stage != tgt_stage || ep.name != tgt_name => {
                    // An entry point was specified, and we are not currently
                    // processing that one, so it is okay not to have this
                    // override.
                    unresolved.entry_points.insert(ep_index, o_handle);
                }
                _ => {
                    // Either we are missing an override for the active entry point,
                    // or no entry point was specified. Either way, this override
                    // is required.
                    return Err(PipelineConstantError::MissingValue(
                        override_key(&overrides[o_handle]).to_string(),
                    ));
                }
            }
        }
    }

    module.entry_points = entry_points;
    module.overrides = overrides;

    // Now that we've rewritten all the expressions, we need to
    // recompute their types and other metadata. For the time being,
    // do a full re-validation.
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let module_info = validator.validate_with_resolved_overrides(&module, &unresolved)?;

    Ok(ProcessOverridesOutput {
        module: Cow::Owned(module),
        info: Cow::Owned(module_info),
        unresolved,
    })
}

/// Process override expressions in the WGSL `@workgroup_size` attribute.
///
/// If all expressions are resolved, returns `Ok(None)`. If any expression could
/// not be resolved due to missing override values, returns `Ok(Some(handle))`,
/// with the handle of the first identified missing override. The caller is
/// responsible for determining whether translation can proceed despite the
/// missing override.
fn process_workgroup_size_override(
    module: &mut Module,
    adjusted_global_expressions: &HandleVec<Expression, Handle<Expression>>,
    ep: &mut crate::EntryPoint,
) -> Result<Option<Handle<Override>>, PipelineConstantError> {
    match ep.workgroup_size_overrides {
        None => {}
        Some(overrides) => {
            for (ov_index, ov) in overrides.iter().enumerate() {
                match *ov {
                    None => continue,
                    Some(h) => {
                        match module
                            .to_ctx()
                            .eval_expr_to_u32(adjusted_global_expressions[h])
                        {
                            Ok(n) => {
                                if n == 0 {
                                    return Err(PipelineConstantError::NegativeWorkgroupSize);
                                } else {
                                    ep.workgroup_size[ov_index] = n;
                                }
                            }
                            Err(U32EvalError::Override(handle)) => {
                                return Ok(Some(handle));
                            }
                            Err(U32EvalError::Runtime) => {
                                return Err(
                                    PipelineConstantError::WorkgroupSizeOverrideEvaluationError,
                                );
                            }
                            Err(U32EvalError::Negative) => {
                                return Err(PipelineConstantError::NegativeWorkgroupSize);
                            }
                        }
                    }
                }
            }
            ep.workgroup_size_overrides = None;
        }
    }
    Ok(None)
}

/// If a value for the override `old_h` is given in `self.pipeline_constants`,
/// add a [`Constant`] for that override to `module`.
///
/// If a value is found, adds the new `Constant` to `override_map` and
/// `adjusted_constant_initializers`, and returns it.
///
/// If no value is found, returns `Ok(None)`. The caller is responsible for
/// determining whether translation can proceed despite the missing override.
fn process_override(
    (old_h, r#override, span): (Handle<Override>, &mut Override, &Span),
    pipeline_constants: &PipelineConstants,
    module: &mut Module,
    override_map: &mut HandleVec<Override, Option<Handle<Constant>>>,
    adjusted_global_expressions: &HandleVec<Expression, Handle<Expression>>,
    adjusted_constant_initializers: &mut HashSet<Handle<Constant>>,
    global_expression_kind_tracker: &mut crate::proc::ExpressionKindTracker,
) -> Result<Option<Handle<Constant>>, PipelineConstantError> {
    let key = override_key(r#override);

    // Generate a global expression for `r#override`'s value, either
    // from the provided `pipeline_constants` table or its initializer
    // in the module.
    let init = if let Some(value) = pipeline_constants.get::<str>(&key) {
        let literal = match module.types[r#override.ty].inner {
            TypeInner::Scalar(scalar) => map_value_to_literal(*value, scalar)?,
            _ => unreachable!(),
        };
        let expr = module
            .global_expressions
            .append(Expression::Literal(literal), Span::UNDEFINED);
        global_expression_kind_tracker.insert(expr, crate::proc::ExpressionKind::Const);
        expr
    } else {
        match r#override.init {
            Some(init) if global_expression_kind_tracker.is_const(init) => {
                adjusted_global_expressions[init]
            }
            _ => {
                override_map.insert(old_h, None);
                return Ok(None);
            }
        }
    };

    // Generate a new `Constant` to represent the override's value.
    let constant = Constant {
        name: r#override.name.clone(),
        ty: r#override.ty,
        init,
    };
    let h = module.constants.append(constant, *span);
    override_map.insert(old_h, Some(h));
    adjusted_constant_initializers.insert(h);
    r#override.init = Some(init);
    Ok(Some(h))
}

/// Replace override expressions in `function` with fully-evaluated constants.
///
/// Replace `Expression::Override`s in `function`'s expression arena with
/// the corresponding `Expression::Constant`s, as given in `override_map`.
/// Replace any expressions whose values are now known with their fully
/// evaluated form.
///
/// If `h` is a `Handle<Override>`, then `override_map[h]` is the
/// `Handle<Constant>` for the override's final value.
///
/// If all override expressions are replaced, returns `Ok(None)`. If any
/// expression could not be replaced due to missing override values, or if
/// the function calls another function that is present in
/// `functions_missing_overrides`, returns `Ok(Some(handle))`, with the handle
/// of the first identified missing override. The caller is responsible for
/// determining whether translation can proceed despite the missing override.
fn process_function(
    module: &mut Module,
    override_map: &HandleVec<Override, Option<Handle<Constant>>>,
    functions_missing_overrides: &FastHashMap<Handle<Function>, Handle<Override>>,
    layouter: &mut crate::proc::Layouter,
    function: &mut Function,
) -> Result<Option<Handle<Override>>, ConstantEvaluatorError> {
    // A map from original local expression handles to
    // handles in the new, local expression arena.
    let mut adjusted_local_expressions = HandleVec::with_capacity(function.expressions.len());

    let mut local_expression_kind_tracker = crate::proc::ExpressionKindTracker::new();

    let mut expressions = mem::take(&mut function.expressions);

    let mut missing_override = None;

    // Dummy `emitter` and `block` for the constant evaluator.
    // We can ignore the concept of emitting expressions here since
    // expressions have already been covered by a `Statement::Emit`
    // in the frontend.
    // The only thing we might have to do is remove some expressions
    // that have been covered by a `Statement::Emit`. See the docs of
    // `filter_emits_in_block` for the reasoning.
    let mut emitter = Emitter::default();
    let mut block = Block::new();

    let mut evaluator = ConstantEvaluator::for_wgsl_function(
        module,
        &mut function.expressions,
        &mut local_expression_kind_tracker,
        layouter,
        &mut emitter,
        &mut block,
        false,
    );

    for (old_h, mut expr, span) in expressions.drain() {
        if let Expression::Override(h) = expr {
            if let Some(&Some(const_h)) = override_map.get(h) {
                expr = Expression::Constant(const_h);
            } else if missing_override.is_none() {
                missing_override = Some(h);
            }
        }
        adjust_expr(&adjusted_local_expressions, &mut expr);
        let h = evaluator
            .try_eval_and_append(expr, span)
            .map_err(|(_expr, err)| err)?;
        adjusted_local_expressions.insert(old_h, h);
    }

    match adjust_block(
        &adjusted_local_expressions,
        functions_missing_overrides,
        &mut function.body,
    ) {
        missing @ Some(_) if missing_override.is_none() => {
            missing_override = missing;
        }
        _ => {}
    }

    filter_emits_in_block(&mut function.body, &function.expressions);

    // Update local expression initializers.
    for (_, local) in function.local_variables.iter_mut() {
        if let &mut Some(ref mut init) = &mut local.init {
            *init = adjusted_local_expressions[*init];
        }
    }

    // We've changed the keys of `function.named_expression`, so we have to
    // rebuild it from scratch.
    let named_expressions = mem::take(&mut function.named_expressions);
    for (expr_h, name) in named_expressions {
        function
            .named_expressions
            .insert(adjusted_local_expressions[expr_h], name);
    }

    Ok(missing_override)
}

/// Replace every expression handle in `expr` with its counterpart
/// given by `new_pos`.
fn adjust_expr(new_pos: &HandleVec<Expression, Handle<Expression>>, expr: &mut Expression) {
    let adjust = |expr: &mut Handle<Expression>| {
        *expr = new_pos[*expr];
    };
    match *expr {
        Expression::Compose {
            ref mut components,
            ty: _,
        } => {
            for c in components.iter_mut() {
                adjust(c);
            }
        }
        Expression::Access {
            ref mut base,
            ref mut index,
        } => {
            adjust(base);
            adjust(index);
        }
        Expression::AccessIndex {
            ref mut base,
            index: _,
        } => {
            adjust(base);
        }
        Expression::Splat {
            ref mut value,
            size: _,
        } => {
            adjust(value);
        }
        Expression::Swizzle {
            ref mut vector,
            size: _,
            pattern: _,
        } => {
            adjust(vector);
        }
        Expression::Load { ref mut pointer } => {
            adjust(pointer);
        }
        Expression::ImageSample {
            ref mut image,
            ref mut sampler,
            ref mut coordinate,
            ref mut array_index,
            ref mut offset,
            ref mut level,
            ref mut depth_ref,
            gather: _,
        } => {
            adjust(image);
            adjust(sampler);
            adjust(coordinate);
            if let Some(e) = array_index.as_mut() {
                adjust(e);
            }
            if let Some(e) = offset.as_mut() {
                adjust(e);
            }
            match *level {
                crate::SampleLevel::Exact(ref mut expr)
                | crate::SampleLevel::Bias(ref mut expr) => {
                    adjust(expr);
                }
                crate::SampleLevel::Gradient {
                    ref mut x,
                    ref mut y,
                } => {
                    adjust(x);
                    adjust(y);
                }
                _ => {}
            }
            if let Some(e) = depth_ref.as_mut() {
                adjust(e);
            }
        }
        Expression::ImageLoad {
            ref mut image,
            ref mut coordinate,
            ref mut array_index,
            ref mut sample,
            ref mut level,
        } => {
            adjust(image);
            adjust(coordinate);
            if let Some(e) = array_index.as_mut() {
                adjust(e);
            }
            if let Some(e) = sample.as_mut() {
                adjust(e);
            }
            if let Some(e) = level.as_mut() {
                adjust(e);
            }
        }
        Expression::ImageQuery {
            ref mut image,
            ref mut query,
        } => {
            adjust(image);
            match *query {
                crate::ImageQuery::Size { ref mut level } => {
                    if let Some(e) = level.as_mut() {
                        adjust(e);
                    }
                }
                crate::ImageQuery::NumLevels
                | crate::ImageQuery::NumLayers
                | crate::ImageQuery::NumSamples => {}
            }
        }
        Expression::Unary {
            ref mut expr,
            op: _,
        } => {
            adjust(expr);
        }
        Expression::Binary {
            ref mut left,
            ref mut right,
            op: _,
        } => {
            adjust(left);
            adjust(right);
        }
        Expression::Select {
            ref mut condition,
            ref mut accept,
            ref mut reject,
        } => {
            adjust(condition);
            adjust(accept);
            adjust(reject);
        }
        Expression::Derivative {
            ref mut expr,
            axis: _,
            ctrl: _,
        } => {
            adjust(expr);
        }
        Expression::Relational {
            ref mut argument,
            fun: _,
        } => {
            adjust(argument);
        }
        Expression::Math {
            ref mut arg,
            ref mut arg1,
            ref mut arg2,
            ref mut arg3,
            fun: _,
        } => {
            adjust(arg);
            if let Some(e) = arg1.as_mut() {
                adjust(e);
            }
            if let Some(e) = arg2.as_mut() {
                adjust(e);
            }
            if let Some(e) = arg3.as_mut() {
                adjust(e);
            }
        }
        Expression::As {
            ref mut expr,
            kind: _,
            convert: _,
        } => {
            adjust(expr);
        }
        Expression::ArrayLength(ref mut expr) => {
            adjust(expr);
        }
        Expression::RayQueryGetIntersection {
            ref mut query,
            committed: _,
        } => {
            adjust(query);
        }
        Expression::Literal(_)
        | Expression::FunctionArgument(_)
        | Expression::GlobalVariable(_)
        | Expression::LocalVariable(_)
        | Expression::CallResult(_)
        | Expression::RayQueryProceedResult
        | Expression::Constant(_)
        | Expression::Override(_)
        | Expression::ZeroValue(_)
        | Expression::AtomicResult {
            ty: _,
            comparison: _,
        }
        | Expression::WorkGroupUniformLoadResult { ty: _ }
        | Expression::SubgroupBallotResult
        | Expression::SubgroupOperationResult { .. } => {}
        Expression::RayQueryVertexPositions {
            ref mut query,
            committed: _,
        } => {
            adjust(query);
        }
    }
}

/// Replace every expression handle in `block` with its counterpart
/// given by `new_pos`.
///
/// On success, returns `Ok(None)`. If `block` calls a function that is present
/// in `functions_missing_overrides`, returns `Ok(Some(handle))`, with the
/// handle of the first identified missing override.
fn adjust_block(
    new_pos: &HandleVec<Expression, Handle<Expression>>,
    functions_missing_overrides: &FastHashMap<Handle<Function>, Handle<Override>>,
    block: &mut Block,
) -> Option<Handle<Override>> {
    let mut missing_override = None;
    for stmt in block.iter_mut() {
        match adjust_stmt(new_pos, functions_missing_overrides, stmt) {
            missing @ Some(_) if missing_override.is_none() => {
                missing_override = missing;
            }
            _ => {}
        }
    }
    missing_override
}

/// Replace every expression handle in `stmt` with its counterpart
/// given by `new_pos`.
///
/// On success, returns `Ok(None)`. If `stmt` calls a function that is present
/// in `functions_missing_overrides`, returns `Ok(Some(handle))`, with the
/// handle of the first identified missing override.
fn adjust_stmt(
    new_pos: &HandleVec<Expression, Handle<Expression>>,
    functions_missing_overrides: &FastHashMap<Handle<Function>, Handle<Override>>,
    stmt: &mut Statement,
) -> Option<Handle<Override>> {
    let mut missing_override = None;
    let adjust = |expr: &mut Handle<Expression>| {
        *expr = new_pos[*expr];
    };
    match *stmt {
        Statement::Emit(ref mut range) => {
            if let Some((mut first, mut last)) = range.first_and_last() {
                adjust(&mut first);
                adjust(&mut last);
                *range = Range::new_from_bounds(first, last);
            }
        }
        Statement::Block(ref mut block) => {
            adjust_block(new_pos, functions_missing_overrides, block);
        }
        Statement::If {
            ref mut condition,
            ref mut accept,
            ref mut reject,
        } => {
            adjust(condition);
            adjust_block(new_pos, functions_missing_overrides, accept);
            adjust_block(new_pos, functions_missing_overrides, reject);
        }
        Statement::Switch {
            ref mut selector,
            ref mut cases,
        } => {
            adjust(selector);
            for case in cases.iter_mut() {
                adjust_block(new_pos, functions_missing_overrides, &mut case.body);
            }
        }
        Statement::Loop {
            ref mut body,
            ref mut continuing,
            ref mut break_if,
        } => {
            adjust_block(new_pos, functions_missing_overrides, body);
            adjust_block(new_pos, functions_missing_overrides, continuing);
            if let Some(e) = break_if.as_mut() {
                adjust(e);
            }
        }
        Statement::Return { ref mut value } => {
            if let Some(e) = value.as_mut() {
                adjust(e);
            }
        }
        Statement::Store {
            ref mut pointer,
            ref mut value,
        } => {
            adjust(pointer);
            adjust(value);
        }
        Statement::ImageStore {
            ref mut image,
            ref mut coordinate,
            ref mut array_index,
            ref mut value,
        } => {
            adjust(image);
            adjust(coordinate);
            if let Some(e) = array_index.as_mut() {
                adjust(e);
            }
            adjust(value);
        }
        Statement::Atomic {
            ref mut pointer,
            ref mut value,
            ref mut result,
            ref mut fun,
        } => {
            adjust(pointer);
            adjust(value);
            if let Some(ref mut result) = *result {
                adjust(result);
            }
            match *fun {
                crate::AtomicFunction::Exchange {
                    compare: Some(ref mut compare),
                } => {
                    adjust(compare);
                }
                crate::AtomicFunction::Add
                | crate::AtomicFunction::Subtract
                | crate::AtomicFunction::And
                | crate::AtomicFunction::ExclusiveOr
                | crate::AtomicFunction::InclusiveOr
                | crate::AtomicFunction::Min
                | crate::AtomicFunction::Max
                | crate::AtomicFunction::Exchange { compare: None } => {}
            }
        }
        Statement::ImageAtomic {
            ref mut image,
            ref mut coordinate,
            ref mut array_index,
            fun: _,
            ref mut value,
        } => {
            adjust(image);
            adjust(coordinate);
            if let Some(ref mut array_index) = *array_index {
                adjust(array_index);
            }
            adjust(value);
        }
        Statement::WorkGroupUniformLoad {
            ref mut pointer,
            ref mut result,
        } => {
            adjust(pointer);
            adjust(result);
        }
        Statement::SubgroupBallot {
            ref mut result,
            ref mut predicate,
        } => {
            if let Some(ref mut predicate) = *predicate {
                adjust(predicate);
            }
            adjust(result);
        }
        Statement::SubgroupCollectiveOperation {
            ref mut argument,
            ref mut result,
            ..
        } => {
            adjust(argument);
            adjust(result);
        }
        Statement::SubgroupGather {
            ref mut mode,
            ref mut argument,
            ref mut result,
        } => {
            match *mode {
                crate::GatherMode::BroadcastFirst => {}
                crate::GatherMode::Broadcast(ref mut index)
                | crate::GatherMode::Shuffle(ref mut index)
                | crate::GatherMode::ShuffleDown(ref mut index)
                | crate::GatherMode::ShuffleUp(ref mut index)
                | crate::GatherMode::ShuffleXor(ref mut index) => {
                    adjust(index);
                }
            }
            adjust(argument);
            adjust(result)
        }
        Statement::Call {
            ref mut arguments,
            ref mut result,
            function,
        } => {
            match functions_missing_overrides.get(&function).copied() {
                missing @ Some(_) if missing_override.is_none() => {
                    missing_override = missing;
                }
                _ => {}
            }
            for argument in arguments.iter_mut() {
                adjust(argument);
            }
            if let Some(e) = result.as_mut() {
                adjust(e);
            }
        }
        Statement::RayQuery {
            ref mut query,
            ref mut fun,
        } => {
            adjust(query);
            match *fun {
                crate::RayQueryFunction::Initialize {
                    ref mut acceleration_structure,
                    ref mut descriptor,
                } => {
                    adjust(acceleration_structure);
                    adjust(descriptor);
                }
                crate::RayQueryFunction::Proceed { ref mut result } => {
                    adjust(result);
                }
                crate::RayQueryFunction::GenerateIntersection { ref mut hit_t } => {
                    adjust(hit_t);
                }
                crate::RayQueryFunction::ConfirmIntersection => {}
                crate::RayQueryFunction::Terminate => {}
            }
        }
        Statement::Break | Statement::Continue | Statement::Kill | Statement::Barrier(_) => {}
    }
    missing_override
}

/// Adjust [`Emit`] statements in `block` to skip [`needs_pre_emit`] expressions we have introduced.
///
/// According to validation, [`Emit`] statements must not cover any expressions
/// for which [`Expression::needs_pre_emit`] returns true. All expressions built
/// by successful constant evaluation fall into that category, meaning that
/// `process_function` will usually rewrite [`Override`] expressions and those
/// that use their values into pre-emitted expressions, leaving any [`Emit`]
/// statements that cover them invalid.
///
/// This function rewrites all [`Emit`] statements into zero or more new
/// [`Emit`] statements covering only those expressions in the original range
/// that are not pre-emitted.
///
/// [`Emit`]: Statement::Emit
/// [`needs_pre_emit`]: Expression::needs_pre_emit
/// [`Override`]: Expression::Override
fn filter_emits_in_block(block: &mut Block, expressions: &Arena<Expression>) {
    let original = mem::replace(block, Block::with_capacity(block.len()));
    for (stmt, span) in original.span_into_iter() {
        match stmt {
            Statement::Emit(range) => {
                let mut current = None;
                for expr_h in range {
                    if expressions[expr_h].needs_pre_emit() {
                        if let Some((first, last)) = current {
                            block.push(Statement::Emit(Range::new_from_bounds(first, last)), span);
                        }

                        current = None;
                    } else if let Some((_, ref mut last)) = current {
                        *last = expr_h;
                    } else {
                        current = Some((expr_h, expr_h));
                    }
                }
                if let Some((first, last)) = current {
                    block.push(Statement::Emit(Range::new_from_bounds(first, last)), span);
                }
            }
            Statement::Block(mut child) => {
                filter_emits_in_block(&mut child, expressions);
                block.push(Statement::Block(child), span);
            }
            Statement::If {
                condition,
                mut accept,
                mut reject,
            } => {
                filter_emits_in_block(&mut accept, expressions);
                filter_emits_in_block(&mut reject, expressions);
                block.push(
                    Statement::If {
                        condition,
                        accept,
                        reject,
                    },
                    span,
                );
            }
            Statement::Switch {
                selector,
                mut cases,
            } => {
                for case in &mut cases {
                    filter_emits_in_block(&mut case.body, expressions);
                }
                block.push(Statement::Switch { selector, cases }, span);
            }
            Statement::Loop {
                mut body,
                mut continuing,
                break_if,
            } => {
                filter_emits_in_block(&mut body, expressions);
                filter_emits_in_block(&mut continuing, expressions);
                block.push(
                    Statement::Loop {
                        body,
                        continuing,
                        break_if,
                    },
                    span,
                );
            }
            stmt => block.push(stmt.clone(), span),
        }
    }
}

fn map_value_to_literal(value: f64, scalar: Scalar) -> Result<Literal, PipelineConstantError> {
    // note that in rust 0.0 == -0.0
    match scalar {
        Scalar::BOOL => {
            // https://webidl.spec.whatwg.org/#js-boolean
            let value = value != 0.0 && !value.is_nan();
            Ok(Literal::Bool(value))
        }
        Scalar::I32 => {
            // https://webidl.spec.whatwg.org/#js-long
            if !value.is_finite() {
                return Err(PipelineConstantError::SrcNeedsToBeFinite);
            }

            let value = value.trunc();
            if value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
                return Err(PipelineConstantError::DstRangeTooSmall);
            }

            let value = value as i32;
            Ok(Literal::I32(value))
        }
        Scalar::U32 => {
            // https://webidl.spec.whatwg.org/#js-unsigned-long
            if !value.is_finite() {
                return Err(PipelineConstantError::SrcNeedsToBeFinite);
            }

            let value = value.trunc();
            if value < f64::from(u32::MIN) || value > f64::from(u32::MAX) {
                return Err(PipelineConstantError::DstRangeTooSmall);
            }

            let value = value as u32;
            Ok(Literal::U32(value))
        }
        Scalar::F32 => {
            // https://webidl.spec.whatwg.org/#js-float
            if !value.is_finite() {
                return Err(PipelineConstantError::SrcNeedsToBeFinite);
            }

            let value = value as f32;
            if !value.is_finite() {
                return Err(PipelineConstantError::DstRangeTooSmall);
            }

            Ok(Literal::F32(value))
        }
        Scalar::F64 => {
            // https://webidl.spec.whatwg.org/#js-double
            if !value.is_finite() {
                return Err(PipelineConstantError::SrcNeedsToBeFinite);
            }

            Ok(Literal::F64(value))
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_map_value_to_literal() {
    let bool_test_cases = [
        (0.0, false),
        (-0.0, false),
        (f64::NAN, false),
        (1.0, true),
        (f64::INFINITY, true),
        (f64::NEG_INFINITY, true),
    ];
    for (value, out) in bool_test_cases {
        let res = Ok(Literal::Bool(out));
        assert_eq!(map_value_to_literal(value, Scalar::BOOL), res);
    }

    for scalar in [Scalar::I32, Scalar::U32, Scalar::F32, Scalar::F64] {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let res = Err(PipelineConstantError::SrcNeedsToBeFinite);
            assert_eq!(map_value_to_literal(value, scalar), res);
        }
    }

    // i32
    assert_eq!(
        map_value_to_literal(f64::from(i32::MIN), Scalar::I32),
        Ok(Literal::I32(i32::MIN))
    );
    assert_eq!(
        map_value_to_literal(f64::from(i32::MAX), Scalar::I32),
        Ok(Literal::I32(i32::MAX))
    );
    assert_eq!(
        map_value_to_literal(f64::from(i32::MIN) - 1.0, Scalar::I32),
        Err(PipelineConstantError::DstRangeTooSmall)
    );
    assert_eq!(
        map_value_to_literal(f64::from(i32::MAX) + 1.0, Scalar::I32),
        Err(PipelineConstantError::DstRangeTooSmall)
    );

    // u32
    assert_eq!(
        map_value_to_literal(f64::from(u32::MIN), Scalar::U32),
        Ok(Literal::U32(u32::MIN))
    );
    assert_eq!(
        map_value_to_literal(f64::from(u32::MAX), Scalar::U32),
        Ok(Literal::U32(u32::MAX))
    );
    assert_eq!(
        map_value_to_literal(f64::from(u32::MIN) - 1.0, Scalar::U32),
        Err(PipelineConstantError::DstRangeTooSmall)
    );
    assert_eq!(
        map_value_to_literal(f64::from(u32::MAX) + 1.0, Scalar::U32),
        Err(PipelineConstantError::DstRangeTooSmall)
    );

    // f32
    assert_eq!(
        map_value_to_literal(f64::from(f32::MIN), Scalar::F32),
        Ok(Literal::F32(f32::MIN))
    );
    assert_eq!(
        map_value_to_literal(f64::from(f32::MAX), Scalar::F32),
        Ok(Literal::F32(f32::MAX))
    );
    assert_eq!(
        map_value_to_literal(-f64::from_bits(0x47efffffefffffff), Scalar::F32),
        Ok(Literal::F32(f32::MIN))
    );
    assert_eq!(
        map_value_to_literal(f64::from_bits(0x47efffffefffffff), Scalar::F32),
        Ok(Literal::F32(f32::MAX))
    );
    assert_eq!(
        map_value_to_literal(-f64::from_bits(0x47effffff0000000), Scalar::F32),
        Err(PipelineConstantError::DstRangeTooSmall)
    );
    assert_eq!(
        map_value_to_literal(f64::from_bits(0x47effffff0000000), Scalar::F32),
        Err(PipelineConstantError::DstRangeTooSmall)
    );

    // f64
    assert_eq!(
        map_value_to_literal(f64::MIN, Scalar::F64),
        Ok(Literal::F64(f64::MIN))
    );
    assert_eq!(
        map_value_to_literal(f64::MAX, Scalar::F64),
        Ok(Literal::F64(f64::MAX))
    );
}
