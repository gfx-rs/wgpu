mod expressions;
mod functions;
mod handle_set_map;
mod statements;
mod types;

use alloc::vec::Vec;

use crate::arena::HandleSet;
use crate::{arena, compact::functions::FunctionTracer};
use handle_set_map::HandleMap;

#[cfg(test)]
use alloc::{format, string::ToString};

/// Remove unused types, expressions, and constants from `module`.
///
/// Assume that the following are used by definition:
/// - global variables
/// - named constants and overrides
/// - special types
/// - functions and entry points, and within those:
///     - arguments
///     - local variables
///     - named expressions
///
/// Given those assumptions, determine which types, constants,
/// overrides, and expressions (both function-local and global
/// constant expressions) are actually used, and remove the rest,
/// adjusting all handles as necessary. The result should always be a
/// module functionally identical to the original.
///
/// This may be useful to apply to modules generated in the snapshot
/// tests. Our backends often generate temporary names based on handle
/// indices, which means that adding or removing unused arena entries
/// can affect the output even though they have no semantic effect.
/// Such meaningless changes add noise to snapshot diffs, making
/// accurate patch review difficult. Compacting the modules before
/// generating snapshots makes the output independent of unused arena
/// entries.
///
/// # Panics
///
/// If `module` would not pass validation, this may panic.
pub fn compact(module: &mut crate::Module) {
    // The trickiest part of compaction is determining what is used and what is
    // not. Once we have computed that correctly, it's easy enough to call
    // `retain_mut` on each arena, drop unused elements, and fix up the handles
    // in what's left.
    //
    // Some arenas' contents are considered used by definition, like
    // `Module::global_variables` and `Module::functions`, so those are never
    // compacted.
    //
    // But for every compactable arena in a `Module`, whether global to the
    // `Module` or local to a function or entry point, the `ModuleTracer` type
    // holds a bitmap indicating which elements of that arena are used. Our task
    // is to populate those bitmaps correctly.
    //
    // First, we mark everything that is considered used by definition, as
    // described in this function's documentation.
    //
    // Since functions and entry points are considered used by definition, we
    // traverse their statement trees, and mark the referents of all handles
    // appearing in those statements as used.
    //
    // Once we've marked which elements of an arena are referred to directly by
    // handles elsewhere (for example, which of a function's expressions are
    // referred to by handles in its body statements), we can mark all the other
    // arena elements that are used indirectly in a single pass, traversing the
    // arena from back to front. Since Naga allows arena elements to refer only
    // to prior elements, we know that by the time we reach an element, all
    // other elements that could possibly refer to it have already been visited.
    // Thus, if the present element has not been marked as used, then it is
    // definitely unused, and compaction can remove it. Otherwise, the element
    // is used and must be retained, so we must mark everything it refers to.
    //
    // The final step is to mark the global expressions and types, which must be
    // traversed simultaneously; see `ModuleTracer::type_expression_tandem`'s
    // documentation for details.
    //
    // # A definition and a rule of thumb
    //
    // In this module, to "trace" something is to mark everything else it refers
    // to as used, on the assumption that the thing itself is used. For example,
    // to trace an `Expression` is to mark its subexpressions as used, as well
    // as any types, constants, overrides, etc. that it refers to. This is what
    // `ExpressionTracer::trace_expression` does.
    //
    // Given that we we want to visit each thing only once (to keep compaction
    // linear in the size of the module), this definition of "trace" implies
    // that things that are not "used by definition" must be marked as used
    // *before* we trace them.
    //
    // Thus, whenever you are marking something as used, it's a good idea to ask
    // yourself how you know that thing will be traced in the future. If you're
    // not sure, then you could be marking it too late to be noticed. The thing
    // itself will be retained by compaction, but since it will not be traced,
    // anything it refers to could be compacted away.
    let mut module_tracer = ModuleTracer::new(module);

    // We treat all globals as used by definition.
    log::trace!("tracing global variables");
    {
        for (_, global) in module.global_variables.iter() {
            log::trace!("tracing global {:?}", global.name);
            module_tracer.types_used.insert(global.ty);
            if let Some(init) = global.init {
                module_tracer.global_expressions_used.insert(init);
            }
        }
    }

    // We treat all special types as used by definition.
    log::trace!("tracing special types");
    module_tracer.trace_special_types(&module.special_types);

    // We treat all named constants as used by definition, unless they have an
    // abstract type as we do not want those reaching the validator.
    log::trace!("tracing named constants");
    for (handle, constant) in module.constants.iter() {
        if constant.name.is_none() || module.types[constant.ty].inner.is_abstract(&module.types) {
            continue;
        }

        log::trace!("tracing constant {:?}", constant.name.as_ref().unwrap());
        module_tracer.constants_used.insert(handle);
        module_tracer.types_used.insert(constant.ty);
        module_tracer.global_expressions_used.insert(constant.init);
    }

    // We treat all named overrides as used by definition.
    log::trace!("tracing named overrides");
    for (handle, r#override) in module.overrides.iter() {
        if r#override.name.is_some() {
            log::trace!("tracing override {:?}", r#override.name.as_ref().unwrap());
            module_tracer.overrides_used.insert(handle);
            module_tracer.types_used.insert(r#override.ty);
            if let Some(init) = r#override.init {
                module_tracer.global_expressions_used.insert(init);
            }
        }
    }

    // We assume that all functions are used.
    //
    // Observe which types, constant expressions, constants, and
    // expressions each function uses, and produce maps for each
    // function from pre-compaction to post-compaction expression
    // handles.
    log::trace!("tracing functions");
    let function_maps: Vec<FunctionMap> = module
        .functions
        .iter()
        .map(|(_, f)| {
            log::trace!("tracing function {:?}", f.name);
            let mut function_tracer = module_tracer.as_function(f);
            function_tracer.trace();
            FunctionMap::from(function_tracer)
        })
        .collect();

    // Similarly, observe what each entry point actually uses.
    log::trace!("tracing entry points");
    let entry_point_maps: Vec<FunctionMap> = module
        .entry_points
        .iter()
        .map(|e| {
            log::trace!("tracing entry point {:?}", e.function.name);

            if let Some(sizes) = e.workgroup_size_overrides {
                for size in sizes.iter().filter_map(|x| *x) {
                    module_tracer.global_expressions_used.insert(size);
                }
            }

            let mut used = module_tracer.as_function(&e.function);
            used.trace();
            FunctionMap::from(used)
        })
        .collect();

    // Treat all named types as used.
    for (handle, ty) in module.types.iter() {
        log::trace!("tracing type {:?}, name {:?}", handle, ty.name);
        if ty.name.is_some() {
            module_tracer.types_used.insert(handle);
        }
    }

    module_tracer.type_expression_tandem();

    // Now that we know what is used and what is never touched,
    // produce maps from the `Handle`s that appear in `module` now to
    // the corresponding `Handle`s that will refer to the same items
    // in the compacted module.
    let module_map = ModuleMap::from(module_tracer);

    // Drop unused types from the type arena.
    //
    // `FastIndexSet`s don't have an underlying Vec<T> that we can
    // steal, compact in place, and then rebuild the `FastIndexSet`
    // from. So we have to rebuild the type arena from scratch.
    log::trace!("compacting types");
    let mut new_types = arena::UniqueArena::new();
    for (old_handle, mut ty, span) in module.types.drain_all() {
        if let Some(expected_new_handle) = module_map.types.try_adjust(old_handle) {
            module_map.adjust_type(&mut ty);
            let actual_new_handle = new_types.insert(ty, span);
            assert_eq!(actual_new_handle, expected_new_handle);
        }
    }
    module.types = new_types;
    log::trace!("adjusting special types");
    module_map.adjust_special_types(&mut module.special_types);

    // Drop unused constant expressions, reusing existing storage.
    log::trace!("adjusting constant expressions");
    module.global_expressions.retain_mut(|handle, expr| {
        if module_map.global_expressions.used(handle) {
            module_map.adjust_expression(expr, &module_map.global_expressions);
            true
        } else {
            false
        }
    });

    // Drop unused constants in place, reusing existing storage.
    log::trace!("adjusting constants");
    module.constants.retain_mut(|handle, constant| {
        if module_map.constants.used(handle) {
            module_map.types.adjust(&mut constant.ty);
            module_map.global_expressions.adjust(&mut constant.init);
            true
        } else {
            false
        }
    });

    // Drop unused overrides in place, reusing existing storage.
    log::trace!("adjusting overrides");
    module.overrides.retain_mut(|handle, r#override| {
        if module_map.overrides.used(handle) {
            module_map.types.adjust(&mut r#override.ty);
            if let Some(ref mut init) = r#override.init {
                module_map.global_expressions.adjust(init);
            }
            true
        } else {
            false
        }
    });

    // Adjust workgroup_size_overrides
    log::trace!("adjusting workgroup_size_overrides");
    for e in module.entry_points.iter_mut() {
        if let Some(sizes) = e.workgroup_size_overrides.as_mut() {
            for size in sizes.iter_mut() {
                if let Some(expr) = size.as_mut() {
                    module_map.global_expressions.adjust(expr);
                }
            }
        }
    }

    // Adjust global variables' types and initializers.
    log::trace!("adjusting global variables");
    for (_, global) in module.global_variables.iter_mut() {
        log::trace!("adjusting global {:?}", global.name);
        module_map.types.adjust(&mut global.ty);
        if let Some(ref mut init) = global.init {
            module_map.global_expressions.adjust(init);
        }
    }

    // Temporary storage to help us reuse allocations of existing
    // named expression tables.
    let mut reused_named_expressions = crate::NamedExpressions::default();

    // Compact each function.
    for ((_, function), map) in module.functions.iter_mut().zip(function_maps.iter()) {
        log::trace!("compacting function {:?}", function.name);
        map.compact(function, &module_map, &mut reused_named_expressions);
    }

    // Compact each entry point.
    for (entry, map) in module.entry_points.iter_mut().zip(entry_point_maps.iter()) {
        log::trace!("compacting entry point {:?}", entry.function.name);
        map.compact(
            &mut entry.function,
            &module_map,
            &mut reused_named_expressions,
        );
    }
}

struct ModuleTracer<'module> {
    module: &'module crate::Module,
    types_used: HandleSet<crate::Type>,
    constants_used: HandleSet<crate::Constant>,
    overrides_used: HandleSet<crate::Override>,
    global_expressions_used: HandleSet<crate::Expression>,
}

impl<'module> ModuleTracer<'module> {
    fn new(module: &'module crate::Module) -> Self {
        Self {
            module,
            types_used: HandleSet::for_arena(&module.types),
            constants_used: HandleSet::for_arena(&module.constants),
            overrides_used: HandleSet::for_arena(&module.overrides),
            global_expressions_used: HandleSet::for_arena(&module.global_expressions),
        }
    }

    fn trace_special_types(&mut self, special_types: &crate::SpecialTypes) {
        let crate::SpecialTypes {
            ref ray_desc,
            ref ray_intersection,
            ref ray_vertex_return,
            ref predeclared_types,
        } = *special_types;

        if let Some(ray_desc) = *ray_desc {
            self.types_used.insert(ray_desc);
        }
        if let Some(ray_intersection) = *ray_intersection {
            self.types_used.insert(ray_intersection);
        }
        if let Some(ray_vertex_return) = *ray_vertex_return {
            self.types_used.insert(ray_vertex_return);
        }
        for (_, &handle) in predeclared_types {
            self.types_used.insert(handle);
        }
    }

    /// Traverse types and global expressions in tandem to determine which are used.
    ///
    /// Assuming that all types and global expressions used by other parts of
    /// the module have been added to [`types_used`] and
    /// [`global_expressions_used`], expand those sets to include all types and
    /// global expressions reachable from those.
    ///
    /// [`types_used`]: ModuleTracer::types_used
    /// [`global_expressions_used`]: ModuleTracer::global_expressions_used
    fn type_expression_tandem(&mut self) {
        // For each type T, compute the latest global expression E that T and
        // its predecessors refer to. Given the ordering rules on types and
        // global expressions in valid modules, we can do this with a single
        // forward scan of the type arena. The rules further imply that T can
        // only be referred to by expressions after E.
        let mut max_dep = Vec::with_capacity(self.module.types.len());
        let mut previous = None;
        for (_handle, ty) in self.module.types.iter() {
            previous = core::cmp::max(
                previous,
                match ty.inner {
                    crate::TypeInner::Array { size, .. }
                    | crate::TypeInner::BindingArray { size, .. } => match size {
                        crate::ArraySize::Constant(_) | crate::ArraySize::Dynamic => None,
                        crate::ArraySize::Pending(pending) => match pending {
                            crate::PendingArraySize::Expression(handle) => Some(handle),
                            crate::PendingArraySize::Override(handle) => {
                                self.module.overrides[handle].init
                            }
                        },
                    },
                    _ => None,
                },
            );
            max_dep.push(previous);
        }

        // Visit types and global expressions from youngest to oldest.
        //
        // The outer loop visits types. Before visiting each type, the inner
        // loop ensures that all global expressions that could possibly refer to
        // it have been visited. And since the inner loop stop at the latest
        // expression that the type could possibly refer to, we know that we
        // have previously visited any types that might refer to each expression
        // we visit.
        //
        // This lets us assume that any type or expression that is *not* marked
        // as used by the time we visit it is genuinely unused, and can be
        // ignored.
        let mut exprs = self.module.global_expressions.iter().rev().peekable();

        for ((ty_handle, ty), dep) in self.module.types.iter().zip(max_dep).rev() {
            while let Some((expr_handle, expr)) = exprs.next_if(|&(h, _)| Some(h) > dep) {
                if self.global_expressions_used.contains(expr_handle) {
                    self.as_const_expression().trace_expression(expr);
                }
            }
            if self.types_used.contains(ty_handle) {
                self.as_type().trace_type(ty);
            }
        }
        // Visit any remaining expressions.
        for (expr_handle, expr) in exprs {
            if self.global_expressions_used.contains(expr_handle) {
                self.as_const_expression().trace_expression(expr);
            }
        }
    }

    fn as_type(&mut self) -> types::TypeTracer {
        types::TypeTracer {
            overrides: &self.module.overrides,
            types_used: &mut self.types_used,
            expressions_used: &mut self.global_expressions_used,
            overrides_used: &mut self.overrides_used,
        }
    }

    fn as_const_expression(&mut self) -> expressions::ExpressionTracer {
        expressions::ExpressionTracer {
            constants: &self.module.constants,
            overrides: &self.module.overrides,
            expressions: &self.module.global_expressions,
            types_used: &mut self.types_used,
            constants_used: &mut self.constants_used,
            expressions_used: &mut self.global_expressions_used,
            overrides_used: &mut self.overrides_used,
            global_expressions_used: None,
        }
    }

    pub fn as_function<'tracer>(
        &'tracer mut self,
        function: &'tracer crate::Function,
    ) -> FunctionTracer<'tracer> {
        FunctionTracer {
            function,
            constants: &self.module.constants,
            overrides: &self.module.overrides,
            types_used: &mut self.types_used,
            constants_used: &mut self.constants_used,
            overrides_used: &mut self.overrides_used,
            global_expressions_used: &mut self.global_expressions_used,
            expressions_used: HandleSet::for_arena(&function.expressions),
        }
    }
}

struct ModuleMap {
    types: HandleMap<crate::Type>,
    constants: HandleMap<crate::Constant>,
    overrides: HandleMap<crate::Override>,
    global_expressions: HandleMap<crate::Expression>,
}

impl From<ModuleTracer<'_>> for ModuleMap {
    fn from(used: ModuleTracer) -> Self {
        ModuleMap {
            types: HandleMap::from_set(used.types_used),
            constants: HandleMap::from_set(used.constants_used),
            overrides: HandleMap::from_set(used.overrides_used),
            global_expressions: HandleMap::from_set(used.global_expressions_used),
        }
    }
}

impl ModuleMap {
    fn adjust_special_types(&self, special: &mut crate::SpecialTypes) {
        let crate::SpecialTypes {
            ref mut ray_desc,
            ref mut ray_intersection,
            ref mut ray_vertex_return,
            ref mut predeclared_types,
        } = *special;

        if let Some(ref mut ray_desc) = *ray_desc {
            self.types.adjust(ray_desc);
        }
        if let Some(ref mut ray_intersection) = *ray_intersection {
            self.types.adjust(ray_intersection);
        }

        if let Some(ref mut ray_vertex_return) = *ray_vertex_return {
            self.types.adjust(ray_vertex_return);
        }

        for handle in predeclared_types.values_mut() {
            self.types.adjust(handle);
        }
    }
}

struct FunctionMap {
    expressions: HandleMap<crate::Expression>,
}

impl From<FunctionTracer<'_>> for FunctionMap {
    fn from(used: FunctionTracer) -> Self {
        FunctionMap {
            expressions: HandleMap::from_set(used.expressions_used),
        }
    }
}

#[test]
fn type_expression_interdependence() {
    let mut module: crate::Module = Default::default();
    let u32 = module.types.insert(
        crate::Type {
            name: None,
            inner: crate::TypeInner::Scalar(crate::Scalar {
                kind: crate::ScalarKind::Uint,
                width: 4,
            }),
        },
        crate::Span::default(),
    );
    let expr = module.global_expressions.append(
        crate::Expression::Literal(crate::Literal::U32(0)),
        crate::Span::default(),
    );
    let type_needs_expression = |module: &mut crate::Module, handle| {
        module.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Array {
                    base: u32,
                    size: crate::ArraySize::Pending(crate::PendingArraySize::Expression(handle)),
                    stride: 4,
                },
            },
            crate::Span::default(),
        )
    };
    let expression_needs_type = |module: &mut crate::Module, handle| {
        module
            .global_expressions
            .append(crate::Expression::ZeroValue(handle), crate::Span::default())
    };
    let expression_needs_expression = |module: &mut crate::Module, handle| {
        module.global_expressions.append(
            crate::Expression::Load { pointer: handle },
            crate::Span::default(),
        )
    };
    let type_needs_type = |module: &mut crate::Module, handle| {
        module.types.insert(
            crate::Type {
                name: None,
                inner: crate::TypeInner::Array {
                    base: handle,
                    size: crate::ArraySize::Dynamic,
                    stride: 0,
                },
            },
            crate::Span::default(),
        )
    };
    let mut type_name_counter = 0;
    let mut type_needed = |module: &mut crate::Module, handle| {
        let name = Some(format!("type{}", type_name_counter));
        type_name_counter += 1;
        module.types.insert(
            crate::Type {
                name,
                inner: crate::TypeInner::Array {
                    base: handle,
                    size: crate::ArraySize::Dynamic,
                    stride: 0,
                },
            },
            crate::Span::default(),
        )
    };
    let mut override_name_counter = 0;
    let mut expression_needed = |module: &mut crate::Module, handle| {
        let name = Some(format!("override{}", override_name_counter));
        override_name_counter += 1;
        module.overrides.append(
            crate::Override {
                name,
                id: None,
                ty: u32,
                init: Some(handle),
            },
            crate::Span::default(),
        )
    };
    let cmp_modules = |mod0: &crate::Module, mod1: &crate::Module| {
        (mod0.types.iter().collect::<Vec<_>>() == mod1.types.iter().collect::<Vec<_>>())
            && (mod0.global_expressions.iter().collect::<Vec<_>>()
                == mod1.global_expressions.iter().collect::<Vec<_>>())
    };
    // borrow checker breaks without the tmp variables as of Rust 1.83.0
    let expr_end = type_needs_expression(&mut module, expr);
    let ty_trace = type_needs_type(&mut module, expr_end);
    let expr_init = expression_needs_type(&mut module, ty_trace);
    expression_needed(&mut module, expr_init);
    let ty_end = expression_needs_type(&mut module, u32);
    let expr_trace = expression_needs_expression(&mut module, ty_end);
    let ty_init = type_needs_expression(&mut module, expr_trace);
    type_needed(&mut module, ty_init);
    let untouched = module.clone();
    compact(&mut module);
    assert!(cmp_modules(&module, &untouched));
    let unused_expr = module.global_expressions.append(
        crate::Expression::Literal(crate::Literal::U32(1)),
        crate::Span::default(),
    );
    type_needs_expression(&mut module, unused_expr);
    assert!(!cmp_modules(&module, &untouched));
    compact(&mut module);
    assert!(cmp_modules(&module, &untouched));
}

#[test]
fn array_length_override() {
    let mut module: crate::Module = Default::default();
    let ty_bool = module.types.insert(
        crate::Type {
            name: None,
            inner: crate::TypeInner::Scalar(crate::Scalar::BOOL),
        },
        crate::Span::default(),
    );
    let ty_u32 = module.types.insert(
        crate::Type {
            name: None,
            inner: crate::TypeInner::Scalar(crate::Scalar::U32),
        },
        crate::Span::default(),
    );
    let one = module.global_expressions.append(
        crate::Expression::Literal(crate::Literal::U32(1)),
        crate::Span::default(),
    );
    let _unused_override = module.overrides.append(
        crate::Override {
            name: None,
            id: Some(40),
            ty: ty_u32,
            init: None,
        },
        crate::Span::default(),
    );
    let o = module.overrides.append(
        crate::Override {
            name: None,
            id: Some(42),
            ty: ty_u32,
            init: Some(one),
        },
        crate::Span::default(),
    );
    let _ty_array = module.types.insert(
        crate::Type {
            name: Some("array<bool, o>".to_string()),
            inner: crate::TypeInner::Array {
                base: ty_bool,
                size: crate::ArraySize::Pending(crate::PendingArraySize::Override(o)),
                stride: 4,
            },
        },
        crate::Span::default(),
    );

    let mut validator = super::valid::Validator::new(
        super::valid::ValidationFlags::all(),
        super::valid::Capabilities::all(),
    );

    assert!(validator.validate(&module).is_ok());
    compact(&mut module);
    assert!(validator.validate(&module).is_ok());
}

/// Test mutual references between types and expressions via override
/// lengths.
#[test]
fn array_length_override_mutual() {
    use crate::Expression as Ex;
    use crate::Scalar as Sc;
    use crate::TypeInner as Ti;

    let nowhere = crate::Span::default();
    let mut module = crate::Module::default();
    let ty_u32 = module.types.insert(
        crate::Type {
            name: None,
            inner: Ti::Scalar(Sc::U32),
        },
        nowhere,
    );

    // This type is only referred to by the override's init
    // expression, so if we visit that too early, this type will be
    // removed incorrectly.
    let ty_i32 = module.types.insert(
        crate::Type {
            name: None,
            inner: Ti::Scalar(Sc::I32),
        },
        nowhere,
    );

    // An override that the other override's init can refer to.
    let first_override = module.overrides.append(
        crate::Override {
            name: None, // so it is not considered used by definition
            id: Some(41),
            ty: ty_i32,
            init: None,
        },
        nowhere,
    );

    // Initializer expression for the override:
    //
    //     (first_override + 0) as u32
    //
    // The `first_override` makes it an override expression; the `0`
    // gets a use of `ty_i32` in there; and the `as` makes it match
    // the type of `second_override` without actually making
    // `second_override` point at `ty_i32` directly.
    let first_override_expr = module
        .global_expressions
        .append(Ex::Override(first_override), nowhere);
    let zero = module
        .global_expressions
        .append(Ex::ZeroValue(ty_i32), nowhere);
    let sum = module.global_expressions.append(
        Ex::Binary {
            op: crate::BinaryOperator::Add,
            left: first_override_expr,
            right: zero,
        },
        nowhere,
    );
    let init = module.global_expressions.append(
        Ex::As {
            expr: sum,
            kind: crate::ScalarKind::Uint,
            convert: None,
        },
        nowhere,
    );

    // Override that serves as the array's length.
    let second_override = module.overrides.append(
        crate::Override {
            name: None, // so it is not considered used by definition
            id: Some(42),
            ty: ty_u32,
            init: Some(init),
        },
        nowhere,
    );

    // Array type that uses the overload as its length.
    // Since this is named, it is considered used by definition.
    let _ty_array = module.types.insert(
        crate::Type {
            name: Some("delicious_array".to_string()),
            inner: Ti::Array {
                base: ty_u32,
                size: crate::ArraySize::Pending(crate::PendingArraySize::Override(second_override)),
                stride: 4,
            },
        },
        nowhere,
    );

    let mut validator = super::valid::Validator::new(
        super::valid::ValidationFlags::all(),
        super::valid::Capabilities::all(),
    );

    assert!(validator.validate(&module).is_ok());
    compact(&mut module);
    assert!(validator.validate(&module).is_ok());
}

#[test]
fn array_length_expression() {
    let mut module: crate::Module = Default::default();
    let ty_u32 = module.types.insert(
        crate::Type {
            name: None,
            inner: crate::TypeInner::Scalar(crate::Scalar::U32),
        },
        crate::Span::default(),
    );
    let _unused_zero = module.global_expressions.append(
        crate::Expression::Literal(crate::Literal::U32(0)),
        crate::Span::default(),
    );
    let one = module.global_expressions.append(
        crate::Expression::Literal(crate::Literal::U32(1)),
        crate::Span::default(),
    );
    let _ty_array = module.types.insert(
        crate::Type {
            name: Some("array<u32, 1>".to_string()),
            inner: crate::TypeInner::Array {
                base: ty_u32,
                size: crate::ArraySize::Pending(crate::PendingArraySize::Expression(one)),
                stride: 4,
            },
        },
        crate::Span::default(),
    );

    let mut validator = super::valid::Validator::new(
        super::valid::ValidationFlags::all(),
        super::valid::Capabilities::all(),
    );

    assert!(validator.validate(&module).is_ok());
    compact(&mut module);
    assert!(validator.validate(&module).is_ok());
}

#[test]
fn global_expression_override() {
    let mut module: crate::Module = Default::default();
    let ty_u32 = module.types.insert(
        crate::Type {
            name: None,
            inner: crate::TypeInner::Scalar(crate::Scalar::U32),
        },
        crate::Span::default(),
    );

    // This will only be retained if we trace the initializers
    // of overrides referred to by `Expression::Override`
    // in global expressions.
    let expr1 = module.global_expressions.append(
        crate::Expression::Literal(crate::Literal::U32(1)),
        crate::Span::default(),
    );

    // This will only be traced via a global `Expression::Override`.
    let o = module.overrides.append(
        crate::Override {
            name: None,
            id: Some(42),
            ty: ty_u32,
            init: Some(expr1),
        },
        crate::Span::default(),
    );

    // This is retained by _p.
    let expr2 = module
        .global_expressions
        .append(crate::Expression::Override(o), crate::Span::default());

    // Since this is named, it will be retained.
    let _p = module.overrides.append(
        crate::Override {
            name: Some("p".to_string()),
            id: None,
            ty: ty_u32,
            init: Some(expr2),
        },
        crate::Span::default(),
    );

    let mut validator = super::valid::Validator::new(
        super::valid::ValidationFlags::all(),
        super::valid::Capabilities::all(),
    );

    assert!(validator.validate(&module).is_ok());
    compact(&mut module);
    assert!(validator.validate(&module).is_ok());
}

#[test]
fn local_expression_override() {
    let mut module: crate::Module = Default::default();
    let ty_u32 = module.types.insert(
        crate::Type {
            name: None,
            inner: crate::TypeInner::Scalar(crate::Scalar::U32),
        },
        crate::Span::default(),
    );

    // This will only be retained if we trace the initializers
    // of overrides referred to by `Expression::Override` in a function.
    let expr1 = module.global_expressions.append(
        crate::Expression::Literal(crate::Literal::U32(1)),
        crate::Span::default(),
    );

    // This will be removed by compaction.
    let _unused_override = module.overrides.append(
        crate::Override {
            name: None,
            id: Some(41),
            ty: ty_u32,
            init: None,
        },
        crate::Span::default(),
    );

    // This will only be traced via an `Expression::Override` in a function.
    let o = module.overrides.append(
        crate::Override {
            name: None,
            id: Some(42),
            ty: ty_u32,
            init: Some(expr1),
        },
        crate::Span::default(),
    );

    let mut fun = crate::Function {
        result: Some(crate::FunctionResult {
            ty: ty_u32,
            binding: None,
        }),
        ..crate::Function::default()
    };

    // This is used by the `Return` statement.
    let o_expr = fun
        .expressions
        .append(crate::Expression::Override(o), crate::Span::default());
    fun.body.push(
        crate::Statement::Return {
            value: Some(o_expr),
        },
        crate::Span::default(),
    );

    module.functions.append(fun, crate::Span::default());

    let mut validator = super::valid::Validator::new(
        super::valid::ValidationFlags::all(),
        super::valid::Capabilities::all(),
    );

    assert!(validator.validate(&module).is_ok());
    compact(&mut module);
    assert!(validator.validate(&module).is_ok());
}

#[test]
fn unnamed_constant_type() {
    let mut module = crate::Module::default();
    let nowhere = crate::Span::default();

    // This type is used only by the unnamed constant.
    let ty_u32 = module.types.insert(
        crate::Type {
            name: None,
            inner: crate::TypeInner::Scalar(crate::Scalar::U32),
        },
        nowhere,
    );

    // This type is used by the named constant.
    let ty_vec_u32 = module.types.insert(
        crate::Type {
            name: None,
            inner: crate::TypeInner::Vector {
                size: crate::VectorSize::Bi,
                scalar: crate::Scalar::U32,
            },
        },
        nowhere,
    );

    let unnamed_init = module
        .global_expressions
        .append(crate::Expression::Literal(crate::Literal::U32(0)), nowhere);

    let unnamed_constant = module.constants.append(
        crate::Constant {
            name: None,
            ty: ty_u32,
            init: unnamed_init,
        },
        nowhere,
    );

    // The named constant is initialized using a Splat expression, to
    // give the named constant a type distinct from the unnamed
    // constant's.
    let unnamed_constant_expr = module
        .global_expressions
        .append(crate::Expression::Constant(unnamed_constant), nowhere);
    let named_init = module.global_expressions.append(
        crate::Expression::Splat {
            size: crate::VectorSize::Bi,
            value: unnamed_constant_expr,
        },
        nowhere,
    );

    let _named_constant = module.constants.append(
        crate::Constant {
            name: Some("totally_named".to_string()),
            ty: ty_vec_u32,
            init: named_init,
        },
        nowhere,
    );

    let mut validator = super::valid::Validator::new(
        super::valid::ValidationFlags::all(),
        super::valid::Capabilities::all(),
    );

    assert!(validator.validate(&module).is_ok());
    compact(&mut module);
    assert!(validator.validate(&module).is_ok());
}

#[test]
fn unnamed_override_type() {
    let mut module = crate::Module::default();
    let nowhere = crate::Span::default();

    // This type is used only by the unnamed override.
    let ty_u32 = module.types.insert(
        crate::Type {
            name: None,
            inner: crate::TypeInner::Scalar(crate::Scalar::U32),
        },
        nowhere,
    );

    // This type is used by the named override.
    let ty_i32 = module.types.insert(
        crate::Type {
            name: None,
            inner: crate::TypeInner::Scalar(crate::Scalar::I32),
        },
        nowhere,
    );

    let unnamed_init = module
        .global_expressions
        .append(crate::Expression::Literal(crate::Literal::U32(0)), nowhere);

    let unnamed_override = module.overrides.append(
        crate::Override {
            name: None,
            id: Some(42),
            ty: ty_u32,
            init: Some(unnamed_init),
        },
        nowhere,
    );

    // The named override is initialized using a Splat expression, to
    // give the named override a type distinct from the unnamed
    // override's.
    let unnamed_override_expr = module
        .global_expressions
        .append(crate::Expression::Override(unnamed_override), nowhere);
    let named_init = module.global_expressions.append(
        crate::Expression::As {
            expr: unnamed_override_expr,
            kind: crate::ScalarKind::Sint,
            convert: None,
        },
        nowhere,
    );

    let _named_override = module.overrides.append(
        crate::Override {
            name: Some("totally_named".to_string()),
            id: None,
            ty: ty_i32,
            init: Some(named_init),
        },
        nowhere,
    );

    let mut validator = super::valid::Validator::new(
        super::valid::ValidationFlags::all(),
        super::valid::Capabilities::all(),
    );

    assert!(validator.validate(&module).is_ok());
    compact(&mut module);
    assert!(validator.validate(&module).is_ok());
}
