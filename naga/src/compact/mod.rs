mod expressions;
mod functions;
mod handle_set_map;
mod statements;
mod types;

use crate::arena::HandleSet;
use crate::{arena, compact::functions::FunctionTracer};
use handle_set_map::HandleMap;

/// Remove unused types, expressions, and constants from `module`.
///
/// Assuming that all globals, named constants, special types,
/// functions and entry points in `module` are used, determine which
/// types, constants, and expressions (both function-local and global
/// constant expressions) are actually used, and remove the rest,
/// adjusting all handles as necessary. The result should be a module
/// functionally identical to the original.
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
/// If `module` has not passed validation, this may panic.
pub fn compact(module: &mut crate::Module) {
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
    module_tracer.trace_special_types(&module.special_types);

    // We treat all named constants as used by definition.
    for (handle, constant) in module.constants.iter() {
        if constant.name.is_some() {
            module_tracer.constants_used.insert(handle);
            module_tracer.global_expressions_used.insert(constant.init);
        }
    }

    // We treat all overrides as used by definition.
    for (_, override_) in module.overrides.iter() {
        module_tracer.types_used.insert(override_.ty);
        if let Some(init) = override_.init {
            module_tracer.global_expressions_used.insert(init);
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

    // Constants' initializers are taken care of already, because
    // expression tracing sees through constants. But we still need to
    // note type usage.
    for (handle, constant) in module.constants.iter() {
        if module_tracer.constants_used.contains(handle) {
            module_tracer.types_used.insert(constant.ty);
        }
    }

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

    // Adjust override types and initializers.
    log::trace!("adjusting overrides");
    for (_, override_) in module.overrides.iter_mut() {
        module_map.types.adjust(&mut override_.ty);
        if let Some(init) = override_.init.as_mut() {
            module_map.global_expressions.adjust(init);
        }
    }

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
    global_expressions_used: HandleSet<crate::Expression>,
}

impl<'module> ModuleTracer<'module> {
    fn new(module: &'module crate::Module) -> Self {
        Self {
            module,
            types_used: HandleSet::for_arena(&module.types),
            constants_used: HandleSet::for_arena(&module.constants),
            global_expressions_used: HandleSet::for_arena(&module.global_expressions),
        }
    }

    fn trace_special_types(&mut self, special_types: &crate::SpecialTypes) {
        let crate::SpecialTypes {
            ref ray_desc,
            ref ray_intersection,
            ref predeclared_types,
        } = *special_types;

        if let Some(ray_desc) = *ray_desc {
            self.types_used.insert(ray_desc);
        }
        if let Some(ray_intersection) = *ray_intersection {
            self.types_used.insert(ray_intersection);
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
            previous = std::cmp::max(
                previous,
                match ty.inner {
                    crate::TypeInner::Array {
                        base: _,
                        size: crate::ArraySize::Pending(crate::PendingArraySize::Expression(expr)),
                        stride: _,
                    }
                    | crate::TypeInner::BindingArray {
                        base: _,
                        size: crate::ArraySize::Pending(crate::PendingArraySize::Expression(expr)),
                    } => Some(expr),
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
        for ((ty_handle, ty), dep) in self.module.types.iter().rev().zip(max_dep.iter().rev()) {
            while let Some((expr_handle, expr)) = exprs.next_if(|&(h, _)| Some(h) > *dep) {
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
            types_used: &mut self.types_used,
            expressions_used: &mut self.global_expressions_used,
        }
    }

    fn as_const_expression(&mut self) -> expressions::ExpressionTracer {
        expressions::ExpressionTracer {
            expressions: &self.module.global_expressions,
            constants: &self.module.constants,
            types_used: &mut self.types_used,
            constants_used: &mut self.constants_used,
            expressions_used: &mut self.global_expressions_used,
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
            types_used: &mut self.types_used,
            constants_used: &mut self.constants_used,
            global_expressions_used: &mut self.global_expressions_used,
            expressions_used: HandleSet::for_arena(&function.expressions),
        }
    }
}

struct ModuleMap {
    types: HandleMap<crate::Type>,
    constants: HandleMap<crate::Constant>,
    global_expressions: HandleMap<crate::Expression>,
}

impl From<ModuleTracer<'_>> for ModuleMap {
    fn from(used: ModuleTracer) -> Self {
        ModuleMap {
            types: HandleMap::from_set(used.types_used),
            constants: HandleMap::from_set(used.constants_used),
            global_expressions: HandleMap::from_set(used.global_expressions_used),
        }
    }
}

impl ModuleMap {
    fn adjust_special_types(&self, special: &mut crate::SpecialTypes) {
        let crate::SpecialTypes {
            ref mut ray_desc,
            ref mut ray_intersection,
            ref mut predeclared_types,
        } = *special;

        if let Some(ref mut ray_desc) = *ray_desc {
            self.types.adjust(ray_desc);
        }
        if let Some(ref mut ray_intersection) = *ray_intersection {
            self.types.adjust(ray_intersection);
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
