use crate::arena::{Arena, Handle};

use super::{flow::*, Error, Instruction, LookupExpression, LookupHelper as _};
use crate::front::Emitter;

pub type BlockId = u32;

#[derive(Copy, Clone, Debug)]
pub struct MergeInstruction {
    pub merge_block_id: BlockId,
    pub continue_block_id: Option<BlockId>,
}
/// Terminator instruction of a SPIR-V's block.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum Terminator {
    ///
    Return {
        value: Option<Handle<crate::Expression>>,
    },
    ///
    Branch { target_id: BlockId },
    ///
    BranchConditional {
        condition: Handle<crate::Expression>,
        true_id: BlockId,
        false_id: BlockId,
    },
    ///
    /// switch(SELECTOR) {
    ///  case TARGET_LITERAL#: {
    ///    TARGET_BLOCK#
    ///  }
    ///  default: {
    ///    DEFAULT
    ///  }
    /// }
    Switch {
        ///
        selector: Handle<crate::Expression>,
        /// Default block of the switch case.
        default_id: BlockId,
        /// Tuples of (literal, target block)
        targets: Vec<(i32, BlockId)>,
    },
    /// Fragment shader discard
    Kill,
    ///
    Unreachable,
}

impl<I: Iterator<Item = u32>> super::Parser<I> {
    // Registers a function call. It will generate a dummy handle to call, which
    // gets resolved after all the functions are processed.
    pub(super) fn add_call(
        &mut self,
        from: spirv::Word,
        to: spirv::Word,
    ) -> Handle<crate::Function> {
        let dummy_handle = self
            .dummy_functions
            .append(crate::Function::default(), Default::default());
        self.deferred_function_calls.push(to);
        self.function_call_graph.add_edge(from, to, ());
        dummy_handle
    }

    pub(super) fn parse_function(&mut self, module: &mut crate::Module) -> Result<(), Error> {
        let start = self.data_offset;
        self.lookup_expression.clear();
        self.lookup_load_override.clear();
        self.lookup_sampled_image.clear();

        let result_type_id = self.next()?;
        let fun_id = self.next()?;
        let _fun_control = self.next()?;
        let fun_type_id = self.next()?;

        let mut fun = {
            let ft = self.lookup_function_type.lookup(fun_type_id)?;
            if ft.return_type_id != result_type_id {
                return Err(Error::WrongFunctionResultType(result_type_id));
            }
            crate::Function {
                name: self.future_decor.remove(&fun_id).and_then(|dec| dec.name),
                arguments: Vec::with_capacity(ft.parameter_type_ids.len()),
                result: if self.lookup_void_type == Some(result_type_id) {
                    None
                } else {
                    let lookup_result_ty = self.lookup_type.lookup(result_type_id)?;
                    Some(crate::FunctionResult {
                        ty: lookup_result_ty.handle,
                        binding: None,
                    })
                },
                local_variables: Arena::new(),
                expressions: self
                    .make_expression_storage(&module.global_variables, &module.constants),
                named_expressions: crate::FastHashMap::default(),
                body: crate::Block::new(),
            }
        };

        // read parameters
        for i in 0..fun.arguments.capacity() {
            let start = self.data_offset;
            match self.next_inst()? {
                Instruction {
                    op: spirv::Op::FunctionParameter,
                    wc: 3,
                } => {
                    let type_id = self.next()?;
                    let id = self.next()?;
                    let handle = fun.expressions.append(
                        crate::Expression::FunctionArgument(i as u32),
                        self.span_from(start),
                    );
                    self.lookup_expression
                        .insert(id, LookupExpression { handle, type_id });
                    //Note: we redo the lookup in order to work around `self` borrowing

                    if type_id
                        != self
                            .lookup_function_type
                            .lookup(fun_type_id)?
                            .parameter_type_ids[i]
                    {
                        return Err(Error::WrongFunctionArgumentType(type_id));
                    }
                    let ty = self.lookup_type.lookup(type_id)?.handle;
                    let decor = self.future_decor.remove(&id).unwrap_or_default();
                    fun.arguments.push(crate::FunctionArgument {
                        name: decor.name,
                        ty,
                        binding: None,
                    });
                }
                Instruction { op, .. } => return Err(Error::InvalidParameter(op)),
            }
        }

        // Read body
        self.function_call_graph.add_node(fun_id);
        let mut flow_graph = FlowGraph::new();
        let mut parameters_sampling =
            vec![super::image::SamplingFlags::empty(); fun.arguments.len()];

        // Scan the blocks and add them as nodes
        loop {
            let fun_inst = self.next_inst()?;
            log::debug!("{:?}", fun_inst.op);
            match fun_inst.op {
                spirv::Op::Label => {
                    // Read the label ID
                    fun_inst.expect(2)?;
                    let block_id = self.next()?;

                    let node = self.next_block(
                        block_id,
                        fun_id,
                        &mut fun.expressions,
                        &mut fun.local_variables,
                        &mut module.constants,
                        &module.types,
                        &module.global_variables,
                        &fun.arguments,
                        &mut parameters_sampling,
                    )?;

                    flow_graph.add_node(node);
                }
                spirv::Op::FunctionEnd => {
                    fun_inst.expect(1)?;
                    break;
                }
                _ => {
                    return Err(Error::UnsupportedInstruction(self.state, fun_inst.op));
                }
            }
        }

        flow_graph.classify();
        flow_graph.remove_phi_instructions(&self.lookup_expression);

        if let Some(ref prefix) = self.options.flow_graph_dump_prefix {
            let dump = flow_graph.to_graphviz().unwrap_or_default();
            let dump_suffix = match self.lookup_entry_point.get(&fun_id) {
                Some(ep) => format!("flow.{:?}-{}.dot", ep.stage, ep.name),
                None => format!("flow.Fun-{}.dot", module.functions.len()),
            };
            let dest = prefix.join(dump_suffix);
            if let Err(e) = std::fs::write(&dest, dump) {
                log::error!("Unable to dump the flow graph into {:?}: {}", dest, e);
            }
        }

        fun.body = flow_graph.convert_to_naga()?;

        // done
        let fun_handle = module.functions.append(fun, self.span_from_with_op(start));
        self.lookup_function.insert(
            fun_id,
            super::LookupFunction {
                handle: fun_handle,
                parameters_sampling,
            },
        );

        if let Some(ep) = self.lookup_entry_point.remove(&fun_id) {
            // create a wrapping function
            let mut function = crate::Function {
                name: Some(format!("{}_wrap", ep.name)),
                arguments: Vec::new(),
                result: None,
                local_variables: Arena::new(),
                expressions: Arena::new(),
                named_expressions: crate::FastHashMap::default(),
                body: crate::Block::new(),
            };

            // 1. copy the inputs from arguments to privates
            for &v_id in ep.variable_ids.iter() {
                let lvar = self.lookup_variable.lookup(v_id)?;
                if let super::Variable::Input(ref arg) = lvar.inner {
                    let span = module.global_variables.get_span(lvar.handle).clone();
                    let arg_expr = function.expressions.append(
                        crate::Expression::FunctionArgument(function.arguments.len() as u32),
                        span.clone(),
                    );
                    let load_expr = if arg.ty == module.global_variables[lvar.handle].ty {
                        arg_expr
                    } else {
                        // The only case where the type is different is if we need to treat
                        // unsigned integer as signed.
                        let mut emitter = Emitter::default();
                        emitter.start(&function.expressions);
                        let handle = function.expressions.append(
                            crate::Expression::As {
                                expr: arg_expr,
                                kind: crate::ScalarKind::Sint,
                                convert: Some(4),
                            },
                            span.clone(),
                        );
                        function.body.extend(emitter.finish(&function.expressions));
                        handle
                    };
                    function.body.push(
                        crate::Statement::Store {
                            pointer: function.expressions.append(
                                crate::Expression::GlobalVariable(lvar.handle),
                                span.clone(),
                            ),
                            value: load_expr,
                        },
                        span,
                    );

                    let mut arg = arg.clone();
                    if ep.stage == crate::ShaderStage::Fragment {
                        if let Some(crate::Binding::Location {
                            interpolation: ref mut interpolation @ None,
                            ..
                        }) = arg.binding
                        {
                            *interpolation = Some(crate::Interpolation::Perspective);
                            // default
                        }
                    }
                    function.arguments.push(arg);
                }
            }
            // 2. call the wrapped function
            let fake_id = !(module.entry_points.len() as u32); // doesn't matter, as long as it's not a collision
            let dummy_handle = self.add_call(fake_id, fun_id);
            function.body.push(
                crate::Statement::Call {
                    function: dummy_handle,
                    arguments: Vec::new(),
                    result: None,
                },
                crate::Span::Unknown,
            );

            // 3. copy the outputs from privates to the result
            let mut members = Vec::new();
            let mut components = Vec::new();
            for &v_id in ep.variable_ids.iter() {
                let lvar = self.lookup_variable.lookup(v_id)?;
                if let super::Variable::Output(ref result) = lvar.inner {
                    let span = module.global_variables.get_span(lvar.handle).clone();
                    let expr_handle = function
                        .expressions
                        .append(crate::Expression::GlobalVariable(lvar.handle), span.clone());
                    match module.types[result.ty].inner {
                        crate::TypeInner::Struct {
                            members: ref sub_members,
                            ..
                        } => {
                            for (index, sm) in sub_members.iter().enumerate() {
                                if sm.binding.is_none() {
                                    // unrecognized binding, skip
                                    continue;
                                }
                                members.push(sm.clone());
                                components.push(function.expressions.append(
                                    crate::Expression::AccessIndex {
                                        base: expr_handle,
                                        index: index as u32,
                                    },
                                    span.clone(),
                                ));
                            }
                        }
                        _ => {
                            members.push(crate::StructMember {
                                name: None,
                                ty: result.ty,
                                binding: result.binding.clone(),
                                offset: 0,
                            });
                            // populate just the globals first, then do `Load` in a
                            // separate step, so that we can get a range.
                            components.push(expr_handle);
                        }
                    }
                }
            }

            for (member_index, member) in members.iter().enumerate() {
                match member.binding {
                    Some(crate::Binding::BuiltIn(crate::BuiltIn::Position))
                        if self.options.adjust_coordinate_space =>
                    {
                        let mut emitter = Emitter::default();
                        emitter.start(&function.expressions);
                        let global_expr = components[member_index];
                        let span = function.expressions.get_span(global_expr).clone();
                        let access_expr = function.expressions.append(
                            crate::Expression::AccessIndex {
                                base: global_expr,
                                index: 1,
                            },
                            span.clone(),
                        );
                        let load_expr = function.expressions.append(
                            crate::Expression::Load {
                                pointer: access_expr,
                            },
                            span.clone(),
                        );
                        let neg_expr = function.expressions.append(
                            crate::Expression::Unary {
                                op: crate::UnaryOperator::Negate,
                                expr: load_expr,
                            },
                            span.clone(),
                        );
                        function.body.extend(emitter.finish(&function.expressions));
                        function.body.push(
                            crate::Statement::Store {
                                pointer: access_expr,
                                value: neg_expr,
                            },
                            span,
                        );
                    }
                    _ => {}
                }
            }

            let mut emitter = Emitter::default();
            emitter.start(&function.expressions);
            for component in components.iter_mut() {
                let load_expr = crate::Expression::Load {
                    pointer: *component,
                };
                let span = function.expressions.get_span(*component).clone();
                *component = function.expressions.append(load_expr, span);
            }

            match &members[..] {
                [] => {}
                [member] => {
                    function.body.extend(emitter.finish(&function.expressions));
                    let span = function.expressions.get_span(components[0]).clone();
                    function.body.push(
                        crate::Statement::Return {
                            value: components.first().cloned(),
                        },
                        span,
                    );
                    function.result = Some(crate::FunctionResult {
                        ty: member.ty,
                        binding: member.binding.clone(),
                    });
                }
                _ => {
                    let span = crate::Span::total_span(
                        components.iter().map(|h| function.expressions.get_span(*h)),
                    );
                    let ty = module.types.append(
                        crate::Type {
                            name: None,
                            inner: crate::TypeInner::Struct {
                                top_level: false,
                                members,
                                span: 0xFFFF, // shouldn't matter
                            },
                        },
                        span.clone(),
                    );
                    let result_expr = function
                        .expressions
                        .append(crate::Expression::Compose { ty, components }, span.clone());
                    function.body.extend(emitter.finish(&function.expressions));
                    function.body.push(
                        crate::Statement::Return {
                            value: Some(result_expr),
                        },
                        span,
                    );
                    function.result = Some(crate::FunctionResult { ty, binding: None });
                }
            }

            module.entry_points.push(crate::EntryPoint {
                name: ep.name,
                stage: ep.stage,
                early_depth_test: ep.early_depth_test,
                workgroup_size: ep.workgroup_size,
                function,
            });
        }

        module.apply_common_default_interpolation();

        Ok(())
    }
}
