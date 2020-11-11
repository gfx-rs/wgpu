use crate::arena::Handle;

use super::flow::*;
use super::*;

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
        default: BlockId,
        /// Tuples of (literal, target block)
        targets: Vec<(i32, BlockId)>,
    },
    /// Fragment shader discard
    Kill,
    ///
    Unreachable,
}

impl<I: Iterator<Item = u32>> super::Parser<I> {
    pub fn parse_function(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Function, inst.op)?;
        inst.expect(5)?;
        let result_type = self.next()?;
        let fun_id = self.next()?;
        let _fun_control = self.next()?;
        let fun_type = self.next()?;

        let mut fun = {
            let ft = self.lookup_function_type.lookup(fun_type)?;
            if ft.return_type_id != result_type {
                return Err(Error::WrongFunctionResultType(result_type));
            }
            crate::Function {
                name: self.future_decor.remove(&fun_id).and_then(|dec| dec.name),
                arguments: Vec::with_capacity(ft.parameter_type_ids.len()),
                return_type: if self.lookup_void_type.contains(&result_type) {
                    None
                } else {
                    Some(self.lookup_type.lookup(result_type)?.handle)
                },
                global_usage: Vec::new(),
                local_variables: Arena::new(),
                expressions: self.make_expression_storage(),
                body: Vec::new(),
            }
        };

        // read parameters
        for i in 0..fun.arguments.capacity() {
            match self.next_inst()? {
                Instruction {
                    op: spirv::Op::FunctionParameter,
                    wc: 3,
                } => {
                    let type_id = self.next()?;
                    let id = self.next()?;
                    let handle = fun
                        .expressions
                        .append(crate::Expression::FunctionArgument(i as u32));
                    self.lookup_expression
                        .insert(id, LookupExpression { type_id, handle });
                    //Note: we redo the lookup in order to work around `self` borrowing

                    if type_id
                        != self
                            .lookup_function_type
                            .lookup(fun_type)?
                            .parameter_type_ids[i]
                    {
                        return Err(Error::WrongFunctionArgumentType(type_id));
                    }
                    let ty = self.lookup_type.lookup(type_id)?.handle;
                    fun.arguments
                        .push(crate::FunctionArgument { name: None, ty });
                }
                Instruction { op, .. } => return Err(Error::InvalidParameter(op)),
            }
        }

        // Read body
        let mut local_function_calls = FastHashMap::default();
        let mut flow_graph = FlowGraph::new();

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
                        &mut fun.expressions,
                        &mut fun.local_variables,
                        &module.types,
                        &module.constants,
                        &module.global_variables,
                        &mut local_function_calls,
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
        fun.body = flow_graph.to_naga()?;

        // done
        fun.fill_global_use(&module.global_variables);

        let source = match self.lookup_entry_point.remove(&fun_id) {
            Some(ep) => {
                module.entry_points.insert(
                    (ep.stage, ep.name.clone()),
                    crate::EntryPoint {
                        early_depth_test: ep.early_depth_test,
                        workgroup_size: ep.workgroup_size,
                        function: fun,
                    },
                );
                DeferredSource::EntryPoint(ep.stage, ep.name)
            }
            None => {
                let handle = module.functions.append(fun);
                self.lookup_function.insert(fun_id, handle);
                DeferredSource::Function(handle)
            }
        };

        if let Some(ref prefix) = self.options.flow_graph_dump_prefix {
            let dump = flow_graph.to_graphviz().unwrap_or_default();
            let suffix = match source {
                DeferredSource::EntryPoint(stage, ref name) => {
                    format!("flow.{:?}-{}.dot", stage, name)
                }
                DeferredSource::Function(handle) => format!("flow.Fun-{}.dot", handle.index()),
            };
            let _ = std::fs::write(prefix.join(suffix), dump);
        }

        for (expr_handle, dst_id) in local_function_calls {
            self.deferred_function_calls.push(DeferredFunctionCall {
                source: source.clone(),
                expr_handle,
                dst_id,
            });
        }

        self.lookup_expression.clear();
        self.lookup_sampled_image.clear();
        Ok(())
    }
}
