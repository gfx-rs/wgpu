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

pub fn parse_function<I: Iterator<Item = u32>>(
    parser: &mut super::Parser<I>,
    inst: Instruction,
    module: &mut crate::Module,
) -> Result<(), Error> {
    parser.switch(ModuleState::Function, inst.op)?;
    inst.expect(5)?;
    let result_type = parser.next()?;
    let fun_id = parser.next()?;
    let _fun_control = parser.next()?;
    let fun_type = parser.next()?;
    let mut fun = {
        let ft = parser.lookup_function_type.lookup(fun_type)?;
        if ft.return_type_id != result_type {
            return Err(Error::WrongFunctionResultType(result_type));
        }
        crate::Function {
            name: parser.future_decor.remove(&fun_id).and_then(|dec| dec.name),
            parameter_types: Vec::with_capacity(ft.parameter_type_ids.len()),
            return_type: if parser.lookup_void_type.contains(&result_type) {
                None
            } else {
                Some(parser.lookup_type.lookup(result_type)?.handle)
            },
            global_usage: Vec::new(),
            local_variables: Arena::new(),
            expressions: parser.make_expression_storage(),
            body: Vec::new(),
        }
    };

    // read parameters
    for i in 0..fun.parameter_types.capacity() {
        match parser.next_inst()? {
            Instruction {
                op: spirv::Op::FunctionParameter,
                wc: 3,
            } => {
                let type_id = parser.next()?;
                let _id = parser.next()?;
                //Note: we redo the lookup in order to work around `parser` borrowing
                if type_id
                    != parser
                        .lookup_function_type
                        .lookup(fun_type)?
                        .parameter_type_ids[i]
                {
                    return Err(Error::WrongFunctionParameterType(type_id));
                }
                let ty = parser.lookup_type.lookup(type_id)?.handle;
                fun.parameter_types.push(ty);
            }
            Instruction { op, .. } => return Err(Error::InvalidParameter(op)),
        }
    }

    // Read body
    let mut local_function_calls = FastHashMap::default();
    let mut flow_graph = FlowGraph::new();

    // Scan the blocks and add them as nodes
    loop {
        let fun_inst = parser.next_inst()?;
        log::debug!("\t\t{:?}", fun_inst.op);
        match fun_inst.op {
            spirv::Op::Label => {
                // Read the label ID
                fun_inst.expect(2)?;
                let block_id = parser.next()?;

                let node = parser.next_block(
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
                return Err(Error::UnsupportedInstruction(parser.state, fun_inst.op));
            }
        }
    }

    flow_graph.classify();
    fun.body = flow_graph.to_naga()?;

    // done
    fun.global_usage =
        crate::GlobalUse::scan(&fun.expressions, &fun.body, &module.global_variables);
    let handle = module.functions.append(fun);
    for (expr_handle, dst_id) in local_function_calls {
        parser.deferred_function_calls.push(DeferredFunctionCall {
            source_handle: handle,
            expr_handle,
            dst_id,
        });
    }

    parser.lookup_function.insert(fun_id, handle);
    parser.lookup_expression.clear();
    parser.lookup_sampled_image.clear();
    Ok(())
}
