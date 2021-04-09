#![allow(dead_code)]

use super::error::Error;
///! see https://en.wikipedia.org/wiki/Control-flow_graph
///! see https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_structuredcontrolflow_a_structured_control_flow
use super::{
    function::{BlockId, MergeInstruction, Terminator},
    LookupExpression, PhiInstruction,
};

use crate::FastHashMap;

use petgraph::{
    algo::has_path_connecting,
    graph::{node_index, NodeIndex},
    visit::EdgeRef,
    Directed, Direction,
};

use std::fmt::Write;

/// Index of a block node in the `ControlFlowGraph`.
type BlockNodeIndex = NodeIndex<u32>;

/// Internal representation of a CFG constisting of function's basic blocks.
type ControlFlowGraph = petgraph::Graph<ControlFlowNode, ControlFlowEdgeType, Directed, u32>;

/// Control flow graph (CFG) containing relationships between blocks.
pub(super) struct FlowGraph {
    ///
    flow: ControlFlowGraph,

    /// Block ID to Node index mapping. Internal helper to speed up the classification.
    block_to_node: FastHashMap<BlockId, BlockNodeIndex>,
}

impl FlowGraph {
    /// Creates empty flow graph.
    pub(super) fn new() -> Self {
        Self {
            flow: ControlFlowGraph::default(),
            block_to_node: FastHashMap::default(),
        }
    }

    /// Add a control flow node.
    pub(super) fn add_node(&mut self, node: ControlFlowNode) {
        let block_id = node.id;
        let node_index = self.flow.add_node(node);
        self.block_to_node.insert(block_id, node_index);
    }

    ///
    /// 1. Creates edges in the CFG.
    /// 2. Classifies types of blocks and edges in the CFG.
    pub(super) fn classify(&mut self) {
        let block_to_node = &mut self.block_to_node;

        // 1.
        // Add all edges
        // Classify Nodes as one of [Header, Loop, Kill, Return]
        for source_node_index in self.flow.node_indices() {
            // Merge edges
            if let Some(merge) = self.flow[source_node_index].merge {
                let merge_block_index = block_to_node[&merge.merge_block_id];

                self.flow[source_node_index].ty = Some(ControlFlowNodeType::Header);
                self.flow[merge_block_index].ty = Some(ControlFlowNodeType::Merge);
                self.flow.add_edge(
                    source_node_index,
                    merge_block_index,
                    ControlFlowEdgeType::ForwardMerge,
                );

                if let Some(continue_block_id) = merge.continue_block_id {
                    let continue_block_index = block_to_node[&continue_block_id];

                    self.flow[source_node_index].ty = Some(ControlFlowNodeType::Loop);
                    self.flow.add_edge(
                        source_node_index,
                        continue_block_index,
                        ControlFlowEdgeType::ForwardContinue,
                    );

                    // Back edge
                    self.flow[continue_block_index].ty = Some(ControlFlowNodeType::Back);
                    self.flow.add_edge(
                        continue_block_index,
                        source_node_index,
                        ControlFlowEdgeType::Back,
                    );
                }
            }

            // Branch Edges
            let terminator = self.flow[source_node_index].terminator.clone();
            match terminator {
                Terminator::Branch { target_id } => {
                    let target_node_index = block_to_node[&target_id];

                    if self.flow[source_node_index].ty != Some(ControlFlowNodeType::Back) {
                        self.flow.add_edge(
                            source_node_index,
                            target_node_index,
                            ControlFlowEdgeType::Forward,
                        );
                    }
                }
                Terminator::BranchConditional {
                    true_id, false_id, ..
                } => {
                    let true_node_index = block_to_node[&true_id];
                    let false_node_index = block_to_node[&false_id];

                    self.flow.add_edge(
                        source_node_index,
                        true_node_index,
                        ControlFlowEdgeType::IfTrue,
                    );
                    self.flow.add_edge(
                        source_node_index,
                        false_node_index,
                        ControlFlowEdgeType::IfFalse,
                    );
                }
                Terminator::Switch {
                    selector: _,
                    default,
                    ref targets,
                } => {
                    let default_node_index = block_to_node[&default];

                    self.flow.add_edge(
                        source_node_index,
                        default_node_index,
                        ControlFlowEdgeType::Forward,
                    );

                    for &(_, target_block_id) in targets.iter() {
                        let target_node_index = block_to_node[&target_block_id];

                        self.flow.add_edge(
                            source_node_index,
                            target_node_index,
                            ControlFlowEdgeType::Forward,
                        );
                    }
                }
                Terminator::Return { .. } => {
                    self.flow[source_node_index].ty = Some(ControlFlowNodeType::Return)
                }
                Terminator::Kill => {
                    self.flow[source_node_index].ty = Some(ControlFlowNodeType::Kill)
                }
                _ => {}
            };
        }

        // 2.
        // Classify Nodes/Edges as one of [Break, Continue, Back]
        for edge_index in self.flow.edge_indices() {
            let (node_source_index, node_target_index) =
                self.flow.edge_endpoints(edge_index).unwrap();

            if self.flow[node_source_index].ty == Some(ControlFlowNodeType::Header)
                || self.flow[node_source_index].ty == Some(ControlFlowNodeType::Loop)
            {
                continue;
            }

            let mut target_incoming_edges = self
                .flow
                .neighbors_directed(node_target_index, Direction::Incoming)
                .detach();
            while let Some((incoming_edge, incoming_source)) =
                target_incoming_edges.next(&self.flow)
            {
                // Loop continue
                if self.flow[incoming_edge] == ControlFlowEdgeType::ForwardContinue {
                    self.flow[node_source_index].ty = Some(ControlFlowNodeType::Continue);
                    self.flow[edge_index] = ControlFlowEdgeType::LoopContinue;
                }
                // Loop break
                if self.flow[incoming_source].ty == Some(ControlFlowNodeType::Loop)
                    && self.flow[incoming_edge] == ControlFlowEdgeType::ForwardMerge
                {
                    self.flow[node_source_index].ty = Some(ControlFlowNodeType::Break);
                    self.flow[edge_index] = ControlFlowEdgeType::LoopBreak;
                }
            }
        }
    }

    /// Removes OpPhi instructions from the control flow graph and turns them into ordinary variables.
    ///
    /// Phi instructions are not supported inside Naga nor do they exist as instructions on CPUs. It is necessary
    /// to remove them and turn into ordinary variables before converting to Naga's IR and shader code.
    pub(super) fn remove_phi_instructions(
        &mut self,
        lookup_expression: &FastHashMap<spirv::Word, LookupExpression>,
    ) {
        for node_index in self.flow.node_indices() {
            let phis = std::mem::replace(&mut self.flow[node_index].phis, Vec::new());
            for phi in phis.iter() {
                for &(variable_id, parent_id) in phi.variables.iter() {
                    let variable = &lookup_expression[&variable_id];
                    let parent_node = &mut self.flow[self.block_to_node[&parent_id]];

                    parent_node.block.push(crate::Statement::Store {
                        pointer: phi.pointer,
                        value: variable.handle,
                    });
                }
            }
            self.flow[node_index].phis = phis;
        }
    }

    /// Traverses the flow graph and returns a list of Naga's statements.
    pub(super) fn to_naga(&self) -> Result<crate::Block, Error> {
        self.naga_traverse(node_index(0), None)
    }

    fn naga_traverse(
        &self,
        node_index: BlockNodeIndex,
        stop_node_index: Option<BlockNodeIndex>,
    ) -> Result<crate::Block, Error> {
        if stop_node_index == Some(node_index) {
            return Ok(vec![]);
        }

        let node = &self.flow[node_index];

        match node.ty {
            Some(ControlFlowNodeType::Header) => match node.terminator {
                Terminator::BranchConditional {
                    condition,
                    true_id,
                    false_id,
                } => {
                    let true_node_index = self.block_to_node[&true_id];
                    let false_node_index = self.block_to_node[&false_id];
                    let merge_node_index = self.block_to_node[&node.merge.unwrap().merge_block_id];

                    let mut result = node.block.clone();

                    if false_node_index != merge_node_index {
                        result.push(crate::Statement::If {
                            condition,
                            accept: self.naga_traverse(true_node_index, Some(merge_node_index))?,
                            reject: self.naga_traverse(false_node_index, Some(merge_node_index))?,
                        });
                    } else {
                        let true_merges_to_false = has_path_connecting(
                            &self.flow,
                            true_node_index,
                            false_node_index,
                            None,
                        );
                        let stop_node_index = if true_merges_to_false {
                            Some(merge_node_index)
                        } else {
                            stop_node_index
                        };

                        result.push(crate::Statement::If {
                            condition,
                            accept: self.naga_traverse(true_node_index, stop_node_index)?,
                            reject: vec![],
                        });
                    }

                    result.extend(self.naga_traverse(merge_node_index, stop_node_index)?);

                    Ok(result)
                }
                Terminator::Switch {
                    selector,
                    default,
                    ref targets,
                } => {
                    let merge_node_index = self.block_to_node[&node.merge.unwrap().merge_block_id];
                    let mut result = node.block.clone();
                    let mut cases = Vec::with_capacity(targets.len());

                    for i in 0..targets.len() {
                        let left_target_node_index = self.block_to_node[&targets[i].1];

                        let fall_through = if i < targets.len() - 1 {
                            let right_target_node_index = self.block_to_node[&targets[i + 1].1];
                            has_path_connecting(
                                &self.flow,
                                left_target_node_index,
                                right_target_node_index,
                                None,
                            )
                        } else {
                            false
                        };

                        cases.push(crate::SwitchCase {
                            value: targets[i].0,
                            body: self
                                .naga_traverse(left_target_node_index, Some(merge_node_index))?,
                            fall_through,
                        });
                    }

                    result.push(crate::Statement::Switch {
                        selector,
                        cases,
                        default: self
                            .naga_traverse(self.block_to_node[&default], Some(merge_node_index))?,
                    });

                    result.extend(self.naga_traverse(merge_node_index, stop_node_index)?);

                    Ok(result)
                }
                _ => Err(Error::InvalidTerminator),
            },
            Some(ControlFlowNodeType::Loop) => {
                let merge_node_index = self.block_to_node[&node.merge.unwrap().merge_block_id];
                let continuing: crate::Block = {
                    let continue_edge = self
                        .flow
                        .edges_directed(node_index, Direction::Outgoing)
                        .find(|&ty| *ty.weight() == ControlFlowEdgeType::ForwardContinue)
                        .unwrap();

                    self.flow[continue_edge.target()].block.clone()
                };

                let mut body = node.block.clone();
                match node.terminator {
                    Terminator::BranchConditional {
                        condition,
                        true_id,
                        false_id,
                    } => {
                        let true_node_index = self.block_to_node[&true_id];
                        let false_node_index = self.block_to_node[&false_id];

                        body.push(crate::Statement::If {
                            condition,
                            accept: if true_node_index == merge_node_index {
                                vec![crate::Statement::Break]
                            } else {
                                self.naga_traverse(true_node_index, Some(merge_node_index))?
                            },
                            reject: if false_node_index == merge_node_index {
                                vec![crate::Statement::Break]
                            } else {
                                self.naga_traverse(false_node_index, Some(merge_node_index))?
                            },
                        });
                    }
                    Terminator::Branch { target_id } => body.extend(
                        self.naga_traverse(self.block_to_node[&target_id], Some(merge_node_index))?,
                    ),
                    _ => return Err(Error::InvalidTerminator),
                };

                let mut result = vec![crate::Statement::Loop { body, continuing }];
                result.extend(self.naga_traverse(merge_node_index, stop_node_index)?);

                Ok(result)
            }
            Some(ControlFlowNodeType::Break) => {
                let mut result = node.block.clone();
                match node.terminator {
                    Terminator::BranchConditional {
                        condition,
                        true_id,
                        false_id,
                    } => {
                        let true_node_id = self.block_to_node[&true_id];
                        let false_node_id = self.block_to_node[&false_id];

                        let true_edge =
                            self.flow[self.flow.find_edge(node_index, true_node_id).unwrap()];
                        let false_edge =
                            self.flow[self.flow.find_edge(node_index, false_node_id).unwrap()];

                        if true_edge == ControlFlowEdgeType::LoopBreak {
                            result.push(crate::Statement::If {
                                condition,
                                accept: vec![crate::Statement::Break],
                                reject: self.naga_traverse(false_node_id, stop_node_index)?,
                            });
                        } else if false_edge == ControlFlowEdgeType::LoopBreak {
                            result.push(crate::Statement::If {
                                condition,
                                accept: self.naga_traverse(true_node_id, stop_node_index)?,
                                reject: vec![crate::Statement::Break],
                            });
                        } else {
                            return Err(Error::InvalidEdgeClassification);
                        }
                    }
                    Terminator::Branch { .. } => {
                        result.push(crate::Statement::Break);
                    }
                    _ => return Err(Error::InvalidTerminator),
                };
                Ok(result)
            }
            Some(ControlFlowNodeType::Continue) => Ok(node.block.clone()),
            Some(ControlFlowNodeType::Back) => Ok(node.block.clone()),
            Some(ControlFlowNodeType::Kill) => {
                let mut result = node.block.clone();
                result.push(crate::Statement::Kill);
                Ok(result)
            }
            Some(ControlFlowNodeType::Return) => {
                let value = match node.terminator {
                    Terminator::Return { value } => value,
                    _ => return Err(Error::InvalidTerminator),
                };
                let mut result = node.block.clone();
                result.push(crate::Statement::Return { value });
                Ok(result)
            }
            Some(ControlFlowNodeType::Merge) | None => match node.terminator {
                Terminator::Branch { target_id } => {
                    let mut result = node.block.clone();
                    result.extend(
                        self.naga_traverse(self.block_to_node[&target_id], stop_node_index)?,
                    );
                    Ok(result)
                }
                _ => Ok(node.block.clone()),
            },
        }
    }

    /// Get the entire graph in a graphviz dot format for visualization. Useful for debugging purposes.
    pub(super) fn to_graphviz(&self) -> Result<String, std::fmt::Error> {
        let mut output = String::new();

        output += "digraph ControlFlowGraph {\n";

        for node_index in self.flow.node_indices() {
            let node = &self.flow[node_index];
            writeln!(
                output,
                "{} [ label = \"%{} {:?}\" ]",
                node_index.index(),
                node.id,
                node.ty
            )?;
        }

        for edge in self.flow.raw_edges() {
            let source = edge.source();
            let target = edge.target();

            let style = match edge.weight {
                ControlFlowEdgeType::Forward => "",
                ControlFlowEdgeType::ForwardMerge => "style=dotted",
                ControlFlowEdgeType::ForwardContinue => "color=green",
                ControlFlowEdgeType::Back => "style=dashed",
                ControlFlowEdgeType::LoopBreak => "color=yellow",
                ControlFlowEdgeType::LoopContinue => "color=green",
                ControlFlowEdgeType::IfTrue => "color=blue",
                ControlFlowEdgeType::IfFalse => "color=red",
                ControlFlowEdgeType::SwitchBreak => "color=yellow",
                ControlFlowEdgeType::CaseFallThrough => "style=dotted",
            };

            writeln!(
                &mut output,
                "{} -> {} [ {} ]",
                source.index(),
                target.index(),
                style
            )?;
        }

        output += "}\n";

        Ok(output)
    }
}

/// Type of an edge(flow) in the `ControlFlowGraph`.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(super) enum ControlFlowEdgeType {
    /// Default
    Forward,

    /// Forward edge to a merge block.
    ForwardMerge,

    /// Forward edge to a OpLoopMerge continue's instruction.
    ForwardContinue,

    /// A back-edge: An edge from a node to one of its ancestors in a depth-first
    /// search from the entry block.
    /// Can only be to a ControlFlowNodeType::Loop.
    Back,

    /// An edge from a node to the merge block of the nearest enclosing loop, where
    /// there is no intervening switch.
    /// The source block is a "break block" as defined by SPIR-V.
    LoopBreak,

    /// An edge from a node in a loop body to the associated continue target, where
    /// there are no other intervening loops or switches.
    /// The source block is a "continue block" as defined by SPIR-V.
    LoopContinue,

    /// An edge from a node with OpBranchConditional to the block of true operand.
    IfTrue,

    /// An edge from a node with OpBranchConditional to the block of false operand.
    IfFalse,

    /// An edge from a node to the merge block of the nearest enclosing switch,
    /// where there is no intervening loop.
    SwitchBreak,

    /// An edge from one switch case to the next sibling switch case.
    CaseFallThrough,
}
/// Type of a node(block) in the `ControlFlowGraph`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum ControlFlowNodeType {
    /// A block whose merge instruction is an OpSelectionMerge.
    Header,

    /// A header block whose merge instruction is an OpLoopMerge.
    Loop,

    /// A block declared by the Merge Block operand of a merge instruction.
    Merge,

    /// A block containing a branch to the Merge Block of a loop header’s merge instruction.
    Break,

    /// A block containing a branch to an OpLoopMerge instruction’s Continue Target.
    Continue,

    /// A block containing an OpBranch to a Loop block.
    Back,

    /// A block containing an OpKill instruction.
    Kill,

    /// A block containing an OpReturn or OpReturnValue branch.
    Return,
}
/// ControlFlowGraph's node representing a block in the control flow.
pub(super) struct ControlFlowNode {
    /// SPIR-V ID.
    pub id: BlockId,

    /// Type of the node. See *ControlFlowNodeType*.
    pub ty: Option<ControlFlowNodeType>,

    /// Phi instructions.
    pub phis: Vec<PhiInstruction>,

    /// Naga's statements inside this block.
    pub block: crate::Block,

    /// Termination instruction of the block.
    pub terminator: Terminator,

    /// Merge Instruction
    pub merge: Option<MergeInstruction>,
}
