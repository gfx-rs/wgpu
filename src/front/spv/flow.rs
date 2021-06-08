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

type ConstructNodeIndex = NodeIndex<u32>;

type ConstructsGraph = rose_tree::RoseTree<ConstructNode>;

/// Control flow graph (CFG) containing relationships between blocks.
pub(super) struct FlowGraph {
    ///
    flow: ControlFlowGraph,

    ///
    constructs: ConstructsGraph,

    /// Block ID to Node index mapping. Internal helper to speed up the classification.
    block_to_node: FastHashMap<BlockId, BlockNodeIndex>,

    /// Reversed post-order traversal of the `flow` graph.
    traversal_order: Vec<BlockNodeIndex>,
}

impl FlowGraph {
    /// Creates empty flow graph.
    pub(super) fn new() -> Self {
        Self {
            flow: ControlFlowGraph::default(),
            constructs: ConstructsGraph::new(ConstructNode {
                ty: ConstructType::Function,
                ..Default::default()
            })
            .0,
            block_to_node: FastHashMap::default(),
            traversal_order: Vec::new(),
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
    /// 2. Computes the traversal order
    /// 3. Finds constructs
    /// 4. Classifies types of blocks and edges in the CFG.
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
                    default_id,
                    ref targets,
                } => {
                    let default_node_index = block_to_node[&default_id];

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
        self.compute_postorder_traverse(Some(node_index(0)));
        self.traversal_order.reverse();

        for (i, node_index) in self.traversal_order.iter().enumerate() {
            self.flow[*node_index].position = i;
        }

        for node_index in self.flow.node_indices() {
            self.flow[node_index].visited = false;
        }

        // 3.
        // The stack of enclosing constructs
        let mut enclosing = vec![node_index(0)];

        let postorder_len = self.traversal_order.len();
        let push_construct = |enclosing: &mut Vec<ConstructNodeIndex>,
                              constructs: &mut ConstructsGraph,
                              ty: ConstructType,
                              begin: BlockNodeIndex,
                              begin_position: usize,
                              end: BlockNodeIndex,
                              end_position: usize,
                              depth: usize|
         -> ConstructNodeIndex {
            let mut parent_index = *enclosing.last().unwrap_or(&node_index(0));

            // A loop construct is added right after its associated continue construct.
            // In that case, adjust the parent up.
            if ty == ConstructType::Loop {
                parent_index = constructs.parent(parent_index).unwrap();
            }

            let end_position = if end_position == 0 {
                postorder_len
            } else {
                end_position
            };

            let node_index = constructs.add_child(
                parent_index,
                ConstructNode {
                    ty,
                    begin,
                    begin_position,
                    end,
                    end_position,
                    depth,
                    ..Default::default()
                },
            );

            let parent = {
                if let Some(parent) = constructs.parent(node_index) {
                    if constructs[parent].depth < depth {
                        Some(constructs[parent])
                    } else {
                        None
                    }
                } else {
                    None
                }
            };

            constructs[node_index].enclosing_loop = {
                if constructs[node_index].ty == ConstructType::Loop {
                    Some(node_index)
                } else if let Some(parent) = parent {
                    parent.enclosing_loop
                } else {
                    None
                }
            };

            constructs[node_index].enclosing_continue = {
                if constructs[node_index].ty == ConstructType::Continue {
                    Some(node_index)
                } else if let Some(parent) = parent {
                    parent.enclosing_continue
                } else {
                    None
                }
            };

            constructs[node_index].enclosing_loop_or_continue_or_switch = {
                if constructs[node_index].ty == ConstructType::Loop
                    || constructs[node_index].ty == ConstructType::Continue
                    || constructs[node_index].ty == ConstructType::Case
                {
                    Some(node_index)
                } else if let Some(parent) = parent {
                    parent.enclosing_loop_or_continue_or_switch
                } else {
                    None
                }
            };

            enclosing.push(node_index);

            node_index
        };

        self.constructs[node_index(0)].begin = self.traversal_order[0];

        for block_index in self.traversal_order.iter() {
            let mut top_construct_index = *enclosing.last().unwrap();

            if top_construct_index != node_index(0) {
                while *block_index == self.constructs[top_construct_index].end {
                    enclosing.pop().unwrap();
                    top_construct_index = *enclosing.last().unwrap();
                }
            }

            if let Some(merge) = self.flow[*block_index].merge {
                let merge_index = self.block_to_node[&merge.merge_block_id];
                let depth = self.constructs[top_construct_index].depth + 1;

                match self.flow[*block_index].ty {
                    Some(ControlFlowNodeType::Loop) => {
                        let continue_index = self.block_to_node[&merge.continue_block_id.unwrap()];

                        top_construct_index = push_construct(
                            &mut enclosing,
                            &mut self.constructs,
                            ConstructType::Continue,
                            continue_index,
                            self.flow[continue_index].position,
                            merge_index,
                            self.flow[merge_index].position,
                            depth,
                        );

                        // A loop header that is its own continue target will have an
                        // empty loop construct. Only create a loop construct when
                        // the continue target is *not* the same as the loop header.
                        if *block_index != continue_index {
                            // From the interval rule, the loop construct consists of blocks
                            // in the block order, starting at the header, until just
                            // before the continue target.
                            top_construct_index = push_construct(
                                &mut enclosing,
                                &mut self.constructs,
                                ConstructType::Loop,
                                *block_index,
                                self.flow[*block_index].position,
                                continue_index,
                                self.flow[continue_index].position,
                                depth,
                            );

                            // If the loop header branches to two different blocks inside the loop
                            // construct, then the loop body should be modeled as an if-selection
                            // construct
                            let neighbors: Vec<BlockNodeIndex> = self
                                .flow
                                .neighbors_directed(*block_index, Direction::Outgoing)
                                .collect();
                            if neighbors.len() == 2 && neighbors[0] != neighbors[1] {
                                let target0_pos = self.flow[neighbors[0]].position;
                                let target1_pos = self.flow[neighbors[1]].position;
                                if self.constructs[top_construct_index]
                                    .contains_position(target0_pos)
                                    && self.constructs[top_construct_index]
                                        .contains_position(target1_pos)
                                {
                                    // Insert a synthetic if-selection
                                    top_construct_index = push_construct(
                                        &mut enclosing,
                                        &mut self.constructs,
                                        ConstructType::Selection,
                                        *block_index,
                                        self.flow[*block_index].position,
                                        continue_index,
                                        self.flow[continue_index].position,
                                        depth + 1,
                                    );
                                }
                            }
                        }
                    }
                    Some(ControlFlowNodeType::Header) => {
                        let ty = match self.flow[*block_index].terminator {
                            Terminator::Switch { .. } => ConstructType::Case,
                            _ => ConstructType::Selection,
                        };

                        top_construct_index = push_construct(
                            &mut enclosing,
                            &mut self.constructs,
                            ty,
                            *block_index,
                            self.flow[*block_index].position,
                            merge_index,
                            self.flow[merge_index].position,
                            depth,
                        );
                    }
                    _ => {}
                }
            }

            self.flow[*block_index].construct = top_construct_index;
        }

        // 2.
        // Classify Nodes/Edges as one of [Break, Continue]
        for edge_index in self.flow.edge_indices() {
            let (node_source_index, node_target_index) =
                self.flow.edge_endpoints(edge_index).unwrap();

            if self.flow[node_source_index].ty == Some(ControlFlowNodeType::Header)
                || self.flow[node_source_index].ty == Some(ControlFlowNodeType::Loop)
            {
                continue;
            }

            let construct_index = self.flow[node_source_index].construct;
            let header_index = self.constructs[construct_index].begin;

            // Loop break/Switch break
            if let Some(enclosing_construct_index) =
                self.constructs[construct_index].enclosing_loop_or_continue_or_switch
            {
                let enclosing_construct = self.constructs[enclosing_construct_index];
                let breakable_header = &self.flow[enclosing_construct.begin];

                if let Some(merge_instruction) = breakable_header.merge {
                    let merge_node_index = self.block_to_node[&merge_instruction.merge_block_id];
                    if node_target_index == merge_node_index {
                        self.flow[node_source_index].ty = Some(ControlFlowNodeType::Break);
                        self.flow[edge_index] = if enclosing_construct.ty == ConstructType::Case {
                            ControlFlowEdgeType::SwitchBreak
                        } else {
                            ControlFlowEdgeType::LoopBreak
                        };
                        continue;
                    }
                }
            }

            // Continue
            if let Some(enclosing_construct_index) = self.constructs[construct_index].enclosing_loop
            {
                let loop_header = &self.flow[self.constructs[enclosing_construct_index].begin];

                if let Some(continue_block_id) =
                    loop_header.merge.and_then(|merge| merge.continue_block_id)
                {
                    if node_target_index == self.block_to_node[&continue_block_id] {
                        self.flow[node_source_index].ty = Some(ControlFlowNodeType::Continue);
                        self.flow[edge_index] = ControlFlowEdgeType::LoopContinue;
                        continue;
                    }
                }
            }

            // If break
            if let Some(merge_instruction) = self.flow[header_index].merge {
                if node_target_index == self.block_to_node[&merge_instruction.merge_block_id] {
                    self.flow[node_source_index].ty = Some(ControlFlowNodeType::Break);
                    self.flow[edge_index] = ControlFlowEdgeType::IfBreak;
                    continue;
                }
            }
        }
    }

    fn header_if_breakable(&self, construct_index: ConstructNodeIndex) -> Option<NodeIndex> {
        match self.constructs[construct_index].ty {
            ConstructType::Loop | ConstructType::Case => {
                Some(self.constructs[construct_index].begin)
            }
            ConstructType::Continue => {
                let continue_target = self.constructs[construct_index].begin;
                Some(
                    self.flow
                        .neighbors_directed(continue_target, Direction::Outgoing)
                        .next()
                        .unwrap(),
                )
            }
            _ => None,
        }
    }

    fn compute_postorder_traverse(&mut self, node_index: Option<BlockNodeIndex>) {
        if node_index.is_none() {
            return;
        }
        let node_index = node_index.unwrap();

        if self.flow[node_index].visited {
            return;
        }
        self.flow[node_index].visited = true;

        if let Some(merge) = self.flow[node_index].merge {
            self.compute_postorder_traverse(Some(self.block_to_node[&merge.merge_block_id]));
        }

        let continue_edge = self
            .flow
            .edges_directed(node_index, Direction::Outgoing)
            .find(|&ty| *ty.weight() == ControlFlowEdgeType::ForwardContinue)
            .map(|continue_edge| continue_edge.target());
        self.compute_postorder_traverse(continue_edge);

        let terminator = self.flow[node_index].terminator.clone();
        match terminator {
            Terminator::BranchConditional {
                condition: _,
                true_id,
                false_id,
            } => {
                self.compute_postorder_traverse(Some(self.block_to_node[&false_id]));
                self.compute_postorder_traverse(Some(self.block_to_node[&true_id]));
            }
            Terminator::Branch { target_id } => {
                self.compute_postorder_traverse(Some(self.block_to_node[&target_id]));
            }
            Terminator::Switch {
                selector: _,
                default_id,
                ref targets,
            } => {
                self.compute_postorder_traverse(Some(self.block_to_node[&default_id]));
                for &(_, target_id) in targets.iter() {
                    self.compute_postorder_traverse(Some(self.block_to_node[&target_id]));
                }
            }
            _ => {}
        };

        self.traversal_order.push(node_index);
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
            let phis = std::mem::take(&mut self.flow[node_index].phis);
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
    pub(super) fn convert_to_naga(&mut self) -> Result<crate::Block, Error> {
        self.convert_to_naga_traverse(node_index(0), std::collections::HashSet::new())
    }

    fn convert_to_naga_traverse(
        &mut self,
        node_index: BlockNodeIndex,
        stop_nodes: std::collections::HashSet<BlockNodeIndex>,
    ) -> Result<crate::Block, Error> {
        if stop_nodes.contains(&node_index) {
            return Ok(vec![]);
        }

        if self.flow[node_index].visited {
            return Err(Error::ControlFlowGraphCycle(self.flow[node_index].id));
        }

        self.flow[node_index].visited = true;

        let terminator = self.flow[node_index].terminator.clone();

        match self.flow[node_index].ty {
            Some(ControlFlowNodeType::Header) => match terminator {
                Terminator::BranchConditional {
                    condition,
                    true_id,
                    false_id,
                } => {
                    let true_node_index = self.block_to_node[&true_id];
                    let false_node_index = self.block_to_node[&false_id];
                    let merge_node_index =
                        self.block_to_node[&self.flow[node_index].merge.unwrap().merge_block_id];

                    let premerge_index: Option<NodeIndex> = {
                        let second_head = self.flow[true_node_index]
                            .position
                            .max(self.flow[false_node_index].position);
                        let end_first_clause = self.traversal_order[second_head - 1];

                        self.flow
                            .edges_directed(end_first_clause, Direction::Outgoing)
                            .find(|e| match *e.weight() {
                                ControlFlowEdgeType::Forward
                                | ControlFlowEdgeType::ForwardContinue
                                | ControlFlowEdgeType::ForwardMerge
                                | ControlFlowEdgeType::IfFalse
                                | ControlFlowEdgeType::IfTrue => self.constructs
                                    [self.flow[node_index].construct]
                                    .contains_position(self.flow[e.target()].position),
                                _ => false,
                            })
                            .map(|e| e.target())
                    };

                    let is_false_inside_construct = self.constructs
                        [self.flow[node_index].construct]
                        .contains_position(self.flow[false_node_index].position);

                    let intended_merge = if let Some(premerge_index) = premerge_index {
                        premerge_index
                    } else {
                        merge_node_index
                    };
                    let then_end_index = if is_false_inside_construct {
                        false_node_index
                    } else {
                        intended_merge
                    };
                    let else_end_index = if let Some(premerge_index) = premerge_index {
                        premerge_index
                    } else {
                        intended_merge
                    };

                    let mut result: crate::Block = std::mem::take(&mut self.flow[node_index].block);

                    let mut accept_stop_nodes = stop_nodes.clone();
                    accept_stop_nodes.insert(merge_node_index);
                    accept_stop_nodes.insert(intended_merge);
                    accept_stop_nodes.insert(then_end_index);

                    let mut reject_stop_nodes = stop_nodes.clone();
                    reject_stop_nodes.insert(merge_node_index);
                    reject_stop_nodes.insert(intended_merge);
                    reject_stop_nodes.insert(else_end_index);

                    let mut accept =
                        self.convert_to_naga_traverse(true_node_index, accept_stop_nodes)?;
                    let mut reject =
                        self.convert_to_naga_traverse(false_node_index, reject_stop_nodes)?;

                    // If the true/false block of a header is breaking from switch or loop we add a break statement after its statements
                    for &mut (target_index, ref mut statements) in [
                        (true_node_index, &mut accept),
                        (false_node_index, &mut reject),
                    ]
                    .iter_mut()
                    {
                        if let Some(ControlFlowNodeType::Break) = self.flow[target_index].ty {
                            let edge = *self
                                .flow
                                .edges_directed(target_index, Direction::Outgoing)
                                .next()
                                .unwrap()
                                .weight();
                            if edge == ControlFlowEdgeType::SwitchBreak
                                || edge == ControlFlowEdgeType::LoopBreak
                            {
                                // Do not add break if already has one as the last statement
                                if let Some(&crate::Statement::Break) = statements.last() {
                                } else {
                                    statements.push(crate::Statement::Break);
                                }
                            }
                        }
                    }

                    result.push(crate::Statement::If {
                        condition,
                        accept,
                        reject,
                    });

                    result.extend(self.convert_to_naga_traverse(merge_node_index, stop_nodes)?);

                    Ok(result)
                }
                Terminator::Switch {
                    selector,
                    default_id,
                    ref targets,
                } => {
                    let merge_node_index =
                        self.block_to_node[&self.flow[node_index].merge.unwrap().merge_block_id];
                    let mut result: crate::Block = std::mem::take(&mut self.flow[node_index].block);
                    let mut cases = Vec::with_capacity(targets.len());

                    let mut stop_nodes_cases = stop_nodes.clone();
                    stop_nodes_cases.insert(merge_node_index);

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
                            body: self.convert_to_naga_traverse(
                                left_target_node_index,
                                stop_nodes_cases.clone(),
                            )?,
                            fall_through,
                        });
                    }

                    result.push(crate::Statement::Switch {
                        selector,
                        cases,
                        default: self.convert_to_naga_traverse(
                            self.block_to_node[&default_id],
                            stop_nodes_cases,
                        )?,
                    });

                    result.extend(self.convert_to_naga_traverse(merge_node_index, stop_nodes)?);

                    Ok(result)
                }
                _ => Err(Error::InvalidTerminator),
            },
            Some(ControlFlowNodeType::Loop) => {
                let merge_node_index =
                    self.block_to_node[&self.flow[node_index].merge.unwrap().merge_block_id];
                let continuing: crate::Block = {
                    let continue_edge = self
                        .flow
                        .edges_directed(node_index, Direction::Outgoing)
                        .find(|&ty| *ty.weight() == ControlFlowEdgeType::ForwardContinue)
                        .unwrap()
                        .target();

                    std::mem::take(&mut self.flow[continue_edge].block)
                };

                let mut body: crate::Block = std::mem::take(&mut self.flow[node_index].block);

                let mut stop_nodes_merge = stop_nodes.clone();
                stop_nodes_merge.insert(merge_node_index);
                match self.flow[node_index].terminator {
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
                                self.convert_to_naga_traverse(
                                    true_node_index,
                                    stop_nodes_merge.clone(),
                                )?
                            },
                            reject: if false_node_index == merge_node_index {
                                vec![crate::Statement::Break]
                            } else {
                                self.convert_to_naga_traverse(false_node_index, stop_nodes_merge)?
                            },
                        });
                    }
                    Terminator::Branch { target_id } => {
                        body.extend(self.convert_to_naga_traverse(
                            self.block_to_node[&target_id],
                            stop_nodes_merge,
                        )?)
                    }
                    _ => return Err(Error::InvalidTerminator),
                };

                let mut result = vec![crate::Statement::Loop { body, continuing }];
                result.extend(self.convert_to_naga_traverse(merge_node_index, stop_nodes)?);

                Ok(result)
            }
            Some(ControlFlowNodeType::Break) => {
                let mut result: crate::Block = std::mem::take(&mut self.flow[node_index].block);
                match self.flow[node_index].terminator {
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

                        if true_edge == ControlFlowEdgeType::LoopBreak
                            || true_edge == ControlFlowEdgeType::IfBreak
                        {
                            result.push(crate::Statement::If {
                                condition,
                                accept: if true_edge == ControlFlowEdgeType::LoopBreak {
                                    vec![crate::Statement::Break]
                                } else {
                                    vec![]
                                },
                                reject: self.convert_to_naga_traverse(false_node_id, stop_nodes)?,
                            });
                        } else if false_edge == ControlFlowEdgeType::LoopBreak
                            || false_edge == ControlFlowEdgeType::IfBreak
                        {
                            result.push(crate::Statement::If {
                                condition,
                                accept: self.convert_to_naga_traverse(true_node_id, stop_nodes)?,
                                reject: if false_edge == ControlFlowEdgeType::LoopBreak {
                                    vec![crate::Statement::Break]
                                } else {
                                    vec![]
                                },
                            });
                        } else {
                            return Err(Error::InvalidEdgeClassification);
                        }
                    }
                    Terminator::Branch { target_id } => {
                        let target_index = self.block_to_node[&target_id];

                        let edge =
                            self.flow[self.flow.find_edge(node_index, target_index).unwrap()];

                        if edge == ControlFlowEdgeType::LoopBreak {
                            result.push(crate::Statement::Break);
                        }
                    }
                    _ => return Err(Error::InvalidTerminator),
                };
                Ok(result)
            }
            Some(ControlFlowNodeType::Continue) | Some(ControlFlowNodeType::Back) => {
                Ok(std::mem::take(&mut self.flow[node_index].block))
            }
            Some(ControlFlowNodeType::Kill) => {
                let mut result: crate::Block = std::mem::take(&mut self.flow[node_index].block);
                result.push(crate::Statement::Kill);
                Ok(result)
            }
            Some(ControlFlowNodeType::Return) => {
                let value = match self.flow[node_index].terminator {
                    Terminator::Return { value } => value,
                    _ => return Err(Error::InvalidTerminator),
                };
                let mut result: crate::Block = std::mem::take(&mut self.flow[node_index].block);
                result.push(crate::Statement::Return { value });
                Ok(result)
            }
            Some(ControlFlowNodeType::Merge) | None => match self.flow[node_index].terminator {
                Terminator::Branch { target_id } => {
                    let mut result: crate::Block = std::mem::take(&mut self.flow[node_index].block);
                    result.extend(
                        self.convert_to_naga_traverse(self.block_to_node[&target_id], stop_nodes)?,
                    );
                    Ok(result)
                }
                _ => Ok(std::mem::take(&mut self.flow[node_index].block)),
            },
        }
    }

    /// Get the entire graph in a graphviz dot format for visualization. Useful for debugging purposes.
    pub(super) fn to_graphviz(&self) -> Result<String, std::fmt::Error> {
        let mut output = String::new();

        output += "digraph ControlFlowGraph {\n";

        for node_index in self.flow.node_indices() {
            let node = &self.flow[node_index];

            let node_name = match node.ty {
                Some(ControlFlowNodeType::Header) => {
                    if self.constructs[node.construct].ty == ConstructType::Case {
                        "Switch"
                    } else {
                        "If"
                    }
                }
                Some(ControlFlowNodeType::Loop) => "Loop",
                Some(ControlFlowNodeType::Merge) => "",
                Some(ControlFlowNodeType::Break) => "Break",
                Some(ControlFlowNodeType::Continue) => "Continue",
                Some(ControlFlowNodeType::Back) => "Back",
                Some(ControlFlowNodeType::Kill) => "Kill",
                Some(ControlFlowNodeType::Return) => "Return",
                None => "Unlabeled",
            };

            writeln!(
                output,
                "{} [ label = \"%{}({}) {}\" shape=ellipse ]",
                node_index.index(),
                node.id,
                node_index.index(),
                node_name,
            )?;
        }

        self.to_graphviz_constructs(&mut output, node_index(0))?;

        for edge in self.flow.raw_edges() {
            let source = edge.source();
            let target = edge.target();

            let style = match edge.weight {
                ControlFlowEdgeType::Forward => "",
                ControlFlowEdgeType::ForwardMerge => "style=dotted",
                ControlFlowEdgeType::ForwardContinue => "color=green",
                ControlFlowEdgeType::Back => "style=dashed",
                ControlFlowEdgeType::LoopContinue => "color=green",
                ControlFlowEdgeType::IfTrue => "color=blue",
                ControlFlowEdgeType::IfFalse => "color=red",
                ControlFlowEdgeType::LoopBreak => "color=orange",
                ControlFlowEdgeType::SwitchBreak => "color=tomato",
                ControlFlowEdgeType::IfBreak => "color=yellow",
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

    fn to_graphviz_constructs(
        &self,
        output: &mut String,
        node_index: ConstructNodeIndex,
    ) -> Result<(), std::fmt::Error> {
        let construct = self.constructs[node_index];

        writeln!(
            output,
            "subgraph cluster_{:?} {{ label = {:?}",
            construct.ty, construct.ty
        )?;

        let traversal_order_begin = self.flow[construct.begin].position;
        let traversal_order_end = if construct.end.index() >= self.traversal_order.len() {
            self.traversal_order.len() - 1
        } else {
            self.flow[construct.end].position
        };

        for i in traversal_order_begin..=traversal_order_end {
            write!(output, "{} ", self.traversal_order[i].index())?;
        }

        for child_index in self.constructs.children(node_index) {
            self.to_graphviz_constructs(output, child_index)?;
        }

        writeln!(output, "}}",)?;

        Ok(())
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

    /// An edge from a node in a loop body to the associated continue target, where
    /// there are no other intervening loops or switches.
    /// The source block is a "continue block" as defined by SPIR-V.
    LoopContinue,

    /// An edge from a node with OpBranchConditional to the block of true operand.
    IfTrue,

    /// An edge from a node with OpBranchConditional to the block of false operand.
    IfFalse,

    /// An edge from a node to the merge block of the nearest enclosing loop, where
    /// there is no intervening switch.
    /// The source block is a "break block" as defined by SPIR-V.
    LoopBreak,

    /// An edge from a node to the merge block of the nearest enclosing switch,
    /// where there is no intervening loop.
    SwitchBreak,

    /// An edge from a node to the merge block of the nearest enclosing structured
    /// construct, but which is neither a SwitchBreak or a LoopBreak.
    /// This can only occur for an "if" selection, i.e. where the selection
    /// header ends in OpBranchConditional.
    IfBreak,

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

    /// Merge Instruction.
    pub merge: Option<MergeInstruction>,

    /// Construct that this block belongs to.
    pub construct: ConstructNodeIndex,

    /// Position in the postorder traversal.
    pub position: usize,

    /// Flag determining whether this node was already visited by the graph traversal while converting a SPIR-V to Naga's IR.
    /// This flag serves as a check in case an incorrect SPIR-V is supplied to the front-end. When a SPIR-V is correct there can be no cycles in it and there is no reason
    /// to visit any node twice.
    pub visited: bool,
}
/// Type of a node(block) in the `ConstructsGraph`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum ConstructType {
    /// Includes all the nodes in the `ControlFlowGraph`.
    Function,
    /// Includes the nodes(blocks) dominated by a `ControlFlowNodeType::Header` node(block), while excluding nodes(blocks) dominated by the selection construct’s `ControlFlowNodeType::Merge` node(block).
    Selection,
    /// Includes the nodes(blocks) dominated by a `ControlFlowNodeType::Loop` Continue Target and post dominated by the corresponding loop’s `ControlFlowNodeType::Back` block, while excluding blocks dominated by that `ControlFlowNodeType::Merge` node(block).
    Continue,
    /// Includes the nodes(blocks) dominated by a `ControlFlowNodeType::Loop`, while excluding both that header’s continue construct and the blocks dominated by the loop’s `ControlFlowNodeType::Merge` node(block).
    Loop,
    /// Includes the nodes(blocks) dominated by an OpSwitch Target or Default (this construct is only defined for those OpSwitch Target or Default that are not equal to the OpSwitch’s corresponding merge block)
    Case,
}

impl Default for ConstructType {
    fn default() -> Self {
        ConstructType::Function
    }
}

/// ConstructsGraph's construct node that encompasses a series of blocks in `ControlFlowGraph`
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub(super) struct ConstructNode {
    ///
    pub ty: ConstructType,

    ///
    pub begin: BlockNodeIndex,

    ///
    pub begin_position: usize,

    ///
    pub end: BlockNodeIndex,

    ///
    pub end_position: usize,

    /// Control flow nesting depth. The entry block is at nesting depth 0.
    pub depth: usize,

    /// The nearest enclosing loop construct, if one exists.  Points to itself
    /// when this is a loop construct.
    pub enclosing_loop: Option<ConstructNodeIndex>,

    /// The nearest enclosing continue construct, if one exists.  Points to
    /// itself when this is a contnue construct.
    pub enclosing_continue: Option<ConstructNodeIndex>,

    /// The nearest enclosing loop construct or continue construct or
    /// switch-selection construct, if one exists. The signficance is
    /// that a high level language "break" will branch to the merge block
    /// of such an enclosing construct. Points to itself when this is
    /// a loop construct, a continue construct, or a switch-selection construct.
    pub enclosing_loop_or_continue_or_switch: Option<ConstructNodeIndex>,
}

impl ConstructNode {
    pub(super) fn contains_position(&self, position: usize) -> bool {
        self.begin_position <= position && position < self.end_position
    }
}
