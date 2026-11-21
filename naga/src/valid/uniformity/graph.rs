//! The uniformity graph and the reachability queries the analysis runs on it.

use alloc::{collections::VecDeque, vec, vec::Vec};

use super::{NonUniformCause, Operation, RequirementTarget};
use crate::Span;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NodeId(u32);

impl NodeId {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// `RequiredToBeUniform.error`, `.warning` and `.info`.
pub(super) const REQUIRED_TO_BE_UNIFORM: [NodeId; 3] = [NodeId(0), NodeId(1), NodeId(2)];
pub(super) const MAY_BE_NON_UNIFORM: NodeId = NodeId(3);
pub(super) const CF_START: NodeId = NodeId(4);
const SPECIAL_NODES: u32 = 5;

/// What a node represents, for error reporting only.
#[derive(Clone, Debug)]
pub(super) enum NodeKind {
    Plain,
    /// An edge from a `RequiredToBeUniform.S` node passes through this node.
    Requirement {
        op: Operation,
        target: RequirementTarget,
        span: Span,
    },
    /// The control flow of a branch, which depends on a condition.
    Condition {
        span: Span,
    },
    /// This node has an edge to [`MAY_BE_NON_UNIFORM`].
    Source {
        cause: NonUniformCause,
        span: Span,
    },
}

#[derive(Debug)]
struct Node {
    edges: Vec<NodeId>,
    kind: NodeKind,
}

#[derive(Debug)]
pub(super) struct Graph {
    nodes: Vec<Node>,
    /// Nodes at or after this index are interior nodes.
    interior_start: usize,
}

/// A path found by [`Graph::reach`] from a `RequiredToBeUniform.S` node to
/// [`MAY_BE_NON_UNIFORM`].
pub(super) struct Violation<'a> {
    pub requirement: Option<(&'a Operation, &'a RequirementTarget, Span)>,
    pub condition: Option<Span>,
    pub cause: Option<(&'a NonUniformCause, Span)>,
}

impl Graph {
    pub fn new() -> Self {
        let mut graph = Graph {
            nodes: Vec::new(),
            interior_start: 0,
        };
        for _ in 0..SPECIAL_NODES {
            graph.add(NodeKind::Plain);
        }
        graph
    }

    /// Mark all nodes added from now on as interior nodes.
    pub const fn begin_interior(&mut self) {
        self.interior_start = self.nodes.len();
    }

    pub fn add(&mut self, kind: NodeKind) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            edges: Vec::new(),
            kind,
        });
        id
    }

    pub fn edge(&mut self, from: NodeId, to: NodeId) {
        if from != to {
            self.nodes[from.index()].edges.push(to);
        }
    }

    pub fn node_to(&mut self, kind: NodeKind, targets: impl IntoIterator<Item = NodeId>) -> NodeId {
        let node = self.add(kind);
        for target in targets {
            self.edge(node, target);
        }
        node
    }

    /// Join `targets` into one node, reusing the node if they are all the same.
    ///
    /// `targets` must not be empty.
    pub fn join(&mut self, targets: &[NodeId]) -> NodeId {
        match *targets {
            [first, ref rest @ ..] if rest.iter().all(|&t| t == first) => first,
            _ => self.node_to(NodeKind::Plain, targets.iter().copied()),
        }
    }

    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Breadth-first search from `start`.
    ///
    /// Interior nodes already marked in `visited` are skipped, and interior
    /// nodes reached are marked. Returns which nodes were reached and, if
    /// [`MAY_BE_NON_UNIFORM`] was reached, a description of the path to it.
    pub fn reach(&self, start: NodeId, visited: &mut [bool]) -> (Vec<bool>, Option<Violation<'_>>) {
        let mut parent = vec![u32::MAX; self.nodes.len()];
        let mut seen = vec![false; self.nodes.len()];
        let mut queue = VecDeque::new();
        let mut reached = Vec::new();
        seen[start.index()] = true;
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            reached.push(node);
            for &next in self.nodes[node.index()].edges.iter() {
                let i = next.index();
                if seen[i] || (i >= self.interior_start && visited[i]) {
                    continue;
                }
                seen[i] = true;
                parent[i] = node.0;
                queue.push_back(next);
            }
        }
        for &node in reached.iter() {
            if node.index() >= self.interior_start {
                visited[node.index()] = true;
            }
        }
        let violation = seen[MAY_BE_NON_UNIFORM.index()].then(|| {
            let mut path = vec![MAY_BE_NON_UNIFORM];
            let mut current = MAY_BE_NON_UNIFORM;
            while current != start {
                current = NodeId(parent[current.index()]);
                path.push(current);
            }
            path.reverse();
            self.violation(&path)
        });
        (seen, violation)
    }

    fn violation(&self, path: &[NodeId]) -> Violation<'_> {
        let mut violation = Violation {
            requirement: None,
            condition: None,
            cause: None,
        };
        for &node in path {
            match self.nodes[node.index()].kind {
                NodeKind::Plain => {}
                NodeKind::Requirement {
                    ref op,
                    ref target,
                    span,
                } => {
                    violation.requirement.get_or_insert((op, target, span));
                }
                NodeKind::Condition { span } => {
                    violation.condition.get_or_insert(span);
                }
                NodeKind::Source { ref cause, span } => violation.cause = Some((cause, span)),
            }
        }
        violation
    }
}
