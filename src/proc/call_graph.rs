use crate::{
    arena::{Arena, Handle},
    proc::{Interface, Visitor},
    Function,
};
use petgraph::{
    graph::{DefaultIx, NodeIndex},
    Graph,
};

pub type CallGraph = Graph<Handle<Function>, ()>;

pub struct CallGraphBuilder<'a> {
    pub functions: &'a Arena<Function>,
}

impl<'a> CallGraphBuilder<'a> {
    pub fn process(&self, func: &Function) -> CallGraph {
        let mut graph = Graph::new();
        let mut children = Vec::new();

        let visitor = CallGraphVisitor {
            children: &mut children,
        };

        let mut interface = Interface {
            expressions: &func.expressions,
            local_variables: &func.local_variables,
            visitor,
        };

        interface.traverse(&func.body);

        for handle in children {
            let id = graph.add_node(handle);
            self.collect(handle, id, &mut graph);
        }

        graph
    }

    fn collect(&self, handle: Handle<Function>, id: NodeIndex<DefaultIx>, graph: &mut CallGraph) {
        let mut children = Vec::new();
        let visitor = CallGraphVisitor {
            children: &mut children,
        };
        let func = &self.functions[handle];

        let mut interface = Interface {
            expressions: &func.expressions,
            local_variables: &func.local_variables,
            visitor,
        };

        interface.traverse(&func.body);

        for handle in children {
            let child_id = graph.add_node(handle);
            graph.add_edge(id, child_id, ());

            self.collect(handle, child_id, graph);
        }
    }
}

struct CallGraphVisitor<'a> {
    children: &'a mut Vec<Handle<Function>>,
}

impl<'a> Visitor for CallGraphVisitor<'a> {
    fn visit_fun(&mut self, func: Handle<Function>) {
        self.children.push(func)
    }
}
