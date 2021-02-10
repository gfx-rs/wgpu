use crate::{
    arena::{Arena, Handle},
    proc::{Interface, Visitor},
};

struct GlobalUseVisitor<'a> {
    usage: &'a mut [crate::GlobalUse],
    functions: &'a Arena<crate::Function>,
}

impl Visitor for GlobalUseVisitor<'_> {
    fn visit_expr(&mut self, expr: &crate::Expression) {
        if let crate::Expression::GlobalVariable(handle) = expr {
            self.usage[handle.index()] |= crate::GlobalUse::READ;
        }
    }

    fn visit_lhs_expr(&mut self, expr: &crate::Expression) {
        if let crate::Expression::GlobalVariable(handle) = expr {
            self.usage[handle.index()] |= crate::GlobalUse::WRITE;
        }
    }

    fn visit_fun(&mut self, fun: Handle<crate::Function>) {
        for (mine, other) in self.usage.iter_mut().zip(&self.functions[fun].global_usage) {
            *mine |= *other;
        }
    }
}

impl crate::Function {
    pub fn fill_global_use(&mut self, globals_num: usize, functions: &Arena<crate::Function>) {
        self.global_usage.clear();
        self.global_usage
            .resize(globals_num, crate::GlobalUse::empty());

        let mut io = Interface {
            expressions: &self.expressions,
            local_variables: &self.local_variables,
            visitor: GlobalUseVisitor {
                usage: &mut self.global_usage,
                functions,
            },
        };
        io.traverse(&self.body);
    }
}

#[test]
fn global_usage_scan() {
    let test_global = crate::GlobalVariable {
        name: None,
        class: crate::StorageClass::Uniform,
        binding: None,
        ty: Handle::new(std::num::NonZeroU32::new(1).unwrap()),
        init: None,
        interpolation: None,
        storage_access: crate::StorageAccess::empty(),
    };
    let mut test_globals = Arena::new();

    let global_1 = test_globals.append(test_global.clone());
    let global_2 = test_globals.append(test_global.clone());
    let global_3 = test_globals.append(test_global.clone());
    let global_4 = test_globals.append(test_global);

    let mut expressions = Arena::new();
    let global_1_expr = expressions.append(crate::Expression::GlobalVariable(global_1));
    let global_2_expr = expressions.append(crate::Expression::GlobalVariable(global_2));
    let global_3_expr = expressions.append(crate::Expression::GlobalVariable(global_3));
    let global_4_expr = expressions.append(crate::Expression::GlobalVariable(global_4));

    let test_body = vec![
        crate::Statement::Return {
            value: Some(global_1_expr),
        },
        crate::Statement::Store {
            pointer: global_2_expr,
            value: global_1_expr,
        },
        crate::Statement::Store {
            pointer: expressions.append(crate::Expression::Access {
                base: global_3_expr,
                index: global_4_expr,
            }),
            value: global_1_expr,
        },
    ];

    let mut function = crate::Function {
        name: None,
        arguments: Vec::new(),
        return_type: None,
        local_variables: Arena::new(),
        expressions,
        global_usage: Vec::new(),
        body: test_body,
    };
    let other_functions = Arena::new();
    function.fill_global_use(test_globals.len(), &other_functions);

    assert_eq!(
        &function.global_usage,
        &[
            crate::GlobalUse::READ,
            crate::GlobalUse::WRITE,
            crate::GlobalUse::WRITE,
            crate::GlobalUse::READ,
        ],
    )
}
