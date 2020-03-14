use crate::arena::{Arena, Handle};

pub struct Typifier {
    types: Vec<Handle<crate::Type>>,
}

impl Typifier {
    pub fn new() -> Self {
        Typifier {
            types: Vec::new(),
        }
    }

    pub fn resolve(
        &mut self,
        expr_handle: Handle<crate::Expression>,
        expressions: &Arena<crate::Expression>,
        types: &Arena<crate::Type>,
        constants: &Arena<crate::Constant>,
        global_vars: &Arena<crate::GlobalVariable>,
        local_vars: &Arena<crate::LocalVariable>,
    ) -> Handle<crate::Type> {
        if self.types.len() <= expr_handle.index() {
            for (_, expr) in expressions.iter().skip(self.types.len()) {
                self.types.push(match *expr {
                    crate::Expression::Access { base, .. } => {
                        match types[self.types[base.index()]].inner {
                            crate::TypeInner::Array { base, .. } => self.types[base.index()],
                            ref other => panic!("Can't access into {:?}", other),
                        }
                    }
                    crate::Expression::AccessIndex { base, index } => {
                        match types[self.types[base.index()]].inner {
                            crate::TypeInner::Array { base, .. } => self.types[base.index()],
                            crate::TypeInner::Struct { ref members } => members[index as usize].ty,
                            ref other => panic!("Can't access into {:?}", other),
                        }
                    }
                    crate::Expression::Constant(h) => constants[h].ty,
                    crate::Expression::Compose { ty, .. } => ty,
                    crate::Expression::FunctionParameter(_) => unimplemented!(),
                    crate::Expression::GlobalVariable(h) => global_vars[h].ty,
                    crate::Expression::LocalVariable(h) => local_vars[h].ty,
                    crate::Expression::Load { .. } => unimplemented!(),
                    crate::Expression::Mul(_, _) => unimplemented!(),
                    crate::Expression::ImageSample { .. } => unimplemented!(),
                    crate::Expression::Unary { .. } => unimplemented!(),
                    crate::Expression::Binary { .. } => unimplemented!(),
                    crate::Expression::Intrinsic { .. } => unimplemented!(),
                    crate::Expression::DotProduct(_, _) => unimplemented!(),
                    crate::Expression::CrossProduct(_, _) => unimplemented!(),
                    crate::Expression::Derivative { .. } => unimplemented!(),
                    crate::Expression::Call { .. } => unimplemented!(),
                });
            };
        }
        self.types[expr_handle.index()]
    }
}
