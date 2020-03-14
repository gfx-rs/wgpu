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
    ) -> Handle<crate::Type> {
        if self.types.len() <= expr_handle.index() {
            for (_, expr) in expressions.iter().skip(self.types.len()) {
                self.types.push(match *expr {
                    crate::Expression::Access { base, .. } |
                    crate::Expression::AccessIndex { base, .. } => {
                        let _ = base;
                        unimplemented!()
                    }
                    crate::Expression::Constant(_) => unimplemented!(),
                    crate::Expression::Compose { ty, .. } => ty,
                    crate::Expression::FunctionParameter(_) => unimplemented!(),
                    crate::Expression::GlobalVariable(_) => unimplemented!(),
                    crate::Expression::LocalVariable(_) => unimplemented!(),
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
