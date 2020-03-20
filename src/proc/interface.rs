use crate::{
    arena::{Arena, Handle},
};

struct Interface<'a> {
    expressions: &'a Arena<crate::Expression>,
    uses: Vec<crate::GlobalUse>,
}

impl<'a> Interface<'a> {
    fn add_inputs(&mut self, handle: Handle<crate::Expression>) {
        use crate::Expression as E;
        match self.expressions[handle] {
            E::Access { base, index } => {
                self.add_inputs(base);
                self.add_inputs(index);
            }
            E::AccessIndex { base, .. } => {
                self.add_inputs(base);
            }
            E::Constant(_) => {}
            E::Compose { ref components, .. } => {
                for &comp in components {
                    self.add_inputs(comp);
                }
            }
            E::FunctionParameter(_) => {},
            E::GlobalVariable(handle) => {
                self.uses[handle.index()] |= crate::GlobalUse::LOAD;
            }
            E::LocalVariable(_) => {}
            E::Load { pointer } => {
                self.add_inputs(pointer);
            }
            E::ImageSample { image, sampler, coordinate } => {
                self.add_inputs(image);
                self.add_inputs(sampler);
                self.add_inputs(coordinate);
            }
            E::Unary { expr, .. } => {
                self.add_inputs(expr);
            }
            E::Binary { left, right, .. } => {
                self.add_inputs(left);
                self.add_inputs(right);
            }
            E::Intrinsic { argument, .. } => {
                self.add_inputs(argument);
            }
            E::DotProduct(left, right) => {
                self.add_inputs(left);
                self.add_inputs(right);
            }
            E::CrossProduct(left, right) => {
                self.add_inputs(left);
                self.add_inputs(right);
            }
            E::Derivative { expr, .. } => {
                self.add_inputs(expr);
            }
            E::Call { ref arguments, .. } => {
                for &argument in arguments {
                    self.add_inputs(argument);
                }
            }
        }
    }

    fn collect(&mut self, block: &[crate::Statement]) {
        for statement in block {
            use crate::Statement as S;
            match *statement {
                S::Empty |
                S::Break |
                S::Continue |
                S::Kill => (),
                S::Block(ref b) => {
                    self.collect(b);
                }
                S::If { condition, ref accept, ref reject } => {
                    self.add_inputs(condition);
                    self.collect(accept);
                    self.collect(reject);
                }
                S::Switch { selector, ref cases, ref default } => {
                    self.add_inputs(selector);
                    for &(ref case, _) in cases.values() {
                        self.collect(case);
                    }
                    self.collect(default);
                }
                S::Loop { ref body, ref continuing } => {
                    self.collect(body);
                    self.collect(continuing);
                }
                S::Return { value } => {
                    if let Some(expr) = value {
                        self.add_inputs(expr);
                    }
                }
                S::Store { pointer, value } => {
                    let mut left = pointer;
                    loop {
                        match self.expressions[left] {
                            crate::Expression::Access { base, index } => {
                                self.add_inputs(index);
                                left = base;
                            }
                            crate::Expression::AccessIndex { base, .. } => {
                                left = base;
                            }
                            crate::Expression::GlobalVariable(handle) => {
                                self.uses[handle.index()] |= crate::GlobalUse::STORE;
                                break;
                            }
                            _ => break,
                        }
                    }
                    self.add_inputs(value);
                }
            }
        }
    }
}

impl crate::GlobalUse {
    pub fn scan(
        expressions: &Arena<crate::Expression>,
        body: &[crate::Statement],
        globals: &Arena<crate::GlobalVariable>,
    ) -> Vec<Self> {
        let mut io = Interface {
            expressions,
            uses: vec![crate::GlobalUse::empty(); globals.len()],
        };
        io.collect(body);
        io.uses
    }
}
