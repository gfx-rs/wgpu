use crate::arena::{Arena, Handle};
use bit_set::BitSet;

pub struct Interface<'a, T> {
    pub visitor: T,
    pub expressions: &'a Arena<crate::Expression>,
    pub local_variables: &'a Arena<crate::LocalVariable>,
    pub mask: &'a mut BitSet,
}

pub trait Visitor {
    fn visit_expr(&mut self, _: Handle<crate::Expression>, _: &crate::Expression) {}
    fn visit_lhs_expr(&mut self, _: Handle<crate::Expression>, _: &crate::Expression) {}
    fn visit_fun(&mut self, _: Handle<crate::Function>) {}
}

impl<'a, T: Visitor> Interface<'a, T> {
    pub fn traverse_expr(&mut self, handle: Handle<crate::Expression>) {
        use crate::Expression as E;

        if !self.mask.insert(handle.index()) {
            return;
        }
        let expr = &self.expressions[handle];

        self.visitor.visit_expr(handle, expr);

        match *expr {
            E::Access { base, index } => {
                self.traverse_expr(base);
                self.traverse_expr(index);
            }
            E::AccessIndex { base, .. } => {
                self.traverse_expr(base);
            }
            E::Constant(_) => {}
            E::Compose { ref components, .. } => {
                for &comp in components {
                    self.traverse_expr(comp);
                }
            }
            E::FunctionArgument(_) | E::GlobalVariable(_) | E::LocalVariable(_) => {}
            E::Load { pointer } => {
                self.traverse_expr(pointer);
            }
            E::ImageSample {
                image,
                sampler,
                coordinate,
                array_index,
                offset: _,
                level,
                depth_ref,
            } => {
                self.traverse_expr(image);
                self.traverse_expr(sampler);
                self.traverse_expr(coordinate);
                if let Some(layer) = array_index {
                    self.traverse_expr(layer);
                }
                match level {
                    crate::SampleLevel::Auto | crate::SampleLevel::Zero => (),
                    crate::SampleLevel::Exact(h) | crate::SampleLevel::Bias(h) => {
                        self.traverse_expr(h);
                    }
                    crate::SampleLevel::Gradient { x, y } => {
                        self.traverse_expr(x);
                        self.traverse_expr(y);
                    }
                }
                if let Some(dref) = depth_ref {
                    self.traverse_expr(dref);
                }
            }
            E::ImageLoad {
                image,
                coordinate,
                array_index,
                index,
            } => {
                self.traverse_expr(image);
                self.traverse_expr(coordinate);
                if let Some(layer) = array_index {
                    self.traverse_expr(layer);
                }
                if let Some(index) = index {
                    self.traverse_expr(index);
                }
            }
            E::ImageQuery { image, query } => {
                self.traverse_expr(image);
                match query {
                    crate::ImageQuery::Size { level: Some(expr) } => self.traverse_expr(expr),
                    crate::ImageQuery::Size { .. }
                    | crate::ImageQuery::NumLevels
                    | crate::ImageQuery::NumLayers
                    | crate::ImageQuery::NumSamples => (),
                }
            }
            E::Unary { expr, .. } => {
                self.traverse_expr(expr);
            }
            E::Binary { left, right, .. } => {
                self.traverse_expr(left);
                self.traverse_expr(right);
            }
            E::Select {
                condition,
                accept,
                reject,
            } => {
                self.traverse_expr(condition);
                self.traverse_expr(accept);
                self.traverse_expr(reject);
            }
            E::Derivative { expr, .. } => {
                self.traverse_expr(expr);
            }
            E::Relational { argument, .. } => {
                self.traverse_expr(argument);
            }
            E::Math {
                arg, arg1, arg2, ..
            } => {
                self.traverse_expr(arg);
                if let Some(arg) = arg1 {
                    self.traverse_expr(arg);
                }
                if let Some(arg) = arg2 {
                    self.traverse_expr(arg);
                }
            }
            E::As { expr, .. } => {
                self.traverse_expr(expr);
            }
            E::Call {
                function,
                ref arguments,
            } => {
                for &argument in arguments {
                    self.traverse_expr(argument);
                }
                self.visitor.visit_fun(function);
            }
            E::ArrayLength(expr) => {
                self.traverse_expr(expr);
            }
        }
    }

    pub fn traverse(&mut self, block: &[crate::Statement]) {
        for statement in block {
            use crate::Statement as S;
            match *statement {
                S::Break | S::Continue | S::Kill => (),
                S::Block(ref b) => {
                    self.traverse(b);
                }
                S::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    self.traverse_expr(condition);
                    self.traverse(accept);
                    self.traverse(reject);
                }
                S::Switch {
                    selector,
                    ref cases,
                    ref default,
                } => {
                    self.traverse_expr(selector);
                    for case in cases.iter() {
                        self.traverse(&case.body);
                    }
                    self.traverse(default);
                }
                S::Loop {
                    ref body,
                    ref continuing,
                } => {
                    self.traverse(body);
                    self.traverse(continuing);
                }
                S::Return { value } => {
                    if let Some(expr) = value {
                        self.traverse_expr(expr);
                    }
                }
                S::Store { pointer, value } => {
                    let mut left = pointer;
                    loop {
                        match self.expressions[left] {
                            crate::Expression::Access { base, index } => {
                                self.traverse_expr(index);
                                left = base;
                            }
                            crate::Expression::AccessIndex { base, .. } => {
                                left = base;
                            }
                            _ => break,
                        }
                    }
                    self.visitor.visit_lhs_expr(left, &self.expressions[left]);
                    self.traverse_expr(value);
                }
                S::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => {
                    self.visitor.visit_lhs_expr(image, &self.expressions[image]);
                    self.traverse_expr(coordinate);
                    if let Some(expr) = array_index {
                        self.traverse_expr(expr);
                    }
                    self.traverse_expr(value);
                }
                S::Call {
                    function,
                    ref arguments,
                } => {
                    for &argument in arguments {
                        self.traverse_expr(argument);
                    }
                    self.visitor.visit_fun(function);
                }
            }
        }
    }
}
