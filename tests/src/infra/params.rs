use std::{future::Future, pin::Pin, sync::Arc};

use crate::{TestParameters, TestingContext};

pub type RunTestAsync =
    Arc<dyn Fn(TestingContext) -> Pin<Box<dyn Future<Output = ()> + Send + Sync>> + Send + Sync>;

#[derive(Clone)]
pub struct GpuTestConfiguration {
    pub name: &'static str,
    pub params: TestParameters,
    pub test: Option<RunTestAsync>,
}

impl GpuTestConfiguration {
    pub fn new() -> Self {
        Self {
            name: "",
            params: TestParameters::default(),
            test: None,
        }
    }

    pub fn name(self, name: &'static str) -> Self {
        Self { name, ..self }
    }

    pub fn name_if_not_set(self, name: &'static str) -> Self {
        if self.name != "" {
            return self;
        }
        Self { name, ..self }
    }

    pub fn parameters(self, parameters: TestParameters) -> Self {
        Self {
            params: parameters,
            ..self
        }
    }

    pub fn run_sync(self, test: impl Fn(TestingContext) + Copy + Send + Sync + 'static) -> Self {
        Self {
            test: Some(Arc::new(move |ctx| Box::pin(async move { test(ctx) }))),
            ..self
        }
    }

    pub fn run_async<F, R>(self, test: F) -> Self
    where
        F: Fn(TestingContext) -> R + Send + Sync + 'static,
        R: Future<Output = ()> + Send + Sync + 'static,
    {
        Self {
            test: Some(Arc::new(move |ctx| Box::pin(test(ctx)))),
            ..self
        }
    }
}
