use std::{future::Future, pin::Pin, sync::Arc};

use wgt::{WasmNotSend, WasmNotSync};

use crate::{TestParameters, TestingContext};

#[cfg(not(target_arch = "wasm32"))]
pub type RunTestAsync =
    Arc<dyn Fn(TestingContext) -> Pin<Box<dyn Future<Output = ()> + Send + Sync>> + Send + Sync>;
#[cfg(target_arch = "wasm32")]
pub type RunTestAsync = Arc<dyn Fn(TestingContext) -> Pin<Box<dyn Future<Output = ()>>>>;

#[derive(Clone)]
pub struct GpuTestConfiguration {
    pub name: String,
    pub params: TestParameters,
    pub test: Option<RunTestAsync>,
}

impl GpuTestConfiguration {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            params: TestParameters::default(),
            test: None,
        }
    }

    pub fn name(self, name: &str) -> Self {
        Self {
            name: String::from(name),
            ..self
        }
    }

    #[doc(hidden)]
    /// Derives the name from a `struct S` in the function initializing the test.
    ///
    /// Does not overwrite a given name if a name has already been set
    pub fn name_from_init_function_typename<S>(self, name: &'static str) -> Self {
        if !self.name.is_empty() {
            return self;
        }
        let type_name = std::any::type_name::<S>();

        // We end up with a string like:
        //
        // module::path::we::want::test_name_initializer::S
        //
        // So we reverse search for the 4th colon from the end, and take everything before that.
        let mut colons = 0;
        let mut colon_4_index = type_name.len();
        for i in (0..type_name.len()).rev() {
            if type_name.as_bytes()[i] == b':' {
                colons += 1;
            }
            if colons == 4 {
                colon_4_index = i;
                break;
            }
        }

        let full = format!("{}::{}", &type_name[..colon_4_index], name);
        Self { name: full, ..self }
    }

    pub fn parameters(self, parameters: TestParameters) -> Self {
        Self {
            params: parameters,
            ..self
        }
    }

    pub fn run_sync(
        self,
        test: impl Fn(TestingContext) + Copy + WasmNotSend + WasmNotSync + 'static,
    ) -> Self {
        Self {
            test: Some(Arc::new(move |ctx| Box::pin(async move { test(ctx) }))),
            ..self
        }
    }

    pub fn run_async<F, R>(self, test: F) -> Self
    where
        F: Fn(TestingContext) -> R + WasmNotSend + WasmNotSync + 'static,
        R: Future<Output = ()> + WasmNotSend + WasmNotSync + 'static,
    {
        Self {
            test: Some(Arc::new(move |ctx| Box::pin(test(ctx)))),
            ..self
        }
    }
}

impl Default for GpuTestConfiguration {
    fn default() -> Self {
        Self::new()
    }
}
