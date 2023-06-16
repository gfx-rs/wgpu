use std::sync::Arc;

use heck::ToSnakeCase;

use crate::{TestParameters, TestingContext};

pub trait GpuTest: Send + Sync + 'static {
    #[allow(clippy::new_ret_no_self)]
    fn new() -> Arc<dyn GpuTest + Send + Sync>
    where
        Self: Sized + Default,
    {
        Self::from_value(Self::default())
    }

    fn from_value(value: Self) -> Arc<dyn GpuTest + Send + Sync>
    where
        Self: Sized,
    {
        Arc::new(value)
    }

    fn name(&self) -> String {
        let name = std::any::type_name::<Self>();
        let (path, type_name) = name.rsplit_once("::").unwrap();
        let snake_case = type_name.to_snake_case();
        let snake_case_trimmed = snake_case.trim_end_matches("_test");
        assert_ne!(
            &snake_case, snake_case_trimmed,
            "Type name of the test must end with \"Test\""
        );
        format!("{path}::{snake_case_trimmed}")
    }

    fn parameters(&self, params: TestParameters) -> TestParameters {
        params
    }

    fn run(&self, ctx: TestingContext);
}

pub struct CpuTest {
    name: &'static str,
    test: Box<dyn FnOnce() + Send + Sync + 'static>,
}

impl CpuTest {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn call(self) {
        (self.test)();
    }
}

// This needs to be generic, otherwise we will get the generic type name of `fn() -> ()`.
pub fn cpu_test<T>(test: T) -> CpuTest
where
    T: FnOnce() + Send + Sync + 'static,
{
    CpuTest {
        name: std::any::type_name::<T>(),
        test: Box::new(test),
    }
}
