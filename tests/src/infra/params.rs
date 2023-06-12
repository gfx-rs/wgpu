use std::sync::Arc;

use heck::ToSnakeCase;

use crate::{TestParameters, TestingContext};

pub trait GpuTest: Send + Sync + 'static {
    fn new() -> Arc<dyn GpuTest + Send + Sync>
    where
        Self: Sized + Default,
    {
        Arc::new(Self::default())
    }

    fn name(&self) -> String {
        let name = std::any::type_name::<Self>();
        let type_name = name.rsplit_once("::").unwrap().1;
        let snake_case = type_name.to_snake_case();
        let snake_case_trimmed = snake_case.trim_end_matches("_test");
        assert_ne!(
            &snake_case, snake_case_trimmed,
            "Type name of the test must end with \"Test\""
        );
        snake_case_trimmed.to_string()
    }

    fn parameters(&self, params: TestParameters) -> TestParameters {
        params
    }

    fn run(&self, ctx: TestingContext);
}
