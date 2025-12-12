use std::time::Duration;

#[derive(Clone, Copy)]
pub enum LoopControl {
    Iterations(u32),
    Time(Duration),
}

impl Default for LoopControl {
    fn default() -> Self {
        LoopControl::Time(Duration::from_secs(2))
    }
}

impl LoopControl {
    pub(crate) fn finished(&self, iterations: u32, elapsed: Duration) -> bool {
        match self {
            LoopControl::Iterations(target) => iterations >= *target,
            LoopControl::Time(target) => elapsed >= *target,
        }
    }
}

pub struct BenchmarkContext {
    pub(crate) override_iters: Option<LoopControl>,
    pub default_iterations: LoopControl,
    pub(crate) is_test: bool,
}

impl BenchmarkContext {
    pub fn is_test(&self) -> bool {
        self.is_test
    }
}
