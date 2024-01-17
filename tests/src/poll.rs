use crate::TestingContext;

impl TestingContext {
    /// Utility to allow future asynchronous polling.
    pub async fn async_poll(&self, maintain: wgpu::Maintain) -> wgpu::MaintainResult {
        self.device.poll(maintain)
    }
}
