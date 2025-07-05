use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

#[gpu_test]
static DROP_FAILED_TIMESTAMP_QUERY_SET: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_sync(|ctx| {
        // Enter an error scope, so the validation catch-all doesn't
        // report the error too early.
        ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);

        // Creating this query set should fail, since we didn't include
        // TIMESTAMP_QUERY in our required features.
        let bad_query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("doomed query set"),
            ty: wgpu::QueryType::Timestamp,
            count: 1,
        });

        // Dropping this should not panic.
        drop(bad_query_set);

        assert!(pollster::block_on(ctx.device.pop_error_scope()).is_some());
    });
