#![cfg(wasm_test)]

use std::{
    cell::RefCell,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([SYNTHETIC_ERROR_SCOPES, SYNTHETIC_ERROR_CALLBACK]);
}

fn create_unsupported_sampler(device: &wgpu::Device) {
    let _sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToBorder,
        ..Default::default()
    });
}

#[apply(gpu_test!)]
static SYNTHETIC_ERROR_SCOPES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default().skip(FailureCase::backend(!wgpu::Backends::BROWSER_WEBGPU)),
    )
    .run_async(|ctx| async move {
        let validation_scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal_scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Internal);

        create_unsupported_sampler(&ctx.device);

        let internal_error = internal_scope.pop();
        let validation_error = validation_scope.pop();
        assert!(internal_error.await.is_none());
        let error = validation_error.await.expect("validation error");
        assert!(matches!(error, wgpu::Error::Validation { .. }));
        assert!(
            error.to_string().contains("Device::create_sampler"),
            "{error}"
        );
    });

#[apply(gpu_test!)]
static SYNTHETIC_ERROR_CALLBACK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default().skip(FailureCase::backend(!wgpu::Backends::BROWSER_WEBGPU)),
    )
    .run_sync(|ctx| {
        thread_local! {
            static CALLBACK_DEVICE: RefCell<Option<wgpu::Device>> = const { RefCell::new(None) };
        }

        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_clone = callback_count.clone();
        CALLBACK_DEVICE.with_borrow_mut(|slot| *slot = Some(ctx.device.clone()));
        ctx.device.on_uncaptured_error(Arc::new(move |error| {
            assert!(matches!(error, wgpu::Error::Validation { .. }));
            callback_count_clone.fetch_add(1, Ordering::Relaxed);
            let device = CALLBACK_DEVICE.with_borrow(|slot| slot.clone()).unwrap();
            device.on_uncaptured_error(Arc::new(|_| {}));
        }));

        create_unsupported_sampler(&ctx.device);
        CALLBACK_DEVICE.with_borrow_mut(|slot| slot.take());

        assert_eq!(callback_count.load(Ordering::Relaxed), 1);
    });
