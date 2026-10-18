use wgpu::*;
use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(DX12_DEVICE_FROM_RAW);
}

#[apply(gpu_test!)]
static DX12_DEVICE_FROM_RAW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default().skip(FailureCase::backend(Backends::all() - Backends::DX12)),
    )
    .run_sync(|ctx| match ctx.adapter_info.backend {
        #[cfg(windows)]
        wgpu::Backend::Dx12 => windows_impl::device_from_raw(&ctx),
        other => unreachable!(
            "test is configured to skip all backends except DX12, but ran on {other:?}"
        ),
    });

#[cfg(windows)]
mod windows_impl {
    use wgpu::hal::dx12;
    use windows::{
        core::Interface as _,
        Win32::{
            Foundation::HANDLE,
            Graphics::{Direct3D::D3D_FEATURE_LEVEL_11_0, Direct3D12::*, Dxgi::*},
        },
    };

    fn native_ref_count(object: &impl windows::core::Interface) -> u32 {
        let raw = object.as_raw();
        // SAFETY: `raw` is a live COM object, and the AddRef is balanced by the Release.
        unsafe {
            let vtable = *(raw as *const *const windows::core::IUnknown_Vtbl);
            ((*vtable).AddRef)(raw);
            ((*vtable).Release)(raw)
        }
    }

    fn native_create_device(adapter: &impl windows::core::Interface) -> ID3D12Device {
        let mut device: Option<ID3D12Device> = None;
        unsafe {
            D3D12CreateDevice(
                &adapter.cast::<IDXGIAdapter>().unwrap(),
                D3D_FEATURE_LEVEL_11_0,
                &mut device,
            )
        }
        .expect("D3D12CreateDevice failed");
        device.unwrap()
    }

    fn native_create_queue(
        device: &ID3D12Device,
        ty: D3D12_COMMAND_LIST_TYPE,
    ) -> ID3D12CommandQueue {
        unsafe {
            device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: ty,
                ..Default::default()
            })
        }
        .expect("CreateCommandQueue failed")
    }

    pub fn device_from_raw(ctx: &wgpu_test::TestingContext) {
        let limits = wgpu::Limits::downlevel_defaults();
        let hal_adapter = unsafe { ctx.adapter.as_hal::<dx12::Api>() }.unwrap();
        let adapter_luid = unsafe { hal_adapter.as_raw().GetDesc2() }
            .unwrap()
            .AdapterLuid;

        let raw_device = native_create_device(hal_adapter.as_raw());
        let raw_queue = native_create_queue(&raw_device, D3D12_COMMAND_LIST_TYPE_DIRECT);

        // A queue that is not a direct queue is rejected.
        let compute_queue = native_create_queue(&raw_device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
        assert!(unsafe {
            hal_adapter.device_from_raw(
                raw_device.clone(),
                compute_queue,
                wgpu::Features::empty(),
                &limits,
                &wgpu::MemoryHints::Performance,
            )
        }
        .is_err());

        // Check a second adapter is available, and if so, check that a device or queue from that adapter is rejected.
        let factory: IDXGIFactory4 =
            unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }.unwrap();
        let other_adapter = (0..)
            .map_while(|i| unsafe { factory.EnumAdapters1(i) }.ok())
            .chain(unsafe { factory.EnumWarpAdapter::<IDXGIAdapter1>() }.ok())
            .find(|adapter| unsafe { adapter.GetDesc1() }.unwrap().AdapterLuid != adapter_luid);
        match other_adapter {
            Some(other_adapter) => {
                let other_device = native_create_device(&other_adapter);
                let other_queue =
                    native_create_queue(&other_device, D3D12_COMMAND_LIST_TYPE_DIRECT);

                // A device of another adapter is rejected.
                assert!(unsafe {
                    hal_adapter.device_from_raw(
                        other_device,
                        other_queue.clone(),
                        wgpu::Features::empty(),
                        &limits,
                        &wgpu::MemoryHints::Performance,
                    )
                }
                .is_err());

                // A queue of another adapter is rejected too.
                assert!(unsafe {
                    hal_adapter.device_from_raw(
                        raw_device.clone(),
                        other_queue,
                        wgpu::Features::empty(),
                        &limits,
                        &wgpu::MemoryHints::Performance,
                    )
                }
                .is_err());
            }
            None => log::warn!("no second adapter, skipping the mismatched adapter checks"),
        }

        let queue_refs = native_ref_count(&raw_queue);

        let hal_device = unsafe {
            hal_adapter.device_from_raw(
                raw_device.clone(),
                raw_queue.clone(),
                wgpu::Features::empty(),
                &limits,
                &wgpu::MemoryHints::Performance,
            )
        }
        .unwrap();

        drop(hal_adapter);
        let (device, queue) = unsafe {
            ctx.adapter.create_device_from_hal(
                hal_device,
                &wgpu::DeviceDescriptor {
                    required_limits: limits,
                    ..Default::default()
                },
            )
        }
        .unwrap();

        // Check .as_hal returns the same raw device and queue
        assert_eq!(
            unsafe { device.as_hal::<dx12::Api>() }
                .unwrap()
                .raw_device()
                .as_raw(),
            raw_device.as_raw()
        );
        assert_eq!(
            unsafe { queue.as_hal::<dx12::Api>() }
                .unwrap()
                .as_raw()
                .as_raw(),
            raw_queue.as_raw()
        );

        queue.submit([device.create_command_encoder(&Default::default()).finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        drop(queue);
        drop(device);

        assert_eq!(
            native_ref_count(&raw_queue),
            queue_refs,
            "wgpu leaked or over-released the queue"
        );

        // The device and queue still work after wgpu is done with them.
        let fence: ID3D12Fence =
            unsafe { raw_device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        unsafe {
            raw_queue.Signal(&fence, 1).unwrap();
            // A null event makes this call block until the fence is signaled.
            fence.SetEventOnCompletion(1, HANDLE::default()).unwrap();
            raw_device.GetDeviceRemovedReason().unwrap();
        }
    }
}
