//! Repro for an Arm Mali driver bug. Arms proprietary Vulkan driver keeps memory for every
//! value a timeline semaphore has ever been signaled with, and only releases it when the
//! semaphore is destroyed.
//!
//! So signaling one long-lived timeline semaphore with 1, 2, 3, ... eventually fails.
//!
//! But according to the Vulkan spec this is a valid use.
//! In the spec the only limit on timeline semaphores is maxTimelineSemaphoreValueDifference, and in
//! the 1, 2, 3, ... sequence the difference is always only 1.
//!
//! wgpu signals one timeline semaphore per submission, so without
//! `Workarounds::DISABLE_TIMELINE_SEMAPHORES` any long-running wgpu application on
//! this Arm Mali driver runs into this issue eventually.
//!
//! We ran this on a Mali-G610 MP4 (Rockchip RK3588, Orange Pi 5 Plus), Android 12,
//! driver `v1.g18p0-01eac0` leaking 16 bytes per signal, allocation failure after ~4.2 million signals.
//!
//! To run this on the an android device, push it via adb to e.g. /data/local/tmp to run it from there.
//! Modes (run on the device, e.g. `adb push` the binary to `/data/local/tmp`; no root needed):
//!

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let mut mode = String::from("raw-host");
    let mut recreate_every = None;
    let mut max_signals: u64 = 8 << 20;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            // destroys and recreates the semaphore every N signals
            "--recreate" => recreate_every = args.next().and_then(|s| s.parse().ok()),
            "--max" => max_signals = args.next().and_then(|s| s.parse().ok()).expect("--max N"),
            other => mode = other.to_owned(),
        }
    }
    match mode.as_str() {
        // no wgpu, just vulkan via ash
        "raw-host" => raw::run(false, recreate_every, max_signals),
        // also vulkan via ash, but signals from empty `vkQueueSubmit` batches and waits for each value with `vkWaitSemaphores`, like a renderer
        "raw-submit" => raw::run(true, recreate_every, max_signals),
        // use wgpu submits
        "wgpu" => wgpu_submits(max_signals),
        other => panic!("unknown mode {other:?}, expected raw-host, raw-submit or wgpu"),
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {
    panic!("this repro needs native Vulkan");
}

#[cfg(not(target_arch = "wasm32"))]
const REPORT_EVERY: u64 = 1 << 18;

#[cfg(not(target_arch = "wasm32"))]
fn wgpu_submits(max_submissions: u64) {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
    });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("no Vulkan adapter");
    let info = adapter.get_info();
    println!(
        "adapter: {:?}, driver {:?} {:?}",
        info.name, info.driver, info.driver_info
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();

    let failed = Arc::new(AtomicBool::new(false));
    let lost = failed.clone();
    device.set_device_lost_callback(move |reason, message| {
        println!("device lost: {reason:?}: {message}");
        lost.store(true, Ordering::SeqCst);
    });
    let errored = failed.clone();
    device.on_uncaptured_error(Arc::new(move |error| {
        println!("uncaptured error: {error}");
        errored.store(true, Ordering::SeqCst);
    }));

    let start = std::time::Instant::now();
    for i in 1..=max_submissions {
        queue.submit([]);
        if i % 64 == 0 {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        if failed.load(Ordering::SeqCst) {
            println!(
                "FAILED after {i} submissions ({:.1}s)",
                start.elapsed().as_secs_f64()
            );
            std::process::exit(1);
        }
        if i % REPORT_EVERY == 0 {
            println!(
                "{i:>9} submissions ({:.1}s) ok",
                start.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "OK: {max_submissions} submissions without error ({:.1}s)",
        start.elapsed().as_secs_f64()
    );
}

#[cfg(not(target_arch = "wasm32"))]
mod raw {
    use ash::vk;
    use std::time::Instant;

    /// Number of 1-page permanently kernel-mapped allocations in this process's Mali contexts,
    /// if the kbase debugfs is readable (root only). Purely informational.
    fn mali_permanent_pages() -> Option<usize> {
        let prefix = format!("{}_", std::process::id());
        let mut total = None;
        for entry in std::fs::read_dir("/sys/kernel/debug/mali0/ctx")
            .ok()?
            .flatten()
        {
            if !entry.file_name().to_string_lossy().starts_with(&prefix) {
                continue;
            }
            let allocs = std::fs::read_to_string(entry.path().join("mem_allocs")).ok()?;
            *total.get_or_insert(0) += allocs.lines().filter(|l| l.contains(" 245c00e,")).count();
        }
        total
    }

    fn pages_suffix() -> String {
        mali_permanent_pages()
            .map(|p| format!(", mali permanently mapped pages: {p}"))
            .unwrap_or_default()
    }

    pub fn run(submit: bool, recreate_every: Option<u64>, max_signals: u64) {
        let entry = unsafe { ash::Entry::load() }.expect("failed to load the Vulkan loader");
        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info),
                None,
            )
        }
        .expect("vkCreateInstance");
        let physical_device = unsafe { instance.enumerate_physical_devices() }.unwrap()[0];

        let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
        let mut timeline_props = vk::PhysicalDeviceTimelineSemaphoreProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut driver_props)
            .push_next(&mut timeline_props);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
        let props = props2.properties;
        println!(
            "device: {:?}, vendor 0x{:x}, api {}.{}.{}, driverVersion 0x{:x}",
            props.device_name_as_c_str().unwrap_or_default(),
            props.vendor_id,
            vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version),
            vk::api_version_patch(props.api_version),
            props.driver_version,
        );
        println!(
            "driver: {:?} {:?} {:?}",
            driver_props.driver_id,
            driver_props.driver_name_as_c_str().unwrap_or_default(),
            driver_props.driver_info_as_c_str().unwrap_or_default(),
        );
        println!(
            "maxTimelineSemaphoreValueDifference: {}",
            timeline_props.max_timeline_semaphore_value_difference
        );

        let queue_family =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .expect("no graphics queue") as u32;
        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let extensions = [ash::khr::timeline_semaphore::NAME.as_ptr()];
        let mut timeline_features =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
        let device = unsafe {
            instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_info)
                    .enabled_extension_names(&extensions)
                    .push_next(&mut timeline_features),
                None,
            )
        }
        .expect("vkCreateDevice");
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let timeline = ash::khr::timeline_semaphore::Device::new(&instance, &device);

        let create_semaphore = || {
            let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            unsafe {
                device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut type_info),
                    None,
                )
            }
            .expect("vkCreateSemaphore")
        };

        println!(
            "signaling one timeline semaphore with value + 1 {}, up to {max_signals} times{}{}",
            if submit {
                "from empty vkQueueSubmit batches"
            } else {
                "from the host (vkSignalSemaphore)"
            },
            match recreate_every {
                Some(n) => format!(", recreating it every {n} signals"),
                None => String::new(),
            },
            pages_suffix()
        );

        let mut semaphore = create_semaphore();
        let mut value: u64 = 0;
        let start = Instant::now();
        for i in 1..=max_signals {
            if recreate_every.is_some_and(|n| i % n == 0) {
                unsafe { device.destroy_semaphore(semaphore, None) };
                semaphore = create_semaphore();
                value = 0;
            }
            value += 1;

            let semaphores = [semaphore];
            let values = [value];
            let result = if submit {
                let mut timeline_info =
                    vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
                let submit_info = vk::SubmitInfo::default()
                    .signal_semaphores(&semaphores)
                    .push_next(&mut timeline_info);
                unsafe { device.queue_submit(queue, &[submit_info], vk::Fence::null()) }.and_then(
                    |()| unsafe {
                        timeline.wait_semaphores(
                            &vk::SemaphoreWaitInfo::default()
                                .semaphores(&semaphores)
                                .values(&values),
                            u64::MAX,
                        )
                    },
                )
            } else {
                unsafe {
                    timeline.signal_semaphore(
                        &vk::SemaphoreSignalInfo::default()
                            .semaphore(semaphore)
                            .value(value),
                    )
                }
            };

            if let Err(err) = result {
                println!(
                    "FAILED after {i} signals ({:.1}s): {err:?}{}",
                    start.elapsed().as_secs_f64(),
                    pages_suffix()
                );
                std::process::exit(1);
            }
            if i % super::REPORT_EVERY == 0 {
                println!(
                    "{i:>9} signals ({:.1}s) ok{}",
                    start.elapsed().as_secs_f64(),
                    pages_suffix()
                );
            }
        }
        println!(
            "OK: {max_signals} signals without error ({:.1}s)",
            start.elapsed().as_secs_f64()
        );
        unsafe {
            device.destroy_semaphore(semaphore, None);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }
}
