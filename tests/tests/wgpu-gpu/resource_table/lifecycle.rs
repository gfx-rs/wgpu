//! Lifetime and slot-reuse-gating tests.
//!
//! Covers the **C2** regression (a table-member texture whose teardown must be
//! deferred while a submission that reaches it through the table is in flight,
//! VVL-clean at retire — see `plans/m0-notes.md` accepted wart on
//! VkImage-vs-VkImageView retire ordering) and the `SlotInUse` reuse gate
//! (Invariant 2): a slot used by a submission cannot be rewritten until the
//! device observes that submission complete.
//!
//! These rely on submissions being genuinely *in flight*: the device's cached
//! completed-submission index only advances in `Device::maintain` (i.e. on
//! `poll`), so between `submit` and the next `poll` every slot of a bound table
//! is gated and every destroyed member texture's teardown is deferred.

use wgpu::*;
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer};

use super::common::{make_red_texture, table_params, texture_red, Sampler};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        RESOURCE_TABLE_DESTROY_MEMBER_TEXTURE_IN_FLIGHT,
        RESOURCE_TABLE_DROP_MEMBER_HANDLE_IN_FLIGHT,
        RESOURCE_TABLE_DESTROY_TABLE_IN_FLIGHT,
        RESOURCE_TABLE_DESTROY_AFTER_COMPLETION,
        RESOURCE_TABLE_SLOT_IN_USE_THEN_POLL,
    ]);
}

/// **C2**: `texture.destroy()` on a table member while a submission that samples
/// it through the table is in flight. The hal teardown of the VkImage/ImageView
/// must be deferred until the submission retires, so the GPU still reads the
/// live image (correct result) and no validation error fires.
#[apply(gpu_test!)]
static RESOURCE_TABLE_DESTROY_MEMBER_TEXTURE_IN_FLIGHT: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            const RED: u8 = texture_red(0);

            let (texture, view) = make_red_texture(&ctx, RED);
            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: None,
                size: 4,
            });
            table.update(0, &view).expect("bind");

            let sampler = Sampler::new(&ctx, &[0]);
            sampler.submit(&ctx, &table);

            // Destroy the member texture while the submission is still in flight
            // (no poll yet). Teardown must be deferred to retire.
            texture.destroy();

            // read() polls to completion; if teardown ordering is wrong the
            // validation canary fires here. The result must still be correct.
            let got = sampler.read(&ctx).await;
            assert_eq!(got, vec![RED as u32]);
        });

/// **C2** (keep-alive variant): drop the last *user* handles to a table member
/// while the referencing submission is in flight. The table holds `Arc`s to its
/// resources (Invariant 5), so the core resource stays alive and the read is
/// correct; nothing is torn down early.
#[apply(gpu_test!)]
static RESOURCE_TABLE_DROP_MEMBER_HANDLE_IN_FLIGHT: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            const RED: u8 = texture_red(1);

            let (texture, view) = make_red_texture(&ctx, RED);
            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: None,
                size: 4,
            });
            table.update(0, &view).expect("bind");

            let sampler = Sampler::new(&ctx, &[0]);
            sampler.submit(&ctx, &table);

            // Drop the last user references while in flight. The table's Arc keeps
            // the resource alive.
            drop(view);
            drop(texture);

            let got = sampler.read(&ctx).await;
            assert_eq!(got, vec![RED as u32]);
        });

/// **C2** (table variant): destroy the *table* itself while a submission that
/// binds it is in flight. The hal descriptor set must survive until the
/// submission retires.
#[apply(gpu_test!)]
static RESOURCE_TABLE_DESTROY_TABLE_IN_FLIGHT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(table_params())
    .run_async(|ctx| async move {
        const RED: u8 = texture_red(2);

        let (_texture, view) = make_red_texture(&ctx, RED);
        let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
            label: None,
            size: 4,
        });
        table.update(0, &view).expect("bind");

        let sampler = Sampler::new(&ctx, &[0]);
        sampler.submit(&ctx, &table);

        // Destroy the table while its submission is in flight.
        table.destroy();

        let got = sampler.read(&ctx).await;
        assert_eq!(got, vec![RED as u32]);
    });

/// The clean-path counterpart: after the submission completes, the slot frees,
/// `remove_binding` succeeds, and destroying the member texture (twice, to
/// confirm idempotence) is a normal, non-deferred operation.
#[apply(gpu_test!)]
static RESOURCE_TABLE_DESTROY_AFTER_COMPLETION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(table_params())
    .run_async(|ctx| async move {
        const RED: u8 = texture_red(3);

        let (texture, view) = make_red_texture(&ctx, RED);
        let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
            label: None,
            size: 4,
        });
        table.update(0, &view).expect("bind");

        let sampler = Sampler::new(&ctx, &[0]);
        sampler.submit(&ctx, &table);

        // Wait for completion, then verify the result.
        let got = sampler.read(&ctx).await;
        assert_eq!(got, vec![RED as u32]);

        // Slot is free now; remove_binding succeeds and destroy is normal.
        table.remove_binding(0).expect("free slot after completion");
        texture.destroy();
        // Destroying again is a no-op.
        texture.destroy();
    });

/// The `SlotInUse` reuse gate (Invariant 2). Submit work that binds a table with
/// slot 0 populated; before polling, `update(0)` must report `SlotInUse`. After
/// a `poll(wait)` observes the submission complete, the update succeeds.
///
/// Also asserts two facts about the marking granularity:
/// * A slot in an *unrelated* table (never submitted) is freely updatable while
///   the first table's submission is in flight — the gate is per-submission, not
///   a global device-wide freeze.
/// * M0 conservatively marks **every** slot of a bound table in use (Invariant 6
///   permits over-approximation), so even an *empty* slot of the in-flight table
///   is gated until the submission completes.
#[apply(gpu_test!)]
static RESOURCE_TABLE_SLOT_IN_USE_THEN_POLL: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(table_params())
    .run_async(|ctx| async move {
        const RED: u8 = texture_red(0);

        let (_texture, view) = make_red_texture(&ctx, RED);
        let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
            label: None,
            size: 4,
        });
        table.update(0, &view).expect("initial bind");

        let sampler = Sampler::new(&ctx, &[0]);
        sampler.submit(&ctx, &table);

        // In flight (not polled): slot 0 is gated.
        let err = table
            .update(0, &view)
            .expect_err("slot 0 is used by an in-flight submission");
        assert!(
            matches!(err, ResourceTableError::SlotInUse { .. }),
            "expected SlotInUse, got {err:?}"
        );

        // Conservative marking: even an empty slot (3) of the same bound table is
        // gated while in flight.
        let err = table
            .update(3, &view)
            .expect_err("empty slots of a bound table are conservatively gated");
        assert!(
            matches!(err, ResourceTableError::SlotInUse { .. }),
            "expected SlotInUse for empty slot, got {err:?}"
        );

        // An unrelated, never-submitted table is not gated.
        let other = ctx.device.create_resource_table(&ResourceTableDescriptor {
            label: Some("unrelated table"),
            size: 2,
        });
        other
            .update(0, &view)
            .expect("an unrelated table's slot is updatable while another is in flight");

        // Observe completion, then the slot is reusable.
        ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
        table
            .update(0, &view)
            .expect("slot reusable after the submission completes");

        // The originally submitted work read the texture correctly.
        let got = sampler.read(&ctx).await;
        assert_eq!(got, vec![RED as u32]);
    });
