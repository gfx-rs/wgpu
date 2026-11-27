//! Acquiring and releasing textures whose memory another device or the kernel
//! owns between uses ([`TextureOwnership::Foreign`]).
//!
//! Each command buffer that uses such a texture acquires it from
//! `VK_QUEUE_FAMILY_FOREIGN_EXT` before its first use and releases it back
//! after its last, so the producer's writes between submissions are seen.
//! Between command buffers the device tracker holds the texture at
//! [`TextureOwnership::FOREIGN_RESTING`], whose layout is `GENERAL`: the one
//! layout an acquire from a foreign queue family keeps the contents in.
//!
//! The pair is never left out, whatever the uses: two command buffers that
//! read the texture the same way still each acquire and release it, since the
//! transfer is of ownership, not a change of use.

use alloc::{sync::Arc, vec::Vec};

use super::{DeviceTextureTracker, TextureTracker};
use crate::{
    resource::{RawResourceAccess as _, Texture, TextureOwnership},
    snatch::SnatchGuard,
};

/// A barrier that transfers all of a foreign texture, resting, to or from its
/// producer, for when the tracker needs no change of use.
fn resting_transfer(
    texture: &dyn hal::DynTexture,
    transfer: hal::QueueFamilyOwnershipTransfer,
) -> hal::TextureBarrier<'_, dyn hal::DynTexture> {
    hal::TextureBarrier {
        texture,
        range: wgt::ImageSubresourceRange {
            aspect: wgt::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        },
        usage: hal::StateTransition {
            from: TextureOwnership::FOREIGN_RESTING,
            to: TextureOwnership::FOREIGN_RESTING,
        },
        queue_family_ownership_transfer: Some(transfer),
    }
}

/// The transfer from the producer to this device's `queue_family`.
fn acquire_transfer(queue_family: u32) -> hal::QueueFamilyOwnershipTransfer {
    hal::QueueFamilyOwnershipTransfer {
        src: hal::QueueFamily::Foreign,
        dst: hal::QueueFamily::Explicit(queue_family),
    }
}

/// Make `barriers`, which bring the device tracker to the start of a command
/// buffer whose tracker is `head`, also acquire each foreign texture `head`
/// uses: the tracker's own barrier for it gains the transfer, and one is added
/// where the tracker needed none.
pub(crate) fn acquire<'a>(
    head: &'a TextureTracker,
    barriers: &mut Vec<hal::TextureBarrier<'a, dyn hal::DynTexture>>,
    snatch_guard: &'a SnatchGuard<'a>,
) {
    for texture in head.used_resources() {
        let TextureOwnership::Foreign { queue_family } = texture.ownership else {
            continue;
        };
        let Ok(raw) = texture.try_raw(snatch_guard) else {
            continue;
        };
        let transfer = acquire_transfer(queue_family);
        let mut found = false;
        for barrier in barriers.iter_mut() {
            if core::ptr::addr_eq(barrier.texture, raw) {
                barrier.queue_family_ownership_transfer = Some(transfer);
                found = true;
            }
        }
        if !found {
            barriers.push(resting_transfer(raw, transfer));
        }
    }
}

/// The foreign textures `head` uses, which the command buffer must release
/// after its last use.
pub(crate) fn used(head: &TextureTracker) -> Vec<Arc<Texture>> {
    head.used_resources()
        .filter(|texture| matches!(texture.ownership, TextureOwnership::Foreign { .. }))
        .cloned()
        .collect()
}

/// The barriers that release each of `foreign` back to its producer after a
/// command buffer, and bring the device tracker back to
/// [`TextureOwnership::FOREIGN_RESTING`] for them.
pub(crate) fn release<'a>(
    device: &mut DeviceTextureTracker,
    foreign: &'a [Arc<Texture>],
    snatch_guard: &'a SnatchGuard<'a>,
) -> Vec<hal::TextureBarrier<'a, dyn hal::DynTexture>> {
    let mut barriers = Vec::new();
    for texture in foreign {
        let TextureOwnership::Foreign { queue_family } = texture.ownership else {
            continue;
        };
        let Ok(raw) = texture.try_raw(snatch_guard) else {
            continue;
        };
        let transfer = hal::QueueFamilyOwnershipTransfer {
            src: hal::QueueFamily::Explicit(queue_family),
            dst: hal::QueueFamily::Foreign,
        };
        let pending = device
            .set_single(
                texture,
                texture.full_range.clone(),
                TextureOwnership::FOREIGN_RESTING,
            )
            .collect::<Vec<_>>();
        if pending.is_empty() {
            barriers.push(resting_transfer(raw, transfer));
        }
        for transition in pending {
            let mut barrier = transition.into_hal(raw);
            barrier.queue_family_ownership_transfer = Some(transfer);
            barriers.push(barrier);
        }
    }
    barriers
}
