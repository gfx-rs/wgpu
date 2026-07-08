//! Unit tests for the TLSF allocator.

use crate::{
    AllocationDesc, AllocationError, AllocationType, CreateError, Strategy, Suballocator, Tlsf,
};

/// Convenience: allocate `size`/`alignment` with the balanced strategy, non-upper,
/// Unknown type, returning `(handle, offset)`.
fn alloc(
    t: &mut Tlsf<u32>,
    size: u64,
    alignment: u64,
    user_data: u32,
) -> Result<(crate::AllocationHandle, u64), AllocationError> {
    let req = t.create_allocation_request(AllocationDesc {
        size,
        alignment,
        ..Default::default()
    })?;
    let offset = req.offset;
    let handle = t.alloc(req, user_data);
    Ok((handle, offset))
}

#[test]
fn new_validates_args() {
    assert_eq!(
        Tlsf::<()>::new(0, 1, true, 0).unwrap_err(),
        CreateError::ZeroSize
    );
    assert_eq!(
        Tlsf::<()>::new(1024, 0, true, 0).unwrap_err(),
        CreateError::ZeroGranularity
    );
    assert_eq!(
        Tlsf::<()>::new(1024, 3, false, 0).unwrap_err(),
        CreateError::GranularityNotPowerOfTwo
    );
    assert_eq!(
        Tlsf::<()>::new(1024, 1, false, 3).unwrap_err(),
        CreateError::DebugMarginNotMultipleOfFour
    );
    assert!(Tlsf::<()>::new(1024, 1, true, 0).is_ok());
}

#[test]
fn basic_alloc_free_cycle() {
    let mut t = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
    assert!(t.is_empty());
    t.validate().unwrap();

    let (h1, o1) = alloc(&mut t, 100, 1, 1).unwrap();
    let (h2, o2) = alloc(&mut t, 200, 1, 2).unwrap();
    let (h3, o3) = alloc(&mut t, 50, 1, 3).unwrap();
    t.validate().unwrap();

    assert_eq!(t.allocation_count(), 3);
    assert!(!t.is_empty());
    assert!(o1 + 100 <= o2 || o2 + 200 <= o1);
    assert!(o2 + 200 <= o3 || o3 + 50 <= o2);

    t.free(h2).unwrap();
    t.validate().unwrap();
    assert_eq!(t.allocation_count(), 2);

    let (h4, _o4) = alloc(&mut t, 150, 1, 4).unwrap();
    t.validate().unwrap();

    t.free(h1).unwrap();
    t.free(h3).unwrap();
    t.free(h4).unwrap();
    t.validate().unwrap();
    assert!(t.is_empty());
    assert_eq!(t.sum_free_size(), 1024);
}

#[test]
fn exact_fit_whole_block() {
    let mut t = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
    let (h, o) = alloc(&mut t, 1024, 1, 1).unwrap();
    assert_eq!(o, 0);
    t.validate().unwrap();
    assert_eq!(t.sum_free_size(), 0);
    assert_eq!(
        alloc(&mut t, 1, 1, 2).unwrap_err(),
        AllocationError::OutOfSpace
    );
    t.free(h).unwrap();
    t.validate().unwrap();
    assert!(t.is_empty());
}

#[test]
fn alignment_padding_creates_filler() {
    let mut t = Tlsf::<u32>::new(4096, 1, true, 0).unwrap();
    let (_h1, o1) = alloc(&mut t, 10, 1, 1).unwrap();
    assert_eq!(o1, 0);
    let (_h2, o2) = alloc(&mut t, 100, 256, 2).unwrap();
    assert_eq!(o2 % 256, 0);
    assert!(o2 >= 10);
    t.validate().unwrap();
    // The gap [10, 256) should be reusable.
    let (_h3, o3) = alloc(&mut t, 8, 1, 3).unwrap();
    assert!((10..256).contains(&o3));
    t.validate().unwrap();
}

#[test]
fn zero_size_and_bad_alignment_error() {
    let mut t = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
    assert_eq!(
        t.create_allocation_request(AllocationDesc {
            size: 0,
            alignment: 1,
            ..Default::default()
        })
        .unwrap_err(),
        AllocationError::InvalidSize
    );
    assert_eq!(
        t.create_allocation_request(AllocationDesc {
            size: 10,
            alignment: 3,
            ..Default::default()
        })
        .unwrap_err(),
        AllocationError::InvalidAlignment
    );
}

#[test]
fn upper_address_rejected() {
    let mut t = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
    assert_eq!(
        t.create_allocation_request(AllocationDesc {
            size: 10,
            alignment: 1,
            upper_address: true,
            ..Default::default()
        })
        .unwrap_err(),
        AllocationError::UpperAddressUnsupported
    );
}

#[test]
fn huge_size_no_overflow() {
    let mut t = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
    assert_eq!(
        t.create_allocation_request(AllocationDesc {
            size: u64::MAX,
            alignment: 1,
            ..Default::default()
        })
        .unwrap_err(),
        AllocationError::OutOfSpace
    );
    // Occupy offset 0 so a subsequent 4096-aligned request must go to offset 4096,
    // which is past the 1024-unit block: must fail cleanly (no overflow/panic).
    let (_h, o) = alloc(&mut t, 10, 1, 1).unwrap();
    assert_eq!(o, 0);
    assert_eq!(
        alloc(&mut t, 10, 4096, 2).unwrap_err(),
        AllocationError::OutOfSpace
    );
    t.validate().unwrap();
}

#[test]
fn near_max_block_size_no_overflow() {
    // A block near u64::MAX exercises the widened top-level bitmap and align_up
    // saturation without panicking.
    let mut t = Tlsf::<u32>::new(u64::MAX, 1, true, 0).unwrap();
    t.validate().unwrap();
    let (_h, o) = alloc(&mut t, 4096, 4096, 1).unwrap();
    assert_eq!(o % 4096, 0);
    t.validate().unwrap();
}

#[test]
fn clear_resets() {
    let mut t = Tlsf::<u32>::new(2048, 1, true, 0).unwrap();
    for i in 0..10 {
        alloc(&mut t, 64, 1, i).unwrap();
    }
    t.validate().unwrap();
    assert_eq!(t.allocation_count(), 10);
    t.clear();
    t.validate().unwrap();
    assert!(t.is_empty());
    assert_eq!(t.sum_free_size(), 2048);
    let (_h, o) = alloc(&mut t, 2048, 1, 99).unwrap();
    assert_eq!(o, 0);
    t.validate().unwrap();
}

#[test]
fn user_data_roundtrip() {
    let mut t = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
    let (h, _o) = alloc(&mut t, 100, 1, 42).unwrap();
    assert_eq!(t.allocation_info(h).unwrap().user_data, 42);
    t.set_user_data(h, 7).unwrap();
    assert_eq!(t.allocation_info(h).unwrap().user_data, 7);
    assert_eq!(t.allocation_info(h).unwrap().size, 100);
}

#[test]
fn iteration_visits_all_allocations() {
    let mut t = Tlsf::<u32>::new(1024, 1, true, 0).unwrap();
    let mut handles = alloc::vec::Vec::new();
    for i in 0..5 {
        let (h, _o) = alloc(&mut t, 64, 1, i).unwrap();
        handles.push(h);
    }
    let mut count = 0;
    let mut cur = t.allocation_list_begin();
    while let Some(h) = cur {
        count += 1;
        assert!(handles.contains(&h));
        cur = t.next_allocation(h);
    }
    assert_eq!(count, 5);
}

#[test]
fn debug_margin_accounting() {
    let mut t = Tlsf::<u32>::new(1024, 1, false, 8).unwrap();
    let (h1, o1) = alloc(&mut t, 100, 1, 1).unwrap();
    let (_h2, o2) = alloc(&mut t, 100, 1, 2).unwrap();
    t.validate().unwrap();
    assert!(o2 >= o1 + 100 + 8);
    assert_eq!(crate::debug_margin_offset(o1, 100), o1 + 100);
    t.free(h1).unwrap();
    t.validate().unwrap();
}

#[test]
fn granularity_rounding_low() {
    let mut t = Tlsf::<u32>::new(4096, 128, false, 0).unwrap();
    let (_h, o) = alloc(&mut t, 10, 1, 1).unwrap();
    assert_eq!(o % 128, 0);
    t.validate().unwrap();
}

#[test]
fn granularity_conflict_page_tracking() {
    let mut t = Tlsf::<u32>::new(8192, 512, false, 0).unwrap();
    let buf = t
        .create_allocation_request(AllocationDesc {
            size: 100,
            alignment: 1,
            alloc_type: AllocationType::Buffer,
            ..Default::default()
        })
        .unwrap();
    let buf_offset = buf.offset;
    let _bh = t.alloc(buf, 1);
    t.validate().unwrap();

    let img = t
        .create_allocation_request(AllocationDesc {
            size: 100,
            alignment: 1,
            alloc_type: AllocationType::ImageOptimal,
            ..Default::default()
        })
        .unwrap();
    let img_offset = img.offset;
    let _ih = t.alloc(img, 2);
    t.validate().unwrap();

    let page = |o: u64| o / 512;
    assert_ne!(page(buf_offset), page(img_offset + 99));
}

#[test]
fn strategies_all_succeed() {
    for strategy in [
        Strategy::Balanced,
        Strategy::MinMemory,
        Strategy::MinTime,
        Strategy::MinOffset,
    ] {
        let mut t = Tlsf::<u32>::new(4096, 1, true, 0).unwrap();
        let mut handles = alloc::vec::Vec::new();
        for i in 0..20 {
            let req = t
                .create_allocation_request(AllocationDesc {
                    size: 50,
                    alignment: 8,
                    strategy,
                    ..Default::default()
                })
                .unwrap();
            handles.push(t.alloc(req, i));
        }
        t.validate().unwrap();
        for h in handles.iter().step_by(2) {
            t.free(*h).unwrap();
        }
        t.validate().unwrap();
        for i in 0..5 {
            let req = t
                .create_allocation_request(AllocationDesc {
                    size: 40,
                    alignment: 8,
                    strategy,
                    ..Default::default()
                })
                .unwrap();
            t.alloc(req, 100 + i);
        }
        t.validate().unwrap();
    }
}

#[test]
fn min_offset_prefers_low() {
    let mut t = Tlsf::<u32>::new(4096, 1, true, 0).unwrap();
    let (h1, _o1) = alloc(&mut t, 100, 1, 1).unwrap();
    let (_h2, _o2) = alloc(&mut t, 100, 1, 2).unwrap();
    let (h3, _o3) = alloc(&mut t, 100, 1, 3).unwrap();
    let (_h4, _o4) = alloc(&mut t, 100, 1, 4).unwrap();
    t.free(h1).unwrap();
    t.free(h3).unwrap();
    t.validate().unwrap();
    let req = t
        .create_allocation_request(AllocationDesc {
            size: 50,
            alignment: 1,
            strategy: Strategy::MinOffset,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(req.offset, 0);
    t.alloc(req, 5);
    t.validate().unwrap();
}
