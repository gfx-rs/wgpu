//! Unit tests for the [`VirtualBlock`](crate::VirtualBlock) facade.

use crate::{Algorithm, AllocationDesc, AllocationError, CreateError, VirtualBlock};

#[test]
fn tlsf_facade_basic() {
    let mut b = VirtualBlock::<u32>::new(1024, Algorithm::Tlsf).unwrap();
    assert!(b.is_empty());
    let (h, o) = b
        .allocate(
            AllocationDesc {
                size: 256,
                alignment: 16,
                ..Default::default()
            },
            5,
        )
        .unwrap();
    assert_eq!(o % 16, 0);
    assert_eq!(b.allocation_count(), 1);
    assert_eq!(b.allocation_offset(h).unwrap(), o);
    assert_eq!(b.allocation_info(h).unwrap().user_data, 5);
    b.validate().unwrap();
    b.free(h).unwrap();
    assert!(b.is_empty());
    b.validate().unwrap();
}

#[test]
fn zero_size_block_errors() {
    assert_eq!(
        VirtualBlock::<()>::new(0, Algorithm::Tlsf).unwrap_err(),
        CreateError::ZeroSize
    );
}

#[test]
fn tlsf_rejects_upper_address() {
    let mut b = VirtualBlock::<()>::new(1024, Algorithm::Tlsf).unwrap();
    assert_eq!(
        b.allocate(
            AllocationDesc {
                size: 100,
                alignment: 1,
                upper_address: true,
                ..Default::default()
            },
            ()
        )
        .unwrap_err(),
        AllocationError::UpperAddressUnsupported
    );
}

#[test]
fn statistics() {
    let mut b = VirtualBlock::<()>::new(1000, Algorithm::Tlsf).unwrap();
    let (_h1, _o1) = b
        .allocate(
            AllocationDesc {
                size: 100,
                alignment: 1,
                ..Default::default()
            },
            (),
        )
        .unwrap();
    let (_h2, _o2) = b
        .allocate(
            AllocationDesc {
                size: 200,
                alignment: 1,
                ..Default::default()
            },
            (),
        )
        .unwrap();
    let stats = b.statistics();
    assert_eq!(stats.block_count, 1);
    assert_eq!(stats.allocation_count, 2);
    assert_eq!(stats.block_bytes, 1000);
    assert_eq!(stats.allocation_bytes, 300);

    let detailed = b.detailed_statistics();
    assert_eq!(detailed.statistics.allocation_count, 2);
    assert_eq!(detailed.allocation_size_min, 100);
    assert_eq!(detailed.allocation_size_max, 200);
    // One trailing free range of 700.
    assert!(detailed.unused_range_count >= 1);
    assert_eq!(detailed.unused_range_size_max, 700);
}

#[test]
fn clear_via_facade() {
    let mut b = VirtualBlock::<u32>::new(2048, Algorithm::Tlsf).unwrap();
    for i in 0..8 {
        b.allocate(
            AllocationDesc {
                size: 64,
                alignment: 1,
                ..Default::default()
            },
            i,
        )
        .unwrap();
    }
    b.clear();
    assert!(b.is_empty());
    b.validate().unwrap();
}
