use alloc::{borrow::Cow, vec};
use core::{mem, ptr};

#[cfg(doc)]
use crate::Device;

const SPIRV_MAGIC_NUMBER: u32 = 0x0723_0203;

/// Treat the given byte slice as a SPIR-V module.
///
/// # Panic
///
/// This function panics if:
///
/// - Input length isn't multiple of 4
/// - Input is longer than [`usize::MAX`]
/// - Input is empty
/// - SPIR-V magic number is missing from beginning of stream
#[cfg(feature = "spirv")]
pub fn make_spirv(data: &[u8]) -> crate::ShaderSource<'_> {
    crate::ShaderSource::SpirV(make_spirv_raw(data))
}

const fn check_spirv_len(data: &[u8]) {
    assert!(
        data.len() % size_of::<u32>() == 0,
        "SPIRV data size must be a multiple of 4."
    );
    assert!(!data.is_empty(), "SPIRV data must not be empty.");
}

const fn verify_spirv_magic(words: &[u32]) {
    assert!(
        words[0] == SPIRV_MAGIC_NUMBER,
        "Wrong magic word in data. Make sure you are using a binary SPIRV file.",
    );
}

/// Version of `make_spirv` intended for use with [`Device::create_shader_module_passthrough`].
/// Returns a raw slice instead of [`ShaderSource`](super::ShaderSource).
///
/// [`Device::create_shader_module_passthrough`]: crate::Device::create_shader_module_passthrough
pub fn make_spirv_raw(data: &[u8]) -> Cow<'_, [u32]> {
    check_spirv_len(data);

    // If the data happens to be aligned, directly use the byte array,
    // otherwise copy the byte array in an owned vector and use that instead.
    let mut words = if data.as_ptr().align_offset(align_of::<u32>()) == 0 {
        let (pre, words, post) = unsafe { data.align_to::<u32>() };
        debug_assert!(pre.is_empty());
        debug_assert!(post.is_empty());
        Cow::from(words)
    } else {
        let mut words = vec![0u32; data.len() / size_of::<u32>()];
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), words.as_mut_ptr() as *mut u8, data.len());
        }
        Cow::from(words)
    };

    // Before checking if the data starts with the magic, check if it starts
    // with the magic in non-native endianness, own & swap the data if so.
    if words[0] == SPIRV_MAGIC_NUMBER.swap_bytes() {
        for word in Cow::to_mut(&mut words) {
            *word = word.swap_bytes();
        }
    }

    verify_spirv_magic(&words);

    words
}

/// Version of `make_spirv_raw` used for implementing [`include_spirv!`] and [`include_spirv_raw!`] macros.
///
/// Not public API. Also, don't even try calling at runtime; you'll get a stack overflow.
///
/// [`include_spirv!`]: crate::include_spirv
#[doc(hidden)]
pub const fn make_spirv_const<const IN: usize, const OUT: usize>(data: [u8; IN]) -> [u32; OUT] {
    #[repr(align(4))]
    struct Aligned<T: ?Sized>(T);

    check_spirv_len(&data);

    // NOTE: to get around lack of generic const expressions
    assert!(IN / 4 == OUT);

    let aligned = Aligned(data);
    let mut words: [u32; OUT] = unsafe { mem::transmute_copy(&aligned) };

    // Before checking if the data starts with the magic, check if it starts
    // with the magic in non-native endianness, own & swap the data if so.
    if words[0] == SPIRV_MAGIC_NUMBER.swap_bytes() {
        let mut idx = 0;
        while idx < words.len() {
            words[idx] = words[idx].swap_bytes();
            idx += 1;
        }
    }

    verify_spirv_magic(&words);

    words
}

#[should_panic = "multiple of 4"]
#[test]
fn make_spirv_le_fail() {
    let _: [u32; 1] = make_spirv_const([0x03, 0x02, 0x23, 0x07, 0x44, 0x33]);
}

#[should_panic = "multiple of 4"]
#[test]
fn make_spirv_be_fail() {
    let _: [u32; 1] = make_spirv_const([0x07, 0x23, 0x02, 0x03, 0x11, 0x22]);
}

#[should_panic = "empty"]
#[test]
fn make_spirv_empty() {
    let _: [u32; 0] = make_spirv_const([]);
}
