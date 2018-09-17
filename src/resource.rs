#[repr(C)]
pub enum TextureFormat {
    R8g8b8a8Unorm = 0,
    R8g8b8a8Uint = 1,
    B8g8r8a8Unorm = 2,
    D32FloatS8Uint = 3,
}

#[repr(C)]
pub enum CompareFunction {
    Never = 0,
    Less = 1,
    Equal = 2,
    LessEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterEqual = 6,
    Always = 7,
}
