/*!
Keywords for Rust.

This is a list of reserved keywords in Rust, including types and language constructs.
*/

pub const RESERVED: &[&str] = &[
    // Fundamental language constructs.
    "let", "fn", "struct", "enum", "impl", "match", "mod", "pub", "crate", "extern", "use", "super",
    "self", "Self", // Primitive types.
    "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize", "f32",
    "f64", "bool", "char", "str", // Other keywords.
    "const", "static", "mut", "ref", "box", "move", "as", "trait", "type", "unsafe", "async",
    "await", "dyn", "loop", "for", "while", "if", "else", "return", "break", "continue", "default",
    "where", "macro", // Macros and special symbols.
    "println!", "print!", "format!", "vec!",
    // Additional control flow keywords.
    "in", "try", "catch", // Future reserved keywords (as of Rust 2021 Edition).
    "abstract", "become", "do", "final", "macro", "override", "priv", "typeof", "unsized",
    "virtual", "yield", // Additional keywords.
    "alignof", "offsetof", "proc", "pure", "sizeof", "typeof",
];
