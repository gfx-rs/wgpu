use spirv::Word;

pub(crate) fn bytes_to_words(bytes: &[u8]) -> Vec<Word> {
    bytes
        .chunks(4)
        .map(|chars| chars.iter().rev().fold(0u32, |u, c| (u << 8) | *c as u32))
        .collect()
}

pub(crate) fn string_to_words(input: &str) -> Vec<Word> {
    let bytes = input.as_bytes();
    let mut words = bytes_to_words(bytes);

    if bytes.len() % 4 == 0 {
        // nul-termination
        words.push(0x0u32);
    }

    words
}
