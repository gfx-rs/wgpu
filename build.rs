use std::{
    env,
    fs::{read_to_string, OpenOptions},
    io::{Result, Write},
};

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=glsl-keywords.txt");

    let keywords = read_to_string("glsl-keywords.txt")?.replace('\n', " ");
    let mut out = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(format!("{}/glsl_keywords.rs", env::var("OUT_DIR").unwrap()))?;

    writeln!(&mut out, "const RESERVED_KEYWORDS: &[&str] = &[")?;

    for keyword in keywords.split(' ') {
        writeln!(&mut out, "\"{}\",", keyword)?;
    }

    writeln!(&mut out, "];")?;

    Ok(())
}
