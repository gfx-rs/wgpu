use std::fmt::{Error as FmtError, Write};

pub struct Options {
}

#[derive(Debug)]
pub enum Error {
    Format(FmtError)
}

impl From<FmtError> for Error {
    fn from(e: FmtError) -> Self {
        Error::Format(e)
    }
}

impl super::Module {
    pub fn to_msl(&self, _options: &Options) -> Result<String, Error> {
        let mut out = String::new();

        writeln!(out, "#include <metal_stdlib>")?;
        writeln!(out, "#include <simd/simd.h>")?;
        writeln!(out, "using namespace metal;")?;

        Ok(out)
    }
}
