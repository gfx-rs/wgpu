use std::{fs::File, path::Path};

use anyhow::Context;

pub(crate) use std::fs::*;

pub(crate) fn open_file(path: impl AsRef<Path>) -> anyhow::Result<File> {
    let path = path.as_ref();
    File::open(path).with_context(|| format!("failed to open {path:?}"))
}
