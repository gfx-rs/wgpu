use std::path::{Path, PathBuf};

pub(crate) fn join_path<P, I>(iter: I) -> PathBuf
where
    P: AsRef<Path>,
    I: IntoIterator<Item = P>,
{
    let mut path = PathBuf::new();
    path.extend(iter);
    path
}
