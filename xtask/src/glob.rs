use std::path::Path;

use anyhow::Context;
use glob::glob;

use crate::result::{ErrorStatus, LogIfError};

pub(crate) fn visit_files(
    path: impl AsRef<Path>,
    glob_expr: &str,
    mut f: impl FnMut(&Path) -> anyhow::Result<()>,
) -> ErrorStatus {
    let path = path.as_ref();
    let glob_expr = path.join(glob_expr);
    let glob_expr = glob_expr.to_str().unwrap();

    let mut status = ErrorStatus::NoFailuresFound;
    glob(glob_expr)
        .context("glob pattern {path:?} is invalid")
        .unwrap()
        .for_each(|path_res| {
            if let Some(path) = path_res
                .with_context(|| format!("error while iterating over glob {path:?}"))
                .log_if_err_found(&mut status)
            {
                if path
                    .metadata()
                    .with_context(|| format!("failed to fetch metadata for {path:?}"))
                    .log_if_err_found(&mut status)
                    .map_or(false, |m| m.is_file())
                {
                    f(&path).log_if_err_found(&mut status);
                }
            }
        });
    status
}
