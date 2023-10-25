use std::{
    ffi::{OsStr, OsString},
    fmt::{self, Display},
    iter::once,
    ops::{Deref, DerefMut},
    process::Command,
};

use anyhow::{ensure, Context};

#[derive(Debug)]
pub(crate) struct EasyCommand {
    inner: Command,
}

impl EasyCommand {
    pub fn new<C>(cmd: C, config: impl FnOnce(&mut Command) -> &mut Command) -> Self
    where
        C: AsRef<OsStr>,
    {
        let mut inner = Command::new(cmd);
        config(&mut inner);
        Self { inner }
    }

    pub fn simple<C, A, I>(cmd: C, args: I) -> Self
    where
        C: AsRef<OsStr>,
        A: AsRef<OsStr>,
        I: IntoIterator<Item = A>,
    {
        Self::new(cmd, |cmd| cmd.args(args))
    }

    pub fn success(&mut self) -> anyhow::Result<()> {
        let Self { inner } = self;
        log::debug!("running {inner:?}");
        let status = inner
            .status()
            .with_context(|| format!("failed to run {self}"))?;
        ensure!(
            status.success(),
            "{self} failed to run; exit code: {:?}",
            status.code()
        );
        Ok(())
    }
}

impl Deref for EasyCommand {
    type Target = Command;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for EasyCommand {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub(crate) fn which(binary_name: &str) -> anyhow::Result<OsString> {
    ::which::which(binary_name)
        .with_context(|| format!("unable to find `{binary_name}` binary"))
        .map(|buf| buf.file_name().unwrap().to_owned())
}

impl Display for EasyCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { inner } = self;
        let prog = inner.get_program().to_string_lossy();
        let args = inner.get_args().map(|a| a.to_string_lossy());
        let shell_words = shell_words::join(once(prog).chain(args));
        write!(f, "`{shell_words}`")
    }
}
