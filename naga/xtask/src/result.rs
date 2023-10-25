#[derive(Clone, Copy, Debug)]
pub(crate) enum ErrorStatus {
    NoFailuresFound,
    OneOrMoreFailuresFound,
}

impl ErrorStatus {
    pub(crate) fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::OneOrMoreFailuresFound, _) | (_, Self::OneOrMoreFailuresFound) => {
                Self::OneOrMoreFailuresFound
            }
            (Self::NoFailuresFound, Self::NoFailuresFound) => Self::NoFailuresFound,
        }
    }
}

pub(crate) trait LogIfError<T> {
    fn log_if_err_found(self, status: &mut ErrorStatus) -> Option<T>;
}

impl<T> LogIfError<T> for anyhow::Result<T> {
    fn log_if_err_found(self, status: &mut ErrorStatus) -> Option<T> {
        match self {
            Ok(t) => Some(t),
            Err(e) => {
                log::error!("{e:?}");
                *status = status.merge(ErrorStatus::OneOrMoreFailuresFound);
                None
            }
        }
    }
}
