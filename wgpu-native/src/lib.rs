/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::sync::Arc;

mod command;
mod device;

pub use self::command::*;
pub use self::device::*;

type Global = core::hub::Global<core::hub::IdentityManagerFactory>;

lazy_static::lazy_static! {
    static ref GLOBAL: Arc<Global> = Arc::new(Global::new("wgpu", core::hub::IdentityManagerFactory));
}
