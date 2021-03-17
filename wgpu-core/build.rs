/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

fn main() {
    // Setup cfg aliases
    cfg_aliases::cfg_aliases! {
        // Vendors/systems
        ios: { target_os = "ios" },
        macos: { target_os = "macos" },
        apple: { any(ios, macos) },

        // Backends
        vulkan: { any(windows, all(unix, not(apple)), feature = "gfx-backend-vulkan") },
        metal: { apple },
        dx12: { windows },
        dx11: { windows },
        gl: { all(not(unix), not(apple), not(windows)) },
    }
}
