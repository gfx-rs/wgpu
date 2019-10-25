/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/* Generated with cbindgen:0.9.1 */

/* DO NOT MODIFY THIS MANUALLY! This file was generated using cbindgen.
 * To generate this file:
 *   1. Get the latest cbindgen using `cargo install --force cbindgen`
 *      a. Alternatively, you can clone `https://github.com/eqrion/cbindgen` and use a tagged release
 *   2. Run `rustup run nightly cbindgen toolkit/library/rust/ --lockfile Cargo.lock --crate wgpu-remote -o dom/webgpu/ffi/wgpu_ffi_generated.h`
 */

typedef void WGPUEmpty;


#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <new>

enum class WGPUPowerPreference {
  WGPUPowerPreference_Default = 0,
  WGPUPowerPreference_LowPower = 1,
  WGPUPowerPreference_HighPerformance = 2,
};

template<typename B>
struct WGPUAdapter;

struct WGPUClient;

template<typename B>
struct WGPUDevice;

struct WGPUGlobal;

using WGPUDummy = WGPUEmpty;

template<typename T>
using WGPUId = uint64_t;

using WGPUAdapterId = WGPUId<WGPUAdapter<WGPUDummy>>;

using WGPUDeviceId = WGPUId<WGPUDevice<WGPUDummy>>;

struct WGPUInfrastructure {
  WGPUClient *client;
  const uint8_t *error;

  bool operator==(const WGPUInfrastructure& aOther) const {
    return client == aOther.client &&
           error == aOther.error;
  }
};

using WGPUBackendBit = uint32_t;

struct WGPURequestAdapterOptions {
  WGPUPowerPreference power_preference;
  WGPUBackendBit backends;

  bool operator==(const WGPURequestAdapterOptions& aOther) const {
    return power_preference == aOther.power_preference &&
           backends == aOther.backends;
  }
};

extern "C" {

WGPU_INLINE
void wgpu_client_delete(WGPUClient *aClient)
WGPU_FUNC;

WGPU_INLINE
uintptr_t wgpu_client_make_adapter_ids(const WGPUClient *aClient,
                                       WGPUAdapterId *aIds,
                                       uintptr_t aIdLength)
WGPU_FUNC;

WGPU_INLINE
WGPUDeviceId wgpu_client_make_device_id(const WGPUClient *aClient,
                                        WGPUAdapterId aAdapterId)
WGPU_FUNC;

WGPU_INLINE
WGPUInfrastructure wgpu_client_new()
WGPU_FUNC;

WGPU_INLINE
void wgpu_server_delete(WGPUGlobal *aGlobal)
WGPU_FUNC;

WGPU_INLINE
WGPUGlobal *wgpu_server_new()
WGPU_FUNC;

WGPU_INLINE
WGPUAdapterId wgpu_server_request_adapter(const WGPUGlobal *aGlobal,
                                          const WGPURequestAdapterOptions *aDesc,
                                          const WGPUAdapterId *aIds,
                                          uintptr_t aIdLength)
WGPU_FUNC;

} // extern "C"
