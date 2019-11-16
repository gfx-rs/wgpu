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


#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum {
  WGPUPowerPreference_Default = 0,
  WGPUPowerPreference_LowPower = 1,
  WGPUPowerPreference_HighPerformance = 2,
} WGPUPowerPreference;

typedef struct WGPUClient WGPUClient;

typedef uint64_t WGPUId_Adapter_Dummy;

typedef WGPUId_Adapter_Dummy WGPUAdapterId;

typedef uint64_t WGPUId_Device_Dummy;

typedef WGPUId_Device_Dummy WGPUDeviceId;

typedef struct {
  WGPUClient *client;
  const uint8_t *error;
} WGPUInfrastructure;

typedef struct {
  bool anisotropic_filtering;
} WGPUExtensions;

typedef struct {
  uint32_t max_bind_groups;
} WGPULimits;

typedef struct {
  WGPUExtensions extensions;
  WGPULimits limits;
} WGPUDeviceDescriptor;

typedef struct {
  WGPUPowerPreference power_preference;
} WGPURequestAdapterOptions;

WGPU_INLINE
void wgpu_client_delete(WGPUClient *aClient)
WGPU_FUNC;

WGPU_INLINE
void wgpu_client_kill_adapter_ids(const WGPUClient *aClient,
                                  const WGPUAdapterId *aIds,
                                  uintptr_t aIdLength)
WGPU_FUNC;

WGPU_INLINE
void wgpu_client_kill_device_id(const WGPUClient *aClient,
                                WGPUDeviceId aId)
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
WGPUInfrastructure wgpu_client_new(void)
WGPU_FUNC;

WGPU_INLINE
void wgpu_server_adapter_request_device(const WGPUGlobal *aGlobal,
                                        WGPUAdapterId aSelfId,
                                        const WGPUDeviceDescriptor *aDesc,
                                        WGPUDeviceId aNewId)
WGPU_FUNC;

WGPU_INLINE
void wgpu_server_delete(WGPUGlobal *aGlobal)
WGPU_FUNC;

WGPU_INLINE
void wgpu_server_device_destroy(const WGPUGlobal *aGlobal,
                                WGPUDeviceId aSelfId)
WGPU_FUNC;

/**
 * Request an adapter according to the specified options.
 * Provide the list of IDs to pick from.
 *
 * Returns the index in this list, or -1 if unable to pick.
 */
WGPU_INLINE
int8_t wgpu_server_instance_request_adapter(const WGPUGlobal *aGlobal,
                                            const WGPURequestAdapterOptions *aDesc,
                                            const WGPUAdapterId *aIds,
                                            uintptr_t aIdLength)
WGPU_FUNC;

WGPU_INLINE
WGPUGlobal *wgpu_server_new(void)
WGPU_FUNC;
