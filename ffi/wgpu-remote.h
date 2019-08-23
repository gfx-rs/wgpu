

/* Generated with cbindgen:0.9.0 */

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

typedef struct WGPUClientFactory WGPUClientFactory;

typedef struct WGPUServer WGPUServer;

typedef uint64_t WGPUId_DeviceHandle;

typedef WGPUId_DeviceHandle WGPUDeviceId;

typedef uint64_t WGPUId_AdapterHandle;

typedef WGPUId_AdapterHandle WGPUAdapterId;

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

typedef struct {
  WGPUClientFactory *factory;
  WGPUServer *server;
  const uint8_t *error;
} WGPUInfrastructure;

WGPUDeviceId wgpu_client_adapter_create_device(const WGPUClient *client,
                                               WGPUAdapterId adapter_id,
                                               const WGPUDeviceDescriptor *desc);

WGPUClient *wgpu_client_create(const WGPUClientFactory *factory);

void wgpu_client_destroy(const WGPUClientFactory *factory, WGPUClient *client);

WGPUAdapterId wgpu_client_get_adapter(const WGPUClient *client,
                                      const WGPURequestAdapterOptions *desc);

WGPUInfrastructure wgpu_initialize(void);

void wgpu_server_process(const WGPUServer *server);

void wgpu_terminate(WGPUClientFactory *factory);
