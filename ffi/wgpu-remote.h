

/* Generated with cbindgen:0.8.3 */

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

typedef struct WGPUServer WGPUServer;

typedef struct WGPUTrackPermit WGPUTrackPermit;

typedef uint32_t WGPUIndex;

typedef uint32_t WGPUEpoch;

typedef struct {
  WGPUIndex _0;
  WGPUEpoch _1;
} WGPUId;

typedef WGPUId WGPUDeviceId;

typedef WGPUId WGPUAdapterId;

typedef struct {
  bool anisotropic_filtering;
} WGPUExtensions;

typedef struct {
  WGPUExtensions extensions;
} WGPUDeviceDescriptor;

typedef struct {
  WGPUPowerPreference power_preference;
} WGPUAdapterDescriptor;

typedef struct {
  WGPUClient *client;
  WGPUServer *server;
  const uint8_t *error;
} WGPUInfrastructure;





WGPUDeviceId wgpu_client_adapter_create_device(const WGPUClient *client,
                                               WGPUAdapterId adapter_id,
                                               const WGPUDeviceDescriptor *desc);

WGPUAdapterId wgpu_client_get_adapter(const WGPUClient *client, const WGPUAdapterDescriptor *desc);

void wgpu_client_terminate(WGPUClient *client);

WGPUInfrastructure wgpu_initialize(void);

void wgpu_server_process(const WGPUServer *server);
