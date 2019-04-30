#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct WGPUClient WGPUClient;

typedef struct WGPUServer WGPUServer;

typedef struct {
  WGPUClient *client;
  WGPUServer *server;
  const uint8_t *error;
} WGPUInfrastructure;

WGPUDeviceId wgpu_adapter_create_device(const WGPUClient *client,
                                        WGPUAdapterId adapter_id,
                                        const WGPUDeviceDescriptor *desc);

WGPUInfrastructure wgpu_initialize(void);

WGPUAdapterId wgpu_instance_get_adapter(const WGPUClient *client,
                                        const WGPUAdapterDescriptor *desc);

void wgpu_server_process(const WGPUServer *server);

void wgpu_terminate(WGPUClient *client);
