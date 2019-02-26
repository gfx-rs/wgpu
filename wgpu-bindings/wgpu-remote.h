#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct WGPUClient WGPUClient;

WGPUDeviceId wgpu_adapter_create_device(const WGPUClient *client,
                                        WGPUAdapterId adapter_id,
                                        const WGPUDeviceDescriptor *desc);

WGPUAdapterId wgpu_instance_get_adapter(const WGPUClient *client,
                                        WGPUInstanceId instance_id,
                                        const WGPUAdapterDescriptor *desc);
