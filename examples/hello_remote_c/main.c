#include "./../../wgpu-bindings/wgpu-native.h"
#include "./../../wgpu-bindings/wgpu-remote.h"
#include <stdio.h>

int main() {
    WGPUInfrastructure infra = wgpu_initialize();

    if (!infra.client || !infra.server || infra.error) {
        printf("Cannot initialize WGPU: %s", infra.error);
        return 1;
    }

    //TODO: do something meaningful

    wgpu_terminate(infra.client);

    return 0;
}
