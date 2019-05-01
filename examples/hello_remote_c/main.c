#include "./../../ffi/wgpu-remote.h"
#include <stdio.h>

int main() {
    WGPUInfrastructure infra = wgpu_initialize();

    if (!infra.factory || !infra.server || infra.error) {
        printf("Cannot initialize WGPU: %s", infra.error);
        return 1;
    }

    WGPUClient* client = wgpu_client_create(infra.factory);

    //TODO: do something meaningful

    wgpu_client_destroy(infra.factory, client);
    wgpu_terminate(infra.factory);

    return 0;
}
