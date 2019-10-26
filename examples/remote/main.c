#define WGPU_INLINE
#define WGPU_FUNC

#include "./../../ffi/wgpu-remote.h"
#include <stdio.h>

int main() {
    WGPUInfrastructure infra = wgpu_client_new();

    if (!infra.client || infra.error) {
        printf("Cannot initialize WGPU client: %s", infra.error);
        return 1;
    }

    WGPUGlobal* server = wgpu_server_new();

    if (!server) {
        printf("Cannot initialize WGPU client: %s", server);
        return 1;
    }

    //TODO: do something meaningful

    wgpu_server_delete(server);
    wgpu_client_delete(infra.client);

    return 0;
}
