/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define WGPU_INLINE
#define WGPU_FUNC

#include "./../../ffi/wgpu-remote.h"
#include <stdio.h>

int main() {
    WGPUInfrastructure infra = wgpu_client_new();
    WGPUClient *client = infra.client;

    if (!client || infra.error) {
        printf("Cannot initialize WGPU client: %s\n", infra.error);
        return 1;
    }

    WGPUGlobal* server = wgpu_server_new();

    if (!server) {
        printf("Cannot initialize WGPU client: %s\n", server);
        return 1;
    }

    WGPUAdapterId adapterId = 0;
    {
        WGPUAdapterId ids[10];
        int count = wgpu_client_make_adapter_ids(client, ids, 10);

        WGPURequestAdapterOptions options = {
            .power_preference = WGPUPowerPreference_LowPower,
        };
        char index = wgpu_server_instance_request_adapter(server, &options, ids, count);
        if (index < 0) {
            printf("No available GPU adapters!\n");
            return 2;
        }

        wgpu_client_kill_adapter_ids(client, ids, index);
        wgpu_client_kill_adapter_ids(client, ids+index+1, count-index-1);
        adapterId = ids[index];
    }

    //TODO: do something meaningful

    if (adapterId) {
        //wgpu_server_destroy_adapter()
        wgpu_client_kill_adapter_ids(client, &adapterId, 1);
    }
    wgpu_server_delete(server);
    wgpu_client_delete(client);

    printf("Done\n");
    return 0;
}
