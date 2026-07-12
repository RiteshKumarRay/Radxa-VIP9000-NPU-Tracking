#include <stdio.h>
#include "vip_lite.h"
int main() {
    vip_status_e status = vip_init();
    if (status != 0) {
        printf("vip_init failed = %d\n", status);
        return 1;
    }
    vip_network network;
    status = vip_create_network("yolov8n_t527.nb", 0, VIP_CREATE_NETWORK_FROM_FILE, &network);
    printf("vip_create_network = %d\n", status);
    return 0;
}
