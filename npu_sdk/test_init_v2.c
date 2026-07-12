#include <stdio.h>
#include "vip_lite.h"
int main() {
    vip_status_e status = vip_init();
    printf("vip_init = %d\n", status);
    return 0;
}
