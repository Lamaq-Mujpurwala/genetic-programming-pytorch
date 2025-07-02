#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <bcrypt.h>

#pragma comment(lib,"bcrypt.lib")

int main() {
    unsigned int rand_val;
    
    // Corrected condition: check if return value == 0 (STATUS_SUCCESS)
    if (BCryptGenRandom(NULL, (PUCHAR)&rand_val, sizeof(rand_val), BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0) {
        printf("Successful Generation : %u\n", rand_val);
    } else {
        printf("Failed Generation\n");
    }

    return 0;
}
