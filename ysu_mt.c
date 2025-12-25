// ysu_mt.c
#include "ysu_mt.h"
#include <stdlib.h>

#if defined(_WIN32)
#include <windows.h>
#endif

int ysu_mt_suggest_threads(void) {
    // 1) Env override: YSU_THREADS
    const char *env = getenv("YSU_THREADS");
    if (env && env[0]) {
        int v = atoi(env);
        if (v > 0) return v;
    }

    // 2) Platform default
#if defined(_WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    int n = (int)sysinfo.dwNumberOfProcessors;
    return (n > 0) ? n : 4;
#else
    // GCC/Clang: fallback
    // (POSIX sysconf kullanmak istersen ekleriz)
    return 8;
#endif
}
