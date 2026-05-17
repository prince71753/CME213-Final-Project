// optional nvtx profiling helpers.
#pragma once

#ifdef CME213_USE_NVTX
#include <nvToolsExt.h>
#endif

struct NvtxRange {
    explicit NvtxRange(const char* name) {
#ifdef CME213_USE_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }

    ~NvtxRange() {
#ifdef CME213_USE_NVTX
        nvtxRangePop();
#endif
    }
};
