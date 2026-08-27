// SPDX-License-Identifier: Apache-2.0 OR MIT
// A flat C entry over the Slang-generated kernel, so ctypes can call it with plain arrays.
// `mtoon_slang_gen.cpp` is generated: slangc mtoon.slang -target cpp -entry shadeMain.

#include "mtoon_slang_gen.cpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace {

template <typename T>
StructuredBuffer<T> ro(const T* data, size_t count)
{
    StructuredBuffer<T> b;
    b.data = const_cast<T*>(data);
    b.count = count;
    return b;
}

template <typename T>
RWStructuredBuffer<T> rw(T* data, size_t count)
{
    RWStructuredBuffer<T> b;
    b.data = data;
    b.count = count;
    return b;
}

}  // namespace

extern "C" SLANG_PRELUDE_SHARED_LIB_EXPORT
void mtoon_shade(uint32_t n,
                 const float* dot_nl,
                 const float* dot_nv,
                 const float* params,  // 22 floats, laid out as MToonParams_0
                 float* out)
{
    MToonParams_0 p;
    std::memcpy(&p, params, sizeof(MToonParams_0));

    EntryPointParams_0 ep;
    ep.dotNL_0 = ro(dot_nl, n);
    ep.dotNV_0 = ro(dot_nv, n);
    ep.params_0 = ro(&p, 1);
    ep.outColor_0 = rw(out, size_t(n) * 3);
    ep.count_0 = ro(&n, 1);

    // The kernel is [numthreads(64,1,1)], so one group covers 64 lanes and the tail group
    // returns early on the count check the shader already carries.
    const uint32_t groups = (n + 63u) / 64u;
    ComputeVaryingInput vi = {};
    vi.startGroupID = uint3{0, 0, 0};
    vi.endGroupID = uint3{groups, 1, 1};
    shadeMain(&vi, &ep, nullptr);
}

extern "C" SLANG_PRELUDE_SHARED_LIB_EXPORT
uint32_t mtoon_params_size(void)
{
    return uint32_t(sizeof(MToonParams_0));
}

extern "C" SLANG_PRELUDE_SHARED_LIB_EXPORT
void mtoon_srgb8(uint32_t n, const float* rgba, uint32_t* out, uint32_t threads)
{
    EntryPointParams_1 ep;
    ep.src_0 = ro(rgba, size_t(n) * 4);
    ep.dst_0 = rw(out, n);
    ep.count_1 = ro(&n, 1);

    const uint32_t groups = (n + 63u) / 64u;
    if (threads == 0)
        threads = std::max(1u, std::thread::hardware_concurrency());
    threads = std::min(threads, groups);

    // The group loop inside srgbMain is serial, so one call uses one core on work that is
    // per-pixel independent. The group range is the split.
    auto run = [&](uint32_t lo, uint32_t hi) {
        ComputeVaryingInput vi = {};
        vi.startGroupID = uint3{lo, 0, 0};
        vi.endGroupID = uint3{hi, 1, 1};
        srgbMain(&vi, &ep, nullptr);
    };
    if (threads <= 1) {
        run(0, groups);
        return;
    }
    std::vector<std::thread> pool;
    const uint32_t span = (groups + threads - 1) / threads;
    for (uint32_t t = 0; t < threads; ++t) {
        const uint32_t lo = t * span;
        const uint32_t hi = std::min(groups, lo + span);
        if (lo < hi)
            pool.emplace_back(run, lo, hi);
    }
    for (auto& th : pool)
        th.join();
}
