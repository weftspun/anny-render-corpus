// SPDX-License-Identifier: Apache-2.0 OR MIT
// A flat C entry over the Slang-generated kernel, so ctypes can call it with plain arrays.
// `mtoon_slang_gen.cpp` is generated: slangc mtoon.slang -target cpp -entry shadeMain.

#include "mtoon_slang_gen.cpp"

#include <cstdint>
#include <cstring>

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
