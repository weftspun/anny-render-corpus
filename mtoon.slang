// SPDX-License-Identifier: Apache-2.0 OR MIT
// MToon 1.0 shading, from the VRMC_materials_mtoon-1.0 pseudocode. mtoon.py is the same
// model in Python and check_mtoon_slang.py holds the two against each other.

struct MToonParams
{
    float3 baseColor;
    float3 shadeColor;
    float3 rimColor;
    float3 lightColor;
    float shadingShiftFactor;
    float shadingToonyFactor;
    float shadingShiftTexture;
    float parametricRimLiftFactor;
    float parametricRimFresnelPowerFactor;
    float rimLightingMixFactor;
};

float linearstep(float a, float b, float t)
{
    if (abs(b - a) <= 1e-6f)
        return t >= a ? 1.0f : 0.0f;
    return saturate((t - a) / (b - a));
}

float mtoonShading(float dotNL, MToonParams p)
{
    float toony = saturate(p.shadingToonyFactor);
    float s = dotNL + p.shadingShiftFactor + p.shadingShiftTexture;
    return linearstep(-1.0f + toony, 1.0f - toony, s);
}

float3 mtoonEvaluate(float dotNL, float dotNV, MToonParams p)
{
    float t = mtoonShading(dotNL, p);
    float3 col = lerp(p.shadeColor, p.baseColor, t) * p.lightColor;

    float r = saturate(1.0f - dotNV + p.parametricRimLiftFactor);
    r = pow(r, max(p.parametricRimFresnelPowerFactor, 1e-6f));
    float3 rim = p.rimColor * r;
    float3 white = float3(1.0f, 1.0f, 1.0f);
    rim = rim * lerp(white, p.lightColor, p.rimLightingMixFactor);
    return col + rim;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void shadeMain(uint3 tid : SV_DispatchThreadID,
               uniform StructuredBuffer<float> dotNL,
               uniform StructuredBuffer<float> dotNV,
               uniform StructuredBuffer<MToonParams> params,
               uniform RWStructuredBuffer<float> outColor,
               uniform StructuredBuffer<uint> count)
{
    uint i = tid.x;
    if (i >= count[0])
        return;
    float3 c = mtoonEvaluate(dotNL[i], dotNV[i], params[0]);
    outColor[i * 3 + 0] = c.x;
    outColor[i * 3 + 1] = c.y;
    outColor[i * 3 + 2] = c.z;
}
