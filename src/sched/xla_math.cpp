// Host ports of libdevice __nv_cosf and __nv_powf exactly as XLA:GPU inlines them into the schedule fusions (jaxlib
// 0.11.1 with jax-cuda13-plugin 0.11.1, PJRT "cuda 13000", libdevice.10.bc of the CUDA 13 wheels, nvidia-nvvm 13.2.78:
// the fixture manifests): the operation sequence is the final NVPTX-bound LLVM IR of the jitted optax schedules
// (`--xla_gpu_dump_llvmir`, S2). Every llvm.fma.f32 is std::fma; every plain fmul/fadd/fsub is one f32 operation
// (NVPTX does not contract them here: the warmup's fmul + fsub matches the logged values only unfused). The file is
// built with -ffp-contract=off (cmake/MarinFlags.cmake), so the host compiler fuses nothing either.
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "marincpp/core.hpp"
#include "sched/sched.hpp"

namespace nm::sched {
namespace {

float f(uint32_t bits) { return std::bit_cast<float>(bits); }
uint32_t u(float x) { return std::bit_cast<uint32_t>(x); }

// cvt.rni (f2i.rn / nvvm.round.f): round to the nearest integer, ties to even (the host's default rounding mode).
float round_even(float x) { return std::nearbyint(x); }

}  // namespace

float xla_cosf(float x) {
  NM_REQUIRE(std::fabs(x) < 105615.0f, "xla_cosf: only the Cody-Waite range of __nv_cosf is ported");
  const float jf = round_even(x * f(0x3F22F983));        // x * 2/pi, f2i.rn
  const int32_t i = static_cast<int32_t>(jf);
  float t = std::fma(jf, f(0xBFC90FDA), x);               // x - j * pi/2 in three parts
  t = std::fma(jf, f(0xB3A22168), t);
  t = std::fma(jf, f(0xA7C234C5), t);
  const float s = t * t;
  const bool even = (i & 1) == 0;                         // even quadrant: cos polynomial; odd: sin polynomial
  const float p86 = even ? 1.0f : t;
  const float p87 = std::fma(s, p86, 0.0f);
  const float p88 = std::fma(s, f(0x37CBAC00), f(0xBAB607ED));
  const float p89 = even ? p88 : f(0xB94D4153);
  const float p90 = even ? f(0x3D2AAABB) : f(0x3C0885E4);
  const float p91 = std::fma(p89, s, p90);
  const float p92 = even ? f(0xBEFFFFFF) : f(0xBE2AAAA8);
  const float p93 = std::fma(p91, s, p92);
  const float p94 = std::fma(p93, p87, p86);
  return ((i + 1) & 2) == 0 ? p94 : 0.0f - p94;
}

float xla_powf(float b, float t) {
  NM_REQUIRE(b >= 0.0f && b < 1.0f && t >= 0.0f && t == std::trunc(t) && t < static_cast<float>(kMaxPowExponent),
             "xla_powf: only 0 <= b < 1 with an integer exponent 0 <= t < 2^24 is ported");
  if (t == 0.0f) return 1.0f;
  if (b == 0.0f) return std::fmod(t, 2.0f) == 1.0f ? b : 0.0f;   // __nv_powf(+-0, t > 0): +-0 for odd t, else +0
  // log2(b) in double-f32 (hi + lo): b = m * 2^e with m in [0.707, 1.414), u = 2(m-1)/(m+1), atanh series.
  const bool sub = b < f(0x00800000);
  const float a = sub ? b * f(0x4B800000) : b;
  const float expo0 = sub ? -24.0f : 0.0f;
  const uint32_t ia = u(a);
  const int32_t e = static_cast<int32_t>((ia - 0x3F3504F3u) & 0xFF800000u);
  const float m = f(ia - static_cast<uint32_t>(e));
  const float expo = std::fma(static_cast<float>(e), f(0x34000000), expo0);
  const float m1 = m - 1.0f;
  const float r = 1.0f / (m + 1.0f);                      // rcp.approx.ftz.f32 (see sched.hpp)
  const float uu = r * (m1 * 2.0f);
  const float u2 = uu * uu;
  float d = (m1 - uu) * 2.0f;
  d = std::fma(-uu, m1, d);
  d = r * d;
  float p = std::fma(u2, f(0x3A2C32E4), f(0x3B52E7DB));
  p = std::fma(p, u2, f(0x3C93BB73));
  p = std::fma(p, u2, f(0x3DF6384F));
  p = p * u2;
  const float hi = std::fma(uu, f(0x3FB8AA3B), expo);
  float lo = expo - hi;
  lo = std::fma(uu, f(0x3FB8AA3B), lo);
  lo = std::fma(d, f(0x3FB8AA3B), lo);
  lo = std::fma(uu, f(0x32A55E34), lo);
  lo = std::fma(p * 3.0f, d, lo);
  lo = std::fma(p, uu, lo);
  const float s = hi + lo;
  const float l2 = lo - (s - hi);
  // t * log2(b) in double-f32, then exp2.
  const float ph = s * t;
  // |t log2 b| > 152: the result is 0 or inf whatever the exp2 polynomial gives (__nv_powf's last select). Decided
  // here, before the f2i of round(ph) below, which would overflow int32 for |ph| >= 2^31 (subnormal b, large t).
  if (std::fabs(ph) > 152.0f) return ph < 0.0f ? 0.0f : std::numeric_limits<float>::infinity();
  float pl = std::fma(s, t, -ph);
  pl = std::fma(l2, t, pl);
  const float rr = round_even(ph);
  const float fr = (ph - rr) + pl;
  float q = std::fma(fr, f(0x391FCB8E), f(0x3AAF85ED));
  q = std::fma(q, fr, f(0x3C1D9856));
  q = std::fma(q, fr, f(0x3D6357BB));
  q = std::fma(q, fr, f(0x3E75FDEC));
  q = std::fma(q, fr, f(0x3F317218));
  q = std::fma(q, fr, 1.0f);
  const int32_t ki = static_cast<int32_t>(rr);             // f2i.rz of an integral value, |rr| <= 152
  const uint32_t sc = rr > 0.0f ? 0u : 0x83000000u;
  q = q * f(0x7F000000u + sc);
  q = q * f((static_cast<uint32_t>(ki) << 23) - sc);
  return q;
}

int64_t ulp_distance(float a, float b) {
  if (std::isnan(a) || std::isnan(b)) return u(a) == u(b) ? 0 : std::numeric_limits<int64_t>::max();
  auto ord = [](float x) -> int64_t {
    const uint32_t i = u(x);
    return (i & 0x80000000u) ? -static_cast<int64_t>(i & 0x7FFFFFFFu) : static_cast<int64_t>(i);
  };
  const int64_t d = ord(a) - ord(b);
  return d < 0 ? -d : d;
}

UlpStats compare(const std::vector<float>& a, const std::vector<float>& b) {
  NM_REQUIRE(a.size() == b.size(), "compare: tables of different length");
  UlpStats s;
  for (size_t k = 0; k < a.size(); ++k)
    if (u(a[k]) != u(b[k])) {
      if (s.first < 0) s.first = static_cast<int64_t>(k);
      ++s.n_diff;
      const int64_t d = ulp_distance(a[k], b[k]);
      if (d > s.max_ulp) s.max_ulp = d;
    }
  return s;
}

}  // namespace nm::sched
