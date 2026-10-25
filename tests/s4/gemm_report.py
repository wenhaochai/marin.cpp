"""Summarize nm_bench_s4_gemm JSON lines: the GemmCache choice per shape and orientation against the E1 bar (the better
of cuBLAS default and XLA, docs/bench/RESULTS.md §2; BUILD_LOG decision 4), and cuBLASLt 12.8 against 13.0 (decision
5, O8). No third-party modules.

    python3 tests/s4/gemm_report.py <gemm_cu128.jsonl> [<gemm_cu130.jsonl>] > gemm_summary.md
    python3 tests/s4/gemm_report.py --runs <run1.jsonl> <run2.jsonl> ... > gemm_runs.md

--runs: one independent search per file; per E1 shape-orientation the TFLOP/s of every run, the median and the best
against the bar. One training run gets one search, so the median is what a run delivers; the best overstates it.
"""
import json
import sys

# (M, K, N) -> {orient: (cuBLAS default, XLA)} TFLOP/s, docs/bench/RESULTS.md §2
E1 = {
    (131072, 512, 1536): {"fwd": (606, 579), "dgrad": (707, 592), "wgrad": (728, 589)},
    (131072, 512, 1792): {"fwd": (645, 593), "dgrad": (715, 598), "wgrad": (712, 571)},
    (131072, 768, 2304): {"fwd": (692, 685), "dgrad": (754, 689), "wgrad": (671, 631)},
    (131072, 768, 2688): {"fwd": (679, 690), "dgrad": (760, 709), "wgrad": (692, 712)},
    (131072, 1024, 3584): {"fwd": (638, 760), "dgrad": (706, 756), "wgrad": (641, 718)},
    (131072, 2048, 7168): {"fwd": (711, 733), "dgrad": (733, 699), "wgrad": (700, 672)},
    (131072, 512, 128256): {"fwd": (557, 671), "dgrad": (651, 685), "wgrad": (659, 696)},
    (131072, 768, 128256): {"fwd": (634, 668), "dgrad": (723, 718), "wgrad": (709, 706)},
    (16384, 512, 128256): {"fwd": (571, 692), "dgrad": (717, 724), "wgrad": (682, 704)},
    (16384, 768, 128256): {"fwd": (641, 665), "dgrad": (740, 690), "wgrad": (708, 676)},
    (8192, 768, 128256): {"fwd": (616, 743), "dgrad": (710, 759), "wgrad": (666, 778)},
}


def load(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                out.append(json.loads(line))
    return out


def runs(paths):
    per = {}
    for p in paths:
        for r in load(p):
            if r.get("set") != "e1" or "error" in r:
                continue
            per.setdefault((r["M"], r["K"], r["N"], r["orient"]), []).append(r["tflops"])
    print("| M | K | N | orient | " + " | ".join(f"run {i + 1}" for i in range(len(paths))) +
          " | median | best | bar | median / bar | best / bar |")
    print("|---|---|---|---|" + "---|" * (len(paths) + 6))
    miss_med, miss_best, n = [], [], 0
    for (M, K, N, o), ts in sorted(per.items()):
        if (M, K, N) not in E1:
            continue
        bar = max(E1[(M, K, N)]["wgrad" if o.startswith("wgrad") else o])
        st = sorted(ts)
        med = st[len(st) // 2] if len(st) % 2 else 0.5 * (st[len(st) // 2 - 1] + st[len(st) // 2])
        best = st[-1]
        n += 1
        if med < bar:
            miss_med.append((M, K, N, o, med / bar))
        if best < bar:
            miss_best.append((M, K, N, o, best / bar))
        print(f"| {M} | {K} | {N} | {o} | " + " | ".join(f"{t:.0f}" for t in ts) +
              f" | {med:.0f} | {best:.0f} | {bar} | {med / bar:.3f} | {best / bar:.3f} |")
    print(f"\nBelow the bar: median {len(miss_med)} of {n}, best {len(miss_best)} of {n}")
    for M, K, N, o, q in miss_med:
        print(f"- median: M {M} K {K} N {N} {o}: {q:.3f}")


def main():
    if sys.argv[1] == "--runs":
        runs(sys.argv[2:])
        return
    a = load(sys.argv[1])
    b = {(r["set"], r["orient"], r["M"], r["K"], r["N"]): r for r in load(sys.argv[2])} if len(sys.argv) > 2 else {}
    print("| set | M | K | N | orient | chosen | candidates | ms | TFLOP/s | re-timed TFLOP/s | E1 bar (cuBLAS / XLA) | "
          "vs bar | cuBLASLt 13.0 TFLOP/s | 13.0 / 12.8 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    miss = []
    ratios = []
    for r in a:
        key = (r["M"], r["K"], r["N"])
        o = r["orient"]
        if "error" in r:
            print(f"| {r['set']} | {r['M']} | {r['K']} | {r['N']} | {o} | error: {r['error'][:60]} | | | | | | | | |")
            continue
        bar = ""
        vs = ""
        if r["set"] == "e1" and key in E1:
            bo = "wgrad" if o.startswith("wgrad") else o
            cu, xla = E1[key][bo]
            best = max(cu, xla)
            bar = f"{best} ({cu} / {xla})"
            vs = f"{r['tflops'] / best:.3f}"
            if r["tflops"] < best:
                miss.append((key, o, r["tflops"], best))
        c13 = b.get((r["set"], o, r["M"], r["K"], r["N"]))
        t13 = f"{c13['tflops']:.0f}" if c13 and "error" not in c13 else ""
        q = ""
        if c13 and "error" not in c13:
            q = f"{c13['tflops'] / r['tflops']:.3f}"
            ratios.append(c13["tflops"] / r["tflops"])
        print(f"| {r['set']} | {r['M']} | {r['K']} | {r['N']} | {o} | `{r['chosen']}` | {r['n_candidates']} | "
              f"{r['ms']:.3f} | {r['tflops']:.0f} | {r.get('retime_tflops', 0):.0f} | {bar} | {vs} | {t13} | {q} |")
    print()
    print(f"Shapes below the E1 bar: {len(miss)}")
    for key, o, t, best in miss:
        print(f"- M {key[0]} K {key[1]} N {key[2]} {o}: {t:.0f} < {best} TFLOP/s ({t / best:.3f})")
    if ratios:
        ratios.sort()
        print(f"\ncuBLASLt 13.0 / 12.8 TFLOP/s over {len(ratios)} shape-orientations: median "
              f"{ratios[len(ratios) // 2]:.3f}, min {ratios[0]:.3f}, max {ratios[-1]:.3f}")


if __name__ == "__main__":
    main()
