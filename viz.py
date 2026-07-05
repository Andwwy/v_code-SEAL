import json, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES_DIR = sys.argv[1] if len(sys.argv) > 1 else "results_math_vector"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
files = sorted(f for f in glob.glob(os.path.join(RES_DIR, "*.json"))
               if os.path.basename(f) != "summary.json")

for f in files:
    data = json.load(open(f))
    bench = data["summary"]["benchmark"]
    rows = data["rows"]
    n = len(rows)
    bacc = sum(r["base_correct"] for r in rows) / n * 100
    sacc = sum(r["steer_correct"] for r in rows) / n * 100
    bt = [r["base_tokens"] for r in rows]
    st = [r["steer_tokens"] for r in rows]
    bavg, savg = sum(bt) / n, sum(st) / n
    tok_red = (1 - savg / bavg) * 100 if bavg else 0

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"SEAL (MATH vector) on {bench.upper()} - {MODEL}", fontweight="bold")

    ax[0, 0].bar(["baseline", "steered"], [bacc, sacc], color=["#9e9e9e", "#1f77e6"])
    for i, v in enumerate([bacc, sacc]):
        ax[0, 0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax[0, 0].set_ylim(0, 100); ax[0, 0].set_title("Accuracy"); ax[0, 0].set_ylabel("Accuracy (%)")

    ax[0, 1].bar(["baseline", "steered"], [bavg, savg], color=["#1f77e6", "#1f77e6"])
    for i, v in enumerate([bavg, savg]):
        ax[0, 1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontweight="bold")
    ax[0, 1].set_title(f"Avg generation length ({-tok_red:.0f}% total)"); ax[0, 1].set_ylabel("avg tokens / problem")

    ax[1, 0].hist(bt, bins=30, alpha=0.5, label=f"baseline (med {sorted(bt)[n//2]})", color="#9e9e9e")
    ax[1, 0].hist(st, bins=30, alpha=0.6, label=f"steered (med {sorted(st)[n//2]})", color="#1f77e6")
    ax[1, 0].set_title("Length distribution"); ax[1, 0].set_xlabel("tokens / problem"); ax[1, 0].set_ylabel("# problems"); ax[1, 0].legend()

    ax[1, 1].plot([bavg, savg], [bacc, sacc], "-", color="#2ca02c")
    ax[1, 1].scatter([bavg], [bacc], color="#9e9e9e", s=90); ax[1, 1].annotate("baseline", (bavg, bacc))
    ax[1, 1].scatter([savg], [sacc], color="#1f77e6", s=90); ax[1, 1].annotate("steered", (savg, sacc))
    ax[1, 1].set_title("Efficiency frontier (up-left is better)")
    ax[1, 1].set_xlabel("avg tokens / problem (lower = cheaper)"); ax[1, 1].set_ylabel("Accuracy (%)")

    fig.tight_layout()
    out = os.path.join(RES_DIR, f"dashboard_{bench}.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}  | {bench}: {bacc:.1f}%->{sacc:.1f}%, tokens {bavg:.0f}->{savg:.0f} (-{tok_red:.0f}%)")
