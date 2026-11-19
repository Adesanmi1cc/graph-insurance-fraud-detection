"""Rebuild the README figures from the CSVs in results/.

Usage:  python src/make_figures.py
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

FAMILY = {"tabular": "#8a8a85", "local": "#2a78d6", "community": "#eb6834", "global": "#1baf7a"}
MODEL_FAMILY = ["tabular", "local", "community", "global", "all"]
MODEL_COL = {**FAMILY, "all": "#3a3a38"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e6e6e2"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

def save(fig, name):
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close(fig)

# 1. Ablation: change vs the M1 baseline, as dots (a bar would need a zero baseline)
ab = pd.read_csv(RES / "ablation_results.csv")
labels = ["M1 Tabular", "M2 +Local", "M3 +Community", "M4 +Global", "M5 All graph"]
cols = [MODEL_COL[f] for f in MODEL_FAMILY]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, metric, title in zip(axes, ["auc_pr", "precision"],
                             ["AUC-PR (primary metric)", "Precision at 0.5 threshold"]):
    v = ab[metric].values
    y = list(range(len(v)))[::-1]
    ax.axvline(v[0], color=MUTED, lw=1, ls="--", zorder=1)
    ax.hlines(y, v[0], v, color=GRID, lw=2, zorder=1)
    ax.scatter(v, y, s=90, c=cols, edgecolor="white", linewidth=2, zorder=3)
    span = v.max() - v.min()
    for yi, x in zip(y, v):
        ax.text(x + span * 0.06, yi, f"{x:.3f}", va="center", fontsize=9)
    ax.set_xlim(v.min() - span * 0.25, v.max() + span * 0.35)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.suptitle("Five-model XGBoost ablation vs M1 baseline (dashed line), held-out test set: 1,623 providers, 152 fraudulent",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "fig1_ablation.png")

# 2. Family-level SHAP: mean per feature (fair comparison across family sizes)
fam = pd.read_csv(RES / "shap_family.csv").sort_values("mean_shap_per_feature")
fig, ax = plt.subplots(figsize=(8, 3.2))
bars = ax.barh(fam["family"].str.capitalize() + " (" + fam["n_features"].astype(str) + ")",
               fam["mean_shap_per_feature"], color=[FAMILY[f] for f in fam["family"]],
               height=0.6, edgecolor="white", linewidth=2)
for b, x in zip(bars, fam["mean_shap_per_feature"]):
    ax.text(x + 0.01, b.get_y() + b.get_height() / 2, f"{x:.3f}", va="center", fontsize=9)
ax.set_xlabel("Mean |SHAP| per feature in family (M5 model)")
ax.set_title("Community features carry the most signal per feature", loc="left", fontweight="bold")
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "fig2_shap_family.png")

# 3. Top 15 features by SHAP
top = pd.read_csv(RES / "shap_top15.csv").iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5.6))
ax.barh(top["feature"], top["mean_abs_shap"], color=[FAMILY[f] for f in top["family"]],
        height=0.66, edgecolor="white", linewidth=2)
for i, (x, f) in enumerate(zip(top["mean_abs_shap"], top["family"])):
    ax.text(x + 0.015, i, f"{x:.2f}", va="center", fontsize=8, color=MUTED)
ax.set_xlabel("Mean |SHAP value| (M5 model, held-out test set)")
ax.set_title("Top 15 features by SHAP importance", loc="left", fontweight="bold")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY.values()]
ax.legend(handles, [k.capitalize() for k in FAMILY], loc="lower right", frameon=False, title="Feature family")
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "fig3_top15_shap.png")

# 4. Detection overlap among the 152 test-set fraud providers
ov = pd.read_csv(RES / "detection_overlap.csv")
names = {"caught_by_both": "Caught by both", "graph_only_M5": "Graph model only (M5)",
         "tabular_only_M1": "Tabular model only (M1)", "missed_by_both": "Missed by both"}
ov = ov.iloc[::-1]
colors = {"caught_by_both": "#3a3a38", "graph_only_M5": FAMILY["community"],
          "tabular_only_M1": FAMILY["tabular"], "missed_by_both": "#c9c9c4"}
fig, ax = plt.subplots(figsize=(8, 3))
bars = ax.barh([names[g] for g in ov["group"]], ov["n_providers"],
               color=[colors[g] for g in ov["group"]], height=0.6, edgecolor="white", linewidth=2)
for b, n, p in zip(bars, ov["n_providers"], ov["pct_of_test_fraud"]):
    ax.text(n + 1, b.get_y() + b.get_height() / 2, f"{n}  ({p}%)", va="center", fontsize=9)
ax.set_xlabel("Fraudulent providers in test set (n = 152)")
ax.set_title("Who catches which fraud cases", loc="left", fontweight="bold")
ax.set_xlim(0, 105)
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "fig4_detection_overlap.png")
print("Figures written to", FIG)
