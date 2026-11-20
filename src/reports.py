"""Métricas, figuras e tabelas LaTeX geradas automaticamente."""
from __future__ import annotations

import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from .models import PRETTY

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


def pretty_class(c: str) -> str:
    return c.replace("_", " ")


def exp_label(model: str, aug: bool) -> str:
    return f"{PRETTY.get(model, model)} {'com' if aug else 'sem'} DA"


# --------------------------------------------------------------- métricas
def compute_metrics(y_true, y_pred, class_names) -> dict:
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = float(p), float(r), float(f)
    out["per_class"] = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names,
        output_dict=True, zero_division=0)
    out["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))).tolist()
    return out


# ---------------------------------------------------------------- figuras
def plot_confusion(cm, class_names, path, title=""):
    cm = np.asarray(cm)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    names = [textwrap.fill(pretty_class(c), 12) for c in class_names]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center",
                    fontsize=6, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Classe predita"); ax.set_ylabel("Classe verdadeira")
    if title:
        ax.set_title(title, fontsize=9)
    fig.savefig(path); plt.close(fig)


def plot_curves_all(histories: dict, path):
    """histories: {exp_label: DataFrame}. Gera loss e acurácia lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for i, (lab, h) in enumerate(histories.items()):
        c = f"C{i}"
        axes[0].plot(h["epoch"], h["train_loss"], c, ls="--", lw=1)
        axes[0].plot(h["epoch"], h["val_loss"], c, lw=1.5, label=lab)
        axes[1].plot(h["epoch"], h["train_acc"], c, ls="--", lw=1)
        axes[1].plot(h["epoch"], h["val_acc"], c, lw=1.5, label=lab)
    axes[0].set_title("Perda (tracejado = treino)"); axes[0].set_xlabel("Época")
    axes[1].set_title("Acurácia (tracejado = treino)"); axes[1].set_xlabel("Época")
    axes[1].legend(fontsize=7, loc="lower right")
    for a in axes:
        a.grid(alpha=0.3)
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.savefig(path); plt.close(fig)


def plot_class_distribution(splits: dict, class_names, path):
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(len(class_names)); w = 0.27
    for k, (name, df) in enumerate(splits.items()):
        counts = df["label"].value_counts().reindex(range(len(class_names)), fill_value=0)
        ax.bar(x + (k - 1) * w, counts.values, w, label=name)
    ax.set_xticks(x, [textwrap.fill(pretty_class(c), 12) for c in class_names],
                  rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Nº de imagens"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    fig.savefig(path); plt.close(fig)


def plot_samples(df, class_names, path, per_class=4, seed=42):
    n = len(class_names)
    fig, axes = plt.subplots(n, per_class, figsize=(per_class * 1.3, n * 1.3))
    axes = np.atleast_2d(axes)
    for i, c in enumerate(class_names):
        sub = df[df["label"] == i].sample(min(per_class, (df["label"] == i).sum()), random_state=seed)
        for j in range(per_class):
            ax = axes[i, j]; ax.axis("off")
            if j < len(sub):
                ax.imshow(Image.open(sub.iloc[j]["path"]).convert("RGB").resize((160, 160)))
        axes[i, 0].text(-10, 80, textwrap.fill(pretty_class(c), 12), ha="right", va="center",
                        fontsize=7)
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig(path); plt.close(fig)


def plot_qualitative(test_df, y_true, y_pred, probs, class_names, path, n=6, seed=42):
    """Linha 1: acertos; linhas 2-3: erros (os de maior confiança primeiro)."""
    rng = np.random.default_rng(seed)
    conf = probs.max(1)
    correct = np.where(y_true == y_pred)[0]
    wrong = np.where(y_true != y_pred)[0]
    correct = rng.choice(correct, size=min(n, len(correct)), replace=False) if len(correct) else []
    wrong = wrong[np.argsort(-conf[wrong])][: 2 * n]
    rows = 1 + int(np.ceil(len(wrong) / n)) if len(wrong) else 1
    fig, axes = plt.subplots(rows, n, figsize=(n * 1.25, rows * 1.55))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")

    def _show(ax, k, ok):
        ax.imshow(Image.open(test_df.iloc[k]["path"]).convert("RGB").resize((160, 160)))
        t = textwrap.fill(pretty_class(class_names[y_true[k]]), 14)
        p = textwrap.fill(pretty_class(class_names[y_pred[k]]), 14)
        ax.set_title(f"V: {t}\nP: {p} ({conf[k]:.0%})", fontsize=5.5,
                     color="green" if ok else "red")

    for j, k in enumerate(correct):
        _show(axes[0, j], k, True)
    for m, k in enumerate(wrong):
        _show(axes[1 + m // n, m % n], k, False)
    plt.subplots_adjust(wspace=0.05, hspace=0.55)
    fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------ tabelas LaTeX
def _cmd(model: str, aug: bool) -> str:
    base = {"resnet50": "ResNet", "vit_b_16": "ViT", "efficientnet_b0": "EffNet",
            "convnext_tiny": "ConvNeXt", "swin_t": "Swin", "resnet18": "ResNetEighteen"}
    return base.get(model, "".join(ch for ch in model if ch.isalpha()).title()) + (
        "Aug" if aug else "NoAug")


def write_latex(summary: pd.DataFrame, best_metrics: dict, splits: dict, stats: dict,
                class_names, out_dir: str, best_label: str):
    os.makedirs(out_dir, exist_ok=True)
    pct = lambda v: f"{100 * v:.2f}"

    # Tabela principal de resultados no teste
    best_f1 = summary["f1_macro"].max()
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Desempenho no conjunto de teste (\%). P, R e F1 com agregação \textit{macro} "
             r"e \textit{weighted}. DA = \textit{data augmentation}. Melhor F1-macro em negrito.}",
             r"\label{tab:resultados}", r"\begin{tabular}{llccccccccc}", r"\toprule",
             r"Modelo & DA & Acc & P$_{mac}$ & R$_{mac}$ & F1$_{mac}$ & P$_{w}$ & R$_{w}$ & F1$_{w}$"
             r" & Parâm. (M) & Tempo (min) \\", r"\midrule"]
    for _, r in summary.iterrows():
        f1 = pct(r["f1_macro"])
        if np.isclose(r["f1_macro"], best_f1):
            f1 = r"\textbf{" + f1 + "}"
        lines.append(
            f"{PRETTY.get(r['model'], r['model'])} & {'Sim' if r['augment'] else 'Não'} & "
            f"{pct(r['accuracy'])} & {pct(r['precision_macro'])} & {pct(r['recall_macro'])} & {f1} & "
            f"{pct(r['precision_weighted'])} & {pct(r['recall_weighted'])} & {pct(r['f1_weighted'])} & "
            f"{r['params_M']:.1f} & {r['train_time_min']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(out_dir, "table_results.tex"), "w").write("\n".join(lines) + "\n")

    # Métricas por classe do melhor modelo
    pc = best_metrics["per_class"]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Métricas por classe no teste para o melhor modelo (" + best_label + r").}",
             r"\label{tab:porclasse}", r"\begin{tabular}{lcccc}", r"\toprule",
             r"Classe & Precisão & Recall & F1 & Suporte \\", r"\midrule"]
    for c in class_names:
        d = pc[c]
        lines.append(f"{pretty_class(c)} & {pct(d['precision'])} & {pct(d['recall'])} & "
                     f"{pct(d['f1-score'])} & {int(d['support'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out_dir, "table_perclass.tex"), "w").write("\n".join(lines) + "\n")

    # Tabela do dataset
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Distribuição das imagens por classe e conjunto após a limpeza.}",
             r"\label{tab:dataset}", r"\begin{tabular}{lcccc}", r"\toprule",
             r"Classe & Treino & Validação & Teste & Total \\", r"\midrule"]
    tot = [0, 0, 0]
    for i, c in enumerate(class_names):
        n = [int((s["label"] == i).sum()) for s in splits.values()]
        tot = [a + b for a, b in zip(tot, n)]
        lines.append(f"{pretty_class(c)} & {n[0]} & {n[1]} & {n[2]} & {sum(n)} \\\\")
    lines += [r"\midrule", f"Total & {tot[0]} & {tot[1]} & {tot[2]} & {sum(tot)} \\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out_dir, "table_dataset.tex"), "w").write("\n".join(lines) + "\n")

    # Macros com os números usados no texto do artigo
    m = {
        "NTotalFiles": stats["n_total_files"], "NAugExcluded": stats.get("n_aug_excluded", 0),
        "NGroups": stats.get("n_groups", stats["n_after_cleaning"]),
        "NNearDup": stats.get("n_in_near_dup_groups", 0), "NDuplicates": stats["n_duplicates_removed"],
        "NConflicting": stats["n_conflicting_removed"], "NCorrupt": stats["n_corrupt"],
        "NClean": stats["n_after_cleaning"], "NClasses": len(class_names),
        "NTrain": len(splits["Treino"]), "NVal": len(splits["Validação"]),
        "NTest": len(splits["Teste"]), "BestLabel": best_label,
    }
    for _, r in summary.iterrows():
        k = _cmd(r["model"], bool(r["augment"]))
        m[f"Acc{k}"] = pct(r["accuracy"])
        m[f"FOne{k}"] = pct(r["f1_macro"])
        m[f"Errors{k}"] = int(r["n_errors"])
        m[f"BestEpoch{k}"] = int(r["best_epoch"])
    for model in summary["model"].unique():
        s = summary[summary["model"] == model].set_index("augment")
        if True in s.index and False in s.index:
            k = _cmd(model, True)[:-3]
            m[f"DeltaAug{k}"] = f"{100 * (s.loc[True, 'f1_macro'] - s.loc[False, 'f1_macro']):+.2f}"
    best = summary.loc[summary["f1_macro"].idxmax()]
    m["BestAcc"] = pct(best["accuracy"]); m["BestFOne"] = pct(best["f1_macro"])
    with open(os.path.join(out_dir, "macros.tex"), "w") as f:
        f.write("% Gerado automaticamente por main.py -- não editar à mão\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(os.path.join(out_dir, "macros.json"), "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
