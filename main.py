"""SIN393 - Classificação de doenças em folhas de arroz com Deep Learning.

Questão experimental investigada:
  (Q1) Arquitetura: CNN (ResNet-50) x Transformer (ViT-B/16), ambos com
       transfer learning do ImageNet;
  (Q2) Impacto do data augmentation em cada arquitetura.
Grade: 2 modelos x {com, sem} DA = 4 experimentos, mesma divisão,
mesmos hiperparâmetros e mesma semente.

Uso (Colab):
    python main.py --data-dir /caminho/do/dataset --epochs 20
Teste rápido:
    python main.py --data-dir dados --epochs 1 --max-per-class 20 --no-pretrained
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import RiceDataset, build_index, get_transforms, split_dataset
from src.engine import fit, predict
from src.models import PRETTY, build_model, count_params
from src import reports


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--paper-dir", default="paper")
    ap.add_argument("--models", nargs="+", default=["resnet50", "vit_b_16"])
    ap.add_argument("--aug", nargs="+", default=["yes", "no"], choices=["yes", "no"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-per-class", type=int, default=None, help="só para testes rápidos")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--keep-aug-files", action="store_true",
                    help="NÃO remover os arquivos aug_* do dataset (não recomendado: vazamento)")
    ap.add_argument("--cache-dir", default="cache", help="imagens redimensionadas (lado maior 512)")
    ap.add_argument("--save-weights", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    fig_dir = os.path.join(args.paper_dir, "figures")
    gen_dir = os.path.join(args.paper_dir, "generated")
    os.makedirs(fig_dir, exist_ok=True)
    print(f"[main] device = {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ------------------------------------------------------------ dados
    df, stats = build_index(args.data_dir, args.max_per_class, not args.keep_aug_files, args.cache_dir)
    class_names = stats["classes"]
    print(f"[data] arquivos aug_* (augmentation offline) removidos: {stats['n_aug_excluded']} | "
          f"grupos de quase-duplicatas: {stats['n_groups']} ({stats['n_in_near_dup_groups']} imgs em grupos >1)")
    print(f"[data] {stats['n_total_files']} arquivos | corrompidos: {stats['n_corrupt']} | "
          f"duplicatas removidas: {stats['n_duplicates_removed']} | "
          f"conflitantes removidas: {stats['n_conflicting_removed']} | final: {len(df)}")
    print(f"[data] classes: {class_names}")
    train_df, val_df, test_df = split_dataset(df, 0.15, 0.15, args.seed)
    splits = {"Treino": train_df, "Validação": val_df, "Teste": test_df}
    for name, s in splits.items():
        s.to_csv(os.path.join(args.out, f"split_{name.lower()}.csv"), index=False)
        print(f"[data] {name}: {len(s)} -> {dict(Counter(s['class_name']))}")
    # garantia de ausência de vazamento
    assert not (set(train_df.md5) & set(test_df.md5)) and not (set(val_df.md5) & set(test_df.md5))
    assert not (set(train_df.group) & set(test_df.group)) and not (set(val_df.group) & set(test_df.group))
    json.dump(stats, open(os.path.join(args.out, "dataset_stats.json"), "w"),
              indent=2, ensure_ascii=False)
    reports.plot_class_distribution(splits, class_names, os.path.join(fig_dir, "class_distribution.png"))
    reports.plot_samples(train_df, class_names, os.path.join(fig_dir, "dataset_samples.png"))

    pin = device.type == "cuda"
    mk = lambda d, tf, sh: DataLoader(RiceDataset(d, tf), batch_size=args.batch_size, shuffle=sh,
                                      num_workers=args.workers, pin_memory=pin)

    # ------------------------------------------------------ experimentos
    rows, histories, preds = [], {}, {}
    for model_name in args.models:
        for aug in [a == "yes" for a in args.aug]:
            exp = f"{model_name}_{'aug' if aug else 'noaug'}"
            label = reports.exp_label(model_name, aug)
            print(f"\n[exp] ===== {label} ({exp}) =====")
            set_seed(args.seed)
            train_tf, eval_tf = get_transforms(args.img_size, aug)
            g = torch.Generator(); g.manual_seed(args.seed)
            train_loader = DataLoader(RiceDataset(train_df, train_tf), batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.workers, pin_memory=pin,
                                      generator=g)
            val_loader, test_loader = mk(val_df, eval_tf, False), mk(test_df, eval_tf, False)

            model = build_model(model_name, len(class_names), not args.no_pretrained).to(device)
            model, hist, info = fit(model, train_loader, val_loader, device, args.epochs, args.lr,
                                    args.weight_decay, args.patience)
            hist = pd.DataFrame(hist)
            exp_dir = os.path.join(args.out, exp); os.makedirs(exp_dir, exist_ok=True)
            hist.to_csv(os.path.join(exp_dir, "history.csv"), index=False)
            if args.save_weights:
                torch.save(model.state_dict(), os.path.join(exp_dir, "best.pt"))

            # Avaliação FINAL no teste (única vez em que o teste é usado)
            t0 = time.time()
            y, p, pr, idx = predict(model, test_loader, device)
            infer_ms = 1000 * (time.time() - t0) / len(y)
            met = reports.compute_metrics(y, p, class_names)
            pred_df = test_df.iloc[idx][["path", "class_name"]].copy()
            pred_df["y_true"], pred_df["y_pred"] = y, p
            pred_df["pred_name"] = [class_names[k] for k in p]
            pred_df["confidence"] = pr.max(1)
            pred_df.to_csv(os.path.join(exp_dir, "test_predictions.csv"), index=False)
            json.dump({**met, **info, "experiment": exp}, open(os.path.join(exp_dir, "metrics.json"), "w"),
                      indent=2)
            reports.plot_confusion(met["confusion_matrix"], class_names,
                                   os.path.join(fig_dir, f"cm_{exp}.png"), label)
            reports.plot_qualitative(test_df.iloc[idx].reset_index(drop=True), y, p, pr, class_names,
                                     os.path.join(fig_dir, f"qualitative_{exp}.png"))
            histories[label] = hist
            preds[exp] = (y, p, pr, idx)
            rows.append({"experiment": exp, "model": model_name, "augment": aug,
                         **{k: v for k, v in met.items() if k not in ("per_class", "confusion_matrix")},
                         "n_errors": int((y != p).sum()), "best_epoch": info["best_epoch"],
                         "epochs_run": len(hist), "best_val_f1": info["best_val_f1"],
                         "train_time_min": info["train_time_s"] / 60,
                         "params_M": count_params(model) / 1e6, "infer_ms_per_img": infer_ms})
            print(f"[exp] TESTE: acc={met['accuracy']:.4f} f1_macro={met['f1_macro']:.4f} "
                  f"f1_weighted={met['f1_weighted']:.4f}")
            del model; torch.cuda.empty_cache() if device.type == "cuda" else None

    # ------------------------------------------------------------ resumo
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\n[resumo]\n", summary[["experiment", "accuracy", "f1_macro", "f1_weighted",
                                   "n_errors", "best_epoch", "train_time_min"]].to_string(index=False))
    reports.plot_curves_all(histories, os.path.join(fig_dir, "curves_all.png"))

    best = summary.loc[summary["f1_macro"].idxmax()]
    best_exp = best["experiment"]
    best_label = reports.exp_label(best["model"], bool(best["augment"]))
    best_metrics = json.load(open(os.path.join(args.out, best_exp, "metrics.json")))
    for src_name, dst_name in [(f"cm_{best_exp}.png", "cm_best.png"),
                               (f"qualitative_{best_exp}.png", "qualitative_best.png")]:
        shutil.copy(os.path.join(fig_dir, src_name), os.path.join(fig_dir, dst_name))

    # Pares de classes mais confundidos (melhor modelo) e erros em comum
    cm = np.array(best_metrics["confusion_matrix"])
    pairs = [(class_names[i], class_names[j], int(cm[i, j]))
             for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: -t[2])
    common = None
    aug_exps = [e for e in preds if e.endswith("_aug")]
    if len(aug_exps) >= 2:
        sets = []
        for e in aug_exps:
            y, p, _, idx = preds[e]
            sets.append(set(idx[y != p].tolist()))
        common = sorted(set.intersection(*sets))
        test_df.iloc[common].to_csv(os.path.join(args.out, "errors_common_aug_models.csv"), index=False)
    analysis = {"best_experiment": best_exp, "top_confusions": pairs[:10],
                "n_errors_common_to_aug_models": None if common is None else len(common)}
    json.dump(analysis, open(os.path.join(args.out, "error_analysis.json"), "w"), indent=2,
              ensure_ascii=False)
    print("[análise] pares mais confundidos:", pairs[:5])
    print("[análise] erros em comum (modelos com DA):", analysis["n_errors_common_to_aug_models"])

    reports.write_latex(summary, best_metrics, splits, stats, class_names, gen_dir, best_label)
    with open(os.path.join(gen_dir, "macros.tex"), "a") as f:
        f.write(f"\\newcommand{{\\NCommonErrors}}{{{analysis['n_errors_common_to_aug_models']}}}\n")
        if pairs:
            a, b, n = pairs[0]
            f.write(f"\\newcommand{{\\TopConfTrue}}{{{reports.pretty_class(a)}}}\n")
            f.write(f"\\newcommand{{\\TopConfPred}}{{{reports.pretty_class(b)}}}\n")
            f.write(f"\\newcommand{{\\TopConfN}}{{{n}}}\n")
        else:
            f.write("\\newcommand{\\TopConfTrue}{--}\n\\newcommand{\\TopConfPred}{--}\n"
                    "\\newcommand{\\TopConfN}{0}\n")
    print(f"\n[main] Pronto. Melhor: {best_label} | tabelas em {gen_dir} | figuras em {fig_dir}")


if __name__ == "__main__":
    main()
