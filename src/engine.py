"""Laços de treino e avaliação."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def _amp_enabled(device: torch.device) -> bool:
    return device.type == "cuda"


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """Executa uma época. Se ``optimizer`` é None, roda em modo avaliação."""
    train = optimizer is not None
    model.train(train)
    total_loss, y_true, y_pred = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=_amp_enabled(device)):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            y_true.append(y.cpu().numpy())
            y_pred.append(logits.argmax(1).cpu().numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    n = len(y_true)
    return {
        "loss": total_loss / n,
        "acc": float((y_true == y_pred).mean()),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def fit(model, train_loader, val_loader, device, epochs=15, lr=1e-4,
        weight_decay=1e-4, patience=5, log=print):
    """Treina com AdamW + cosine annealing e early stopping.

    O melhor modelo é escolhido pelo F1-macro na VALIDAÇÃO; o conjunto
    de teste não é usado aqui.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=_amp_enabled(device))

    history, best_f1, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        te = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        va = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {"epoch": ep, "train_loss": tr["loss"], "train_acc": tr["acc"],
               "val_loss": va["loss"], "val_acc": va["acc"], "val_f1_macro": va["f1_macro"],
               "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": time.time() - te}
        history.append(row)
        log(f"  ep {ep:02d} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.4f} f1 {va['f1_macro']:.4f} "
            f"| {row['epoch_time_s']:.0f}s")
        if va["f1_macro"] > best_f1:
            best_f1, best_epoch, bad = va["f1_macro"], ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stopping na época {ep} (melhor: {best_epoch})")
                break
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "best_val_f1": best_f1,
                            "train_time_s": time.time() - t0}


@torch.no_grad()
def predict(model, loader, device):
    """Retorna rótulos verdadeiros, previstos, probabilidades e índices."""
    model.eval()
    ys, ps, probs, idxs = [], [], [], []
    for x, y, i in loader:
        x = x.to(device)
        with torch.autocast(device_type=device.type, enabled=_amp_enabled(device)):
            logits = model(x)
        pr = torch.softmax(logits.float(), 1).cpu().numpy()
        ys.append(y.numpy()); ps.append(pr.argmax(1)); probs.append(pr); idxs.append(i.numpy())
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(probs),
            np.concatenate(idxs))
