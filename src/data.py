"""Carregamento, verificação e divisão do Rice Disease Dataset.

Responsabilidades:
  * localizar as pastas de classe dentro do diretório baixado do Kaggle;
  * remover duplicatas exatas (mesmo hash MD5) ANTES da divisão, para
    evitar vazamento de dados entre treino/validação/teste;
  * dividir em treino/validação/teste com ``train_test_split``
    estratificado e semente fixa;
  * definir as transformações (com e sem data augmentation).
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Estatísticas do ImageNet: os pesos pré-treinados do torchvision esperam
# entradas normalizadas com estes valores.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _is_image(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in IMG_EXTS


def find_class_root(root: str) -> str:
    """Retorna o diretório cujos subdiretórios são as classes.

    O download do Kaggle pode vir aninhado (ex.: ``root/Rice_Leaf/...``).
    Procuramos o diretório mais raso que tenha >= 2 subpastas contendo
    imagens diretamente.
    """
    candidates = []
    for dirpath, dirnames, _ in os.walk(root):
        n_class_dirs = 0
        for d in dirnames:
            sub = os.path.join(dirpath, d)
            try:
                if any(_is_image(f) for f in os.listdir(sub)):
                    n_class_dirs += 1
            except OSError:
                pass
        if n_class_dirs >= 2:
            depth = dirpath[len(root):].count(os.sep)
            candidates.append((depth, dirpath, n_class_dirs))
    if not candidates:
        raise FileNotFoundError(f"Nenhuma pasta de classes encontrada em {root}")
    candidates.sort()
    if len(candidates) > 1:
        print("[data] Mais de um diretório candidato encontrado:")
        for depth, path, n in candidates:
            print(f"   depth={depth} classes={n} -> {path}")
        print(f"[data] Usando o mais raso: {candidates[0][1]}")
        print("[data] (use --data-dir para escolher outro explicitamente)")
    return candidates[0][1]


def _md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _dhash8(img: Image.Image, n: int = 8) -> np.ndarray:
    """dHash nas 8 orientações (rotações de 90° e espelhamento).

    Permite encontrar quase-duplicatas mesmo quando uma é versão
    girada/espelhada/redimensionada/recomprimida da outra.
    """
    im = img.convert("L").resize((64, 64), Image.BILINEAR)
    out = []
    for flip in (False, True):
        b = im.transpose(Image.FLIP_LEFT_RIGHT) if flip else im
        for r in (0, 90, 180, 270):
            a = np.asarray(b.rotate(r).resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
            out.append((a[:, 1:] > a[:, :-1]).flatten())
    return np.stack(out)


def _near_duplicate_groups(hashes: np.ndarray, labels: np.ndarray, thr: int) -> np.ndarray:
    """Agrupa (union-find) imagens da mesma classe com distância dHash <= thr."""
    n = len(hashes)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    base = hashes[:, 0, :]  # orientação original
    for i in range(n):
        d = (hashes[i + 1:] != base[i][None, None, :]).sum(-1).min(-1)
        for j in np.where(d <= thr)[0] + i + 1:
            if labels[i] == labels[j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def build_index(root: str, max_per_class: int | None = None, exclude_aug: bool = True,
                cache_dir: str | None = None, max_side: int = 512,
                near_dup_thr: int = 6) -> tuple[pd.DataFrame, dict]:
    """Lista, limpa e agrupa as imagens -> DataFrame(path, label, class_name, md5, group).

    Etapas (todas antes da divisão, para evitar vazamento de dados):
      1) remove arquivos de augmentation offline já presentes no dataset
         (nome iniciado por ``aug_``): são cópias transformadas das fotos
         originais e, se caíssem em conjuntos diferentes da original,
         o teste conteria versões de imagens vistas no treino;
      2) remove arquivos corrompidos;
      3) remove duplicatas exatas (MD5);
      4) redimensiona para cache (lado maior = max_side) para acelerar o treino;
      5) agrupa quase-duplicatas (dHash invariante a rotação/espelhamento) para
         que cada grupo fique inteiro em um único conjunto.
    """
    class_root = find_class_root(root)
    classes = sorted(
        d for d in os.listdir(class_root)
        if os.path.isdir(os.path.join(class_root, d))
        and any(_is_image(f) for f in os.listdir(os.path.join(class_root, d)))
    )
    rows, n_aug = [], 0
    for idx, cname in enumerate(classes):
        cdir = os.path.join(class_root, cname)
        files = sorted(f for f in os.listdir(cdir) if _is_image(f))
        if exclude_aug:
            n_aug += sum(f.lower().startswith("aug_") for f in files)
            files = [f for f in files if not f.lower().startswith("aug_")]
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            rows.append({"path": os.path.join(cdir, f), "label": idx, "class_name": cname})
    n_total = len(rows) + n_aug
    df = pd.DataFrame(rows)

    ok = []
    for p in df["path"]:
        try:
            with Image.open(p) as im:
                im.verify()
            ok.append(True)
        except Exception:
            ok.append(False)
    n_corrupt = int((~pd.Series(ok)).sum())
    df = df[ok].reset_index(drop=True)

    df["md5"] = [_md5(p) for p in df["path"]]
    nun = df.groupby("md5")["label"].nunique()
    conflict = set(nun[nun > 1].index)
    dup_conflict = int(df["md5"].isin(conflict).sum())
    df = df[~df["md5"].isin(conflict)]
    before = len(df)
    df = df.drop_duplicates(subset="md5", keep="first").reset_index(drop=True)
    dup_same = before - len(df)

    # cache redimensionado + hashes perceptuais
    hashes, new_paths = [], []
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    for i, p in enumerate(df["path"]):
        img = Image.open(p).convert("RGB")
        img = ImageOps.exif_transpose(img) if hasattr(ImageOps, "exif_transpose") else img
        hashes.append(_dhash8(img))
        if cache_dir:
            img.thumbnail((max_side, max_side), Image.BICUBIC)
            q = os.path.join(cache_dir, f"{df.at[i, 'label']}_{df.at[i, 'md5']}.jpg")
            if not os.path.exists(q):
                img.save(q, quality=95)
            new_paths.append(q)
    if cache_dir:
        df["orig_path"] = df["path"]
        df["path"] = new_paths
    groups = _near_duplicate_groups(np.stack(hashes), df["label"].to_numpy(), near_dup_thr)
    df["group"] = groups
    gsize = df.groupby("group")["path"].transform("size")

    stats = {
        "class_root": class_root,
        "classes": classes,
        "n_total_files": n_total,
        "n_aug_excluded": n_aug,
        "n_corrupt": n_corrupt,
        "n_duplicates_removed": dup_same,
        "n_conflicting_removed": dup_conflict,
        "n_after_cleaning": len(df),
        "n_groups": int(df["group"].nunique()),
        "n_in_near_dup_groups": int((gsize > 1).sum()),
        "per_class": dict(Counter(df["class_name"])),
    }
    return df, stats


def split_dataset(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15,
                  seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divisão hold-out estratificada em duas etapas com semente fixa.

    A divisão é feita sobre os GRUPOS de quase-duplicatas (coluna ``group``),
    estratificada pela classe; depois cada grupo é expandido para as suas
    imagens. Assim, fotos quase iguais nunca ficam em conjuntos diferentes.
      1) separa o teste (test_size dos grupos);
      2) do restante, separa a validação (val_size do total).
    """
    if "group" not in df.columns:
        df = df.assign(group=np.arange(len(df)))
    g = df.groupby("group")["label"].first().reset_index()
    g_trval, g_test = train_test_split(
        g, test_size=test_size, stratify=g["label"], random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    g_train, g_val = train_test_split(
        g_trval, test_size=rel_val, stratify=g_trval["label"], random_state=seed)
    pick = lambda gg: df[df["group"].isin(set(gg["group"]))].reset_index(drop=True)
    return pick(g_train), pick(g_val), pick(g_test)


def get_transforms(img_size: int = 224, augment: bool = True):
    """Transformações de treino e de avaliação.

    Data augmentation (apenas no treino) e justificativa:
      * RandomResizedCrop(scale 0.7-1.0): simula variação de distância e
        enquadramento da câmera sem cortar demais a lesão;
      * flips horizontal/vertical e rotação de até 20°: a orientação da
        folha na foto é arbitrária, a doença não depende dela;
      * ColorJitter leve (brilho/contraste/saturação 0.2, matiz 0.02):
        simula iluminação de campo (sol, sombra, nublado). O matiz varia
        muito pouco porque a COR é sintoma diagnóstico (ex.: manchas
        marrons x folha verde saudável); alterá-lo muito criaria
        exemplos com rótulo incorreto.
    Avaliação: apenas redimensiona para img_size x img_size (sem corte
    central, para não perder lesões perto das bordas) e normaliza.
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    if not augment:
        return eval_tf, eval_tf
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


class RiceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i], i
