"""Training loop for generative models (VAE, CVAE, CVAE+Pred)."""

import torch
import numpy as np
import openpyxl as xl
import time
import os
from pathlib import Path

from torch.utils.data import DataLoader, TensorDataset

from cvae_amp.config.defaults import (
    SEED, GEN_EPOCHS, GEN_BATCH_SIZE, GEN_LR, KL_CYCLE, PROP_WEIGHT,
)
from cvae_amp.config.paths import DATA_RAW, MODELS_GEN, RESULTS, ensure_dirs
from cvae_amp.generation.models import VAE, CVAE, CVAE_Pred, DEVICE
from cvae_amp.generation.losses import vae_loss_fn, cvae_loss_fn, cvae_pred_loss_fn
from cvae_amp.generation.regularizers import get_kl_weight
from cvae_amp.generation.vocab import seq_to_onehot, process_labels


def load_data(path: str, vae_mode: bool = False) -> tuple[np.ndarray, np.ndarray]:
    wb = xl.load_workbook(path)
    ws = wb.active
    data_x, data_y = [], []

    for row in ws.iter_rows(min_row=2, values_only=True):
        seq, labels = row[1], [row[2], row[3], row[4]]
        if seq and len(seq) <= 50:
            data_x.append(seq_to_onehot(list(seq)))
            data_y.append(labels)

    x_np = np.array(data_x, dtype=np.float32)
    y_np = process_labels(data_y)

    if vae_mode:
        mask = (y_np == 1.0).any(axis=1)
        x_np, y_np = x_np[mask], y_np[mask]

    return x_np, y_np


def train_one_model(
    model,
    loss_fn,
    model_name: str,
    data_path: str,
    is_vae: bool = False,
    has_predictor: bool = False,
    epochs: int = GEN_EPOCHS,
    batch_size: int = GEN_BATCH_SIZE,
    lr: float = GEN_LR,
) -> tuple:
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_np, y_np = load_data(data_path, vae_mode=is_vae)
    print(f"Data: {len(x_np)} samples")

    x_tensor = torch.FloatTensor(x_np)
    y_tensor = torch.FloatTensor(y_np)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"total": [], "recon": [], "kl": [], "prop": []}
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_e = recon_e = kl_e = prop_e = 0.0
        kl_weight = get_kl_weight(epoch, cycle=KL_CYCLE)

        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()

            if is_vae:
                logits, mu, logvar = model(x_batch)
                recon, kl = loss_fn(logits, x_batch, mu, logvar)
                loss = recon + kl_weight * kl
            elif has_predictor:
                logits, mu, logvar, pred_labels = model(x_batch, y_batch)
                recon, kl, prop = loss_fn(
                    logits, x_batch, mu, logvar, pred_labels, y_batch,
                    prop_weight=PROP_WEIGHT,
                )
                loss = recon + kl_weight * kl + PROP_WEIGHT * prop
                prop_e += prop.item()
            else:
                logits, mu, logvar = model(x_batch, y_batch)
                recon, kl = loss_fn(logits, x_batch, mu, logvar)
                loss = recon + kl_weight * kl

            loss.backward()
            optimizer.step()
            total_e += loss.item()
            recon_e += recon.item()
            kl_e += kl.item()

        n = len(loader)
        history["total"].append(total_e / n)
        history["recon"].append(recon_e / n)
        history["kl"].append(kl_e / n)
        if has_predictor:
            history["prop"].append(prop_e / n)

        if epoch % 50 == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            prop_str = f"Prop: {history['prop'][-1]:.4f} | " if has_predictor else ""
            print(
                f"Epoch {epoch:4d} | Total: {history['total'][-1]:.4f} | "
                f"Recon: {history['recon'][-1]:.4f} | KL: {history['kl'][-1]:.4f} | "
                f"{prop_str}Time: {elapsed:.0f}s"
            )

    total_time = time.time() - start_time
    print(f"{model_name} done in {total_time:.0f}s")
    return model, history


def train_all_models(
    data_path: str | None = None,
    output_dir: str | None = None,
) -> dict[str, str]:
    if data_path is None:
        data_path = str(DATA_RAW / "VAE_train_real.xlsx")
    if output_dir is None:
        output_dir = str(MODELS_GEN)

    os.makedirs(output_dir, exist_ok=True)
    ensure_dirs()

    # Train VAE
    model_vae, hist_vae = train_one_model(
        VAE(), vae_loss_fn, "VAE (unconditional)",
        data_path, is_vae=True, has_predictor=False,
    )
    vae_path = os.path.join(output_dir, "vae_model.pth")
    torch.save(model_vae.state_dict(), vae_path)

    # Train CVAE
    model_cvae, hist_cvae = train_one_model(
        CVAE(), cvae_loss_fn, "CVAE (conditional, no predictor)",
        data_path, is_vae=False, has_predictor=False,
    )
    cvae_path = os.path.join(output_dir, "cvae_model.pth")
    torch.save(model_cvae.state_dict(), cvae_path)

    # Train CVAE+Pred
    model_cvae_pred, hist_cvae_pred = train_one_model(
        CVAE_Pred(), cvae_pred_loss_fn, "CVAE+Pred (conditional, with predictor)",
        data_path, is_vae=False, has_predictor=True,
    )
    cvae_pred_path = os.path.join(output_dir, "cvae_pred_model.pth")
    torch.save(model_cvae_pred.state_dict(), cvae_pred_path)

    # Save histories
    hist_path = os.path.join(str(RESULTS), "training_histories.npz")
    np.savez(
        hist_path,
        vae_total=hist_vae["total"], vae_recon=hist_vae["recon"], vae_kl=hist_vae["kl"],
        cvae_total=hist_cvae["total"], cvae_recon=hist_cvae["recon"], cvae_kl=hist_cvae["kl"],
        cvae_pred_total=hist_cvae_pred["total"], cvae_pred_recon=hist_cvae_pred["recon"],
        cvae_pred_kl=hist_cvae_pred["kl"], cvae_pred_prop=hist_cvae_pred["prop"],
    )

    print(f"\n{'='*60}")
    print("All models trained. Outputs:")
    print(f"  {vae_path}")
    print(f"  {cvae_path}")
    print(f"  {cvae_pred_path}")
    print(f"  {hist_path}")
    print(f"{'='*60}")

    return {"vae": vae_path, "cvae": cvae_path, "cvae_pred": cvae_pred_path}
