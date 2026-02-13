import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_preparation import get_dataloaders
from metrics import evaluate_reconstruction, InceptionV3FeatureExtractor
from github.face_reconstruction_gan import FaceReconstructionGAN


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 0.5) + 0.5


def save_sample_results(gan: FaceReconstructionGAN, val_loader, epoch: int, save_dir: str, device: str):
    os.makedirs(save_dir, exist_ok=True)

    gan.context_encoder.eval()
    gan.tan_generator.eval()

    batch = next(iter(val_loader))
    real = batch["real_image"][:8].to(device)
    masked = batch["masked_image"][:8].to(device)
    mask = batch["mask"][:8].to(device)

    with torch.no_grad():
        recon = gan.reconstruct(masked)

    composite = masked * mask + recon * (1 - mask)

    real = denormalize(real).cpu()
    masked = denormalize(masked).cpu()
    recon = denormalize(recon).cpu()
    composite = denormalize(composite).cpu()

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))

    for i in range(8):
        axes[0, i].imshow(real[i].permute(1, 2, 0).numpy())
        axes[1, i].imshow(masked[i].permute(1, 2, 0).numpy())
        axes[2, i].imshow(recon[i].permute(1, 2, 0).numpy())
        axes[3, i].imshow(composite[i].permute(1, 2, 0).numpy())
        for r in range(4):
            axes[r, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Masked", fontsize=12)
    axes[2, 0].set_ylabel("Reconstructed", fontsize=12)
    axes[3, 0].set_ylabel("Composite", fontsize=12)

    plt.suptitle(f"Epoch {epoch} - Face Reconstruction Results", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"epoch_{epoch:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved samples → {save_path}")


def plot_training_history(history: dict, save_dir: str):
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    epochs = range(1, len(history["d_loss"]) + 1)

    def _save_lineplot(y, title, fname, ylabel="Value"):
        plt.figure(figsize=(10, 6))
        plt.plot(list(epochs), y, linewidth=2)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()

    _save_lineplot(history["d_loss"], "Discriminator Loss", "1_discriminator_loss.png", ylabel="Loss")

    _save_lineplot(history["g_loss"], "Generator Total Loss", "2_generator_total_loss.png", ylabel="Loss")

    _save_lineplot(history["l_adv"], "Generator Adversarial Loss", "3_generator_adversarial_loss.png", ylabel="Loss")

    _save_lineplot(history["l_cc"], "Generator Context (L1 on Occluded) Loss", "4_generator_context_loss.png", ylabel="Loss")

    _save_lineplot(history["l_id"], "Generator Identity Loss", "5_generator_identity_loss.png", ylabel="Loss")

    _save_lineplot(history["psnr"], "PSNR (Occluded Regions)", "6_psnr.png", ylabel="PSNR (dB)")

    _save_lineplot(history["ssim"], "SSIM (Occluded Regions)", "7_ssim.png", ylabel="SSIM")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(list(epochs), history["d_loss"], linewidth=2)
    axes[0, 0].set_title("D Loss", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(list(epochs), history["g_loss"], linewidth=2)
    axes[0, 1].set_title("G Total Loss", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(list(epochs), history["l_adv"], linewidth=2)
    axes[1, 0].set_title("G Adv Loss", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(list(epochs), history["l_cc"], linewidth=2)
    axes[1, 1].set_title("G Context (L1) Loss", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Combined Losses", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "9_combined_losses.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(list(epochs), history["psnr"], linewidth=2, marker="o", markersize=4)
    axes[0].set_title("PSNR", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(epochs), history["ssim"], linewidth=2, marker="s", markersize=4)
    axes[1].set_title("SSIM", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Combined Metrics", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "10_combined_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*60}")
    print(f"All plots saved to: {plots_dir}/")
    print(f"{'='*60}")


def train(
    image_dir: str,
    epochs: int = 25,
    batch_size: int = 8,
    image_size: int = 128,
    mask_type: str = "mixed",
    train_split: float = 0.8,
    num_workers: int = 0,     
    save_interval: int = 5,
    save_dir: str = "output_new_2",
):
    device = get_device()
    print("\nUsing device:", device)

    os.makedirs(save_dir, exist_ok=True)
    samples_dir = os.path.join(save_dir, "samples")
    models_dir = os.path.join(save_dir, "models")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        image_dir=image_dir,
        batch_size=batch_size,
        image_size=image_size,
        mask_type=mask_type,
        train_split=train_split,
        num_workers=num_workers
    )

    gan = FaceReconstructionGAN(device=device)


    history = {
        "d_loss": [],
        "g_loss": [],
        "l_adv": [],
        "l_cc": [],
        "l_id": [],
        "psnr": [],
        "ssim": [],
    }

    print("\nStarting training...\n")

    for epoch in range(1, epochs + 1):
        gan.context_encoder.train()
        gan.tan_generator.train()
        gan.discriminator.train()

        epoch_d = []
        epoch_g = []
        epoch_adv = []
        epoch_cc = []
        epoch_id = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            real = batch["real_image"].to(device)
            masked = batch["masked_image"].to(device)
            mask = batch["mask"].to(device)

            losses = gan.train_step(real, masked, mask)

            epoch_d.append(losses["d_loss"])
            epoch_g.append(losses["g_loss"])
            epoch_adv.append(losses["l_adv"])
            epoch_cc.append(losses["l_cc"])
            epoch_id.append(losses["l_id"])

            pbar.set_postfix({
                "D": f"{losses['d_loss']:.3f}",
                "G": f"{losses['g_loss']:.3f}"
            })

        gan.context_encoder.eval()
        gan.tan_generator.eval()

        psnr_list = []
        ssim_list = []

        real_feats = []
        fake_feats = []

        with torch.no_grad():
            for batch in val_loader:
                real = batch["real_image"].to(device)
                masked = batch["masked_image"].to(device)
                mask = batch["mask"].to(device)

                fake = gan.reconstruct(masked)

                metrics = evaluate_reconstruction(real, fake, mask)
                psnr_list.append(metrics["psnr_occluded"])
                ssim_list.append(metrics["ssim_occluded"])


        avg_d = float(np.mean(epoch_d))
        avg_g = float(np.mean(epoch_g))
        avg_adv = float(np.mean(epoch_adv))
        avg_cc = float(np.mean(epoch_cc))
        avg_id = float(np.mean(epoch_id))
        avg_psnr = float(np.mean(psnr_list))
        avg_ssim = float(np.mean(ssim_list))

        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)
        history["l_adv"].append(avg_adv)
        history["l_cc"].append(avg_cc)
        history["l_id"].append(avg_id)
        history["psnr"].append(avg_psnr)
        history["ssim"].append(avg_ssim)

        print("\nEpoch", epoch)
        print(f"D Loss: {avg_d:.4f}")
        print(f"G Loss: {avg_g:.4f}")
        print(f"  L_adv: {avg_adv:.4f}")
        print(f"  L_cc : {avg_cc:.4f}")
        print(f"  L_id : {avg_id:.4f}")
        print(f"PSNR: {avg_psnr:.2f} dB")
        print(f"SSIM: {avg_ssim:.4f}")

        if epoch % save_interval == 0 or epoch == 1:
            save_sample_results(
                gan=gan,
                val_loader=val_loader,
                epoch=epoch,
                save_dir=samples_dir,
                device=device
            )

            torch.save(gan.context_encoder.state_dict(), os.path.join(models_dir, f"context_encoder_epoch_{epoch:04d}.pth"))
            torch.save(gan.tan_generator.state_dict(), os.path.join(models_dir, f"tan_generator_epoch_{epoch:04d}.pth"))
            torch.save(gan.discriminator.state_dict(), os.path.join(models_dir, f"discriminator_epoch_{epoch:04d}.pth"))
            print(f"Saved checkpoint models → epoch {epoch}\n")

    plot_training_history(history, save_dir)

    np.save(os.path.join(save_dir, "history.npy"), history, allow_pickle=True)

    print("\nTraining finished.")
    print(f"Samples: {samples_dir}")
    print(f"Models:  {models_dir}")
    print(f"Plots:   {os.path.join(save_dir, 'plots')}")


if __name__ == "__main__":
    train(
        image_dir="celeba_1000",
        epochs=25,
        batch_size=8,
        image_size=128,
        mask_type="mixed",
        save_interval=5,
        save_dir="output_new_2"
    )
