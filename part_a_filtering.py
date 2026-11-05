import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

os.makedirs("results/part_a", exist_ok=True)
os.makedirs("report_latex/figures", exist_ok=True)

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(image.astype(np.float32) + gauss, 0, 255)
    return noisy.astype(np.uint8)

def add_salt_pepper_noise(image, salt_prob=0.04, pepper_prob=0.04):
    noisy = image.copy()
    rand = np.random.rand(*image.shape)
    noisy[rand < salt_prob] = 255
    noisy[(rand >= salt_prob) & (rand < salt_prob + pepper_prob)] = 0
    return noisy

def laplacian_sharpening(image, ksize=3, alpha=0.25):
    lap = cv2.Laplacian(image, cv2.CV_32F, ksize=ksize)
    sharp = np.clip(image.astype(np.float32) - alpha * lap, 0, 255)
    return sharp.astype(np.uint8)

def run_part_a(img_path="data/reference_image.jpg"):
    clean_bgr = cv2.imread(img_path)
    clean = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
    clean = cv2.resize(clean, (640, 480))
    cv2.imwrite("data/clean_gray.png", clean)

    np.random.seed(42)
    noises = {
        "Gaussian": add_gaussian_noise(clean, sigma=25),
        "SaltPepper": add_salt_pepper_noise(clean, 0.04, 0.04)
    }

    kernel_sizes = [3, 5, 7]
    records = []

    for noise_name, noisy_img in noises.items():
        cv2.imwrite(f"results/part_a/noisy_{noise_name}.png", noisy_img)
        fig, axes = plt.subplots(4, 4, figsize=(15, 13))
        fig.suptitle(f"Part A: Filtering Comparison on {noise_name} Noise", fontsize=15, fontweight='bold')

        axes[0, 0].imshow(clean, cmap='gray')
        axes[0, 0].set_title("Original Clean Image")
        axes[0, 1].imshow(noisy_img, cmap='gray')
        axes[0, 1].set_title(f"{noise_name} Noisy\nPSNR: {psnr(clean, noisy_img, data_range=255):.2f} dB")
        axes[0, 2].axis('off')
        axes[0, 3].axis('off')
        axes[0, 0].axis('off')
        axes[0, 1].axis('off')

        for idx, k in enumerate(kernel_sizes):
            # 1. Mean Filter
            t0 = time.perf_counter()
            mean_img = cv2.blur(noisy_img, (k, k))
            t_mean = (time.perf_counter() - t0) * 1000

            # 2. Gaussian Filter
            sigma = round(0.8 + 0.4 * idx, 1)
            t0 = time.perf_counter()
            gauss_img = cv2.GaussianBlur(noisy_img, (k, k), sigmaX=sigma)
            t_gauss = (time.perf_counter() - t0) * 1000

            # 3. Median Filter
            t0 = time.perf_counter()
            med_img = cv2.medianBlur(noisy_img, k)
            t_med = (time.perf_counter() - t0) * 1000

            # 4. Laplacian Sharpening (on denoised image to enhance architectural edges)
            base_denoised = gauss_img if noise_name == "Gaussian" else med_img
            alpha_val = 0.25 if k == 3 else (0.12 if k == 5 else 0.03)
            t0 = time.perf_counter()
            sharp_img = laplacian_sharpening(base_denoised, ksize=k, alpha=alpha_val)
            t_sharp = (time.perf_counter() - t0) * 1000

            outputs = [
                (f"Mean ({k}x{k})", mean_img, t_mean),
                (f"Gaussian ({k}x{k}, sigma={sigma})", gauss_img, t_gauss),
                (f"Median ({k}x{k})", med_img, t_med),
                (f"Laplacian ({k}x{k})", sharp_img, t_sharp)
            ]

            for col_idx, (f_name, out_img, exec_t) in enumerate(outputs):
                val_mse = mse(clean, out_img)
                val_psnr = psnr(clean, out_img, data_range=255)
                val_ssim = ssim(clean, out_img, data_range=255)

                records.append({
                    "Noise Type": noise_name,
                    "Filter": f_name,
                    "Kernel": f"{k}x{k}",
                    "MSE": round(val_mse, 2),
                    "PSNR (dB)": round(val_psnr, 2),
                    "SSIM": round(val_ssim, 4),
                    "Time (ms)": round(exec_t, 3)
                })

                ax = axes[idx + 1, col_idx]
                ax.imshow(out_img, cmap='gray')
                ax.set_title(f"{f_name}\nPSNR: {val_psnr:.2f}dB | SSIM: {val_ssim:.3f}", fontsize=9.5)
                ax.axis('off')

        plt.tight_layout()
        out_fig = f"results/part_a/grid_{noise_name}.png"
        plt.savefig(out_fig, dpi=180)
        plt.savefig(f"report_latex/figures/grid_{noise_name}.png", dpi=180)
        plt.close()

    df = pd.DataFrame(records)
    df.to_csv("results/part_a/quantitative_part_a.csv", index=False)
    print(df.to_string(index=False))

    plot_metric_bars(df)
    sigma_sensitivity(clean, noises)

def plot_metric_bars(df):
    filters = ["Mean", "Gaussian", "Median", "Laplacian"]
    kernels = ["3x3", "5x5", "7x7"]
    width = 0.2
    x = np.arange(len(kernels))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, noise_name in enumerate(["Gaussian", "SaltPepper"]):
        sub = df[df["Noise Type"] == noise_name]
        for col, metric in enumerate(["PSNR (dB)", "SSIM"]):
            ax = axes[row, col]
            for i, f in enumerate(filters):
                vals = [sub[(sub["Filter"].str.startswith(f)) & (sub["Kernel"] == k)][metric].values[0] for k in kernels]
                ax.bar(x + (i - 1.5) * width, vals, width, label=f)
            ax.set_xticks(x)
            ax.set_xticklabels(kernels)
            ax.set_title(f"{metric} - {noise_name} noise", fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            if row == 0 and col == 0:
                ax.legend()
    plt.tight_layout()
    plt.savefig("results/part_a/metrics_bar_chart.png", dpi=180)
    plt.savefig("report_latex/figures/metrics_bar_chart.png", dpi=180)
    plt.close()

def sigma_sensitivity(clean, noises, sigmas=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
    records = []
    for noise_name, noisy_img in noises.items():
        for s in sigmas:
            k = 2 * int(np.ceil(3 * s)) + 1
            t0 = time.perf_counter()
            out = cv2.GaussianBlur(noisy_img, (k, k), sigmaX=s)
            exec_t = (time.perf_counter() - t0) * 1000
            records.append({
                "Noise Type": noise_name,
                "sigma": s,
                "Kernel": f"{k}x{k}",
                "MSE": round(mse(clean, out), 2),
                "PSNR (dB)": round(psnr(clean, out, data_range=255), 2),
                "SSIM": round(ssim(clean, out, data_range=255), 4),
                "Time (ms)": round(exec_t, 3)
            })

    df = pd.DataFrame(records)
    df.to_csv("results/part_a/sigma_sensitivity.csv", index=False)
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for noise_name in noises:
        sub = df[df["Noise Type"] == noise_name]
        axes[0].plot(sub["sigma"], sub["PSNR (dB)"], marker='o', label=noise_name)
        axes[1].plot(sub["sigma"], sub["SSIM"], marker='o', label=noise_name)
    for ax, metric in zip(axes, ["PSNR (dB)", "SSIM"]):
        ax.set_xlabel("Gaussian filter sigma")
        ax.set_ylabel(metric)
        ax.set_title(f"Gaussian filter: {metric} vs. sigma", fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig("results/part_a/sigma_sensitivity.png", dpi=180)
    plt.savefig("report_latex/figures/sigma_sensitivity.png", dpi=180)
    plt.close()

if __name__ == "__main__":
    run_part_a()
