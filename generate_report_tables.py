import os
import pandas as pd

OUT_DIR = "report_latex/tables"
os.makedirs(OUT_DIR, exist_ok=True)

def fmt(v, digits):
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)

def write_table(name, df, columns, digits, align, bold_rows=None, midrule_every=None):
    bold_rows = bold_rows or set()
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(h for _, h in columns) + r" \\", r"\midrule"]
    for i, idx in enumerate(df.index):
        if midrule_every and i > 0 and i % midrule_every == 0:
            lines.append(r"\midrule")
        cells = [fmt(df.at[idx, c], digits.get(c, 2)) for c, _ in columns]
        if idx in bold_rows:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT_DIR, f"{name}.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

def best_index(df, group_col, metric, maximize=True):
    grouped = df.groupby(group_col)[metric]
    return set((grouped.idxmax() if maximize else grouped.idxmin()).values)

def part_a():
    df = pd.read_csv("results/part_a/quantitative_part_a.csv")
    df["Noise Type"] = df["Noise Type"].map({"Gaussian": r"Gaussian ($\sigma_n$=25)", "SaltPepper": "S\\&P (8\\%)"})
    df["Filter"] = df["Filter"].str.replace(r" \(\d+x\d+(, )?", " (", regex=True).str.replace(" ()", "", regex=False)
    df["Filter"] = df["Filter"].str.replace("sigma=", r"$\sigma$=", regex=False)
    df["Kernel"] = df["Kernel"].str.replace("x", r"$\times$", regex=False)
    df = df.sort_values(["Noise Type", "Kernel"], kind="stable").reset_index(drop=True)
    write_table("part_a_main", df,
                [("Noise Type", "Noise"), ("Filter", "Filter"), ("Kernel", "Kernel"), ("MSE", r"MSE $\downarrow$"),
                 ("PSNR (dB)", r"PSNR (dB) $\uparrow$"), ("SSIM", r"SSIM $\uparrow$"), ("Time (ms)", r"Time (ms) $\downarrow$")],
                {"MSE": 2, "PSNR (dB)": 2, "SSIM": 4, "Time (ms)": 3}, "llcrrrr",
                bold_rows=best_index(df, "Noise Type", "PSNR (dB)"), midrule_every=4)

    df = pd.read_csv("results/part_a/sigma_sensitivity.csv")
    df["Noise Type"] = df["Noise Type"].map({"Gaussian": "Gaussian", "SaltPepper": "S\\&P"})
    df["Kernel"] = df["Kernel"].str.replace("x", r"$\times$", regex=False)
    write_table("part_a_sigma", df,
                [("Noise Type", "Noise"), ("sigma", r"$\sigma$"), ("Kernel", "Kernel"), ("MSE", "MSE"),
                 ("PSNR (dB)", "PSNR (dB)"), ("SSIM", "SSIM"), ("Time (ms)", "Time (ms)")],
                {"sigma": 1, "MSE": 2, "PSNR (dB)": 2, "SSIM": 4, "Time (ms)": 3}, "lccrrrr",
                bold_rows=best_index(df, "Noise Type", "PSNR (dB)") | best_index(df, "Noise Type", "SSIM"),
                midrule_every=6)

def part_b():
    df = pd.read_csv("results/part_b/fundamental_matrix.csv")
    write_table("part_b_fundamental", df,
                [("Ratio-test Matches", "Ratio-test matches"), ("RANSAC Inliers", "RANSAC inliers"),
                 ("Inlier Ratio (%)", r"Inlier ratio (\%)"), ("Mean Epipolar Error (px)", "Epipolar err. (px)"),
                 ("Mean Sampson Error (px)", "Sampson err. (px)")],
                {"Inlier Ratio (%)": 2, "Mean Epipolar Error (px)": 4, "Mean Sampson Error (px)": 4}, "rrrrr")

    df = pd.read_csv("results/part_b/quantitative_part_b.csv")
    write_table("part_b_main", df,
                [("Method", "Method"), ("blockSize", "blockSize"), ("numDisparities", "numDisp."),
                 ("Valid Disparity (%)", r"Valid (\%) $\uparrow$"), ("RMSE (px)", r"RMSE (px) $\downarrow$"),
                 ("Bad-Pixel >2px (%)", r"Bad $>$2px (\%) $\downarrow$"), ("Runtime (ms)", r"Runtime (ms) $\downarrow$")],
                {"Valid Disparity (%)": 2, "RMSE (px)": 2, "Bad-Pixel >2px (%)": 2, "Runtime (ms)": 1}, "lccrrrr",
                bold_rows={df["RMSE (px)"].idxmin()})

    df = pd.read_csv("results/part_b/parameter_sensitivity.csv")
    write_table("part_b_sensitivity", df,
                [("Method", "Method"), ("Varied", "Varied"), ("blockSize", "blockSize"), ("numDisparities", "numDisp."),
                 ("Valid Disparity (%)", r"Valid (\%)"), ("RMSE (px)", "RMSE (px)"),
                 ("Bad-Pixel >2px (%)", r"Bad $>$2px (\%)"), ("Runtime (ms)", "Runtime (ms)")],
                {"Valid Disparity (%)": 2, "RMSE (px)": 2, "Bad-Pixel >2px (%)": 2, "Runtime (ms)": 1}, "llccrrrr")

def part_c():
    cols = [("Detector", "Detector"), ("RANSAC Thresh (px)", "RANSAC (px)"), ("Total Keypoints", "Keypoints"),
            ("Initial Matches", "Matches"), ("Inliers", "Inliers"), ("Inlier Ratio (%)", r"Inlier (\%)"),
            ("Mean RMSE (px)", "RMSE (px)"), ("Stitching Time (ms)", "Time (ms)")]
    digits = {"RANSAC Thresh (px)": 1, "Inlier Ratio (%)": 2, "Mean RMSE (px)": 3, "Stitching Time (ms)": 1}
    df = pd.read_csv("results/part_c/quantitative_part_c.csv")
    write_table("part_c_main", df, cols, digits, "lcrrrrrr", bold_rows={df["Mean RMSE (px)"].idxmin()})

    df = pd.read_csv("results/part_c/blending_comparison.csv")
    df["Blending"] = df["Blending"].map({"none": "None (overwrite)", "feather": "Feather (distance transform)",
                                         "multiband": "Laplacian pyramid (5 levels)"})
    write_table("part_c_blending", df,
                [("Blending", "Blending method"), ("Blending Time (ms)", "Blending time (ms)"),
                 ("Stitching Time (ms)", "Total time (ms)")],
                {"Blending Time (ms)": 1, "Stitching Time (ms)": 1}, "lrr")

if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    print(f"LaTeX tables written to {OUT_DIR}/")
