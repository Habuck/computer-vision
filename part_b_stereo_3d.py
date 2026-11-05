import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/part_b", exist_ok=True)
os.makedirs("report_latex/figures", exist_ok=True)

def create_stereo_from_reference(ref_path="data/reference_image.jpg"):
    img = cv2.imread(ref_path)
    img = cv2.resize(img, (640, 480))
    h, w = img.shape[:2]

    # Build a realistic 3D depth/disparity ground-truth map for the architectural wall:
    # Slanted decorative wall (disparity increases from left=14px to right=36px) + Foreground structural column on left (disparity=46px)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    gt_disp = 14.0 + 22.0 * (xx / w) + 4.0 * ((h - yy) / h)

    # Left window frame / pillar in foreground (approx x < 170 - 0.18*y)
    pillar_mask = xx < (185 - 0.22 * yy)
    gt_disp[pillar_mask] = 44.0

    # Central raised geometric emblem relief
    relief_mask = ((xx - 350)**2 / 95**2 + (yy - 240)**2 / 120**2) <= 1.0
    gt_disp[relief_mask] += 10.0

    gt_disp = cv2.GaussianBlur(gt_disp, (9, 9), 2.0)

    # Synthesize right stereo view via backward-compatible horizontal disparity remap: I_R(x - d, y) = I_L(x, y)
    map_x = (xx + gt_disp).astype(np.float32)
    map_y = yy.astype(np.float32)
    imgL = img.copy()
    # In stereo, a point at (x, y) in Left image appears at (x - d, y) in Right image -> I_R(x, y) = I_L(x + d, y)
    imgR = cv2.remap(imgL, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    cv2.imwrite("data/stereo_left.png", imgL)
    cv2.imwrite("data/stereo_right.png", imgR)
    return imgL, imgR, gt_disp

def compute_epipolar_geometry(imgL, imgR):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(grayL, None)
    kp2, des2 = sift.detectAndCompute(grayR, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.5, 0.99)
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]

    lines1 = cv2.computeCorrespondEpilines(inliers2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(inliers1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    err1 = np.abs(lines1[:, 0]*inliers1[:, 0] + lines1[:, 1]*inliers1[:, 1] + lines1[:, 2]) / np.hypot(lines1[:, 0], lines1[:, 1])
    err2 = np.abs(lines2[:, 0]*inliers2[:, 0] + lines2[:, 1]*inliers2[:, 1] + lines2[:, 2]) / np.hypot(lines2[:, 0], lines2[:, 1])
    mean_epi_err = float(np.mean(err1 + err2) / 2.0)

    visL, visR = imgL.copy(), imgR.copy()
    w = imgL.shape[1]
    np.random.seed(12)
    indices = np.linspace(0, len(inliers1) - 1, min(30, len(inliers1)), dtype=int)
    for idx in indices:
        r1, r2, pt1, pt2 = lines1[idx], lines2[idx], inliers1[idx], inliers2[idx]
        color = tuple(np.random.randint(40, 255, 3).tolist())
        x0, y0 = map(int, [0, -r1[2]/r1[1]])
        x1, y1 = map(int, [w, -(r1[2]+r1[0]*w)/r1[1]])
        cv2.line(visL, (x0, y0), (x1, y1), color, 2)
        cv2.circle(visL, tuple(np.int32(pt1)), 5, color, -1)

        x0, y0 = map(int, [0, -r2[2]/r2[1]])
        x1, y1 = map(int, [w, -(r2[2]+r2[0]*w)/r2[1]])
        cv2.line(visR, (x0, y0), (x1, y1), color, 2)
        cv2.circle(visR, tuple(np.int32(pt2)), 5, color, -1)

    epi_pair = np.hstack([visL, visR])
    cv2.imwrite("results/part_b/epipolar_lines.png", epi_pair)
    cv2.imwrite("report_latex/figures/epipolar_lines.png", epi_pair)
    return F, len(pts1), len(inliers1), mean_epi_err

def write_ply(filename, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    ply_header = f"""ply
format ascii 1.0
element vertex {len(verts)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, 'w') as f:
        f.write(ply_header)
        for v, c in zip(verts, colors):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def run_part_b():
    imgL, imgR, gt_disp = create_stereo_from_reference()
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    F, total_matches, num_inliers, epi_err = compute_epipolar_geometry(imgL, imgR)
    print(f"Stereo Matches: {total_matches}, Inliers: {num_inliers}, Mean Epipolar Error: {epi_err:.4f} px")

    configs = [
        ("BM", 15, 64),
        ("BM", 21, 64),
        ("SGBM", 5, 64),
        ("SGBM", 9, 64),
        ("SGBM", 5, 80)
    ]

    records = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    axes[0].imshow(gt_disp, cmap='plasma', vmin=0, vmax=64)
    axes[0].set_title("Ground Truth Disparity Map", fontweight='bold')
    axes[0].axis('off')

    best_disp = None

    for idx, (algo, bsize, ndisp) in enumerate(configs):
        t0 = time.perf_counter()
        if algo == "BM":
            matcher = cv2.StereoBM_create(numDisparities=ndisp, blockSize=bsize)
        else:
            matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=ndisp,
                blockSize=bsize,
                P1=8 * 3 * bsize**2,
                P2=32 * 3 * bsize**2,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        raw_disp = matcher.compute(grayL, grayR).astype(np.float32) / 16.0
        runtime = (time.perf_counter() - t0) * 1000

        valid_mask = raw_disp > 0
        valid_ratio = (np.sum(valid_mask) / raw_disp.size) * 100.0

        eval_mask = valid_mask & (np.arange(raw_disp.shape[1])[None, :] >= ndisp)
        rmse_val = float(np.sqrt(np.mean((raw_disp[eval_mask] - gt_disp[eval_mask])**2)))
        bad_pixel_rate = float(np.mean(np.abs(raw_disp[eval_mask] - gt_disp[eval_mask]) > 2.0) * 100.0)

        records.append({
            "Method": algo,
            "blockSize": bsize,
            "numDisparities": ndisp,
            "Valid Disparity (%)": round(valid_ratio, 2),
            "Epipolar Error (px)": round(epi_err, 4),
            "RMSE (px)": round(rmse_val, 2),
            "Bad-Pixel >2px (%)": round(bad_pixel_rate, 2),
            "Runtime (ms)": round(runtime, 2)
        })

        ax = axes[idx + 1]
        ax.imshow(raw_disp, cmap='plasma', vmin=0, vmax=64)
        ax.set_title(f"{algo} (b={bsize}, d={ndisp})\nValid: {valid_ratio:.1f}% | RMSE: {rmse_val:.2f}px")
        ax.axis('off')

        if algo == "SGBM" and bsize == 9 and ndisp == 64:
            best_disp = raw_disp

    plt.suptitle("Part B: Dense Disparity Map Comparison (BM vs. SGBM)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/part_b/disparity_comparison.png", dpi=180)
    plt.savefig("report_latex/figures/disparity_comparison.png", dpi=180)
    plt.close()

    df = pd.DataFrame(records)
    df.to_csv("results/part_b/quantitative_part_b.csv", index=False)
    print(df.to_string(index=False))

    # Reconstruct 3D Point Cloud
    h, w = grayL.shape
    f_len = 0.8 * w
    Q = np.float32([
        [1, 0, 0, -0.5 * w],
        [0, -1, 0, 0.5 * h],
        [0, 0, 0, -f_len],
        [0, 0, 1.0 / 0.12, 0]
    ])
    points_3d = cv2.reprojectImageTo3D(best_disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = (best_disp > 8.0) & (np.arange(w)[None, :] >= 64)

    out_points = points_3d[mask]
    out_colors = colors[mask]

    write_ply("results/part_b/reconstructed_cloud.ply", out_points[::4], out_colors[::4])

    fig = plt.figure(figsize=(13, 5.5))
    step = max(1, len(out_points) // 10000)
    pts_sub = out_points[::step]
    col_sub = out_colors[::step] / 255.0

    views = [
        (22, -55, "3D Point Cloud - Oblique Perspective View"),
        (68, -90, "3D Point Cloud - Top-Down Depth Profile")
    ]
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.scatter(pts_sub[:, 0], pts_sub[:, 2], pts_sub[:, 1], c=col_sub, s=1.8, alpha=0.85)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Depth Z (m)")
        ax.set_zlabel("Y (m)")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontweight='bold')

    plt.tight_layout()
    plt.savefig("results/part_b/point_cloud_views.png", dpi=180)
    plt.savefig("report_latex/figures/point_cloud_views.png", dpi=180)
    plt.close()

if __name__ == "__main__":
    run_part_b()
