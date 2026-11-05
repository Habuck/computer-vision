import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/part_c", exist_ok=True)
os.makedirs("report_latex/figures", exist_ok=True)

def create_4_overlapping_views(ref_path="data/reference_image.jpg"):
    img = cv2.imread(ref_path)
    # Resize to a wide high-res canvas (1200 x 650)
    canvas = cv2.resize(img, (1200, 650))
    H, W = canvas.shape[:2]

    # Extract 4 overlapping sub-views with subtle camera pitch/yaw perspective warps
    views = []
    crop_w, crop_h = 480, 560
    x_offsets = [0, 230, 460, 690]
    y_offsets = [40, 25, 45, 30]

    for i, (x0, y0) in enumerate(zip(x_offsets, y_offsets)):
        patch = canvas[y0:y0+crop_h, x0:x0+crop_w].copy()
        # Apply a slight perspective homography to simulate real handheld multi-view capture
        src_pts = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])
        shift = (-1)**i * 6.0
        dst_pts = np.float32([
            [max(0, shift), max(0, -shift)],
            [crop_w - max(0, -shift), max(0, shift)],
            [crop_w - max(0, shift), crop_h - max(0, -shift)],
            [max(0, -shift), crop_h - max(0, shift)]
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_view = cv2.warpPerspective(patch, M, (crop_w, crop_h), borderMode=cv2.BORDER_REFLECT)
        cv2.imwrite(f"data/view_{i+1}.png", warped_view)
        views.append(warped_view)

    # Save 4 input views figure
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for i, v in enumerate(views):
        axes[i].imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Overlapping View {i+1}", fontweight='bold')
        axes[i].axis('off')
    plt.suptitle("Part C: Four Overlapping Input Views from Reference Architectural Image", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/part_c/input_4_views.png", dpi=180)
    plt.savefig("report_latex/figures/input_4_views.png", dpi=180)
    plt.close()
    return views

def detect_and_match(img1, img2, detector_name="SIFT", ratio_thresh=0.75, ransac_thresh=4.0):
    if detector_name == "SIFT":
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
    elif detector_name == "AKAZE":
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    else:
        detector = cv2.ORB_create(nfeatures=3000)
        norm = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(norm)
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransac_thresh)
    inlier_mask = mask.ravel().astype(bool)

    inliers1 = pts1[inlier_mask]
    inliers2 = pts2[inlier_mask]

    proj_pts1 = cv2.perspectiveTransform(inliers2, H)
    reproj_errors = np.linalg.norm(inliers1 - proj_pts1, axis=2)
    rmse = float(np.sqrt(np.mean(reproj_errors**2)))

    return {
        "kp1": kp1, "kp2": kp2,
        "num_kp": len(kp1) + len(kp2),
        "initial_matches": len(good_matches),
        "inliers": int(np.sum(inlier_mask)),
        "inlier_ratio": (np.sum(inlier_mask) / max(1, len(good_matches))) * 100.0,
        "rmse": rmse,
        "H": H,
        "good_matches": good_matches,
        "mask": mask.ravel().tolist()
    }

def feather_blend(base_img, warped_img):
    mask_base = (np.sum(base_img, axis=2) > 0).astype(np.float32)
    mask_warp = (np.sum(warped_img, axis=2) > 0).astype(np.float32)

    dist_base = cv2.distanceTransform((mask_base * 255).astype(np.uint8), cv2.DIST_L2, 3)
    dist_warp = cv2.distanceTransform((mask_warp * 255).astype(np.uint8), cv2.DIST_L2, 3)

    total_weight = dist_base + dist_warp + 1e-6
    w1 = (dist_base / total_weight)[..., None]
    w2 = (dist_warp / total_weight)[..., None]

    blended = base_img.astype(np.float32) * w1 + warped_img.astype(np.float32) * w2
    return np.clip(blended, 0, 255).astype(np.uint8)

def stitch_4_images(views, detector_name="SIFT", ransac_thresh=4.0):
    t_start = time.perf_counter()

    m01 = detect_and_match(views[1], views[0], detector_name, ransac_thresh=ransac_thresh)
    m12 = detect_and_match(views[1], views[2], detector_name, ransac_thresh=ransac_thresh)
    m23 = detect_and_match(views[2], views[3], detector_name, ransac_thresh=ransac_thresh)

    inlier_matches = [m for m, keep in zip(m12["good_matches"], m12["mask"]) if keep]
    match_vis = cv2.drawMatches(
        views[1], m12["kp1"], views[2], m12["kp2"],
        inlier_matches[:80], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(f"results/part_c/inliers_{detector_name}_r{int(ransac_thresh)}.png", match_vis)

    H_0_to_1 = m01["H"]
    H_1_to_1 = np.eye(3)
    H_2_to_1 = m12["H"]
    H_3_to_1 = m12["H"] @ m23["H"]
    homographies = [H_0_to_1, H_1_to_1, H_2_to_1, H_3_to_1]

    all_corners = []
    for img, H in zip(views, homographies):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped_corners)

    all_corners = np.concatenate(all_corners, axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    T_shift = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    pano_w, pano_h = min(2400, xmax - xmin), min(1400, ymax - ymin)

    panorama = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
    for img, H in zip(views, homographies):
        warped = cv2.warpPerspective(img, T_shift @ H, (pano_w, pano_h))
        panorama = feather_blend(panorama, warped)

    total_time = (time.perf_counter() - t_start) * 1000
    cv2.imwrite(f"results/part_c/panorama_{detector_name}_r{int(ransac_thresh)}.png", panorama)

    pair_stats = [m01, m12, m23]
    return {
        "Detector": detector_name,
        "RANSAC Thresh (px)": ransac_thresh,
        "Total Keypoints": sum(s["num_kp"] for s in pair_stats),
        "Initial Matches": sum(s["initial_matches"] for s in pair_stats),
        "Inliers": sum(s["inliers"] for s in pair_stats),
        "Inlier Ratio (%)": round(np.mean([s["inlier_ratio"] for s in pair_stats]), 2),
        "Mean RMSE (px)": round(np.mean([s["rmse"] for s in pair_stats]), 3),
        "Stitching Time (ms)": round(total_time, 2)
    }, match_vis, panorama

def run_part_c():
    views = create_4_overlapping_views()
    configs = [
        ("SIFT", 2.0),
        ("SIFT", 4.0),
        ("ORB", 2.0),
        ("ORB", 4.0),
        ("AKAZE", 4.0)
    ]
    records = []
    vis_dict = {}
    for det, r_thresh in configs:
        res, match_vis, pano = stitch_4_images(views, detector_name=det, ransac_thresh=r_thresh)
        records.append(res)
        if r_thresh == 4.0:
            vis_dict[det] = (match_vis, pano, res)

    df = pd.DataFrame(records)
    df.to_csv("results/part_c/quantitative_part_c.csv", index=False)
    print(df.to_string(index=False))

    # Create comparison figure for report
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    for idx, det in enumerate(["SIFT", "ORB", "AKAZE"]):
        m_vis, pano_img, st = vis_dict[det]
        axes[idx, 0].imshow(cv2.cvtColor(m_vis, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f"{det} Inlier Matches (Inliers: {st['Inliers']}, Ratio: {st['Inlier Ratio (%)']:.1f}%)", fontweight='bold')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title(f"{det} Stitched 4-View Panorama (RMSE: {st['Mean RMSE (px)']:.3f}px, Time: {st['Stitching Time (ms)']:.1f}ms)", fontweight='bold')
        axes[idx, 1].axis('off')

    plt.suptitle("Part C: Multi-Detector Feature Matching & 4-View Panorama Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/part_c/stitching_comparison.png", dpi=180)
    plt.savefig("report_latex/figures/stitching_comparison.png", dpi=180)
    plt.close()

if __name__ == "__main__":
    run_part_c()
