# INS3155 Computer Vision - Midterm Project
**Traditional Image Processing for Filtering, 3D Reconstruction, and Image Stitching**

Ha Manh Duc - 23070287

## 1. Directory Structure
- `part_a_filtering.py`: Part A. Synthesizes Gaussian and Salt-and-Pepper noise, then compares Mean, Gaussian, Median and Laplacian sharpening over 3x3, 5x5 and 7x7 kernels (PSNR, SSIM, MSE, runtime), plus a Gaussian sigma sensitivity sweep.
- `part_b_stereo_3d.py`: Part B. SIFT + RANSAC fundamental matrix with epipolar lines (epipolar and Sampson error), Block Matching vs. SGBM disparity against a known ground truth (valid ratio, RMSE, bad-pixel rate, runtime), a blockSize / numDisparities sensitivity study, and 3D point cloud export (`.ply`).
- `part_c_stitching.py`: Part C. 4-view panorama stitching comparing SIFT, ORB and AKAZE (KNN + Lowe's ratio test, RANSAC homography at 2 px and 4 px), keypoint visualization, and a blending comparison (none vs. distance-transform feathering vs. Laplacian pyramid blending).
- `generate_report_tables.py`: Converts the CSV results into LaTeX tables in `report_latex/tables/`.
- `data/`: Reference photograph (`IMG_0891.jpg` / `reference_image.jpg`) and the derived inputs: `clean_gray.png`, stereo pair (`stereo_left.png`, `stereo_right.png`) and 4 overlapping views (`view_1.png` ... `view_4.png`).
- `results/`: All generated figures (`.png`), the point cloud (`part_b/reconstructed_cloud.ply`), the fundamental matrix, and CSV tables for every experiment.
- `report_latex/`: LaTeX report (`main.tex`, figures, generated tables) and the compiled PDF `23070287_HaManhDuc_Report.pdf`.
- `requirements_docs/`: Project specification and the official report template.

## 2. Environment Setup
Python 3.10+ is required. OpenCV must be 4.x because AKAZE is not included in the OpenCV 5 main package.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. How to Run the Code
Run from the project root folder. Each script regenerates its inputs in `data/`, its figures and CSVs in `results/`, and the figures used by the report:
```bash
python part_a_filtering.py
python part_b_stereo_3d.py
python part_c_stitching.py
python generate_report_tables.py
```

## 4. Building the Report
```bash
cd report_latex
tectonic main.tex
```
`pdflatex main.tex` (run twice) also works with a full TeX Live / MacTeX installation.
