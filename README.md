# INS3155 Computer Vision - Midterm Project
**Traditional Image Processing for Filtering, 3D Reconstruction, and Image Stitching**

## 1. Directory Structure
- `part_a_filtering.py`: Source code for Part A (Gaussian & Salt-and-Pepper noise synthesis, Mean, Gaussian, Median, and Laplacian Sharpening across 3x3, 5x5, 7x7 kernels).
- `part_b_stereo_3d.py`: Source code for Part B (SIFT + RANSAC Fundamental Matrix estimation, epipolar lines visualization, Block Matching vs. SGBM dense disparity comparison, and 3D point cloud `.ply` reconstruction).
- `part_c_stitching.py`: Source code for Part C (4-view panoramic stitching comparing SIFT, ORB, and AKAZE with KNN + Lowe's ratio test, RANSAC Homography, and Distance-Transform Feather Blending).
- `data/`: Contains the reference image (`reference_image.jpg` / `IMG_0891.jpg`), stereo pair (`stereo_left.png`, `stereo_right.png`), and 4 overlapping views (`view_1.png` ... `view_4.png`).
- `results/`: Contains all generated visual figures (`.png`), 3D point cloud (`reconstructed_cloud.ply`), and quantitative evaluation tables (`quantitative_part_a.csv`, `quantitative_part_b.csv`, `quantitative_part_c.csv`).
- `report_latex/`: Contains the complete LaTeX report (`main.tex`) formatted according to the official template and the compiled PDF (`StudentID_FullName_Report.pdf`).
- `requirements_docs/`: Contains the original project specification PDF (`Midterm_Exam_Computer_Vision_HK1.26-27.pdf`) and LaTeX report template (`Midterm_Exam_Report_Template.zip`).

## 2. Environment Setup
Install the required traditional computer vision libraries:
```bash
pip install -r requirements.txt
```

## 3. How to Run the Code
Run all three parts sequentially from the project root folder:
```bash
python part_a_filtering.py
python part_b_stereo_3d.py
python part_c_stitching.py
```
