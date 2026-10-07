# SIFT under pixel noise: airplane, 200 images

Noise: I' = clip(round(I + U(-a, a))), per pixel and channel, 0-255 scale. low: a = 4 (std 2.31); high: a = 20 (std 11.55). COLMAP 3.9.1 SIFT (CPU, max 8192 features) + exhaustive matching with two-view verification; no reconstruction. Object/background from the GT depth mask.

| | clean | low | high |
|---|---|---|---|
| keypoints per image, median | 410 | 457 | 679 |
|   on the object | 364 | 399 | 573 |
|   on the background | 45 | 53 | 107 |
| background share of all keypoints | 11.3% | 11.5% | 16.0% |
| object keypoints with a verified match | 83.9% | 81.7% | 62.4% |
| background keypoints with a verified match | 83.0% | 79.4% | 49.2% |
| image pairs with raw matches | 10239 | 10338 | 9390 |
| verified image pairs | 8494 | 8553 | 7369 |
| verified inliers per pair, median | 32 | 31 | 29 |
| verified inliers, total | 405887 | 404029 | 305968 |
| SIFT scale, object keypoints, median (px) | 1.50 | 1.41 | 1.35 |
| repeatability of clean object keypoints (<= 1.5 px) | - | 83.9% | 57.6% |
| shift of repeated keypoints, median (px) | - | 0.09 | 0.30 |

Figures: `NN_<image>.png` (clean | low | high; green = has a verified match, red = none), `summary.png`. Per-image numbers: `per_image.csv`.
