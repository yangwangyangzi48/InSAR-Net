# InSAR-Net

## Integrating Physical Modeling and Data-driven Learning: A DCNN for InSAR Phase Filtering and Applications

---

## Authors

**Wang Yang**<sup>1,2,3,4</sup>, **Yi He**<sup>1,2,4,*</sup>, **Juan M. Lopez-Sanchez**<sup>3</sup>, **Qing Zhu**<sup>1,2,5</sup>, **Alejandro Mestre-Quereda**<sup>3</sup>, **Lifeng Zhang**<sup>1,2,6</sup>, **Wenhui Wang**<sup>1,2,4</sup>, **Youdong Chen**<sup>1,2,6</sup>

<sup>*</sup> Corresponding author

---

## Affiliations

<sup>1</sup> Faculty of Geomatics, Lanzhou Jiaotong University, Lanzhou 730070, Gansu, China

<sup>2</sup> National-Local Joint Engineering Research Center of Technologies and Applications for National Geographic State Monitoring, Lanzhou 730070, Gansu, China

<sup>3</sup> Institute for Computer Research (IUII), University of Alicante, 03080 Alicante, Spain

<sup>4</sup> Gansu Provincial Key Laboratory of Science and Technology in Surveying & Mapping, Lanzhou 730070, Gansu, China

<sup>5</sup> Faculty of Geosciences and Environmental Engineering, Southwest Jiaotong University, Chengdu 611756, China

<sup>6</sup> School of Civil and Hydraulic Engineering, Lanzhou University of Technology, Lanzhou 730050, China

---

## Abstract

Interferometric synthetic aperture radar (**InSAR**) phase filtering is crucial for stable phase unwrapping and accurate topographic mapping and deformation retrieval. However, conventional spatial, frequency, and nonlocal filters, as well as existing learning-based methods, often struggle to balance noise suppression and detail preservation, especially under extremely low coherence and high fringe density, where oversmoothing and artifacts are prone to occur.

This paper proposes **InSAR-Net**, a deep neural network for phase filtering targeting severely noisy InSAR scenarios. The network is trained on a high-fidelity physics-based synthetic dataset that spans a wide range of coherence levels, fringe densities, and deformation patterns, with enriched extremely low-coherence samples to improve robustness.

InSAR-Net integrates:

* **Feature Separation Coding (FSC)** to enhance phase-related physical representations;
* **Residual dense encoding with a Feature Pyramid Network (FPN)** for multi-scale modeling;
* **Adaptive Feature Fusion (AFF)** with phase consistency loss to suppress pseudo fringes while preserving edges and fine structures.

Experiments on simulated and real SAR datasets demonstrate that InSAR-Net achieves superior performance over representative state-of-the-art methods in terms of **RMSE**, **PSNR**, and **SSIM**, with more pronounced gains under strong noise conditions.

Further validation using low-coherence masking, detrended residual phase analysis, and unwrapped phase stability assessment shows that InSAR-Net reduces random phase fluctuations and scattering-related noise in valid coherent areas, improves phase unwrapping reliability, and avoids physical interpretation of signal-void regions.

---

## Repository

The dataset and trained model will be made publicly available at:

https://github.com/yangwangyangzi48/InSAR-Net.git

---

## Keywords

InSAR; phase filtering; deep convolutional neural network; physical modeling; data-driven learning; phase unwrapping; low-coherence areas; synthetic aperture radar

## Model
<img width="849" height="651" alt="image" src="https://github.com/user-attachments/assets/6f86f29a-9405-43fa-a000-6682f74d525f" />
## results
<img width="837" height="968" alt="image" src="https://github.com/user-attachments/assets/c2402d98-88c8-44de-90d9-3854a60ae9f3" />
<img width="859" height="580" alt="image" src="https://github.com/user-attachments/assets/36c239e7-441d-4b6e-8d85-5180b58ef007" />


