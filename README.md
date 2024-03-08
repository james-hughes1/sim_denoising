# SIM Denoising Project

## Description

This repository contains software that helps to denoise SIM microscopy images with low signal-to-noise ratios, in order to improve the final reconstructed image.

The aim of the implementation is to explore the reproducibility of a recent piece of research published in Nature Biotechnology [[1]](#key_paper).

## Acknowledgements

The source code has been adapted from earlier work that implemented the RCAN deep neural network architecture for a range of applications within fluorescence microscopy [[2]](#rcan).

The original version of that code can be found [here](https://github.com/AiviaCommunity/3D-RCAN) and was released with the following copyright and licensing:

> Copyright © 2021 [SVision Technologies LLC.](https://www.aivia-software.com/)
> Copyright © 2021-2022 Leica Microsystems, Inc.
>
> Released under Creative Commons Attribution-NonCommercial 4.0 International Public License ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))

This original code has been migrated to PyTorch, with parts of the PyTorch training pipeline adapted from another repository which can be found [here](https://github.com/edward-n-ward/ML-OS-SIM/tree/master); this software is also associated to recent research investigating the use of machine learning to improve the SIM reconstruction process [[3]](#ml_os_sim).

## References

<a id="key_paper">[1]</a>
Li, X., Wu, Y., Su, Y. et al. Three-dimensional structured illumination microscopy with enhanced axial resolution. Nat Biotechnol 41, 1307–1319 (2023). [https://doi.org/10.1038/s41587-022-01651-1](https://doi.org/10.1038/s41587-022-01651-1)

<a id="rcan">[2]</a>
Chen, J., Sasaki, H., Lai, H. et al. Three-dimensional residual channel attention networks denoise and sharpen fluorescence microscopy image volumes. Nat Methods 18, 678–687 (2021). [https://doi.org/10.1038/s41592-021-01155-x](https://doi.org/10.1038/s41592-021-01155-x)


<a id="ml_os_sim">[3]</a> Edward N. Ward, Rebecca M. McClelland, Jacob R. Lamb, Roger Rubio-Sánchez, Charles N. Christensen, Bismoy Mazumder, Sofia Kapsiani, Luca Mascheroni, Lorenzo Di Michele, Gabriele S. Kaminski Schierle, and Clemens F. Kaminski, "Fast, multicolour optical sectioning over extended fields of view with patterned illumination and machine learning," Biomed. Opt. Express 15, 1074-1088 (2024) [https://doi.org/10.1364/BOE.510912](https://doi.org/10.1364/BOE.510912)