# bdpca

Principal Component Analysis (PCA) is a widely used statistical technique for dimensionality reduction and feature extraction. It finds applications in various fields, including data analysis, image processing, pattern recognition, and machine learning. A variant of PCA that has gained significant interest in applications such as face recognition and image compression is 2D-PCA. Zuo et al. proposed a generalized formulation of 2D-PCA, which they named Bi-Directional Principal Component Analysis (BDPCA).

This repository contains three versions of BDPCA:

* A naive implementation based on the article, available in the file 'bdpca.m'. This implementation has been tested on Octave and Matlab.
* A C-based mex-file, 'blas_lapack.c', which utilizes AVX (if available), SSE (if available), Matlab's BLAS, and Matlab's Lapack libraries.
* Another mex-file named 'fast_bdpca_cuda.cu' that leverages custom CUDA kernels, cuBlas, and cuSolver to accelerate the bdpca implementation. The necessary libraries for the second mex-file are located in the 'cuda' folder.
The two mex-files and the library required for the second mex-file are automatically compiled using the script 'compileme.m'.

**At this moment the execution of the mex-file and compileme.m has only be tested in MatLab on an Ubuntu 22.10.**

The code is distributed under FreeBSD License.

## References

- Zuo, W., Wang, K., & Zhang, D. (2005). Bi-directional PCA with assembled matrix distance metric. In IEEE International Conference on Image Processing 2005 (Vol. 2, pp. II-958). IEEE.

 
