#ifndef CUBLAS_BDPCA_H
#define CUBLAS_BDPCA_H

//#include "cuda_buffer"

#include <tuple>

namespace cuda
{

template<class T>
class cudaBuffer;

/**
 * @brief bdpca_2d : Compute the bi-directional PCA for given matrix
 * @param X : device memory segments containing the data organize of a matrix.
 * The organization of the data is assume to be columnwisely.
 * @param rows : number of rows.
 * @param cols : number of columns.
 * @param krows : number of rows to keep.
 * @param kcols : number of columns to keep.
 * @param Y : Compressed matrix.
 * @param X_ : Mean of every columns.
 * @param Wrt : right eigenvectors of the rows scatter matrix.
 * @param Wct : right eigenvectors of the columns scatter matrix.
 */
template<class T>
void bdpca_2d(const cudaBuffer<T>& X, const int& rows, const int& cols, const int& krows, const int& kcols, cudaBuffer<T>& Y, cudaBuffer<T>& X_, cudaBuffer<T>& Wrt, cudaBuffer<T>& Wct);

/**
 * @brief bdpca_2d : Compute the bi-directional PCA for given matrix
 * @param X : device memory segments containing the data organize of a matrix.
 * The organization of the data is assume to be columnwisely.
 * @param rows : number of rows.
 * @param cols : number of columns.
 * @param krows : number of rows to keep.
 * @param kcols : number of columns to keep.
 * @return 4 matrices.
 * The first matrix is the compressed version of the inputs.
 * The second matrix is the Mean of every columns.
 * The third matrix is the right eigenvectors of the rows scatter matrix.
 * The fourth matrix is the right eigenvectors of the columns scatter matrix.
 */
template<class T>
std::tuple<cudaBuffer<T>, cudaBuffer<T>, cudaBuffer<T>, cudaBuffer<T> > bdpca_2d(const cudaBuffer<T>& X, const int& rows, const int& cols, const int& krows, const int& kcols);

/**
 * @brief bdpca_3d : Compute the bi-directional PCA for given matrix
 * @param X : device memory segments containing the data organize of a matrix.
 * The organization of the data is assume to be columnwisely.
 * @param rows : number of rows.
 * @param cols : number of columns.
 * @param frames : number of frames.
 * @param krows : number of rows to keep.
 * @param kcols : number of columns to keep.
 * @param Y : Compressed matrix.
 * @param X_ : Mean of every columns.
 * @param Wrt : right eigenvectors of the rows scatter matrix.
 * @param Wct : right eigenvectors of the columns scatter matrix.
 */
template<class T>
void bdpca_3d(const cudaBuffer<T>& X, const int& rows, const int& cols, const int& frames, const int& krows, const int& kcols, cudaBuffer<T>& Y, cudaBuffer<T>& X_, cudaBuffer<T>& Wrt, cudaBuffer<T>& Wct);

/**
 * @brief bdpca_3d : Compute the bi-directional PCA for given matrix
 * @param X : device memory segments containing the data organize of a matrix.
 * The organization of the data is assume to be columnwisely.
 * @param rows : number of rows.
 * @param cols : number of columns.
 * @param krows : number of rows to keep.
 * @param kcols : number of columns to keep.
 * @return 4 matrices.
 * The first matrix is the compressed version of the inputs.
 * The second matrix is the Mean of every columns.
 * The third matrix is the right eigenvectors of the rows scatter matrix.
 * The fourth matrix is the right eigenvectors of the columns scatter matrix.
 */
template<class T>
std::tuple<cudaBuffer<T>, cudaBuffer<T>, cudaBuffer<T>, cudaBuffer<T> > bdpca_3d(const cudaBuffer<T>& X, const int& rows, const int& cols, const int& frames, const int& krows, const int& kcols);



} // cuda

#endif // CUBLAS_BDPCA_H
