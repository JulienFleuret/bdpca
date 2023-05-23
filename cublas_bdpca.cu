#include <cuda_runtime.h>
#include <vector_functions.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cstdio>


#include "safe_types.h"
#include "cuda_buffer"
#include "traits.h"
#include "utils.h"

#if 1
#include <iostream>
#endif

namespace cuda
{

#if 1
template<class T>
__host__ void print_mat_host(const T* ptr, const int& rows, const int& cols)
{
    std::vector<T> buffer(rows * cols);

    buffer.shrink_to_fit();

    CHECK_EXECUTION(cudaMemcpy(buffer.data(), ptr, buffer.size() * (size_t)sizeof(T), cudaMemcpyDeviceToHost));

    for(int r=0; r<rows; r++)
    {
        for(int c=0; c<cols; c++)
            std::cout<<buffer[c*rows + r]<<" ";
        std::cout<<std::endl;
    }
    std::cout<<std::endl<<std::endl;
}

template<class T>
__host__ void print_mat_host(const T* ptr, const int& rows, const int& cols, const int& frames)
{
   int offset = rows * cols;

   for(int frame = 0; frame < frames; frame++, ptr+=offset)
   {
       print_mat_host(ptr, rows, cols);
       std::cout<<std::endl;
   }
}


template<class T>
void print_mat_host(const cudaBuffer<T>& ptr, const int& rows, const int& cols)
{
    print_mat_host(static_cast<const T*>(ptr), rows, cols);
}

template<class T>
void print_mat_host(const cudaBuffer<T>& ptr, const int& rows, const int& cols, const int& frames)
{
    print_mat_host(static_cast<const T*>(ptr), rows, cols, frames);
}
#endif

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
void bdpca_2d(const cudaBuffer<T>& _X, const int& rows, const int& cols, const int& krows, const int& kcols, cudaBuffer<T>& Y, cudaBuffer<T>& X_, cudaBuffer<T>& Wrt, cudaBuffer<T>& Wct)
{
    cudaBuffer<T> X, Srt, Sct;

    safe_cudaStream_t stream;
    safe_cublasHandle_t cublas_handle(stream);
    safe_cusolverDnHandle_t cusolver_handle(stream);


    X = _X;

    X_.allocate(rows, 1);


    // Step 1) centre the values of X around its mean.
    utils::reduce_mean(stream, cublas_handle, X, rows, cols, 0, X_);

    utils::centre(stream, X, rows, cols, 0, X_);

    // Step 2) Compute the scatter matrices and eigenvectors matrices.
    utils::compute_scatter_matrices_2d(cublas_handle, stream, X, rows, cols, Srt, Sct);

    Wrt = Srt;
    utils::eig(cusolver_handle, stream, cols, Wrt);

    Srt.deallocate();

    Wct = Sct;
    utils::eig(cusolver_handle, stream, cols, Wct);

    Sct.deallocate();

    // Step 3) Reduce the dimensionality of the eigenvectors matrices.

    utils::col_range(stream, Wrt, cols, cols, 0, kcols);
    utils::col_range(stream, Wct, rows, rows, 0, krows);

    utils::sort(stream, Wrt, cols, kcols);
    utils::sort(stream, Wct, rows, krows);

    Y.allocate(krows * kcols);

    utils::compute_Y_2d(cublas_handle, Wct, X, Wrt, rows, cols, krows, kcols, Y);
}

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
std::tuple<cudaBuffer<T>, cudaBuffer<T>, cudaBuffer<T>, cudaBuffer<T> > bdpca_2d(const cudaBuffer<T>& X, const int& rows, const int& cols, const int& krows, const int& kcols)
{
    cudaBuffer<T> Y, X_, Wrt, Wct;

    bdpca_2d(X, rows, cols, krows, kcols, Y, X_, Wrt, Wct);

    return std::make_tuple(std::move(Y), std::move(X_), std::move(Wrt), std::move(Wct));
}

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
void bdpca_3d(const cudaBuffer<T>& _X, const int& rows, const int& cols, const int& frames, const int& krows, const int& kcols, cudaBuffer<T>& Y, cudaBuffer<T>& X_, cudaBuffer<T>& Wrt, cudaBuffer<T>& Wct)
{
    cudaBuffer<T> X, Srt, Sct;

    safe_cudaStream_t stream;
    safe_cublasHandle_t cublas_handle(stream);
    safe_cusolverDnHandle_t cusolver_handle(stream);

    X = _X;

    utils::rescale(stream, cublas_handle, X, rows * cols * frames, 1);

    X_.allocate(rows, cols);

    // Step 1) centre the values of X around its mean.
    utils::reduce_mean(stream, cublas_handle, X, rows * cols, frames, 1, X_);

    utils::centre(stream, X, rows, cols, 1, X_);

    // Step 2) Compute the scatter matrices and eigenvectors matrices.
    cuda::utils::compute_scatter_matrices_3d(cublas_handle, stream, X, rows, cols, frames, Srt, Sct);

    Wrt = Srt;
    utils::eig(cusolver_handle, stream, cols, Wrt);

    Srt.deallocate();

    Wct = Sct;

    Sct.deallocate();

    // Step 3) Reduce the dimensionality of the eigenvectors matrices.

    utils::col_range(stream, Wrt, cols, cols, 0, kcols);
    utils::col_range(stream, Wct, rows, rows, 0, krows);

    utils::sort(stream, Wrt, cols, kcols);
    utils::sort(stream, Wct, rows, krows);

    Y.allocate(krows * kcols);

    utils::compute_Y_3d(cublas_handle, Wct, X, Wrt, rows, cols, frames, krows, kcols, Y);
}

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
std::tuple<cudaBuffer<T>, cudaBuffer<T>, cudaBuffer<T>, cudaBuffer<T> > bdpca_3d(const cudaBuffer<T>& X, const int& rows, const int& cols, const int& frames, const int& krows, const int& kcols)
{
    cudaBuffer<T> Y, X_, Wrt, Wct;

    bdpca_3d(X, rows, cols, frames, krows, kcols, Y, X_, Wrt, Wct);

    return std::make_tuple(std::move(Y), std::move(X_), std::move(Wrt), std::move(Wct));
}

#if 0
template void print_mat_host<float>(const float* ptr, const int& rows, const int& cols);
template void print_mat_host<double>(const double* ptr, const int& rows, const int& cols);
#endif

template void bdpca_2d<float>(const cudaBuffer<float>& _X, const int& rows, const int& cols, const int& krows, const int& kcols, cudaBuffer<float>& Y, cudaBuffer<float>& X_, cudaBuffer<float>& Wrt, cudaBuffer<float>& Wct);
template void bdpca_2d<double>(const cudaBuffer<double>& _X, const int& rows, const int& cols, const int& krows, const int& kcols, cudaBuffer<double>& Y, cudaBuffer<double>& X_, cudaBuffer<double>& Wrt, cudaBuffer<double>& Wct);

template std::tuple<cudaBuffer<float>, cudaBuffer<float>, cudaBuffer<float>, cudaBuffer<float> > bdpca_2d(const cudaBuffer<float>& X, const int& rows, const int& cols, const int& krows, const int& kcols);
template std::tuple<cudaBuffer<double>, cudaBuffer<double>, cudaBuffer<double>, cudaBuffer<double> > bdpca_2d(const cudaBuffer<double>& X, const int& rows, const int& cols, const int& krows, const int& kcols);

template void bdpca_3d<float>(const cudaBuffer<float>& _X, const int& rows, const int& cols, const int& frames, const int& krows, const int& kcols, cudaBuffer<float>& Y, cudaBuffer<float>& X_, cudaBuffer<float>& Wrt, cudaBuffer<float>& Wct);
template void bdpca_3d<double>(const cudaBuffer<double>& _X, const int& rows, const int& cols, const int& frames, const int& krows, const int& kcols, cudaBuffer<double>& Y, cudaBuffer<double>& X_, cudaBuffer<double>& Wrt, cudaBuffer<double>& Wct);

template std::tuple<cudaBuffer<float>, cudaBuffer<float>, cudaBuffer<float>, cudaBuffer<float> > bdpca_3d(const cudaBuffer<float>& X, const int& rows, const int& cols, const int& frames, const int& krows, const int& kcols);
template std::tuple<cudaBuffer<double>, cudaBuffer<double>, cudaBuffer<double>, cudaBuffer<double> > bdpca_3d(const cudaBuffer<double>& X, const int& rows, const int& cols, const int& frames, const int& krows, const int& kcols);


} // cuda

