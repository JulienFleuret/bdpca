#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>

#include <cmath>

#include "cuda_buffer"

#include <tuple>

namespace cuda
{

namespace utils
{

/**
 * @brief divUp : return the ceil of the division of "num" by "den".
 * @param num : numerator.
 * @param den : denominator.
 * @return ceil of the division of "num" by "den".
 */
__device__ __host__ __forceinline__ int divUp(const int& num, const int& den)
{
    return static_cast<int>(std::ceil(static_cast<float>(num) / static_cast<float>(den)));
}

/**
 * @brief getMinMax : return the min and max value of a cudaBuffer object.
 * @param handle : cublas handle.
 * @param src : memory segment to process
 * @param min : minimum value of the segment
 * @param max " maximum value of the segment
 * @note this function only work single and double precision
 * floating point type, and call cublasIsmax, cublasIsmin, cublasIdmin, cublasIdmax.
 */
template<class T>
__host__ void getMinMax(safe_cublasHandle_t& handle, const cudaBuffer<T>& src, T& min, T& max);

/**
 * Specialization for single precision floating point type.
 * @brief getMinMax : return the min and max value of a cudaBuffer object.
 * @param handle : cublas handle.
 * @param src : memory segment to process
 * @param min : minimum value of the segment
 * @param max " maximum value of the segment
 * @note this function only work single and double precision
 * floating point type, and call cublasIsmax, cublasIsmin.
 */
template<> __host__ void getMinMax<float>(safe_cublasHandle_t& handle, const cudaBuffer<float>& src, float& min, float& max);

/**
 * Specialization for single precision floating point type.
 * @brief getMinMax : return the min and max value of a cudaBuffer object.
 * @param handle : cublas handle.
 * @param src : memory segment to process
 * @param min : minimum value of the segment
 * @param max " maximum value of the segment
 * @note this function only work single and double precision
 * floating point type, and call cublasIsmax, cublasIsmin, cublasIdmin, cublasIdmax.
 */
template<> __host__ void getMinMax<double>(safe_cublasHandle_t& handle, const cudaBuffer<double>& src, double& min, double& max);

/**
 * @brief getMinMax : return the min and max value of a cudaBuffer object.
 * @param handle : cublas handle.
 * @param src : memory segment to process
 * @return a tuple containing the min and max values.
 * @note this function only work single and double precision
 * floating point type, and call cublasIsmax, cublasIsmin, cublasIdmin, cublasIdmax.
 */
template<class T>
__host__ std::tuple<T, T> getMinMax(safe_cublasHandle_t& handle, const cudaBuffer<float>& src);

/**
 * @brief rescale : Resecale the input data between 0 and 1
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param handle : cublas handle.
 * @param src : memory segment to process
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @note this function only work single and double precision
 * floating point type, and call cublasIsmax, cublasIsmin, cublasIdmin, cublasIdmax.
 */
template<class T>
__host__ void rescale(safe_cudaStream_t& stream, safe_cublasHandle_t& handle, cudaBuffer<T>& src, const int& rows, const int& cols);

/**
 * @brief reduce_mean : compute a reduce mean on a given axis.
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param handle : cublas handle.
 * @param src : memory segment to process
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @param axis : should the mean be computed along the columns or the rows
 * @param dst : memory segment that store the mean.
 * @note this function only work single and double precision
 * floating point type, and call cublasSaxpy, cublasScal, cublasDaxpy, cublasDscal.
 */
template<class T>
__host__ void reduce_mean(
                safe_cudaStream_t& stream,
                safe_cublasHandle_t& handle,
                 const cudaBuffer<T>& src,
                 const int& rows,
                 const int& cols,
                 const int& axis,
                 cudaBuffer<T>& dst
                 );

/**
 * @brief centre : subtract the mean to each rows or column depending on the given axis.
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param handle : cublas handle.
 * @param src : memory segment to process and return.
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @param axis : should the mean be computed along the columns or the rows
 * @param mean : memory segment that store the mean.
 * @note this function only work single and double precision
 */
template<class T>
__host__ void centre(safe_cudaStream_t& stream, cudaBuffer<T>& src, const int& rows, const int& cols, const int& axis, const cudaBuffer<T>& mean);

/**
 * @brief compute_scatter_matrices_2d : compute two scatter matrices, one along the rows (X x X') and one along the columns (X' x X), where "x" is the matrix multiplication.
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param handle : cublas handle.
 * @param X : memory segment to process
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @param Srt : memory segment storing the scatter matrix computed along the rows.
 * @param Sct : memory segment storing the scatter matrix computed along the columns.
 * @note this function only work single and double precision
 * floating point type, and call cublasSsyrk, cublasDsyrk.
 */
template<class T>
__host__ void compute_scatter_matrices_2d(safe_cublasHandle_t& handle, safe_cudaStream_t& stream, const cudaBuffer<T>& X, const int& rows, const int& cols, cudaBuffer<T>& Srt, cudaBuffer<T>& Sct);

/**
 * @brief eig : compute and return the right eigenvector of the given square matrix.
 * @param handle : cudnn handle.
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param rows_cols : number of rows and columns of the matrix.
 * @param eigenvectors : memory segment storing the eigenvector.
 * @note this function only work single and double precision
 * floating point type, and call cusolverDnSsyevd_bufferSize, cusolverDnDsyevd_bufferSize, cusolverDnSsyevd, cusolverDnDsyevd.
 */
template<class T>
__host__ void eig(safe_cusolverDnHandle_t& handle, safe_cudaStream_t& stream, const int& rows_cols, cudaBuffer<T>& eigenvectors);

/**
 * @brief col_range : reduce the input segment from a size of rows x cols to a size of rows x (end - start)
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param buffer : memory segment to process and return.
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @param start : index of the first columns of the range to keep
 * @param end : index of the first columns of the range to keep
 * @note this function only work single and double precision floating point type.
 */
template<class T>
__host__ void col_range(safe_cudaStream_t& stream, cudaBuffer<T>& buffer, const int& rows, const int& cols, const int& start, const int& end);

/**
 * @brief sort : sort every columns in descending order.
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param buffer : memory segment to process and return.
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @note this function only work single and double precision floating point type.
 */
template<class T>
__host__ void sort(safe_cudaStream_t& stream, cudaBuffer<T>& buffer, const int& rows, const int& cols);

/**
 * @brief compute_Y_2d : compute Y = Wct' x X x Wrt, where "'" is the transposition operator, and "x" is the matrix multiplication
 * @param handle : cublas handle.
 * @param Wct : memory segment storing eigenvector the scatter matrix computed along the rows.
 * @param X : memory segment of the input data.
 * @param Wrt : memory segment storing eigenvector the scatter matrix computed along the rolumns.
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @param krows : number of rows to keep.
 * @param kcols : number of columns to keep.
 * @param Y : memory to store the result of the computation.
 * @note this function only work single and double precision
 * floating point type, and call cublasSgemm, cublasDgemm.
 */
template<class T>
__host__ void compute_Y_2d(safe_cublasHandle_t& handle,
               const cudaBuffer<T>& Wct,
               const cudaBuffer<T>& X,
               const cudaBuffer<T>& Wrt,
               const int& rows, const int& cols,
               const int& krows, const int& kcols,
               cudaBuffer<T>& Y);


/**
 * @brief compute_scatter_matrices_3d : compute two scatter matrices, one along the rows (X x X') and one along the columns (X' x X), where "x" is the matrix multiplication.
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param handle : cublas handle.
 * @param X : memory segment to process.
 * @param rows : number of rows in the 3d array.
 * @param cols : number of columns in the 3d array.
 * @param frames : number of frames in the 3d array.
 * @param Srt : memory segment storing the scatter matrix computed along the rows.
 * @param Sct : memory segment storing the scatter matrix computed along the columns.
 * @note this function only work single and double precision
 * floating point type, and call cublasSsyrk, cublasDsyrk.
 */
template<class T>
__host__ void compute_scatter_matrices_3d(safe_cublasHandle_t& handle, safe_cudaStream_t& stream, const cudaBuffer<T>& X, const int& rows, const int& cols, const int& frames, cudaBuffer<T>& Srt, cudaBuffer<T>& Sct);


/**
 * @brief compute_Y_2d : compute Y = Wct' x X x Wrt, where "'" is the transposition operator, and "x" is the matrix multiplication
 * @param handle : cublas handle
 * @param Wct : memory segment storing eigenvector the scatter matrix computed along the rows.
 * @param X : memory segment of the input data.
 * @param Wrt : memory segment storing eigenvector the scatter matrix computed along the rolumns.
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @param frames : number of frames in the 3d array.
 * @param krows : number of rows to keep.
 * @param kcols : number of columns to keep.
 * @param Y : memory to store the result of the computation.
 * @note this function only work single and double precision
 * floating point type, and call cublasSgemm, cublasDgemm.
 */
template<class T>
__host__ void compute_Y_3d(safe_cublasHandle_t& handle,
               const cudaBuffer<T>& Wct,
               const cudaBuffer<T>& X,
               const cudaBuffer<T>& Wrt,
               const int& rows, const int& cols, const int& frames,
               const int& krows, const int& kcols,
               cudaBuffer<T>& Y);

} //utils

} //cuda


#endif // UTILS_H
