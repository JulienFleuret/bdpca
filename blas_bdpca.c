#include "mex.h"
#ifndef OCTAVE
#include "blas.h"
#include "lapack.h"
#else
#include <blas.h>
#include <lapack.h>
#endif

#include <x86intrin.h>

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>


#ifndef CALLABLE_FUNCTION
#define CALLABLE_FUNCTION
#endif // CALLABLE_FUNCTION

#ifndef HELPER_FUNCTION
#define HELPER_FUNCTION static
#endif // HELPER_FUNCTION

mxArray* getInputArguments(const int nrhs, const mxArray **prhs, int*krows, int*kcols);
mxArray* createX_(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID);
mxArray* createY(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID);
mxArray* createWrt(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID);
mxArray* createWct(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID);

typedef mxArray* (*creation_function_type)(const int, const mwSize* __restrict__, const mwSize, const mwSize, const mxClassID);

void bdpca_2d_32f(const float* __restrict__ X, const int rows, const int cols, const int krows, const int kcols, float** __restrict__ _Y, float** __restrict__ _X_, float** __restrict__ _Wrt, float** __restrict__ _Wct);
void bdpca_2d_64f(const double* __restrict__ X, const int rows, const int cols, const int krows, const int kcols, double** __restrict__ _Y, double** __restrict__ _X_, double** __restrict__ _Wrt, double** __restrict__ _Wct);
void bdpca_3d_32f(const float* __restrict__ X, const int rows, const int cols, const int frames, const int krows, const int kcols, float** __restrict__ _Y, float** __restrict__ _X_, float** __restrict__ _Wrt, float** __restrict__ _Wct);
void bdpca_3d_64f(const double* __restrict__ X, const int rows, const int cols, const int frames, const int krows, const int kcols, double** __restrict__ _Y, double** __restrict__ _X_, double** __restrict__ _Wrt, double** __restrict__ _Wct);


void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    const creation_function_type init[] = {createY, createX_, createWrt, createWct};

    int krows, kcols;

    krows = kcols = 0;

    // Check and get the input arguments.
    // If the input argument is not from a floating point type,
    // it is converted.
    mxArray* X = getInputArguments(nrhs, prhs, &krows, &kcols);

    // get the dimensions.
    int nb_dims = mxGetNumberOfDimensions(X);

    const mwSize* dims = mxGetDimensions(X);

    mxClassID classID = mxGetClassID(X);

    // Create the outputs.
    for(int i=0;i<nlhs;i++)
        plhs[i] = init[i](nb_dims, dims, krows, kcols, classID);

    // Process
    if(classID == mxSINGLE_CLASS)
    {
        float* tmp[4] = {NULL, NULL, NULL, NULL};

        // Get the pointer of outputs.
        for(int i=0;i<nlhs;i++)
            tmp[i] = (float*)mxGetData(plhs[i]);

        // Compute.
        if(nb_dims == 2)
            bdpca_2d_32f((const float*)mxGetData(X),(int)dims[0], (int)dims[1], (int)krows, (int)kcols, &tmp[0], &tmp[1], &tmp[2], &tmp[3]);
        else
            bdpca_3d_32f((const float*)mxGetData(X),(int)dims[0], (int)dims[1], (int)dims[2], (int)krows, (int)kcols, &tmp[0], &tmp[1], &tmp[2], &tmp[3]);

    }
    else
    {
        double* tmp[4] = {NULL, NULL, NULL, NULL};

        // Get the pointer of outputs.
        for(int i=0;i<nlhs;i++)
            tmp[i] = (double*)mxGetData(plhs[i]);

        // Compute.        
        if(nb_dims == 2)
            bdpca_2d_64f((const double*)mxGetData(X),(int)dims[0], (int)dims[1], (int)krows, (int)kcols, &tmp[0], &tmp[1], &tmp[2], &tmp[3]);
        else
            bdpca_3d_64f((const double*)mxGetData(X),(int)dims[0], (int)dims[1], (int)dims[2], (int)krows, (int)kcols, &tmp[0], &tmp[1], &tmp[2], &tmp[3]);
    }

    mxDestroyArray(X);

}


CALLABLE_FUNCTION mxArray* getInputArguments(const int nrhs, const mxArray **prhs,
                                             int*krows, int*kcols)
{
    // Check and get the input arguments.
    if(nrhs<2)
    {
        mexErrMsgTxt("Not enough arguments");
    }
    else if(nrhs>3)
    {
        mexErrMsgTxt("To much arguments");
    }

    // Get the number of components to keep.
    if(!mxIsScalar(prhs[1]))
        mexErrMsgTxt("Wrong input type, for the second input argument. It should be a scalar");

    *krows = (int)(mxGetScalar(prhs[1]));

    if(nrhs == 3)
    {
        if(!mxIsScalar(prhs[2]))
            mexErrMsgTxt("Wrong input type");

        *kcols = (int)(mxGetScalar(prhs[2]));
    }
    else
    {
        *kcols = *krows;
    }


    // Get the input and ensure its data type is floating point.
    // Convert the input otherwise.
    int nb_dims = mxGetNumberOfDimensions(prhs[0]);

    if( (nb_dims > 3) || (nb_dims == 1) )
    {
        mexErrMsgTxt("The first argument, has an incomputible dimensionality. The dimensionality can only be 2d or 3d");
    }

    // Take X as a copy of the input.
    mxArray* X = mxDuplicateArray(prhs[0]);

    // If the input matrix is not from a floating point type,
    // then it is converted before being returned.
    mxClassID input_id = mxGetClassID(X);

    if(input_id != mxSINGLE_CLASS && input_id != mxDOUBLE_CLASS)
    {
        mxArray* tmp = NULL;
        mexCallMATLAB(1, &tmp, 1, &X, "single");

        mxDestroyArray(X);

        X = mxDuplicateArray(tmp);

        mxDestroyArray(tmp);
    }

    // If the matrix has 2D but the first dimension is larger than
    // the second, it should be transposed.
    const mwSize* dims = mxGetDimensions(prhs[0]);

    if( (nb_dims == 2) && (dims[0] < dims[1]) )
    {
        mxArray* tmp = NULL;
        mexCallMATLAB(1, &tmp, 1, &X, "transpose");

        mxDestroyArray(X);

        X = mxDuplicateArray(tmp);

        mxDestroyArray(tmp);
    }

    return X;
}
//
CALLABLE_FUNCTION mxArray* createX_(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID)
{
    (void)krows;
    (void)kcols;

    if(nb_dims == 2)
        return mxCreateNumericMatrix(1, dims[1], classID, mxREAL);
    else
        return mxCreateNumericMatrix(dims[0], dims[1], classID, mxREAL);
}

CALLABLE_FUNCTION mxArray* createY(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID)
{
    if(nb_dims == 2)
        return mxCreateNumericMatrix(krows, kcols, classID, mxREAL);
    // Implicit else
    const mwSize y_dims[] = {krows, kcols, dims[2]};

    return mxCreateNumericArray(nb_dims, y_dims, classID, mxREAL);
}

CALLABLE_FUNCTION mxArray* createWrt(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID)
{
    (void)nb_dims;
    (void)krows;

    return mxCreateNumericMatrix(dims[1], kcols, classID, mxREAL);
}

CALLABLE_FUNCTION mxArray* createWct(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID)
{
    (void)nb_dims;
    (void)kcols;

    return mxCreateNumericMatrix(dims[0], krows, classID, mxREAL);
}

////////////////////////////////////////////////////////////////////////

HELPER_FUNCTION void* kbyte_alloc(const size_t size)
{
    return aligned_alloc(0x400, size<0x400 ? 0x400 : size%0x400 ?
                         (size_t)(ceilf((float)size/1024.f)) * 0x400 :
                         size );
}

/**
 * Function for single precision floating points
 * @brief reduce_sum : compute a reduce sum on a given axis.
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
HELPER_FUNCTION void reduce_sum_32f(const float* __restrict__ src,
                                    const int rows,
                                    const int cols,
                                    const int axis,
                                    float* __restrict__ dst)
{
    const ptrdiff_t n = rows;
    const ptrdiff_t inc = 1;
    const ptrdiff_t inc_null = 0;
    const float alpha = 1.f;

    if(axis)
    {
        // Compute the sum of every column.
        for(int c=0;c<cols;c++, src+=rows)
            saxpy(&n, &alpha, src, &inc, dst, &inc);
    }
    else
    {
        // Compute the sum of every rows.        
        const float one = 1.f;
        for(int c=0; c<cols; c++, src+=rows, dst++)
            *dst = sdot(&n, src, &inc, &one, &inc_null);
    }
}

/**
 * Function for double precision floating points
 * @brief reduce_sum : compute a reduce sum on a given axis.
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
HELPER_FUNCTION void reduce_sum_64f(const double* __restrict__ src,
                                    const int rows,
                                    const int cols,
                                    const int axis,
                                    double* __restrict__ dst)
{
    const ptrdiff_t n = rows;
    const ptrdiff_t inc = 1;
    const ptrdiff_t inc_null = 0;
    const double alpha = 1.;

    if(axis)
    {
        // Compute the sum of every column.
        for(int c=0;c<cols;c++, src+=rows)
            daxpy(&n, &alpha, src, &inc, dst, &inc);
    }
    else
    {
        // Compute the sum of every rows.                
        const double one = 1.;
        for(int c=0; c<cols; c++, src+=rows, dst++)
            *dst = ddot(&n, src, &inc, &one, &inc_null);
    }
}

/**
 * Function for single precision floating points.
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
HELPER_FUNCTION void reduce_mean_32f(const float* __restrict__ src,
                                     const int rows,
                                     const int cols,
                                     const int axis,
                                     float* __restrict__ dst)
{

    const ptrdiff_t n = rows;
    const ptrdiff_t inc = 1;
    const ptrdiff_t inc_null = 0;
    const float alpha = 1.f;
    const float den = 1.f / (float)(axis ? cols : rows);

    if(axis)
    {
        // Compute the mean of every column.        
        for(int c=0;c<cols;c++, src+=rows)
            saxpy(&n, &alpha, src, &inc, dst, &inc);
    }
    else
    {
        // Compute the mean of every rows.        
        float* __restrict__ it_output = dst;
        const float once = 1.f;
        for(int c=0; c<cols; c++, src+=rows, it_output++)
            *it_output = sdot(&n, src, &inc, &once, &inc_null);
    }

    // Normalize the sum either by the number of columns or rows
    // depending on the axis.
    sscal(&n, &den, dst, &inc);
}

/**
 * Function for double precision floating points.
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
HELPER_FUNCTION void reduce_mean_64f(const double* __restrict__ src,
                                     const int rows,
                                     const int cols,
                                     const int axis,
                                     double* __restrict__ dst)
{

    const ptrdiff_t n = rows;
    const ptrdiff_t inc = 1;
    const ptrdiff_t inc_null = 0;
    const double alpha = 1.;
    const double den = 1. / (double)(axis ? cols : rows);

    if(axis)
    {
        // Compute the mean of every column.        
        for(int c=0;c<cols;c++, src+=rows)
            daxpy(&n, &alpha, src, &inc, dst, &inc);
    }
    else
    {
        // Compute the mean of every rows.
        double* __restrict__ it_output = dst;
        const double once = 1.f;
        for(int c=0; c<cols; c++, src+=rows, it_output++)
            *it_output = ddot(&n, src, &inc, &once, &inc_null);
    }

    // Normalize the sum either by the number of columns or rows
    // depending on the axis.    
    dscal(&n, &den, dst, &inc);
}

/**
 * Function for single precision floating points.
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
HELPER_FUNCTION void centre_32f(float* __restrict__ X,
                                const int rows,
                                const int cols,
                                const int axis,
                                const float* __restrict__ X_)
{
    const ptrdiff_t n = rows;
    const ptrdiff_t inc = 1;
    const ptrdiff_t inc_null = 0;
    const float alpha = -1.f;

    if(axis)
    {

#if defined(__AVX__)
        int simd_rows = rows - (rows%8);
#elif defined(__SSE__)
        int simd_rows = rows - (rows%4);
#endif

        for(int c=0; c<cols; c++, X+=rows)
        {
            const float* __restrict__ it_X_ = X_;
            float* __restrict__ it_X = X;

            int r=0;

#ifdef __AVX__

            for(; r<simd_rows; r+=8, it_X+=8, it_X_+=8)
            {
                __m256 vx = _mm256_loadu_ps(it_X);
                __m256 vx_bar = _mm256_loadu_ps(it_X_);

                vx = _mm256_sub_ps(vx, vx_bar);
                _mm256_storeu_ps(it_X, vx);
            }

            _mm256_zeroupper();

            __m128 vx = _mm_loadu_ps(it_X);
            __m128 vx_bar = _mm_loadu_ps(it_X_);

            vx = _mm_sub_ps(vx, vx_bar);

            _mm_storeu_ps(it_X, vx);

            it_X+=4;
            it_X_+=4;


#elif defined(__SSE__)
            for(; r<simd_rows; r+=4, it_X+=4, it_X_+=4)
            {
                __m128 vx = _mm_loadu_ps(it_X);
                __m128 vx_bar = _mm_loadu_ps(it_X_);

                vx = _mm_sub_ps(vx, vx_bar);
                _mm_storeu_ps(it_X, vx);
            }
#endif

            for(;r<rows; r++, it_X++, it_X_++)
                *it_X -= *it_X_;
        }

    }
    else
    {
        for(int c=0;c<cols;c++, X+=rows, X_++)
            saxpy(&n, &alpha, X_, &inc_null, X, &inc);
    }
}

/**
 * Function for double precision floating points.
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
HELPER_FUNCTION void centre_64f(double* __restrict__ X,
                                const int rows,
                                const int cols,
                                const int axis,
                                const double* __restrict__ X_)
{
    const ptrdiff_t n = rows;
    const ptrdiff_t inc = 1;
    const ptrdiff_t inc_null = 0;
    const double alpha = -1.;

    if(axis)
    {

        #if defined(__AVX__)
        int simd_rows = rows - (rows%4);
        #elif defined(__SSE__)
        int simd_rows = rows - (rows%2);
        #endif

        for(int c=0; c<cols; c++, X+=rows)
        {
            const double* __restrict__ it_X_ = X_;
            double* __restrict__ it_X = X;

            int r=0;

            #ifdef __AVX__

            for(; r<simd_rows; r+=4, it_X+=4, it_X_+=4)
            {
                __m256d vx = _mm256_loadu_pd(it_X);
                __m256d vx_bar = _mm256_loadu_pd(it_X_);

                vx = _mm256_sub_pd(vx, vx_bar);
                _mm256_storeu_pd(it_X, vx);
            }

            _mm256_zeroupper();

            __m128d vx = _mm_loadu_pd(it_X);
            __m128d vx_bar = _mm_loadu_pd(it_X_);

            vx = _mm_sub_pd(vx, vx_bar);

            _mm_storeu_pd(it_X, vx);

            it_X+=2;
            it_X_+=2;


            #elif defined(__SSE__)
            for(; r<simd_rows; r+=2, it_X+=2, it_X_+=2)
            {
                __m128d vx = _mm_loadu_pd(it_X);
                __m128d vx_bar = _mm_loadu_pd(it_X_);

                vx = _mm_sub_pd(vx, vx_bar);
                _mm_storeu_pd(it_X, vx);
            }
            #endif

            for(;r<rows; r++, it_X++, it_X_++)
                *it_X -= *it_X_;
        }

    }
    else
    {

        for(int c=0;c<cols;c++, X+=rows, X_++)
            daxpy(&n, &alpha, X_, &inc_null, X, &inc);
    }
}

/**
 * Function for single precision floating points.
 * @brief fill_symetric : fill the lower part of the scatter matrix.
 * @param X : address of the memory segment to process.
 * @param N : Square root of the total size allocated to X.
 * Because X represents a square matrix.
 */
HELPER_FUNCTION void fill_symetric_32f(float* __restrict__ X, const int N)
{
    const float* __restrict__ src = X;
    for(int j=0, jj=0, i=0, ii=0;j<N;j++, jj+=N, src+=N, i=0, ii=0)
    {
        const float* __restrict__ it_src = src;
        float* __restrict__ it_dst = X + j;
        for(; i<j;i++, ii+=N, it_dst+=N, it_src++)
            *it_dst = *it_src;
    }
}

/**
 * Function for double precision floating points.
 * @brief fill_symetric : fill the lower part of the scatter matrix.
 * @param X : address of the memory segment to process.
 * @param N : Square root of the total size allocated to X.
 * Because X represents a square matrix.
 */
HELPER_FUNCTION void fill_symetric_64f(double* __restrict__ X, const int N)
{
    const double* __restrict__ src = X;
    for(int j=0, jj=0, i=0, ii=0;j<N;j++, jj+=N, src+=N, i=0, ii=0)
    {
        const double* __restrict__ it_src = src;
        double* __restrict__ it_dst = X + j;
        for(; i<j;i++, ii+=N, it_dst+=N, it_src++)
            *it_dst = *it_src;
    }
}

/**
 * Function for single precision floating points.
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
HELPER_FUNCTION void compute_scatter_matrices_2d_32f(
    const float* __restrict__ X,
    const int rows, const int cols,
    float* __restrict__ Srt,
    float* __restrict__ Sct)
{
    const char storage_mode = 'U';
    const char do_transpose = 'T';
    const char do_not_transpose = 'N';
    ptrdiff_t n = (ptrdiff_t)rows;
    ptrdiff_t k = (ptrdiff_t)cols;
    const float zero = 0.f;
    const float inv_rows = 1.f/((float)rows);
    const float inv_cols = 1.f/((float)cols);
    //    const float alpha_beta = 1.f;

    ssyrk(&storage_mode, &do_transpose, &k, &n, &inv_rows, X, &n, &zero, Srt, &k);
    fill_symetric_32f(Srt, cols);

    ssyrk(&storage_mode, &do_not_transpose, &n, &k, &inv_cols, X, &n, &zero, Sct, &n);
    fill_symetric_32f(Sct, rows);
}

/**
 * Function for double precision floating points.
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
HELPER_FUNCTION void compute_scatter_matrices_2d_64f(
    const double* __restrict__ X,
    const int rows, const int cols,
    double* __restrict__ Srt,
    double* __restrict__ Sct)
{
    const char storage_mode = 'U';
    const char do_transpose = 'T';
    const char do_not_transpose = 'N';
    ptrdiff_t n = (ptrdiff_t)rows;
    ptrdiff_t k = (ptrdiff_t)cols;
    const double zero = 0.;
    const double inv_rows = 1./((double)rows);
    const double inv_cols = 1./((double)cols);

    dsyrk(&storage_mode, &do_transpose, &k, &n, &inv_rows, X, &n, &zero, Srt, &k);
    fill_symetric_64f(Srt, cols);

    dsyrk(&storage_mode, &do_not_transpose, &n, &k, &inv_cols, X, &n, &zero, Sct, &n);
    fill_symetric_64f(Sct, rows);
}

/**
 * Function for single precision floating points.
 * @brief eig : compute and return the right eigenvector of the given square matrix.
 * @param handle : cudnn handle.
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param rows_cols : number of rows and columns of the matrix.
 * @param eigenvectors : memory segment storing the eigenvector.
 * @note this function only work single and double precision
 * floating point type, and call cusolverDnSsyevd_bufferSize, cusolverDnDsyevd_bufferSize, cusolverDnSsyevd, cusolverDnDsyevd.
 */
HELPER_FUNCTION void eig_f32(float* __restrict__ src,
                             const int rows_cols,
                             float* __restrict__ eigen_vectors)
{
    const char do_compute_right_ev = 'V';
    const char do_not_compute_left_ev = 'N';

    // Allocate memory for eigenvalues

    const size_t eigen_values_size = (size_t)(rows_cols * (int)sizeof(double));

    float* eigen_values_real = (float*)kbyte_alloc(eigen_values_size);
    float* eigen_values_imaginary = (float*)kbyte_alloc(eigen_values_size);

    // Compute optimal workspace size
    const ptrdiff_t n = (ptrdiff_t)rows_cols;
    const ptrdiff_t lwork_query = -1;
    float work_query = 0.f;
    ptrdiff_t lwork, info;


    sgeev(
        &do_not_compute_left_ev,
        &do_compute_right_ev,
        &n,
        src,
        &n,
        eigen_values_real,
        eigen_values_imaginary,
        NULL,
        &n,
        eigen_vectors,
        &n,
        &work_query,
        &lwork_query,
        &info);
    lwork = (ptrdiff_t)work_query;

    float* workspace = (float*)kbyte_alloc(lwork * sizeof(float));


    // Compute eigenvectors
    sgeev(
        &do_not_compute_left_ev,
        &do_compute_right_ev,
        &n,
        src,
        &n,
        eigen_values_real,
        eigen_values_imaginary,
        NULL,
        &n,
        eigen_vectors,
        &n,
        workspace,
        &lwork,
        &info);

    // Free memory
    free(eigen_values_real);
    free(eigen_values_imaginary);
    free(workspace);
}

/**
 * Function for double precision floating points.
 * @brief eig : compute and return the right eigenvector of the given square matrix.
 * @param handle : cudnn handle.
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param rows_cols : number of rows and columns of the matrix.
 * @param eigenvectors : memory segment storing the eigenvector.
 * @note this function only work single and double precision
 * floating point type, and call cusolverDnSsyevd_bufferSize, cusolverDnDsyevd_bufferSize, cusolverDnSsyevd, cusolverDnDsyevd.
 */
HELPER_FUNCTION void eig_f64(double* __restrict__ src,
                             const int rows_cols,
                             double* __restrict__ eigen_vectors)
{
    const char do_compute_right_ev = 'V';
    const char do_not_compute_left_ev = 'N';

    // Allocate memory for eigenvalues

    const size_t eigen_values_size = (size_t)(rows_cols * (int)sizeof(double));

    double* eigen_values_real = (double*)kbyte_alloc(eigen_values_size);
    double* eigen_values_imaginary = (double*)kbyte_alloc(eigen_values_size);

    // Compute optimal workspace size
    const ptrdiff_t n = (ptrdiff_t)rows_cols;
    const ptrdiff_t lwork_query = -1;
    double work_query = 0.;
    ptrdiff_t lwork, info;


    dgeev(
        &do_not_compute_left_ev,
        &do_compute_right_ev,
        &n,
        src,
        &n,
        eigen_values_real,
        eigen_values_imaginary,
        NULL,
        &n,
        eigen_vectors,
        &n,
        &work_query,
        &lwork_query,
        &info);
    lwork = (ptrdiff_t)work_query;

    double* workspace = (double*)kbyte_alloc(lwork * sizeof(double));


    // Compute eigenvectors
    dgeev(
        &do_not_compute_left_ev,
        &do_compute_right_ev,
        &n,
        src,
        &n,
        eigen_values_real,
        eigen_values_imaginary,
        NULL,
        &n,
        eigen_vectors,
        &n,
        workspace,
        &lwork,
        &info);

    // Free memory
    free(eigen_values_real);
    free(eigen_values_imaginary);
    free(workspace);
}

/**
 * Function for single precision floating points.
 * @brief col_range : reduce the input segment from a size of rows x cols to a size of rows x (end - start)
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param buffer : memory segment to process and return.
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @param start : index of the first columns of the range to keep
 * @param end : index of the first columns of the range to keep
 * @note this function only work single and double precision floating point type.
 */
HELPER_FUNCTION void col_range_32f(float** __restrict__ ptr,
                                   const int rows, const int cols,
                                   const int start, const int end)
{
    (void)cols;

    size_t mem_size = (size_t)((end-start) * rows * (int)sizeof(float));

    float* tmp = (float*)kbyte_alloc(mem_size);

    memcpy(tmp, *ptr + start * rows, mem_size);

    free(*ptr);
    *ptr = tmp;
}

/**
 * Function for double precision floating points.
 * @brief col_range : reduce the input segment from a size of rows x cols to a size of rows x (end - start)
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param buffer : memory segment to process and return.
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @param start : index of the first columns of the range to keep
 * @param end : index of the first columns of the range to keep
 * @note this function only work single and double precision floating point type.
 */
HELPER_FUNCTION void col_range_64f(double** __restrict__ ptr,
                                   const int rows, const int cols,
                                   const int start, const int end)
{
    (void)cols;

    size_t mem_size = (size_t)((end-start) * rows * (int)sizeof(double));

    double* tmp = (double*)kbyte_alloc(mem_size);

    memcpy(tmp, *ptr + start * rows, mem_size);

    free(*ptr);
    *ptr = tmp;
}

// Function to swap two elements
HELPER_FUNCTION void swap_32f(float* __restrict__ a, float* __restrict__ b)
{
    float t = *a;
    *a = *b;
    *b = t;
}

HELPER_FUNCTION void swap_64f(double* __restrict__ a, double* __restrict__ b)
{
    double t = *a;
    *a = *b;
    *b = t;
}

// Partition the array using the last element as the pivot
HELPER_FUNCTION int partition_32f(float* __restrict__ arr, const int low, const int high)
{
    // Choosing the pivot
    float pivot = arr[high];

    // Index of smaller element and indicates
    // the right position of pivot found so far
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++)
    {

        // If current element is smaller than the pivot
        if (arr[j] > pivot)
            {

            // Increment index of smaller element
            i++;
            swap_32f(&arr[i], &arr[j]);
        }
    }
    swap_32f(&arr[i + 1], &arr[high]);
    return (i + 1);
}

HELPER_FUNCTION int partition_64f(double* __restrict__ arr, const int low, const int high)
{
    // Choosing the pivot
    double pivot = arr[high];

    // Index of smaller element and indicates
    // the right position of pivot found so far
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++)
    {

        // If current element is smaller than the pivot
        if (arr[j] > pivot)
        {

            // Increment index of smaller element
            i++;
            swap_64f(&arr[i], &arr[j]);
        }
    }
    swap_64f(&arr[i + 1], &arr[high]);
    return (i + 1);
}

/**
 * Function for single precision floating points.
 * @brief sort : sort every columns in descending order.
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param buffer : memory segment to process and return.
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @note this function only work single and double precision floating point type.
 */
HELPER_FUNCTION void quickSort_32f(float* __restrict__ arr, const int low, const int high)
{
    if (low < high)
    {

        // pi is partitioning index, arr[p]
        // is now at right place
        int pi = partition_32f(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        quickSort_32f(arr, low, pi - 1);
        quickSort_32f(arr, pi + 1, high);
    }
}

HELPER_FUNCTION void quickSort_64f(double* __restrict__ arr,
                                   const int low, const int high)
{
    if (low < high) {

        // pi is partitioning index, arr[p]
        // is now at right place
        int pi = partition_64f(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        quickSort_64f(arr, low, pi - 1);
        quickSort_64f(arr, pi + 1, high);
    }
}


/**
 * Function for single precision floating points.
 * @brief sort : sort every columns in descending order.
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param buffer : memory segment to process and return.
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @note this function only work single and double precision floating point type.
 */
HELPER_FUNCTION void sort_32f(float* __restrict__ ptr, const int rows, const int cols)
{
    for(int c=0; c<cols; c++, ptr+=rows)
        quickSort_32f(ptr, 0, cols - 1);
}

/**
 * Function for double precision floating points.
 * @brief sort : sort every columns in descending order.
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param buffer : memory segment to process and return.
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @note this function only work single and double precision floating point type.
 */
HELPER_FUNCTION void sort_64f(double* __restrict__ ptr, const int rows, const int cols)
{
    for(int c=0; c<cols; c++, ptr+=rows)
        quickSort_64f(ptr, 0, cols - 1);
}

/**
 * Function for single precision floating points.
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
HELPER_FUNCTION void compute_Y_2d_32f(const float* __restrict__ Wct,
                                      const float* __restrict__ X,
                                      const float* __restrict__ Wrt,
                                      const int rows,
                                      const int cols,
                                      const int krows,
                                      const int kcols,
                                      float* __restrict__ Y)
{
    const ptrdiff_t n = (ptrdiff_t)rows;
    const ptrdiff_t k = (ptrdiff_t)cols;
    const ptrdiff_t m = (ptrdiff_t)krows;
    const ptrdiff_t l = (ptrdiff_t)kcols;

    const char do_transpose = 'T';
    const char do_not_transpose = 'N';
    const float alpha = 1.f;
    const float beta = 0.f;

    const size_t tmp_size = (size_t)(krows * cols * (int)sizeof(float));

    float *tmp = (float*)kbyte_alloc(tmp_size );

    sgemm(&do_transpose, &do_not_transpose, &m, &n, &k, &alpha, Wct, &n, X, &n, &beta, tmp, &m);

    sgemm(&do_not_transpose, &do_not_transpose, &m, &k, &l, &alpha, tmp, &m, Wrt, &k, &beta, Y, &m);

    free(tmp);
}

/**
 * Function for double precision floating points.
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
HELPER_FUNCTION void compute_Y_2d_64f(const double* __restrict__ Wct,
                                      const double* __restrict__ X,
                                      const double* __restrict__ Wrt,
                                      const int rows,
                                      const int cols,
                                      const int krows,
                                      const int kcols,
                                      double* __restrict__ Y)
{

    const ptrdiff_t n = (ptrdiff_t)rows;
    const ptrdiff_t k = (ptrdiff_t)cols;
    const ptrdiff_t m = (ptrdiff_t)krows;
    const ptrdiff_t l = (ptrdiff_t)kcols;

    const char do_transpose = 'T';
    const char do_not_transpose = 'N';
    const double alpha = 1.;
    const double beta = 0.;

    const size_t tmp_size = (size_t)(krows * cols * (int)sizeof(double));

    double *tmp = (double*)kbyte_alloc(tmp_size );

    dgemm(&do_transpose, &do_not_transpose, &m, &n, &k, &alpha, Wct, &n, X, &n, &beta, tmp, &m);

    dgemm(&do_not_transpose, &do_not_transpose, &m, &k, &l, &alpha, tmp, &m, Wrt, &k, &beta, Y, &m);

    free(tmp);
}

/**
 * Function for single precision floating points.
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
CALLABLE_FUNCTION void bdpca_2d_32f(const float* __restrict__ _X,
                                    const int rows, const int cols,
                                    const int krows, const int kcols,
                                    float** __restrict__ _Y,
                                    float** __restrict__ _X_,
                                    float** __restrict__ _Wrt,
                                    float** __restrict__ _Wct)
{
    // Fool proofing.
    if(!_Y && !_X_ && !_Wrt && !_Wct)
        return;

    const int elem_size = (int)sizeof(float);
    const size_t X_size = (size_t)(rows * cols * elem_size);
    const size_t X_bar_size = (size_t)(cols * elem_size);

    const size_t nb_elems_srt = (size_t)(cols * cols * elem_size);
    const size_t nb_elems_sct = (size_t)(rows * rows * elem_size);

    const size_t Y_size = (size_t)(krows * kcols * elem_size);

    float* X = (float*)kbyte_alloc(X_size);
    float* X_ = (float*)kbyte_alloc(X_bar_size);

    float* Srt = (float*)kbyte_alloc(nb_elems_srt );
    float* Sct = (float*)kbyte_alloc(nb_elems_sct );

    float* Wrt = (float*)kbyte_alloc(nb_elems_srt);
    float* Wct = (float*)kbyte_alloc(nb_elems_sct);

    float* Y = (float*)kbyte_alloc(Y_size);

    memcpy(X, _X, X_size);

    // Step 1) centre the values of X around its mean.
    reduce_mean_32f(X, rows, cols, 0, X_);

    centre_32f(X, rows, cols, 0, X_);

    // Step 2) Compute the scatter matrices and eigenvectors matrices.
    compute_scatter_matrices_2d_32f(X, rows, cols, Srt, Sct);


    eig_f32(Srt, cols, Wrt);
    free(Srt);

    eig_f32(Sct, rows, Wct);
    free(Sct);

    // Step 3) Reduce the dimensionality of the eigenvectors matrices.
    col_range_32f(&Wrt, cols, cols, 0, kcols);
    col_range_32f(&Wct, rows, rows, 0, krows);


    sort_32f(Wrt, cols, kcols);
    sort_32f(Wct, rows, krows);


    // Step 4) Compute Y (Y = Wct' x X x Wrt)
    compute_Y_2d_32f(Wct, X, Wrt, rows, cols, krows, kcols, Y);


    if(_Y)
    {
        memcpy(*_Y, Y, Y_size);
    }

    if(_X_)
    {
        memcpy(*_X_, X_, X_bar_size);
    }

    if(_Wrt)
    {
        memcpy(*_Wrt, Wrt, nb_elems_srt);
    }

    if(_Wct)
    {
        memcpy(*_Wct, Wct, nb_elems_sct);
    }

    free(X);
    free(X_);
    free(Wrt);
    free(Wct);
    free(Y);
}

/**
 * Function for double precision floating points.
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
CALLABLE_FUNCTION void bdpca_2d_64f(const double* __restrict__ _X,
                                    const int rows, const int cols,
                                    const int krows, const int kcols,
                                    double** __restrict__ _Y,
                                    double** __restrict__ _X_,
                                    double** __restrict__ _Wrt,
                                    double** __restrict__ _Wct)
{
    // Fool proofing.
    if(!_Y && !_X_ && !_Wrt && !_Wct)
        return;

    const int elem_size = (int)sizeof(double);
    const size_t X_size = (size_t)(rows * cols * elem_size);
    const size_t X_bar_size = (size_t)(cols * elem_size);

    const size_t nb_elems_srt = (size_t)(cols * cols * elem_size);
    const size_t nb_elems_sct = (size_t)(rows * rows * elem_size);

    const size_t Y_size = (size_t)(krows * kcols * elem_size);

    double* X = (double*)kbyte_alloc(X_size);
    double* X_ = (double*)kbyte_alloc(X_bar_size);

    double* Srt = (double*)kbyte_alloc(nb_elems_srt );
    double* Sct = (double*)kbyte_alloc(nb_elems_sct );

    double* Wrt = (double*)kbyte_alloc(nb_elems_srt);
    double* Wct = (double*)kbyte_alloc(nb_elems_sct);

    double* Y = (double*)kbyte_alloc(Y_size);

    memcpy(X, _X, X_size);

    // Step 1) centre the values of X around its mean.
    reduce_mean_64f(X, rows, cols, 0, X_);

    centre_64f(X, rows, cols, 0, X_);

    // Step 2) Compute the scatter matrices and eigenvectors matrices.
    compute_scatter_matrices_2d_64f(X, rows, cols, Srt, Sct);

    eig_f64(Srt, cols, Wrt);
    free(Srt);

    eig_f64(Sct, rows, Wct);
    free(Sct);

    // Step 3) Reduce the dimensionality of the eigenvectors matrices.
    col_range_64f(&Wrt, cols, cols, 0, kcols);
    col_range_64f(&Wct, rows, rows, 0, krows);


    sort_64f(Wrt, cols, kcols);
    sort_64f(Wct, rows, krows);

    // Step 4) Compute Y (Y = Wct' x X x Wrt)
    compute_Y_2d_64f(Wct, X, Wrt, rows, cols, krows, kcols, Y);


    if(_Y)
    {
        memcpy(*_Y, Y, Y_size);
    }

    if(_X_)
    {
        memcpy(*_X_, X_, X_bar_size);
    }

    if(_Wrt)
    {
        memcpy(*_Wrt, Wrt, nb_elems_srt);
    }

    if(_Wct)
    {
        memcpy(*_Wct, Wct, nb_elems_sct);
    }

    free(X);
    free(X_);
    free(Wrt);
    free(Wct);
    free(Y);
}

/**
 * Function for single precision floating points.
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
HELPER_FUNCTION void compute_scatter_matrices_3d_32f(const float* __restrict__ X,
                                                     const int rows, const int cols,
                                                     const int frames,
                                                     float* __restrict__ Srt,
                                                     float* __restrict__ Sct)
{

    const int elem_size = (int)sizeof(float);
    const size_t srt_elem = (size_t)(cols * cols * elem_size);
    const size_t sct_elem = (size_t)(rows * rows * elem_size);

    float* tmp_srt = (float*)kbyte_alloc(srt_elem);
    float* tmp_sct = (float*)kbyte_alloc(sct_elem);

    const size_t stride = rows * cols;

    const float one = 1.f;
    const float inv_frames = 1.f / (float)frames;
    const ptrdiff_t n = (ptrdiff_t)(cols * cols);
    const ptrdiff_t m = (ptrdiff_t)(rows * rows);
    const ptrdiff_t inc_unit = 1;

    for(int i=0; i<frames; i++, X+=stride)
    {
        compute_scatter_matrices_2d_32f(X, rows, cols, tmp_srt, tmp_sct);

        saxpy(&n, &one, tmp_srt, &inc_unit, Srt, &inc_unit);
        saxpy(&m, &one, tmp_sct, &inc_unit, Sct, &inc_unit);
    }

    sscal(&n, &inv_frames, Srt, &inc_unit);
    sscal(&m, &inv_frames, Sct, &inc_unit);

    free(tmp_srt);
    free(tmp_sct);
}

/**
 * Function for double precision floating points.
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
HELPER_FUNCTION void compute_scatter_matrices_3d_64f(const double* __restrict__ X,
                                                     const int rows, const int cols,
                                                     const int frames,
                                                     double* __restrict__ Srt,
                                                     double* __restrict__ Sct)
{

    const int elem_size = (int)sizeof(double);
    const size_t srt_elem = (size_t)(cols * cols * elem_size);
    const size_t sct_elem = (size_t)(rows * rows * elem_size);

    double* tmp_srt = (double*)kbyte_alloc(srt_elem);
    double* tmp_sct = (double*)kbyte_alloc(sct_elem);

    const size_t stride = rows * cols;

    const double one = 1.;
    const double inv_frames = 1. / (double)frames;
    const ptrdiff_t n = (ptrdiff_t)(cols * cols);
    const ptrdiff_t m = (ptrdiff_t)(rows * rows);
    const ptrdiff_t inc_unit = 1;

    for(int i=0; i<frames; i++, X+=stride)
    {
        compute_scatter_matrices_2d_64f(X, rows, cols, tmp_srt, tmp_sct);

        daxpy(&n, &one, tmp_srt, &inc_unit, Srt, &inc_unit);
        daxpy(&m, &one, tmp_sct, &inc_unit, Sct, &inc_unit);
    }

    dscal(&n, &inv_frames, Srt, &inc_unit);
    dscal(&m, &inv_frames, Sct, &inc_unit);

    free(tmp_srt);
    free(tmp_sct);
}

/**
 * Function for single precision floating points.
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
HELPER_FUNCTION void compute_Y_3d_32f(const float* __restrict__ Wct,
                                      const float* __restrict__ X,
                                      const float* __restrict__ Wrt,
                                      const int rows,
                                      const int cols,
                                      const int frames,
                                      const int krows,
                                      const int kcols,
                                      float* __restrict__ Y)
{
    const size_t stride_X = (size_t)(rows * cols);
    const size_t stride_Y = (size_t)(krows * kcols);

    for(int i=0; i<frames;i++, X+=stride_X, Y+=stride_Y)
        compute_Y_2d_32f(Wct, X, Wrt, rows, cols, krows, kcols, Y);
}

/**
 * Function for double precision floating points.
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
HELPER_FUNCTION void compute_Y_3d_64f(const double* __restrict__ Wct,
                                      const double* __restrict__ X,
                                      const double* __restrict__ Wrt,
                                      const int rows,
                                      const int cols,
                                      const int frames,
                                      const int krows,
                                      const int kcols,
                                      double* __restrict__ Y)
{
    const size_t stride_X = (size_t)(rows * cols);
    const size_t stride_Y = (size_t)(krows * kcols);

    for(int i=0; i<frames;i++, X+=stride_X, Y+=stride_Y)
        compute_Y_2d_64f(Wct, X, Wrt, rows, cols, krows, kcols, Y);
}

/**
 * Function for single precision floating points.
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
CALLABLE_FUNCTION void bdpca_3d_32f(const float* __restrict__ _X,
                                    const int rows, const int cols, const int frames,
                                    const int krows, const int kcols,
                                    float** __restrict__ _Y,
                                    float** __restrict__ _X_,
                                    float** __restrict__ _Wrt,
                                    float** __restrict__ _Wct)
{

    // Fool proofing.
    if(!_Y && !_X_ && !_Wrt && !_Wct)
        return;

    const int elem_size = (int)sizeof(float);
    const size_t X_size = (size_t)(rows * cols * frames * elem_size);
    const size_t X_bar_size = (size_t)(rows * cols * elem_size);

    const size_t nb_elems_srt = (size_t)(cols * cols * elem_size);
    const size_t nb_elems_sct = (size_t)(rows * rows * elem_size);

    const size_t Y_size = (size_t)(krows * kcols * frames * elem_size);

    float* X = (float*)kbyte_alloc(X_size);
    float* X_ = (float*)kbyte_alloc(X_bar_size);

    float* Srt = (float*)kbyte_alloc(nb_elems_srt );
    float* Sct = (float*)kbyte_alloc(nb_elems_sct );

    float* Wrt = (float*)kbyte_alloc(nb_elems_srt);
    float* Wct = (float*)kbyte_alloc(nb_elems_sct);

    float* Y = (float*)kbyte_alloc(Y_size);

    memcpy(X, _X, X_size);

    // Step 1) centre the values of X around its mean.
    reduce_mean_32f(X, rows * cols, frames, 1, X_);

    centre_32f(X, rows * cols, frames, 1, X_);

    // Step 2) Compute the scatter matrices and eigenvectors matrices.
    compute_scatter_matrices_3d_32f(X, rows, cols, frames, Srt, Sct);


    eig_f32(Srt, cols, Wrt);
    free(Srt);

    eig_f32(Sct, rows, Wct);
    free(Sct);

    // Step 3) Reduce the dimensionality of the eigenvectors matrices.
    col_range_32f(&Wrt, cols, cols, 0, kcols);
    col_range_32f(&Wct, rows, rows, 0, krows);


    sort_32f(Wrt, cols, kcols);
    sort_32f(Wct, rows, krows);


    // Step 4) Compute Y (Y = Wct' x X x Wrt)
    compute_Y_3d_32f(Wct, X, Wrt, rows, cols, frames, krows, kcols, Y);


    if(_Y)
    {
        memcpy(*_Y, Y, Y_size);
    }

    if(_X_)
    {
        memcpy(*_X_, X_, X_bar_size);
    }

    if(_Wrt)
    {
        memcpy(*_Wrt, Wrt, nb_elems_srt);
    }

    if(_Wct)
    {
        memcpy(*_Wct, Wct, nb_elems_sct);
    }

    free(X);
    free(X_);
    free(Wrt);
    free(Wct);
    free(Y);
}

/**
 * Function for double precision floating points.
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
CALLABLE_FUNCTION void bdpca_3d_64f(const double* __restrict__ _X,
                                    const int rows, const int cols,
                                    const int frames,
                                    const int krows, const int kcols,
                                    double** __restrict__ _Y,
                                    double** __restrict__ _X_,
                                    double** __restrict__ _Wrt,
                                    double** __restrict__ _Wct)
{

    // Fool proofing.
    if(!_Y && !_X_ && !_Wrt && !_Wct)
        return;

    const int elem_size = (int)sizeof(double);
    const size_t X_size = (size_t)(rows * cols * frames * elem_size);
    const size_t X_bar_size = (size_t)(rows * cols * elem_size);

    const size_t nb_elems_srt = (size_t)(cols * cols * elem_size);
    const size_t nb_elems_sct = (size_t)(rows * rows * elem_size);

    const size_t Y_size = (size_t)(krows * kcols * frames * elem_size);

    double* X = (double*)kbyte_alloc(X_size);
    double* X_ = (double*)kbyte_alloc(X_bar_size);

    double* Srt = (double*)kbyte_alloc(nb_elems_srt );
    double* Sct = (double*)kbyte_alloc(nb_elems_sct );

    double* Wrt = (double*)kbyte_alloc(nb_elems_srt);
    double* Wct = (double*)kbyte_alloc(nb_elems_sct);

    double* Y = (double*)kbyte_alloc(Y_size);

    memcpy(X, _X, X_size);

    // Step 1) centre the values of X around its mean.
    reduce_mean_64f(X, rows * cols, frames, 1, X_);

    centre_64f(X, rows * cols, frames, 1, X_);

    // Step 2) Compute the scatter matrices and eigenvectors matrices.
    compute_scatter_matrices_3d_64f(X, rows, cols, frames, Srt, Sct);


    eig_f64(Srt, cols, Wrt);
    free(Srt);

    eig_f64(Sct, rows, Wct);
    free(Sct);

    // Step 3) Reduce the dimensionality of the eigenvectors matrices.
    col_range_64f(&Wrt, cols, cols, 0, kcols);
    col_range_64f(&Wct, rows, rows, 0, krows);


    sort_64f(Wrt, cols, kcols);
    sort_64f(Wct, rows, krows);


    // Step 4) Compute Y (Y = Wct' x X x Wrt)
    compute_Y_3d_64f(Wct, X, Wrt, rows, cols, frames, krows, kcols, Y);


    if(_Y)
    {
        memcpy(*_Y, Y, Y_size);
    }

    if(_X_)
    {
        memcpy(*_X_, X_, X_bar_size);
    }

    if(_Wrt)
    {
        memcpy(*_Wrt, Wrt, nb_elems_srt);
    }

    if(_Wct)
    {
        memcpy(*_Wct, Wct, nb_elems_sct);
    }

    free(X);
    free(X_);
    free(Wrt);
    free(Wct);
    free(Y);
}
