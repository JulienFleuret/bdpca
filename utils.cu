#include "utils.h"
#include "utils_vt.h"

#include <iostream>

namespace cuda
{

namespace utils
{

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// GET MIN MAX ////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
template<> __host__  void getMinMax<float>(safe_cublasHandle_t& handle, const cudaBuffer<float>& src, float& min, float& max)
{
    int argmin(0), argmax(0);

    // Step 1) Get the argmin and argmax.

    CHECK_EXECUTION(cublasIsamin(handle, src.length(), src, 1, &argmin));
    CHECK_EXECUTION(cublasIsamax(handle, src.length(), src, 1, &argmax));

    // Step 2) Shift the argmin and argmax by one to fit the proper range.

    argmin--;
    argmax--;

    // Step 3) Copy the minimum and maximum value in their respective variables.

    CHECK_EXECUTION(cudaMemcpy(&min, src.getPtr() + argmin, static_cast<size_t>(sizeof(float)), cudaMemcpyDeviceToHost));
    CHECK_EXECUTION(cudaMemcpy(&max, src.getPtr() + argmax, static_cast<size_t>(sizeof(float)), cudaMemcpyDeviceToHost));

}

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
template<> __host__ void getMinMax<double>(safe_cublasHandle_t& handle, const cudaBuffer<double>& src, double& min, double& max)
{
    int argmin(0), argmax(0);

    // Step 1) Get the argmin and argmax.

    CHECK_EXECUTION(cublasIdamin(handle, src.length(), src, 1, &argmin));
    CHECK_EXECUTION(cublasIdamax(handle, src.length(), src, 1, &argmax));

    // Step 2) Shift the argmin and argmax by one to fit the proper range.

    argmin--;
    argmax--;

    // Step 3) Copy the minimum and maximum value in their respective variables.

    CHECK_EXECUTION(cudaMemcpy(&min, src.getPtr() + argmin, static_cast<size_t>(sizeof(double)), cudaMemcpyDeviceToHost));
    CHECK_EXECUTION(cudaMemcpy(&max, src.getPtr() + argmax, static_cast<size_t>(sizeof(double)), cudaMemcpyDeviceToHost));
}

/**
 * @brief getMinMax : return the min and max value of a cudaBuffer object.
 * @param handle : cublas handle.
 * @param src : memory segment to process
 * @return a tuple containing the min and max values.
 * @note this function only work single and double precision
 * floating point type, and call cublasIsmax, cublasIsmin, cublasIdmin, cublasIdmax.
 */
template<class T>
__host__ std::tuple<T, T> getMinMax(safe_cublasHandle_t& handle, const cudaBuffer<float>& src)
{
    T min(static_cast<T>(0)), max(static_cast<T>(0));

    getMinMax(handle, src, min, max);

    return std::make_tuple(min, max);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// RESCALE ////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Custom CUDA Kernel
 * @brief rescale_kernel_vec : Rescale the value considering I * alpah + beta, where
 * alpha = min(I) / (max(I) - min(I)) and beta = 1 / (max(I) - min(I)).
 * @param ptr : address of the memory segment to compute.
 * @param rows : number of rows of the matrix.
 * @param cols : number of columns of the matrix.
 * @param alpha : min(I) / (max(I) - min(I)).
 * @param beta : 1 / (max(I) - min(I)).
 */
template<class T>
__global__ void rescale_kernel(T* ptr, const int rows, const int cols, const T alpha, const T beta)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(y>=rows || x>=cols)
        return;

    const int offset = x * rows + y;

    ptr += offset;

    *ptr = fma_op(*ptr, alpha, beta);
}

/**
 * Custom CUDA Kernel, using vectorized type
 * @brief rescale_kernel_vec : Rescale the value considering I * alpah + beta, where
 * alpha = min(I) / (max(I) - min(I)) and beta = 1 / (max(I) - min(I)).
 * @param ptr : address of the memory segment to compute.
 * @param rows : number of rows of the matrix.
 * @param cols : number of columns of the matrix.
 * @param alpha : min(I) / (max(I) - min(I)).
 * @param beta : 1 / (max(I) - min(I)).
 */
template<class T, class VT>
__global__ void rescale_kernel_vec(T* ptr, const int rows, const int cols, const T alpha, const T beta)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y * vector_type_traits<VT>::stride_width;

    if(y>=rows || x>=cols)
        return;

    const int offset = x * rows + y;

    ptr += offset;

    v_store(ptr, fma_op(v_load<VT>(ptr), make_type<VT>(alpha), make_type<VT>(beta)));
}

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
__host__ void rescale(safe_cudaStream_t& stream, safe_cublasHandle_t& handle, cudaBuffer<T>& src, const int& rows, const int& cols)
{

    T mn(static_cast<T>(0)), mx(static_cast<T>(0));

    getMinMax(handle, src, mn, mx);


    mx -= mn;

    mn /= mx;
    mx = static_cast<T>(1) / mx;

    // TODO investigate where the issue comes from with 3D matrices.

//    int best_div = !(rows%4) ? 4 : (rows%3) ? 3 : !(rows%2) ? 2 : 1;
    int best_div = 1;
    bool use_vectorized_kernel( best_div != 1);

    dim3 grid(32,8);
    dim3 block(divUp(cols, grid.x), divUp(rows / best_div, grid.y));

    if(use_vectorized_kernel)
    {
        switch (best_div) {
        case 4:
        {
            CHECK_EXECUTION(cudaFuncSetCacheConfig(rescale_kernel_vec<T, typename create_type<T, 4>::type>, cudaFuncCachePreferL1));
            rescale_kernel_vec<T, typename create_type<T, 4>::type><<<grid, block, 0,stream.stream>>>(src, rows, cols, mn, mx);
        }
            break;

        case 3:
        {
            CHECK_EXECUTION(cudaFuncSetCacheConfig(rescale_kernel_vec<T, typename create_type<T, 3>::type>, cudaFuncCachePreferL1));
            rescale_kernel_vec<T, typename create_type<T, 3>::type><<<grid, block, 0,stream.stream>>>(src, rows, cols, mn, mx);
        }
            break;

        case 2:
        {
            CHECK_EXECUTION(cudaFuncSetCacheConfig(rescale_kernel_vec<T, typename create_type<T, 2>::type>, cudaFuncCachePreferL1));
            rescale_kernel_vec<T, typename create_type<T, 2>::type><<<grid, block, 0,stream.stream>>>(src, rows, cols, mn, mx);
        }
            break;
        }
    }
    else
    {
            CHECK_EXECUTION(cudaFuncSetCacheConfig(rescale_kernel<T>, cudaFuncCachePreferL1));
            rescale_kernel<T><<<grid, block, 0,stream.stream>>>(src, rows, cols, mn, mx);
    }

    CHECK_EXECUTION(cudaGetLastError());

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// REDUCE MEAN ////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{
/**
 * Convinience structure to manage the computation of the mean of every rows.
 *@brief reduce_mean_axis_1_op
 */
template<class T>
struct reduce_mean_axis_1_op;

/**
 * Convinience structure to manage the computation of the mean of every rows.
 * @brief The reduce_mean_axis_1_op class, specialization for the single precision floating point type.
 * @note unlike the name and maybe the reference axis
 */
template<>
struct reduce_mean_axis_1_op<float>
{

    static void op(cublasHandle_t& handle, const int& cols, const int& n, const float& alpha, const float* __restrict__ x, const int& inc_x, float* __restrict__ y, const int& inc_y)
    {
        // Doing the sum of every columns.
        // This approach might slightly slower
        // than calling a custom kernels, but it
        // avoid to have synchronization.
        for(int c=0; c<cols; c++, x+=n)
            CHECK_EXECUTION(cublasSaxpy_v2(handle, n, &alpha, x, inc_x, y, inc_y));
    }
};

/**
 * Convinience structure to manage the computation of the mean of every rows.
 * @brief The reduce_mean_axis_1_op class, specialization for the double precision floating point type.
 */
template<>
struct reduce_mean_axis_1_op<double>
{
    static void op(cublasHandle_t& handle, const int& cols, const int& n, const double& alpha, const double* __restrict__ x, const int& inc_x, double* __restrict__ y, const int& inc_y)
    {
        // Doing the sum of every columns.
        // This approach might slightly slower
        // than calling a custom kernels, but it
        // avoid to have synchronization.
        for(int c=0; c<cols; c++, x+=n)
            CHECK_EXECUTION(cublasDaxpy_v2(handle, n, &alpha, x, inc_x, y, inc_y));
    }
};




/**
 * @brief reduce_mean_axis_0_kernel : compute the mean of every column.
 * @param src : address of origin of the memory segment to process.
 * @param rows : number of rows of the matrix to process.
 * @param cols : number of columns of the matrix to process.
 * @param dst : address of origin of the memory to store the processing.
 */
template<class T>
__global__ void reduce_mean_axis_0_kernel(const T* __restrict__ src, const int rows, const int cols, T* __restrict__ dst)
{

    using vector_type=typename create_type<T,4>::type;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    //Check if x and y are in the range of the rows and column
    if(x>=cols || y>=rows)
        return;

    // Properly offset the pointers.
    src += x * rows;

    dst += x;

    // Get the number of rows that can be process using either a float4 or double4 vector type.
    const int simd_rows = !(rows%4) ? rows : (rows-4 + 3) & -4;

    int r=0;

    // Set the vectorized sum to 0.
    vector_type v_sum = v_setzeros<vector_type>();

    // Compute the sum using vector type
    for(;r<simd_rows; r+=4, src+=4)
        v_sum += v_load<vector_type>(src);

    // Compute a horizontal reduce sum.
    T sum = v_reduce_sum(v_sum);

    // Compute the sum of the remaining elements.
    for(; r<rows; r++, src++)
        sum += *src;

    // Normalize by the number of rows.
    sum /= static_cast<T>(rows);

    // Assign to the destination.
    *dst = sum;
}

/** Convinience function.
 * @brief scal : manage the call of cublasSscal or cublasDscal depending on the template type.
 * @param handle : cublas resourse handle.
 * @param n : number of elements to scale
 * @param den : coeficient to apply on the elements of dst.
 * @param dst : data to process
 * @param inc : increment to use.
 */
template<class T>
__host__ void scal(safe_cublasHandle_t& handle, const int& n, const T& den, cudaBuffer<T>& dst, const int &inc);

/** Convinience function, specialize for single precision floating points type.
 * @brief scal : manage the call of cublasSscal or cublasDscal depending on the template type.
 * @param handle : cublas resourse handle.
 * @param n : number of elements to scale
 * @param den : coeficient to apply on the elements of dst.
 * @param dst : data to process
 * @param inc : increment to use.
 */
template<> __host__ void scal<float>(safe_cublasHandle_t& handle, const int& n, const float& den, cudaBuffer<float>& dst, const int &inc)
{
    CHECK_EXECUTION(cublasSscal_v2(handle, n, &den, dst, inc));
}

/** Convinience function, specialize for double precision floating points type.
 * @brief scal : manage the call of cublasSscal or cublasDscal depending on the template type.
 * @param handle : cublas resourse handle.
 * @param n : number of elements to scale
 * @param den : coeficient to apply on the elements of dst.
 * @param dst : data to process
 * @param inc : increment to use.
 */
template<> __host__ void scal<double>(safe_cublasHandle_t& handle, const int& n, const double& den, cudaBuffer<double>& dst, const int &inc)
{
    CHECK_EXECUTION(cublasDscal_v2(handle, n, &den, dst, inc));
}

} // anonymous

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
                 )
{
        const int n = rows;
        const int inc = 1;
        const T alpha = static_cast<T>(1);
        const T den = static_cast<T>(1) / static_cast<T>(axis ? cols : rows);

        if(axis)
        {
            reduce_mean_axis_1_op<T>::op(handle.handle, cols, n, alpha, src.getPtr(), inc, dst.getPtr(), inc);
        }
        else
        {
            dim3 block (256);
            dim3 grid (divUp (cols, block.x));

            CHECK_EXECUTION(cudaFuncSetCacheConfig(reduce_mean_axis_0_kernel<T>, cudaFuncCachePreferL1));
            reduce_mean_axis_0_kernel<T><<<grid, block, 0,stream>>>(src.getPtr(), rows, cols, dst.getPtr());
            CHECK_EXECUTION(cudaGetLastError());

            if(!stream)
                CHECK_EXECUTION(cudaDeviceSynchronize());
        }

        scal(handle, n, den, dst, inc);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// CENTRE /////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{

/**
 * Custom CUDA Kernel.
 * @brief centre_axis_0_kernel : cuda kernel computing the subtraction of x by mu.
 * @param X : address of the device segment to process.
 * @param rows : number of rows of the matrix.
 * @param cols : number of columns of the matrix.
 * @param mu : address of the memory segment storing the mean.
 */
template<class T>
__global__ void centre_axis_0_kernel(T* __restrict__ X, const int rows, const int cols, const T* __restrict__ mu)
{
    using vector_type=typename create_type<T,4>::type;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x>=cols || y>=rows)
        return;

    X += x * rows + y;

    mu += x;

    *X -= *mu;
}

/**
 * Custom CUDA Kernel, using vectorized type.
 * @brief centre_axis_0_kernel : cuda kernel computing the subtraction of x by mu.
 * @param X : address of the device segment to process.
 * @param rows : number of rows of the matrix.
 * @param cols : number of columns of the matrix.
 * @param mu : address of the memory segment storing the mean.
 */
template<class T, class VT>
__global__ void centre_axis_0_kernel_vec(T* __restrict__ X, const int rows, const int cols, const T* __restrict__ mu)
{
    using vector_type=typename create_type<T,4>::type;

    int x = threadIdx.x + blockIdx.x * blockDim.x ;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * vector_type_traits<VT>::stride_width;

    if(x>=cols || y>=rows)
        return;

    X += x * rows + y;

    mu += x;

    *reinterpret_cast<VT*>(X) -= make_type<VT>(*mu);
}

/**
* Custom CUDA Kernel.
 * @brief centre_axis_0_kernel : cuda kernel computing the subtraction of x by mu.
 * @param X : address of the device segment to process.
 * @param rows : number of rows of the matrix.
 * @param cols : number of columns of the matrix.
 * @param mu : address of the memory segment storing the mean.
 */
template<class T>
__global__ void centre_axis_1_kernel(T* __restrict__ X, const int rows, const int cols, const T* __restrict__ mu)
{
    using vector_type=typename create_type<T,4>::type;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;


    if(x>=cols || y>=rows)
        return;

    X += x * rows + y;

    mu += y;

    *X -= *mu;
}


/**
 * Custom CUDA Kernel, using vectorized type.
 * @brief centre_axis_0_kernel
 * @param X : address of the device segment to process.
 * @param rows : number of rows of the matrix.
 * @param cols : number of columns of the matrix.
 * @param mu : address of the memory segment storing the mean.
 */
template<class T, class VT>
__global__ void centre_axis_1_kernel_vec(T* __restrict__ X, const int rows, const int cols, const T* __restrict__ mu)
{
    using vector_type=typename create_type<T,4>::type;

    int x = threadIdx.x + blockIdx.x * blockDim.x ;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * vector_type_traits<VT>::stride_width;

    if(x>=cols || y>=rows)
        return;

    X += x * rows + y;

    mu += y;

    *reinterpret_cast<VT*>(X) -= *reinterpret_cast<const VT*>(mu);
}

} // anonymous

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
void centre(safe_cudaStream_t& stream, cudaBuffer<T>& src, const int& rows, const int& cols, const int& axis, const cudaBuffer<T>& dst)
{

    //the restrict keywords tell the compiler that X and mu are not the same pointer nor part of the same allocation.
    typedef void (*global_function_ptr_t)(T* __restrict__ , const int, const int, const T* __restrict__);

    // This management stategy is not allways the fastest compare with 8 if else or a swith.
    // However it offers the benefits to be easy to read and to use.
    static const global_function_ptr_t funcs[2][4] = { {
                                                        centre_axis_0_kernel<T>,
                                                        centre_axis_0_kernel_vec<T, typename create_type<T, 2>::type>,
                                                        centre_axis_0_kernel_vec<T, typename create_type<T, 3>::type>,
                                                        centre_axis_0_kernel_vec<T, typename create_type<T, 4>::type>,
                                                       },
                                                       {
                                                        centre_axis_1_kernel<T>,
                                                        centre_axis_1_kernel_vec<T, typename create_type<T, 2>::type>,
                                                        centre_axis_1_kernel_vec<T, typename create_type<T, 3>::type>,
                                                        centre_axis_1_kernel_vec<T, typename create_type<T, 4>::type>,
                                                       } };

    dim3 block (32, 8);

    int best_stride = axis ? !(rows%4) ? 4 : !(rows%3) ? 3 : !(rows%2) ? 2 : 1 : !(cols%4) ? 4 : !(cols%3) ? 3 : !(cols%2) ? 2 : 1;

    dim3 grid(utils::divUp(axis ? cols : cols / best_stride, block.x),
              utils::divUp(axis ? rows / best_stride : rows, block.y));

    global_function_ptr_t fun = funcs[axis][best_stride-1];

    CHECK_EXECUTION(cudaFuncSetCacheConfig(fun, cudaFuncCachePreferL1));
    fun<<<grid, block, 0,stream>>>(src, rows, cols, dst);
    CHECK_EXECUTION(cudaGetLastError());

    if(!stream)
        CHECK_EXECUTION(cudaDeviceSynchronize());
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// COMPUTE SCATTER MATRICES 2D /////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{
/**
 * CUSTOM CUDA KERNEL.
 * @brief fill_symetric_kernel : fill the lower part of the scatter matrix.
 * @param X : address of the memory segment to process.
 * @param N : Square root of the total size allocated to X.
 * Because X represents a square matrix.
 */
template<class T>
__global__ void fill_symetric_kernel(T* X, const int N)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x>=N || y>=N || y>x)
        return;

    // Conceptually: X(y,x) = X(x,y) under the condition that y < x.
    int offset_x_y = x * N + y;
    int offset_y_x = y * N + x;

    *(X + offset_y_x) = *(X + offset_x_y);
}

/**
 * @brief fill_symetric : fill the lower part of the scatter matrix.
 * @param X : address of the memory segment to process.
 * @param N : Square root of the total size allocated to X.
 * Because X represents a square matrix.
 */
template<class T>
__host__ void fill_symetric(safe_cudaStream_t& stream, cudaBuffer<T>& X, const int& N)
{
    // Prepare the block and the grid.
    dim3 block (32, 8);
    dim3 grid (utils::divUp (N, block.x), utils::divUp (N, block.y));

    // Call the kernel.
    CHECK_EXECUTION(cudaFuncSetCacheConfig(fill_symetric_kernel<T>, cudaFuncCachePreferL1));
    fill_symetric_kernel<T><<<grid, block, 0,stream.stream>>>(X, N);
    CHECK_EXECUTION(cudaGetLastError());

    if(!stream)
        CHECK_EXECUTION(cudaDeviceSynchronize());
}

/**
 * @brief syrk : cublas functions to compute either Y = alpha x matmul(X',X) + beta x Y or Y = alpha x matmul(X,X') + beta x Y,
 *  where "x" is the element wise multiplication operator.
 * @param handle : cublas resources handle.
 * @param uplo : should the results be wrote in the upper or lower part of the matrix.
 * @param trans : should the first input be considered transpose.
 * @param n : trans == CUBLAS_OP_N (do not transpose) ? rows : cols
 * @param k : trans == CUBLAS_OP_N (do not transpose) ? cols : rows
 * @param alpha : weight coefficient of the result.
 * @param A : device memory address of the data to process.
 * @param lda : leading dimension of the input data. The function expect the data to be stored columnwisely.
 * @param beta : weight coefficient to apply on the existing output before adding the result of the dot product.
 * @param C : device memory to store the result of the operations
 * @param ldc : leading dimension of the output data. The function expect the data to be stored columnwisely.
 */
template<class T>
__host__ void syrk(safe_cublasHandle_t& handle, cublasFillMode_t& uplo, cublasOperation_t& trans, const int& n, const int& k, const T& alpha, const cudaBuffer<T>& A, int lda, const T& beta, cudaBuffer<T>& C, const int& ldc);

/**
 * Specialization for single precision data type.
 * @brief syrk : cublas functions to compute either Y = alpha x matmul(X',X) + beta x Y or Y = alpha x matmul(X,X') + beta x Y,
 *  where "x" is the element wise multiplication operator.
 * @param handle : cublas resources handle.
 * @param uplo : should the results be wrote in the upper or lower part of the matrix.
 * @param trans : should the first input be considered transpose.
 * @param n : trans == CUBLAS_OP_N (do not transpose) ? rows : cols
 * @param k : trans == CUBLAS_OP_N (do not transpose) ? cols : rows
 * @param alpha : weight coefficient of the result.
 * @param A : device memory address of the data to process.
 * @param lda : leading dimension of the input data. The function expect the data to be stored columnwisely.
 * @param beta : weight coefficient to apply on the existing output before adding the result of the dot product.
 * @param C : device memory to store the result of the operations
 * @param ldc : leading dimension of the output data. The function expect the data to be stored columnwisely.
 */
template<>
__host__ __forceinline__ void syrk<float>(safe_cublasHandle_t& handle, cublasFillMode_t& uplo, cublasOperation_t& trans, const int& n, const int& k, const float& alpha, const cudaBuffer<float>& A, int lda, const float& beta, cudaBuffer<float>& C, const int& ldc)
{
CHECK_EXECUTION(cublasSsyrk_v2(handle, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc));
}

/**
 * Specialization for double precision data type.
 * @brief syrk : cublas functions to compute either Y = alpha x matmul(X',X) + beta x Y or Y = alpha x matmul(X,X') + beta x Y,
 *  where "x" is the element wise multiplication operator.
 * @param handle : cublas resources handle.
 * @param uplo : should the results be wrote in the upper or lower part of the matrix.
 * @param trans : should the first input be considered transpose.
 * @param n : trans == CUBLAS_OP_N (do not transpose) ? rows : cols
 * @param k : trans == CUBLAS_OP_N (do not transpose) ? cols : rows
 * @param alpha : weight coefficient of the result.
 * @param A : device memory address of the data to process.
 * @param lda : leading dimension of the input data. The function expect the data to be stored columnwisely.
 * @param beta : weight coefficient to apply on the existing output before adding the result of the dot product.
 * @param C : device memory to store the result of the operations
 * @param ldc : leading dimension of the output data. The function expect the data to be stored columnwisely.
 */
template<>
__host__ __forceinline__  void syrk<double>(safe_cublasHandle_t& handle, cublasFillMode_t& uplo, cublasOperation_t& trans, const int& n, const int& k, const double& alpha, const cudaBuffer<double>& A, int lda, const double& beta, cudaBuffer<double>& C, const int& ldc)
{
CHECK_EXECUTION(cublasDsyrk_v2(handle, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc));
}


} // anonymous

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
__host__ void compute_scatter_matrices_2d(safe_cublasHandle_t& handle, safe_cudaStream_t& stream, const cudaBuffer<T>& X, const int& rows, const int& cols, cudaBuffer<T>& Srt, cudaBuffer<T>& Sct)
{
    int n = rows;
    int k = cols;
    const T zero = static_cast<T>(0);
    const T inv_rows = static_cast<T>(1)/static_cast<T>(rows);
    const T inv_cols = static_cast<T>(1)/static_cast<T>(cols);

    Srt.allocate(cols, cols);
    Sct.allocate(rows, rows);

    cublasFillMode_t storage_mode = CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t do_transpose = CUBLAS_OP_T;
    cublasOperation_t do_not_transpose = CUBLAS_OP_N;

    syrk(handle, storage_mode, do_transpose, k, n, inv_rows, X, n, zero, Srt, k);
    fill_symetric(stream, Srt, cols);

    syrk(handle, storage_mode, do_not_transpose, n, k, inv_rows, X, n, zero, Sct, n);
    fill_symetric(stream, Sct, rows);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// EIG ///////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{

/**
 * @brief The compute_right_eigenvector_helper class :
 * Convinience class in order to manage both the steps
 * and the differents types and the different functions
 * associate with each type.
 */
template<class T>
struct compute_right_eigenvector_helper
{
public:

    using buffer_type = cudaBuffer<T>;
    using buffer_type_ref = cudaBuffer<T>&;

    /**
     * Parametric Constructor.
     * @brief compute_right_eigenvector_helper
     * @param _handle : cuDNN resources handle.
     */
    inline compute_right_eigenvector_helper(safe_cusolverDnHandle_t& _handle):
        handle(_handle),
        jobz(CUSOLVER_EIG_MODE_VECTOR),
        uplo(CUBLAS_FILL_MODE_UPPER),
        lwork(0),
        info(1)
    {}


    ~compute_right_eigenvector_helper() = default;


    /**
     * @brief allocate_buffer_size : get the size required by the appropriate SYEVD function and allocate the buffer associate.
     * @param n : number of rows (or columns) of the input square matrix.
     * @param V : eigenValue device memory segment.
     * @param W : device memory segment which store the data to process.
     */
    void allocate_buffer_size(const int& n, buffer_type_ref V, buffer_type_ref W);

    /**
     * @brief compute : compute the eigen values and eigen vectors
     * @param n : number of rows (or columns) of the input square matrix.
     * @param V : eigenValue device memory segment.
     * @param W : device memory segment which store the data to process.
     * These data will be updated with the eigenvectors.
     */
    void compute(const int& n, buffer_type_ref V, buffer_type_ref W);

private:

    // cuSolver handle
    safe_cusolverDnHandle_t& handle;

    // What should be computed? -> Eigenvalues and Eigenvectors
    cusolverEigMode_t jobz;
    // Fill the upper triangular part of the matrix
    cublasFillMode_t uplo;

    // How much quantity of memory is needed for the computations.
    int lwork;
    // Did the computation happens correctuly (memory on device mandatory)
    cudaBuffer<int> info;
    // Temporaty device memory buffer.
    buffer_type buffer;

};

template<>
__host__ void compute_right_eigenvector_helper<float>::allocate_buffer_size(const int &n, buffer_type_ref V, buffer_type_ref W)
{
    // "lwork" variable will contains the amount of memory need SYEVD to compute both the eigenvalue and the eigenvectors.
    CHECK_EXECUTION(cusolverDnSsyevd_bufferSize(this->handle, this->jobz, this->uplo, n, V, n, W, &this->lwork));

    // Allocate enough memory for the operations to happen safely.
    this->buffer.allocate(this->lwork);
}

template<>
__host__ void compute_right_eigenvector_helper<double>::allocate_buffer_size(const int &n, buffer_type_ref V, buffer_type_ref W)
{
    // "lwork" variable will contains the amount of memory need SYEVD to compute both the eigenvalue and the eigenvectors.
    CHECK_EXECUTION(cusolverDnDsyevd_bufferSize(this->handle, this->jobz, this->uplo, n, V, n, W, &this->lwork));

    // Allocate enough memory for the operations to happen safely.
    this->buffer.allocate(this->lwork);
}

template<>
__host__ void compute_right_eigenvector_helper<float>::compute(const int &n, buffer_type_ref V, buffer_type_ref W)
{
    // Compute the eigenvalue and eigenvector.
    CHECK_EXECUTION(cusolverDnSsyevd(this->handle, this->jobz, this->uplo, n, W, n, V, this->buffer.getPtr(), this->lwork, this->info));

    // Deallocate the info memory used on the device.
    this->info.deallocate();
    // Deallocate the memory needed by SYEVD.
    this->buffer.deallocate();
}

template<>
__host__ void compute_right_eigenvector_helper<double>::compute(const int &n, buffer_type_ref V, buffer_type_ref W)
{
    // Compute the eigenvalue and eigenvector.
    CHECK_EXECUTION(cusolverDnDsyevd(this->handle, this->jobz, this->uplo, n, W, n, V, this->buffer.getPtr(), this->lwork, this->info));

    // Deallocate the info memory used on the device.
    this->info.deallocate();
    // Deallocate the memory needed by SYEVD.
    this->buffer.deallocate();
}

} // anonymous.

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
__host__ void eig(safe_cusolverDnHandle_t& handle, safe_cudaStream_t& stream, const int& rows_cols, cudaBuffer<T>& eigenvectors)
{
    // Because the computation of the eigenvector, must be done with
    // the computation of the eigenvalues.
    cudaBuffer<T> eigenvalues(rows_cols * rows_cols);

    compute_right_eigenvector_helper<T> obj(handle);

    // Step 1) Check what quantity of memory is needed for SYSEVD to execute safely.
    obj.allocate_buffer_size(rows_cols, eigenvalues, eigenvectors);
    // Step 2) Compute the eigenvalues and eigenvectors.
    obj.compute(rows_cols, eigenvalues, eigenvectors);

    if(!stream)
        CHECK_EXECUTION(cudaDeviceSynchronize());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// COL RANGE ///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
__host__ void col_range(safe_cudaStream_t& stream, cudaBuffer<T>& buffer, const int& rows, const int& cols, const int& start, const int& end)
{
    cudaBuffer<T> tmp;

    // Step 1) Properly allocate the memory.
    T* src_ptr = buffer.getPtr();

    size_t nb_elems = static_cast<size_t>(rows) * static_cast<size_t>(end - start);

    tmp.allocate(nb_elems);

    if(start)
        src_ptr += start * rows;

    // Step 2) Make a deep-copy of the memory.
    tmp.copyFrom(src_ptr, nb_elems, cudaMemcpyDeviceToDevice, &stream);

    buffer = std::move(tmp);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// SORT ///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{

// The quicksort algorithm is a modified version of:

// Seriously?
template<class T>
__device__ __forceinline__ void swap(T& a, T& b)
{
    T c = a;
    a = b;
    b = c;
}

// Partition the array using the last element as the pivot
template<class T>
__device__ int partition(T* __restrict__ arr, const int low, const int high)
{
    // Choosing the pivot
    T pivot = arr[high];

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
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}

template<class T>
__device__ void quickSort(T* __restrict__ arr, const int start, const int end)
{
    if (start < end)
    {
        // "pivot" is partitioning index, arr[p]
        // is now at right place
        int pivot = partition(arr, start, end);

        // Separately sort elements before
        // partition and after partition
        int current_pivot = pivot - 1;

        while(current_pivot > start)
        {
            current_pivot = partition(arr, start, current_pivot) - 1;
        }

        current_pivot = pivot + 1;

        while(current_pivot <= end)
        {
            current_pivot = partition(arr, current_pivot, end) + 1;
        }
    }
}

/**
 * @brief sort_kernel : sort every rows for a given column in descending order
 * @param ptr : adress of the device memory segment that store the matrix.
 * @param rows : number of rows of the matrix
 * @param cols : number of columns of the matrix
 */
template<class T>
__global__ void sort_kernel(T* ptr, const int rows, const int cols)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    if(x>=cols)
        return;

    ptr += x * rows;

    quickSort(ptr, 0, rows - 1);
}

} // anonymous

/**
 * @brief sort : sort every columns in descending order.
 * @param stream : if a copy or a kernel can be asynchronous, is there any existing stream?
 * @param buffer : memory segment to process and return.
 * @param rows : number of rows in the matrix.
 * @param cols : number of columns in the matrix.
 * @note this function only work single and double precision floating point type.
 */
template<class T>
__host__ void sort( safe_cudaStream_t& stream, cudaBuffer<T>& buffer, const int& rows, const int& cols)
{
    dim3 block(256,1);
    dim3 grid(utils::divUp(cols, block.x));

    CHECK_EXECUTION(cudaFuncSetCacheConfig(sort_kernel<T>, cudaFuncCachePreferL1));
    sort_kernel<T><<<grid, block, 0,stream.stream>>>(buffer.getPtr(), rows, cols);
    CHECK_EXECUTION(cudaGetLastError());
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// COMPUTE Y ///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{

/**
 * @brief The compute_Y_2d_helper class
 * Convinient class to compute Y = Wct' x X x Wrt,
 * where "'" is the transposition operator, and
 * "x" is the matrix multiplication operator.
 */
template<class T>
struct compute_Y_2d_helper
{

    // Convinient internal types.
    using buffer_type = cudaBuffer<T>;
    using buffer_type_ref = buffer_type&;
    using const_buffer_type_ref = const buffer_type&;

    // Cublas resources handle
    safe_cublasHandle_t& handle;

    // m -> krows -> number of columns of Wct, and number of rows of Y.
    // n -> rows ->  number of rows of X, and Wct.
    // k -> cols -> number of columns of X, and number of rows of Wrt.
    // l -> kcols -> number of columns of Wrt and Y.
    const int m,n,k,l;

    // GEMM operators computes: D = alpha x matmul(A, B) + beta x C,
    // where "x" is the elementwise multiplication operator.
    T alpha, beta;

    // Store the flags no notify that an input should be transpose or not.
    cublasOperation_t do_transpose, do_not_transpose;

    // As the operation cannot be done in one time, a buffer is needed
    // to store the result of: Wct' x X, where "x" is the matrix multiplication
    // operator.
    cudaBuffer<T> buffer;

    /**
     * Parametric constructor.
     * @brief compute_Y_2d_helper
     * @param _handle : cublas resource handle
     * @param rows : number of columns of Wct, and number of rows of Y.
     * @param cols : number of rows of X, and Wct.
     * @param krows : number of columns of X, and number of rows of Wrt.
     * @param kcols : number of columns of Wrt and Y.
     */
    compute_Y_2d_helper(safe_cublasHandle_t& _handle, const int& rows, const int& cols, const int& krows, const int& kcols):
        handle(_handle),
        m(krows),
        n(rows),
        k(cols),
        l(kcols),
        alpha(static_cast<T>(1)),
        beta(static_cast<T>(0)),
        do_transpose(CUBLAS_OP_T),
        do_not_transpose(CUBLAS_OP_N),
        buffer(krows, cols)
    {}

    ~compute_Y_2d_helper() = default;

    /**
     * @brief compute : does the computation of Y = Wct' x X x Wrt.
     * @param Wct : memory segment storing eigenvector the scatter matrix computed along the rows.
     * @param X : memory segment of the input data.
     * @param Wrt : memory segment storing eigenvector the scatter matrix computed along the rolumns.
     * @param Y : retults Wct' x X x Wrt.
     */
    void compute(const_buffer_type_ref Wct, const_buffer_type_ref X, const_buffer_type_ref Wrt, buffer_type_ref Y);
};


/**
 * Specialization of the compute method for single precision floating point data.
 * @brief compute_Y_2d_helper::compute : does the computation of Y = Wct' x X x Wrt.
     * @param Wct : memory segment storing eigenvector the scatter matrix computed along the rows.
     * @param X : memory segment of the input data.
     * @param Wrt : memory segment storing eigenvector the scatter matrix computed along the rolumns.
     * @param Y : retults Wct' x X x Wrt.
 */
template<>
void compute_Y_2d_helper<float>::compute(const_buffer_type_ref Wct, const_buffer_type_ref X, const_buffer_type_ref Wrt, buffer_type_ref Y)
{
    // Step 1) buffer <- Wct' x X
    CHECK_EXECUTION(cublasSgemm(this->handle, this->do_transpose, this->do_not_transpose,
                                       this->m, this->n, this->k,
                                       &this->alpha, Wct, this->n,
                                       X, this->n,
                                       &this->beta, this->buffer, this->m));

    // Step 2) Y <- buffer x Wrt
    CHECK_EXECUTION(cublasSgemm(this->handle, this->do_not_transpose, this->do_not_transpose,
                                   this->m, this->k, this->l,
                                   &this->alpha, this->buffer, this->m,
                                   Wrt, this->k,
                                   &this->beta, Y, this->m));

    // Clear the buffer.
    this->buffer.deallocate();
}

/**
 * Specialization of the compute method for double precision floating point data.
 * @brief compute_Y_2d_helper::compute : does the computation of Y = Wct' x X x Wrt.
     * @param Wct : memory segment storing eigenvector the scatter matrix computed along the rows.
     * @param X : memory segment of the input data.
     * @param Wrt : memory segment storing eigenvector the scatter matrix computed along the rolumns.
     * @param Y : retults Wct' x X x Wrt.
 */
template<>
void compute_Y_2d_helper<double>::compute(const_buffer_type_ref Wct, const_buffer_type_ref X, const_buffer_type_ref Wrt, buffer_type_ref Y)
{
        // Step 1) buffer <- Wct' x X
    CHECK_EXECUTION(cublasDgemm(this->handle, this->do_transpose, this->do_not_transpose,
                                       this->m, this->n, this->k,
                                       &this->alpha, Wct, this->n,
                                       X, this->n,
                                       &this->beta, this->buffer, this->m));

    // Step 2) Y <- buffer x Wrt
    CHECK_EXECUTION(cublasDgemm(this->handle, this->do_not_transpose, this->do_not_transpose,
                                   this->m, this->k, this->l,
                                   &this->alpha, this->buffer, this->m,
                                   Wrt, this->k,
                                   &this->beta, Y, this->m));

    // Clear the buffer.
    this->buffer.deallocate();
}

} // anonymous

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
               cudaBuffer<T>& Y)
{

    // Create a helper object.
    compute_Y_2d_helper<T> obj(handle, rows, cols, krows, kcols);

    // Do the computation.
    obj.compute(Wct, X, Wrt, Y);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// COMPUTE SCATTER MATRICES 3D /////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


namespace
{

template<class T>
__host__ void update_scater_matrices(safe_cublasHandle_t& handle, const int& rows, const int& cols, const cudaBuffer<T>& tmp_Srt, const cudaBuffer<T>& tmp_Sct, cudaBuffer<T>& Srt, cudaBuffer<T>& Sct);

template<>
__host__ void update_scater_matrices<float>(safe_cublasHandle_t& handle, const int& rows, const int& cols, const cudaBuffer<float>& tmp_Srt, const cudaBuffer<float>& tmp_Sct, cudaBuffer<float>& Srt, cudaBuffer<float>& Sct)
{
    float alpha(1.f / static_cast<float>(rows * rows));
    float beta(1.f / static_cast<float>(cols * cols));

    CHECK_EXECUTION(cublasSaxpy_v2(handle, rows * rows, &alpha, tmp_Sct, 1, Sct, 1));
    CHECK_EXECUTION(cublasSaxpy_v2(handle, cols * cols, &beta, tmp_Srt, 1, Srt, 1));
}

template<>
__host__ void update_scater_matrices<double>(safe_cublasHandle_t& handle, const int& rows, const int& cols, const cudaBuffer<double>& tmp_Srt, const cudaBuffer<double>& tmp_Sct, cudaBuffer<double>& Srt, cudaBuffer<double>& Sct)
{
    double alpha(1. / static_cast<double>(rows * rows));
    double beta(1. / static_cast<double>(cols * cols));

    CHECK_EXECUTION(cublasDaxpy_v2(handle, rows * rows, &alpha, tmp_Sct, 1, Sct, 1));
    CHECK_EXECUTION(cublasDaxpy_v2(handle, cols * cols, &alpha, tmp_Srt, 1, Srt, 1));
}

} // anonymous

template<class T>
__host__ void compute_scatter_matrices_3d(safe_cublasHandle_t& handle, safe_cudaStream_t& stream, const cudaBuffer<T>& X, const int& rows, const int& cols, const int& frames, cudaBuffer<T>& Srt, cudaBuffer<T>& Sct)
{
    cudaBuffer<T> tmp_Srt, tmp_Sct;

    Srt.allocate(cols, cols);
    Sct.allocate(rows, rows);

    CHECK_EXECUTION(cudaMemset(Srt, 0, Srt.length_in_bytes()));
    CHECK_EXECUTION(cudaMemset(Sct, 0, Sct.length_in_bytes()));

    int offset = rows * cols;

    int inc = 1;
    int rows_sq = rows * rows;
    int cols_sq = cols * cols;
    T den = static_cast<T>(1) / static_cast<T>(frames);

    for(int frame=0, start=0, end = offset; frame<frames; frame++, start+=offset, end+=offset)
    {
        cudaBuffer<T> currentFrame = X.getRange(start, end);

        compute_scatter_matrices_2d(handle, stream, currentFrame, rows, cols, tmp_Srt, tmp_Sct);

        update_scater_matrices(handle, rows, cols, tmp_Srt, tmp_Sct, Srt, Sct);
    }

    scal(handle, rows_sq, den, Sct, inc);
    scal(handle, cols_sq, den, Srt, inc);

    tmp_Srt.deallocate();
    tmp_Sct.deallocate();
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// COMPUTE Y 3D ///////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
__host__ void compute_Y_3d(safe_cublasHandle_t& handle,
               const cudaBuffer<T>& Wct,
               const cudaBuffer<T>& X,
               const cudaBuffer<T>& Wrt,
               const int& rows, const int& cols, const int& frames,
               const int& krows, const int& kcols,
               cudaBuffer<T>& Y)
{

    Y.allocate(krows, kcols, frames);

    int offset_X = rows * cols;
    int offset_Y = krows * kcols;

    for(int frame=0, start_X=0, end_X = offset_X, start_Y=0, end_Y=offset_Y; frame<frames; frame++, start_X+=offset_X, end_X+=offset_X, start_Y+=offset_Y, end_Y+=offset_Y)
    {
        cudaBuffer<T> currentFrame = X.getRange(start_X, end_X); // aka Xi
        cudaBuffer<T> currentY = Y.getRange(start_Y, end_Y);

        compute_Y_2d(handle, Wct, currentFrame, Wrt, rows, cols, krows, kcols, currentY);
    }

}

} // utils

} // cuda

#define DECL_ALL(type)\
    template void cuda::utils::getMinMax(safe_cublasHandle_t& handle, const cudaBuffer<type>& src, type& min, type& max); \
    template void cuda::utils::rescale<type>(safe_cudaStream_t& stream, safe_cublasHandle_t& handle, cudaBuffer<type>& src, const int& rows, const int& cols); \
    template void cuda::utils::reduce_mean<type>(safe_cudaStream_t&, safe_cublasHandle_t&, const cudaBuffer<type>&, const int&, const int&, const int&, cudaBuffer<type>&);\
    template void cuda::utils::centre<type>(safe_cudaStream_t& stream, cudaBuffer<type>& src, const int& rows, const int& cols, const int& axis, const cudaBuffer<type>& dst); \
    template void cuda::utils::compute_scatter_matrices_2d<type>(safe_cublasHandle_t&, safe_cudaStream_t&, const cudaBuffer<type>&, const int&, const int&, cudaBuffer<type>&, cudaBuffer<type>&); \
    template void cuda::utils::eig<type>(safe_cusolverDnHandle_t&, safe_cudaStream_t&, const int&, cudaBuffer<type>&); \
    template void cuda::utils::col_range<type>(safe_cudaStream_t&, cudaBuffer<type>&, const int&, const int&, const int&, const int&); \
    template void cuda::utils::sort<type>( safe_cudaStream_t&, cudaBuffer<type>&, const int&, const int&); \
    template void cuda::utils::compute_Y_2d<type>(safe_cublasHandle_t&, const cudaBuffer<type>&, const cudaBuffer<type>&, const cudaBuffer<type>&, const int&, const int&, const int&, const int&, cudaBuffer<type>&); \
    template void cuda::utils::compute_scatter_matrices_3d<type>(safe_cublasHandle_t&, safe_cudaStream_t&, const cudaBuffer<type>&, const int&, const int&, const int&, cudaBuffer<type>&, cudaBuffer<type>&); \
    template void cuda::utils::compute_Y_3d<type>(safe_cublasHandle_t&, const cudaBuffer<type>&, const cudaBuffer<type>&, const cudaBuffer<type>&, const int&, const int&, const int&, const int&, const int&, cudaBuffer<type>&);

DECL_ALL(float)
DECL_ALL(double)

#undef DECL_ALL

