#ifndef SAFE_TYPES_H
#define SAFE_TYPES_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "check_marcos.h"

namespace cuda
{


/**
 * @brief The safe_cudaStream_t class.
 * The goal of this class is to properly manage a cudaStream_t object in situation of abnormal execution.
 * This class properly destroy stream each time its destructor is call, and properly create the stream
 * each time its constructor is called.
 *
 */
struct safe_cudaStream_t
{
    cudaStream_t stream;

    /**
     * Default Ctor
     * @brief safe_cudaStream_t
     * create the stream.
     */
    inline safe_cudaStream_t()
    {
        CHECK_EXECUTION(cudaStreamCreate(&this->stream));
    }

    /**
     * Dtor
     * @brief safely destroy the stream.
     */
    ~safe_cudaStream_t()
    {
        CHECK_EXECUTION(cudaStreamDestroy(this->stream));
    }

    /**
     * Cast operator
     * @brief operator cudaStream_t
     * return the current stream.
     */
    inline operator cudaStream_t() { return this->stream; }
//    inline operator const cudaStream_t() const {return this->stream; }
};

/**
 * @brief The safe_cublasHandle_t class
 * The goal of this class is to properly manage a cublasHandle_t object in situation of abnormal execution.
 * This class properly destroy the handle each time its destructor is call, and properly create the handle
 * and initialize the stream, each time its constructor is called.
 */
struct safe_cublasHandle_t
{
    cublasHandle_t handle;

    /**
     * Parametric constructor.
     * @brief safe_cublasHandle_t
     *  create the handle and set the stream
     * @param safe_stream : stream object to initialize the stream from
     */
    inline safe_cublasHandle_t(safe_cudaStream_t& safe_stream)
    {
        CHECK_EXECUTION(cublasCreate_v2(&this->handle));
        CHECK_EXECUTION(cublasSetStream_v2(this->handle, safe_stream.stream));
    }

    /**
     * Dtor
     * @brief safely destroy the handle.
     */
    ~safe_cublasHandle_t()
    {
        CHECK_EXECUTION(cublasDestroy_v2(this->handle));
    }

    /**
     * Cast operator
     * @brief operator cublasHandle_t
     * return the current handle.
     */
    inline operator cublasHandle_t() { return this->handle; }
//    inline operator const cublasHandle_t() const {return this->handle; }
};

/**
 * @brief The cusolverDnHandle_t class
 * The goal of this class is to properly manage a cusolverDnHandle_t object in situation of abnormal execution.
 * This class properly destroy the handle each time its destructor is call, and properly create the handle
 * and initialize the stream, each time its constructor is called.
 */
struct safe_cusolverDnHandle_t
{
    cusolverDnHandle_t handle;

    /**
     * Parametric constructor.
     * @brief safe_cusolverDnHandle_t
     *  create the handle and set the stream
     * @param safe_stream : stream object to initialize the stream from
     */
    inline safe_cusolverDnHandle_t(safe_cudaStream_t& safe_stream)
    {
        CHECK_EXECUTION(cusolverDnCreate(&this->handle));
        CHECK_EXECUTION(cusolverDnSetStream(this->handle, safe_stream.stream));
    }

    /**
     * Dtor
     * @brief safely destroy the handle.
     */
    ~safe_cusolverDnHandle_t()
    {
        CHECK_EXECUTION(cusolverDnDestroy(this->handle));
    }

    /**
     * Cast operator
     * @brief operator cusolverDnHandle_t
     * return the current handle.
     */
    inline operator cusolverDnHandle_t() { return this->handle; }
//    inline operator const cusolverDnHandle_t() const {return this->handle; }
};


} // cuda

#endif // SAFE_TYPES_H
