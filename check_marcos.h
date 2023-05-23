#ifndef CHECK_MARCOS_H
#define CHECK_MARCOS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <sstream>

namespace cuda
{


template<class T>
inline bool check_if_success(const T& err){ return err == 0;}


template<class T>
std::string getErrorString(const T& err);

namespace
{
    std::string cusolverGetStatusString(const cusolverStatus_t& status)
    {
        const std::string errors[]=
        {
                "SUCCESS",
                "NOT_INITIALIZED",
                "ALLOC_FAILED",
                "INVALID_VALUE",
                "ARCH_MISMATCH",
                "MAPPING_ERROR",
                "EXECUTION_FAILED",
                "INTERNAL_ERROR",
                "MATRIX_TYPE_NOT_SUPPORTED",
                "NOT_SUPPORTED",
                "ZERO_PIVOT",
                "INVALID_LICENSE",
                "IRS_PARAMS_NOT_INITIALIZED",
                "IRS_PARAMS_INVALID",
                "IRS_PARAMS_INVALID_PREC",
                "IRS_PARAMS_INVALID_REFINE",
                "IRS_PARAMS_INVALID_MAXITER",
                "IRS_INTERNAL_ERROR",
                "IRS_NOT_SUPPORTED",
                "IRS_OUT_OF_RANGE",
                "IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES",
                "IRS_INFOS_NOT_INITIALIZED",
                "IRS_INFOS_NOT_DESTROYED",
                "IRS_MATRIX_SINGULAR",
                "INVALID_WORKSPACE"
        };

        return errors[static_cast<int>(status)];
    }
} // anonymous

template<> inline std::string getErrorString<cudaError_t>(const cudaError_t& err){ return cudaGetErrorString(err); }
template<> inline std::string getErrorString<cublasStatus_t>(const cublasStatus_t& err){ return cublasGetStatusString(err); }
template<> inline std::string getErrorString<cusolverStatus_t>(const cusolverStatus_t& err){ return cusolverGetStatusString(err); }

template<class T>
std::string getWhereTheErrorComesFrom();

template<> inline std::string getWhereTheErrorComesFrom<cudaError_t>(){ return "CUDA";}
template<> inline std::string getWhereTheErrorComesFrom<cublasStatus_t>(){ return "CUBLAS";}
template<> inline std::string getWhereTheErrorComesFrom<cusolverStatus_t>(){ return "CUSOLVER";}

template<class T>
void check_status_errors(const T& err, const char* const func, const char* const file, const int line)
{
    if (!check_if_success(err))
    {
        std::stringstream sstream;

        sstream << getWhereTheErrorComesFrom<T>() << " Runtime Error at: " << file << ":" << line<< std::endl << getErrorString(err) << " " << func << std::endl;

        throw std::runtime_error(sstream.str());
    }
}

} // cuda

#define CHECK_EXECUTION(val) cuda::check_status_errors((val), #val, __FILE__, __LINE__)


#endif // CHECK_MARCOS_H
