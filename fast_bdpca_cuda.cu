#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <cuda/cuda_buffer>
#include <cuda/cublas_bdpca.h>

#include <tuple>
#include <vector>


using namespace cuda; // This namespace comes from the files in the folder "cuda" not from NVIDIA.


mxGPUArray* getInputArguments(const int nrhs, const mxArray **prhs, int*krows, int*kcols);




mxGPUArray* createX_(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID);
mxGPUArray* createY(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID);
mxGPUArray* createWrt(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID);
mxGPUArray* createWct(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID);

typedef mxGPUArray* (*creation_function_type)(const int, const mwSize* __restrict__, const mwSize, const mwSize, const mxClassID);
                                                

template<class T>
std::tuple<cudaBuffer<T>, cudaBuffer<T>, cudaBuffer<T>, cudaBuffer<T> > compute_bdpca(const cudaBuffer<T>& X, const int& nb_dims, const mwSize* dims, const int& krows, const int& kcols);

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[])
{

    const creation_function_type init[] = {createY, createX_, createWrt, createWct};

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    mxGPUArray* X;
    int krows, kcols;

    krows = kcols = 0;

    X = getInputArguments(nrhs,prhs, &krows, &kcols);

    mexPrintf("krows: %d, kcols: %d\n", krows, kcols);

    // get the dimensions.
    int nb_dims = mxGPUGetNumberOfDimensions(X);

    const mwSize* dims = mxGPUGetDimensions(X);

    mxClassID classID = mxGPUGetClassID(X);

   
    // Create the outputs.
    mxGPUArray* output[4];

    mexPrintf("Preparation of the Outputs\n");

    for(int i=0;i<nlhs;i++)
        output[i] = init[i](nb_dims, dims, krows, kcols, classID);

    // Process
    if(classID == mxSINGLE_CLASS)
    {        
        cudaBuffer<float> tmpX( static_cast<size_t>(mxGPUGetNumberOfElements(X)));
        tmpX.copyFrom(reinterpret_cast<float*>(mxGPUGetData(X)));
        
        // mexcuda does not like C++ 14
        // auto [Y, Xbar, Wrt, Wct] = compute_bdpca(tmpX, nb_dims, dims, krows, kcols);
        // 
        // tmp[0] = std::move(Y);
        // tmp[1] = std::move(Xbar);        
        // tmp[2] = std::move(Wrt);
        // tmp[3] = std::move(Wct);        

        auto bdpca_outputs = std::move(compute_bdpca(tmpX, nb_dims, dims, krows, kcols));

        cudaBuffer<float> tmp[4] =
        {
            std::move(std::get<0>(bdpca_outputs)), // Y
            std::move(std::get<1>(bdpca_outputs)), // Xbar
            std::move(std::get<2>(bdpca_outputs)), // Wrt
            std::move(std::get<3>(bdpca_outputs)) // Wct
        };

        for(int i=0; i<nlhs; i++)        
            tmp[i].copyTo(reinterpret_cast<float*>(mxGPUGetData(output[i])));


    }
    else
    {        
        cuda::cudaBuffer<double> tmpX( static_cast<size_t>(mxGPUGetNumberOfElements(X)));        
        tmpX.copyFrom(reinterpret_cast<double*>(mxGPUGetData(X)));

        // mexcuda does not like C++ 14        
        // auto [Y, Xbar, Wrt, Wct] = compute_bdpca(tmpX, nb_dims, dims, krows, kcols);
        // 
        // cuda::cudaBuffer<double> tmp[4];        
        // 
        // tmp[0] = std::move(Y);
        // tmp[1] = std::move(Xbar);        
        // tmp[2] = std::move(Wrt);
        // tmp[3] = std::move(Wct);

        auto bdpca_outputs = std::move(compute_bdpca(tmpX, nb_dims, dims, krows, kcols));

        cudaBuffer<double> tmp[4] =
        {
            std::move(std::get<0>(bdpca_outputs)), // Y
            std::move(std::get<1>(bdpca_outputs)), // Xbar
            std::move(std::get<2>(bdpca_outputs)), // Wrt
            std::move(std::get<3>(bdpca_outputs)) // Wct
        };

        for(int i=0; i<nlhs; i++)        
            tmp[i].copyTo(reinterpret_cast<double*>(mxGPUGetData(output[i])));
    }

    for(int i=0; i<nlhs; i++)
        plhs[i] = mxGPUCreateMxArrayOnGPU(output[i]);

}

mxGPUArray* getInputArguments(const int nrhs, const mxArray **prhs,
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

    // If the input matrix is not a gpuArray, the input is converted.
    if(!mxIsGPUArray(X))
    {
        mxArray* tmp = NULL;
        mexCallMATLAB(1, &tmp, 1, &X, "gpuArray");

        mxDestroyArray(X);

        X = mxDuplicateArray(tmp);

        mxDestroyArray(tmp);
    }

    return mxGPUCopyFromMxArray(X);
}

//
mxGPUArray* createX_(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID)
{
    (void)krows;
    (void)kcols;

    const mwSize local_dims[2] = {1, dims[1]};

    return mxGPUCreateGPUArray(2, local_dims, classID, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
}

mxGPUArray* createY(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID)
{
    mwSize local_dims[nb_dims];

    local_dims[0] = krows;
    local_dims[1] = kcols;

    if(nb_dims > 2)
        local_dims[2] = dims[2];

    return mxGPUCreateGPUArray(nb_dims, local_dims, classID, mxREAL, MX_GPU_DO_NOT_INITIALIZE);    
}

mxGPUArray* createWrt(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID)
{
    (void)nb_dims;
    (void)krows;

    const mwSize local_dims[2] = {dims[1], kcols};

    return mxGPUCreateGPUArray(2, local_dims, classID, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
}

mxGPUArray* createWct(const int nb_dims, const mwSize* __restrict__ dims, const mwSize krows, const mwSize kcols, const mxClassID classID)
{
    (void)nb_dims;
    (void)kcols;

    const mwSize local_dims[2] = {dims[0], krows};

    return mxGPUCreateGPUArray(2, local_dims, classID, mxREAL, MX_GPU_DO_NOT_INITIALIZE);    
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
std::tuple<cudaBuffer<T>, cudaBuffer<T>, cudaBuffer<T>, cudaBuffer<T> > compute_bdpca(const cudaBuffer<T>& X, const int& nb_dims, const mwSize* dims, const int& krows, const int& kcols)
{
    return nb_dims == 2 ?
        cuda::bdpca_2d(X, static_cast<int>(dims[0]), static_cast<int>(dims[1]), krows, kcols) :
        cuda::bdpca_3d(X, static_cast<int>(dims[0]), static_cast<int>(dims[1]), static_cast<int>(dims[2]), krows, kcols);
}