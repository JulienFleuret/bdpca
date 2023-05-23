
if isunix() && ~ismac()

    mex -v blas_bdpca.c -I/usr/include/x86_64-linux-gnu -lmwblas -lmwlapack

    if ~isempty(which('mexcuda'))

        cd cuda

        compilelib

        if isempty(getenv('BDPCA_CUDA'))

            setenv('BDPCA_CUDA', pwd);

            warning("A folder has been added to the environment." + ...
                " In case the dynamic library 'libcublas_bdpca.so' is" + ...
                " not found, it is advised to exit MatLab and update" + ...
                " the LD_LIBRARY_PATH locally. To do so open a terminal" + ...
                " go to the folder where the library file is, and write" + ...
                " the following command: 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD'" + ...
                " DO NOT add the keyword 'export' before the left LD_LIBRARY_PATH" + ...
                " unless you wish to make the change global. Then start matlab" + ...
                " by calling 'matlab&' ");
        end

        cd ..

        mexcuda fast_bdpca_cuda.cu -I./ -L./cuda -lcublas_bdpca

    end

end