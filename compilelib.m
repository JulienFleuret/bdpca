

mexcuda -Dynamic -g -G -I/usr/local/cuda/include  -lcudart -lcuda -lcublas -lcusolver -c  ./cublas_bdpca.cu
mexcuda -Dynamic -g -G -I/usr/local/cuda/include  -lcudart -lcuda -lcublas -lcusolver -c ./utils.cu

%TODO Test on MacOS and Windows
if(isunix() && ~ismac()) 
compiler = mex.getCompilerConfigurations('C++').Location;

if exist("libcublas_bdpca.so.1.0","file")
    delete("libcublas_bdpca.so.1.0")
end

if exist("libcublas_bdpca.so.1","file")
    delete("libcublas_bdpca.so.1")
end

if exist("libcublas_bdpca.so","file")
    delete("libcublas_bdpca.so")
end

if exist("../libcublas_bdpca.so","file")
    delete("../libcublas_bdpca.so")
end

if exist("../libcublas_bdpca.so.1","file")
    delete("../libcublas_bdpca.so.1")
end

if exist("../libcublas_bdpca.so.1.0","file")
    delete("../libcublas_bdpca.so.1.0")
end

if exist("../libcublas_bdpca.so.1.0.0","file")
    delete("../libcublas_bdpca.so.1.0.0")
end

system(sprintf("%s -shared -Wl,-soname,libcublas_bdpca.so -o libcublas_bdpca.so cublas_bdpca.o utils.o  -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas -lcusolver", compiler));
% system("ln -s libcublas_bdpca.so.1.0.0 libcublas_bdpca.so.1.0");
% system("ln -s libcublas_bdpca.so.1.0.0 libcublas_bdpca.so.1");
% system("ln -s libcublas_bdpca.so.1.0.0 libcublas_bdpca.so");

if(isempty(getenv("BDPCA_CUDA")))
setenv("BDPCA_CUDA",pwd);
end

% system("ln -s $PWD/libcublas_bdpca.so.1.0.0 $PWD/../libcublas_bdpca.so");
% system("ln -s $PWD/libcublas_bdpca.so.1.0.0 $PWD/../libcublas_bdpca.so.1");
% system("ln -s $PWD/libcublas_bdpca.so.1.0.0 $PWD/../libcublas_bdpca.so.1.0");
% system("ln -s $PWD/libcublas_bdpca.so.1.0.0 $PWD/../libcublas_bdpca.so.1.0.0");
end

%mex -shared -o libcuda_bdpca_lib.so.1.0.0 ./cublas_bdpca_cuda.o ./utils_cuda.o  -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas -lcusolver