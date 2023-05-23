#ifndef CUDA_BUFFER_IMPL_H
#define CUDA_BUFFER_IMPL_H


namespace cuda
{

/**
 * Create a new object with uninitialized memory.
 * @brief Default constructor. Set the size to 0, the ownership attribute to true, and the internal pointers to nullptr
 * @see cudaBuffer(const size_t& size)
 * @see cudaBuffer(const int& rows, const int& cols)
 * @see cudaBuffer(const int& rows, const int& cols, const int& frames)
 * @see cudaBuffer(const cudaBuffer& obj, const bool& async, safe_cudaStream_t* stream)
 * @see cudaBuffer(cudaBuffer&& obj)
 * @see cudaBuffer(pointer ptr, const size_t& size, const bool& acquire)
 */
template<class T>
cudaBuffer<T>::cudaBuffer():
    ptr(nullptr),
    ptrStart(nullptr),
    size(0),
    own(true)
{}

/**
 * Create a new object initialized memory container.
 * @brief Initialized constructor.
 * @param size : number of elements of the template type to allocate.
 * @see cudaBuffer()
 * @see cudaBuffer(const int& rows, const int& cols)
 * @see cudaBuffer(const int& rows, const int& cols, const int& frames)
 * @see cudaBuffer(const cudaBuffer& obj, const bool& async, safe_cudaStream_t* stream)
 * @see cudaBuffer(cudaBuffer&& obj)
 * @see cudaBuffer(pointer ptr, const size_t& size, const bool& acquire)
 */
template<class T>
cudaBuffer<T>::cudaBuffer(const size_t& size):
    cudaBuffer()
{
    this->allocate(size);
}

/**
 * Create a new object initialized memory container.
 * @brief Initialized constructor.
 * @param rows : number of rows of the matrix.
 * @param cols : number of columns of the matrix.
 * @note This constructor will allocate rows * cols elements of the template type, but it does not have attributes "rows" and "cols".
 * @see cudaBuffer()
 * @see cudaBuffer(const size_t& size)
 * @see cudaBuffer(const int& rows, const int& cols, const int& frames)
 * @see cudaBuffer(const cudaBuffer& obj, const bool& async, safe_cudaStream_t* stream)
 * @see cudaBuffer(cudaBuffer&& obj)
 * @see cudaBuffer(pointer ptr, const size_t& size, const bool& acquire)
 */
template<class T>
cudaBuffer<T>::cudaBuffer(const int& rows, const int& cols):
    cudaBuffer()
{
    this->allocate(rows, cols);
}

/**
 * Create a new object initialized memory container.
 * @brief Initialized constructor.
 * @param rows : number of rows of the 3d array.
 * @param cols : number of columns of the 3d array.
 * @param frames : number of frames of the 3d array.
 * @note This constructor will allocate rows * cols elements of the template type, but it does not have attributes "rows", "cols" nor "frames".
 * @see cudaBuffer()
 * @see cudaBuffer(const size_t& size)
 * @see cudaBuffer(const int& rows, const int& cols)
 * @see cudaBuffer(const cudaBuffer& obj, const bool& async, safe_cudaStream_t* stream)
 * @see cudaBuffer(cudaBuffer&& obj)
 * @see cudaBuffer(pointer ptr, const size_t& size, const bool& acquire)
 */
template<class T>
cudaBuffer<T>::cudaBuffer(const int& rows, const int& cols, const int& frames):
    cudaBuffer()
{
    this->allocate(rows, cols, frames);
}


/**
 * @brief cudaBuffer
 * @param ptr : pointer of the memory to copy from.
 * @param size : number of elements of the template type to copy.
 * @param acquire : should a deep copy be made or not.
 * @see cudaBuffer()
 * @see cudaBuffer(const size_t& size)
 * @see cudaBuffer(const int& rows, const int& cols)
 * @see cudaBuffer(const int& rows, const int& cols, const int& frames)
 * @see cudaBuffer(const cudaBuffer& obj, const bool& async, safe_cudaStream_t* stream)
 * @see cudaBuffer(cudaBuffer&& obj)
 */
template<class T>
cudaBuffer<T>::cudaBuffer(pointer ptr, const size_t& size, const bool& acquire)
{
    this->ptr = this->ptrStart = ptr;
    this->size = size;
    this->own = acquire;

}

/**
 * Make a deep copy of the first input.
 * @brief Copy constructor.
 * @param obj : object to duplicate.
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @see cudaBuffer()
 * @see cudaBuffer(const size_t& size)
 * @see cudaBuffer(const int& rows, const int& cols)
 * @see cudaBuffer(const int& rows, const int& cols, const int& frames)
 * @see cudaBuffer(cudaBuffer&& obj)
 * @see cudaBuffer(pointer ptr, const size_t& size, const bool& acquire)
 */
template<class T>
cudaBuffer<T>::cudaBuffer(const cudaBuffer& obj, const bool& async, safe_cudaStream_t* stream):
    cudaBuffer(obj.size)
{
    this->copyFrom(obj, cudaMemcpyDeviceToDevice, async, stream);
}

/**
 * Acquire the memory the first input.
 * @brief Move constructor.
 * @param obj : object to acquire.
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @note the object provided as first input will be reset.
 * @see cudaBuffer()
 * @see cudaBuffer(const size_t& size)
 * @see cudaBuffer(const int& rows, const int& cols)
 * @see cudaBuffer(const int& rows, const int& cols, const int& frames)
 * @see cudaBuffer(cudaBuffer&& obj)
 * @see cudaBuffer(pointer ptr, const size_t& size, const bool& acquire)
 */
template<class T>
cudaBuffer<T>::cudaBuffer(cudaBuffer&& obj):
    ptr(obj.ptr),
    ptrStart(obj.ptrStart),
    size(obj.size),
    own(obj.own)
{
    obj.ptr = nullptr;
    obj.ptrStart = nullptr;
    obj.size = 0;
    obj.own = false;
}

/**
 * Make a deep copy of the input.
 * @brief Assignation Operator.
 * @param obj : object to duplicate.
 * @note the object provided as first input will be reset.
 * @see cudaBuffer& operator=(cudaBuffer&& obj)
 */
template<class T>
cudaBuffer<T>& cudaBuffer<T>::operator =(const cudaBuffer& obj)
{
    if(std::addressof(obj) != this)
    {
        this->deallocate();
        this->copyFrom(obj);
    }

    return (*this);
}

/**
 * Make a deep copy of the input.
 * @brief Move Assignment Operator.
 * @param obj : object to acquire.
 * @note The memory and attributes values of the current object and the input will be swapped.
 * @see cudaBuffer& operator=(const cudaBuffer& obj)
 */
template<class T>
cudaBuffer<T>& cudaBuffer<T>::operator=(cudaBuffer&& obj)
{
    if(std::addressof(obj) != this)
    {
        std::swap(this->ptr, obj.ptr);
        std::swap(this->ptrStart, obj.ptrStart);
        std::swap(this->size, obj.size);
    }
    return (*this);
}

/**
 * Allocate the memory.
 * @brief Initialized constructor.
 * @param size : number of elements of the template type to allocate.
 * @see void allocate(const int& rows, const int& cols)
 * @see void allocate(const int& rows, const int& cols, const int& frames)
 */
template<class T>
void cudaBuffer<T>::allocate(const size_t& nb_elems)
{
    if(this->size && this->size != nb_elems)
        this->deallocate();

    if(!this->size)
    {
        CHECK_EXECUTION(cudaMalloc(&this->ptr, nb_elems * static_cast<size_t>(sizeof(T))));
        this->size = nb_elems;
        this->ptrStart = this->ptr;
        this->own = true;
    }
}

/**
 * Allocate the memory.
 * @brief Initialized constructor.
 * @param rows : number of rows of the matrix.
 * @param cols : number of columns of the matrix.
 * @note This constructor will allocate rows * cols elements of the template type, but it does not have attributes "rows" and "cols".
 * @see void allocate(const size_t& nb_elems)
 * @see void allocate(const int& rows, const int& cols, const int& frames)
 */
template<class T>
void cudaBuffer<T>::allocate(const int& rows, const int& cols)
{
    this->allocate(static_cast<size_t>(rows) * static_cast<size_t>(cols));
}

/**
 * Allocate the memory.
 * @brief Initialized constructor.
 * @param rows : number of rows of the 3d array.
 * @param cols : number of columns of the 3d array.
 * @param frames : number of frames of the 3d array.
 * @note This constructor will allocate rows * cols elements of the template type, but it does not have attributes "rows", "cols" nor "frames".
 * @see void allocate(const size_t& nb_elems)
 * @see void allocate(const int& rows, const int& cols, const int& frames)
 */
template<class T>
void cudaBuffer<T>::allocate(const int& rows, const int& cols, const int& frames)
{
    this->allocate(static_cast<size_t>(rows) * static_cast<size_t>(cols) * static_cast<size_t>(frames));
}

/**
 * Deallocate the memory if owner and some memory was allocated.
 * @brief Deallocate the memory if owned and some memory was allocated. Reset attributes to default values otherwise.
 */
template<class T>
void cudaBuffer<T>::deallocate()
{
    if(this->size && this->own)
        CHECK_EXECUTION(cudaFree(this->ptrStart));

    this->ptrStart = this->ptr = nullptr;
    this->size = 0;
}

/**
 * Make a deep copy of the first input.
 * @brief Method to copy data from a pointer.
 * @param data : pointer of the memory to copy from.
 * @param nb_elems : number of elements of the template type to copy.
 * @param kind : should the copy be from Device to Device or From Device to Host?
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @note If not enough memory is allocated, "nb_elems" elements are allocated.
 * @see void copyFrom(const_pointer data, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
 * @see void copyFrom(const self& buffer, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
 */
template<class T>
void cudaBuffer<T>::copyFrom(const_pointer data, const size_t& nb_elems, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
{
    if(this->size != nb_elems)
        this->allocate(nb_elems);
    if(async)
        CHECK_EXECUTION(cudaMemcpyAsync(this->ptr, data, nb_elems * static_cast<size_t>(sizeof(T)), kind, safe_stream ? safe_stream->stream : nullptr));
    else
        CHECK_EXECUTION(cudaMemcpy(this->ptr, data, nb_elems * static_cast<size_t>(sizeof(T)), kind ) );
}

/**
 * Make a deep copy of the first input.
 * @brief Method to copy data from a pointer.
 * @param data : pointer of the memory to copy from.
 * @param kind : should the copy be from Device to Device or From Device to Host?
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @note this method requires the memory to be allocated in advance. The only copy of the number of elements pre-allocated will be made.
 * @see void copyFrom(const_pointer data, const size_t& nb_elems, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
 * @see void copyFrom(const self& buffer, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
 */
template<class T>
void cudaBuffer<T>::copyFrom(const_pointer data, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
{
    this->copyFrom(data, this->size, kind, async, safe_stream);
}

/**
 * Make a deep copy of the first input.
 * @brief Method to copy data from an object.
 * @param buffer : object to copy from.
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @note this method will allocate enough memory to make the copy, if necessary.
 * @see void copyFrom(const_pointer data, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
 * @see void copyFrom(const self& buffer, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
 */
template<class T>
void cudaBuffer<T>::copyFrom(const self& buffer, const bool& async, safe_cudaStream_t* safe_stream)
{
    if(this->size!=buffer.size)
        this->allocate(buffer.size);

    this->copyFrom(buffer.ptr, buffer.size, cudaMemcpyDeviceToDevice, async, safe_stream);
}

/**
 * Make a deep copy of the first input.
 * @brief Method to copy data to a pointer.
 * @param data : pointer of the memory to copy to.
 * @param nb_elems : number of elements of the template type to copy.
 * @param kind : should the copy be from Device to Device or From Device to Host?
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @note If not enough memory is allocated, "nb_elems" elements are allocated.
 * @see void copyTo(pointer data, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
 * @see void copyTo(self& buffer, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
 */
template<class T>
void cudaBuffer<T>::copyTo(pointer data, const size_t& nb_elems, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream) const
{
    if(async)
        CHECK_EXECUTION(cudaMemcpyAsync(data, this->ptr, nb_elems * static_cast<size_t>(sizeof(T)), kind, safe_stream ? safe_stream->stream : nullptr));
    else
        CHECK_EXECUTION(cudaMemcpy(data, this->ptr, nb_elems * static_cast<size_t>(sizeof(T)), kind ) );
}

/**
 * Make a deep copy of the first input.
 * @brief Method to copy data to a pointer.
 * @param data : pointer of the memory to copy to.
 * @param kind : should the copy be from Device to Device or From Device to Host?
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @note this method requires the memory to be allocated in advance. The only copy of the number of elements pre-allocated will be made.
 * @see void copyFrom(const_pointer data, const size_t& nb_elems, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
 * @see void copyFrom(const self& buffer, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
 */
template<class T>
void cudaBuffer<T>::copyTo(pointer data, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream) const
{
    this->copyTo(data, this->size, kind, async, safe_stream);
}

/**
 * Make a deep copy of the first input.
 * @brief Method to copy data to an object.
 * @param buffer : object to copy to.
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @note this method will allocate enough memory to make the copy, if necessary.
 * @see void copyFrom(const_pointer data, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
 * @see void copyFrom(const self& buffer, cudaMemcpyKind kind, const bool& async, safe_cudaStream_t* safe_stream)
 */
template<class T>
void cudaBuffer<T>::copyTo(self& buffer, const bool& async, safe_cudaStream_t* safe_stream) const
{
    buffer.allocate(this->size);

    this->copyTo(buffer.ptr, this->size, async, safe_stream);
}

/**
 * Make a deep copy of the first input, from host to device memory.
 * @brief Method to copy upload data from a pointer, from host memory to device memoryr.
 * @param data : pointer of the memory to copy from.
 * @param nb_elems : number of elements of the template type to copy.
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @note If not enough memory is allocated, "nb_elems" elements are allocated.
 * @see void upload(const std::vector<value_type, Alloc<value_type> >& obj, const bool& async, safe_cudaStream_t* safe_stream)
 * @see void upload(const Cont<value_type, Alloc<value_type> >& obj, const bool& async, safe_cudaStream_t* safe_stream)
 */
template<class T>
void cudaBuffer<T>::upload(const_pointer ptr, const size_t& size, const bool& async, safe_cudaStream_t* safe_stream)
{
    this->copyFrom(ptr, size, cudaMemcpyHostToDevice,async, safe_stream);
}

/**
 * Make a deep copy of the first input, from host to device memory.
 * @brief Method to copy upload data from an continuous memory allocated object, from host memory to device memory.
 * @param obj : object to copy from.
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @note If not enough memory is allocated, "nb_elems" elements are allocated.
 * @see void upload(const_pointer ptr, const size_t& size, const bool& async, safe_cudaStream_t* safe_stream)
 * @see void upload(const Cont<value_type, Alloc<value_type> >& obj, const bool& async, safe_cudaStream_t* safe_stream)
 */
template<class T>
template<template<class>class Alloc>
void cudaBuffer<T>::upload(const std::vector<value_type, Alloc<value_type> >& obj, const bool& async, safe_cudaStream_t* safe_stream)
{
    this->upload(obj.data(), obj.size(), async, safe_stream);
}

/**
 * Make a deep copy of the first input, from host to device memory.
 * @brief Method to copy upload data from an object, from host memory to device memory.
 * @param obj : object to copy from.
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @note The difference between this method and the previous one is that, this method also manage non-contiguous memory.
 * @see void upload(const_pointer ptr, const size_t& size, const bool& async, safe_cudaStream_t* safe_stream)
 * @see void upload(const std::vector<value_type, Alloc<value_type> >& obj, const bool& async, safe_cudaStream_t* safe_stream)
 */
template<class T>
template<template<class, class>class Cont, template<class>class Alloc>
void cudaBuffer<T>::upload(const Cont<value_type, Alloc<value_type> >& obj, const bool& async, safe_cudaStream_t* safe_stream)
{
    this->upload(std::vector<value_type>(obj.begin(), obj.end()), async, safe_stream);
}

/**
 * Make a deep copy of the first input, from device to host memory.
 * @brief Method to copy upload data from a pointer, from device memory to host memoryr.
 * @param data : pointer of the memory to copy from.
 * @param nb_elems : number of elements of the template type to copy.
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @note If not enough memory is allocated, "nb_elems" elements are allocated.
 * @see void upload(const std::vector<value_type, Alloc<value_type> >& obj, const bool& async, safe_cudaStream_t* safe_stream)
 * @see void upload(const Cont<value_type, Alloc<value_type> >& obj, const bool& async, safe_cudaStream_t* safe_stream)
 */
template<class T>
void cudaBuffer<T>::download(pointer ptr, const bool& async, safe_cudaStream_t* safe_stream) const
{
    this->copyTo(ptr, this->size, cudaMemcpyDeviceToHost, async, safe_stream);
}

/**
 * Make a deep copy of the first input, from device to host memory.
 * @brief Method to copy upload data from an continuous memory allocated object, from device memory to host memory.
 * @param obj : object to copy from.
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @note If not enough memory is allocated, "nb_elems" elements are allocated.
 * @see void upload(const_pointer ptr, const size_t& size, const bool& async, safe_cudaStream_t* safe_stream)
 * @see void upload(const Cont<value_type, Alloc<value_type> >& obj, const bool& async, safe_cudaStream_t* safe_stream)
 */
template<class T>
template<template<class>class Alloc>
void cudaBuffer<T>::download(std::vector<value_type, Alloc<value_type> >& obj, const bool& async, safe_cudaStream_t* safe_stream) const
{
    if(!obj.empty() && obj.size() != this->size)
    {
        obj.clear();
        obj.reserve(this->size);
        obj.resize(obj.capacity());
    }

    this->download(obj.data(), async, safe_stream);
}

/**
 * Make a deep copy of the first input, from device to host memory.
 * @brief Method to copy upload data from an object, from device memory to host memory.
 * @param obj : object to copy from.
 * @param async : should the copy been asynchronous.
 * @param stream : if the copy should be asynchronous, is there any existing stream.
 * @note The difference between this method and the previous one is that, this method also manage non-contiguous memory.
 * @see void upload(const_pointer ptr, const size_t& size, const bool& async, safe_cudaStream_t* safe_stream)
 * @see void upload(const std::vector<value_type, Alloc<value_type> >& obj, const bool& async, safe_cudaStream_t* safe_stream)
 */
template<class T>
template<template<class, class>class Cont, template<class>class Alloc>
void cudaBuffer<T>::download(Cont<value_type, Alloc<value_type> >& obj, const bool& async, safe_cudaStream_t* safe_stream) const
{
    std::vector<T> tmp(obj.begin(), obj.end());

    tmp.shrink_to_fit();

    this->download(tmp, async, safe_stream);
}

/**
 * Return an object that does not own the memory, with a size that fits the range.
 * @brief getRange
 * @param start : element index to of the begining of the memory segment.
 * @param end : element index to of the ending of the memory segment.
 * @return object of size (end - start), that does not own the memory.
 */
template<class T>
cudaBuffer<T> cudaBuffer<T>::getRange(const int& start, const int& end) const
{
    cudaBuffer<T> obj;

    obj.ptrStart = obj.ptr = this->ptr + start;
    obj.size = end - start;
    obj.own = false;

    return obj;
}



/**
 * Return the number of elements of the current object.
 * @brief length
 * @return number of elements of the current object.
 */
template<class T>
size_t cudaBuffer<T>::length() const{ return this->size;}

/**
 * Return the number of bytes of the current object.
 * @brief length
 * @return number of bytes of the current object.
 */
template<class T>
size_t cudaBuffer<T>::length_in_bytes() const { return this->size * static_cast<size_t>(sizeof(T));}



/**
 * Take the ownership of the given pointer as first argument.
 * @brief takeOwnership
 * @param ptr : pointer to take ownership
 * @param size : size of the memory associate with the pointer to take ownership from.
 * @note taking ownership means that the method "deallocate" or the destructor will free the memory.
 */
template<class T>
void cudaBuffer<T>::takeOwnership(pointer& ptr, const size_t& size)
{
    this->ptrStart = this->ptr = ptr;
    this->size = size;
}

/**
 * Give the ownership of the given pointer as first argument.
 * @brief giveOwnership
 * @param ptr : pointer to give ownership
 * @param size : size of the memory associate with the pointer to give ownership to.
 * @note giving ownership means that neither the method "deallocate" nor the destructor will attempt to free the memory.
 */
template<class T>
void cudaBuffer<T>::giveOwnership(pointer& ptr, size_t& size)
{
    ptr = this->ptrStart;
    size = this->size;

    this->ptr = this->ptrStart = nullptr;
    this->size = 0;
}


} // cuda

#endif // CUDA_BUFFER_IMPL_H
