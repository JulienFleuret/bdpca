#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

#include <cuda_runtime.h>

#include "safe_types.h"

#include <iterator>
#include <vector>

namespace cuda
{

/**
 * @brief memory storage class of CUDA device memory. Provide a convinient and light interface for memory manipulation, and linear allocation.
 */
template<class T>
class cudaBuffer
{
public:

    using self = cudaBuffer;
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;

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
    cudaBuffer();

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
    cudaBuffer(const size_t& size);

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
    cudaBuffer(const int& rows, const int& cols);

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
    cudaBuffer(const int& rows, const int& cols, const int& frames);

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
    cudaBuffer(pointer ptr, const size_t& size, const bool& acquire=false);

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
    cudaBuffer(const cudaBuffer& obj, const bool& async=false, safe_cudaStream_t* stream=nullptr);

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
    cudaBuffer(cudaBuffer&& obj);

    /**
     * Deallocate the memory if owned and some memory was allocated.
     * @brief Destructor.
     */
    ~cudaBuffer(){ this->deallocate();}

    /**
     * Make a deep copy of the input.
     * @brief Assignation Operator.
     * @param obj : object to duplicate.
     * @note the object provided as first input will be reset.
     * @see cudaBuffer& operator=(cudaBuffer&& obj)
     */
    cudaBuffer& operator=(const cudaBuffer& obj);

    /**
     * Make a deep copy of the input.
     * @brief Move Assignment Operator.
     * @param obj : object to acquire.
     * @note The memory and attributes values of the current object and the input will be swapped.
     * @see cudaBuffer& operator=(const cudaBuffer& obj)
     */
    cudaBuffer& operator=(cudaBuffer&& obj);

    /**
     * Allocate the memory.
     * @brief Initialized constructor.
     * @param size : number of elements of the template type to allocate.
     * @see void allocate(const int& rows, const int& cols)
     * @see void allocate(const int& rows, const int& cols, const int& frames)
     */
    void allocate(const size_t& nb_elems);

    /**
     * Allocate the memory.
     * @brief Initialized constructor.
     * @param rows : number of rows of the matrix.
     * @param cols : number of columns of the matrix.
     * @note This constructor will allocate rows * cols elements of the template type, but it does not have attributes "rows" and "cols".
     * @see void allocate(const size_t& nb_elems)
     * @see void allocate(const int& rows, const int& cols, const int& frames)
     */
    void allocate(const int& rows, const int& cols);

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
    void allocate(const int& rows, const int& cols, const int& frames);

    /**
     * Deallocate the memory if owner and some memory was allocated.
     * @brief Deallocate the memory if owned and some memory was allocated. Reset attributes to default values otherwise.
     */
    void deallocate();


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
    void copyFrom(const_pointer data, const size_t& nb_elems, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice, const bool& async=false, safe_cudaStream_t* safe_stream = nullptr);

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
    void copyFrom(const_pointer data, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice, const bool& async=false, safe_cudaStream_t* safe_stream = nullptr);

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
    void copyFrom(const self& buffer, const bool& async=false, safe_cudaStream_t* safe_stream = nullptr);


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
    void copyTo(pointer data, const size_t& nb_elems, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice, const bool& async=false, safe_cudaStream_t* safe_stream = nullptr) const;

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
    void copyTo(pointer data, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice, const bool& async=false, safe_cudaStream_t* safe_stream = nullptr) const;

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
    void copyTo(self& buffer, const bool& async=false, safe_cudaStream_t* safe_stream = nullptr) const;

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
    void upload(const_pointer ptr, const size_t& size, const bool& async=false, safe_cudaStream_t* safe_stream = nullptr);

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
    template<template<class>class Alloc>
    void upload(const std::vector<value_type, Alloc<value_type> >& obj, const bool& async=false, safe_cudaStream_t* safe_stream = nullptr);

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
    template<template<class, class>class Cont, template<class>class Alloc>
    void upload(const Cont<value_type, Alloc<value_type> >& obj, const bool& async=false, safe_cudaStream_t* safe_stream = nullptr);

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
    void download(pointer ptr, const bool& async=false, safe_cudaStream_t* safe_stream = nullptr) const;

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
    template<template<class>class Alloc>
    void download(std::vector<value_type, Alloc<value_type> >& obj, const bool& async=false, safe_cudaStream_t* safe_stream = nullptr) const;

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
    template<template<class, class>class Cont, template<class>class Alloc>
    void download(Cont<value_type, Alloc<value_type> >& obj, const bool& async=false, safe_cudaStream_t* safe_stream = nullptr) const;


    /**
     * Return an object that does not own the memory, with a size that fits the range.
     * @brief getRange
     * @param start : element index to of the begining of the memory segment.
     * @param end : element index to of the ending of the memory segment.
     * @return object of size (end - start), that does not own the memory.
     */
    cudaBuffer getRange(const int& start, const int& end) const;


    /**
     * Return the number of elements of the current object.
     * @brief length
     * @return number of elements of the current object.
     */
    size_t length() const;

    /**
     * Return the number of bytes of the current object.
     * @brief length
     * @return number of bytes of the current object.
     */
    size_t length_in_bytes() const;


    /**
     * Return the address of the internal pointer.
     * @brief getPtr
     * @return the address of the internal pointer.
     */
    inline pointer getPtr(){ return this->ptr; }

    /**
     * Return the address of the internal pointer.
     * @brief getPtr
     * @return the address of the internal pointer.
     */
    inline const_pointer getPtr() const{ return this->ptr; }

    /**
     * Take the ownership of the given pointer as first argument.
     * @brief takeOwnership
     * @param ptr : pointer to take ownership
     * @param size : size of the memory associate with the pointer to take ownership from.
     * @note taking ownership means that the method "deallocate" or the destructor will free the memory.
     */
    void takeOwnership(pointer& ptr, const size_t& size);

    /**
     * Give the ownership of the given pointer as first argument.
     * @brief giveOwnership
     * @param ptr : pointer to give ownership
     * @param size : size of the memory associate with the pointer to give ownership to.
     * @note giving ownership means that neither the method "deallocate" nor the destructor will attempt to free the memory.
     */
    void giveOwnership(pointer& ptr, size_t& size);


    /**
     * Cast operator
     * @brief operator pointer
     * @return the address of the internal pointer.
     */
    inline operator pointer(){ return this->ptr; }

    /**
     * Cast operator
     * @brief operator pointer
     * @return the address of the internal pointer.
     */
    inline operator const_pointer() const { return this->ptr; }

private:

    T* ptr, *ptrStart;
    size_t size;
    bool own;
};

/**
 * @brief operator <<
 * @param ostr : output stream to use to display the content of the buffer object. (oftenly std::cout or std::wcout)
 * @param buffer : object to display the content from.
 * @return ostr
 */
template<class T, class CharT, class Traits>
inline std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& ostr, const cudaBuffer<T>& buffer)
{
    std::vector<T> tmp(buffer.length());

    tmp.shrink_to_fit();

    buffer.download(tmp.data());

    std::copy(tmp.begin(), tmp.end(), std::ostream_iterator<T, CharT, Traits>(ostr,", "));
    ostr<<std::endl;

    return ostr;
}

/**
 * Overload of the insertion operator for upload purpose.
 * @brief operator <<
 * @param device : device memory object to update.
 * @param host : memory to upload on the device.
 * @return device
 */
template<class T, class Alloc, template<class, class>class Cont>
inline cudaBuffer<T>& operator<<(cudaBuffer<T>& device, const Cont<T, Alloc>& host)
{
    device.upload(host);

    return device;
}

/**
 * Overload of the extraction operator for upload purpose.
 * @brief operator >>
 * @param device : device memory to copy on the host object.
 * @param host : host memory to update.
 * @return
 */
template<class T, class Alloc, template<class, class>class Cont>
inline cudaBuffer<T>& operator>>(cudaBuffer<T>& device, Cont<T, Alloc>& host)
{
    device.download(host);

    return device;
}

} //cuda


#endif // CUDA_BUFFER_H
