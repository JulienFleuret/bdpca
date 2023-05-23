#ifndef TRAITS_H
#define TRAITS_H

#include <vector_types.h>

/**
 * @brief Convinient macro that is helpful for extend the declaration of specialized functions and operators overloads.
 */
#define DECL_ALL_VEC_TYPES(macro)\
    macro(uchar)\
    macro(char)\
    macro(ushort)\
    macro(short)\
    macro(uint)\
    macro(int)\
    macro(long)\
    macro(ulong)\
    macro(longlong)\
    macro(ulonglong)\
    macro(float)\
    macro(double)


namespace cuda
{

// Common type shortening.
using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned;
using ulong = unsigned long;
using longlong = long long;
using ulonglong = unsigned long long;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// VECTOR TYPE TRAITS /////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief traits structure for GPU vector type.
 * @note this structure contains the element_type (e.g. float for a float4 vector type), and the width of the stride.
 */
template<class T>
struct __device__ __host__ vector_type_traits;

#define SPEC_VECTOR_TYPE_TRAITS_(type) \
    template<> \
    struct  vector_type_traits<type ## 4> \
    { \
        using element_type = type; \
        static constexpr int stride_width = 4;\
    }; \
    template<> \
    struct  vector_type_traits<type ## 3> \
    { \
        using element_type = type; \
        static constexpr int stride_width = 3;\
    }; \
    template<> \
    struct  vector_type_traits<type ## 2> \
    { \
        using element_type = type; \
        static constexpr int stride_width = 2;\
    }; \
    template<> \
    struct  vector_type_traits<type ## 1> \
    { \
        using element_type = type; \
        static constexpr int stride_width = 1;\
    };    \
    template<> \
    struct  vector_type_traits<type> \
    { \
        using element_type = type; \
        static constexpr int stride_width = 1;\
    };

DECL_ALL_VEC_TYPES(SPEC_VECTOR_TYPE_TRAITS_)

#undef SPEC_VECTOR_TYPE_TRAITS_


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// CREATE TYPE ////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 *@brief allow to create vector_type, based on the element type and the number of stride size.
 */
template<class T, int N>
struct __device__ __host__ create_type;

#define IMPL_CREATE_TYPE_SPECS_(_type)\
template<> struct create_type<_type, 4>{ using type= _type ## 4;};\
template<> struct create_type<_type, 3>{ using type= _type ## 3;};\
template<> struct create_type<_type, 2>{ using type= _type ## 2;};\
template<> struct create_type<_type, 1>{ using type= _type ## 1;};

DECL_ALL_VEC_TYPES(IMPL_CREATE_TYPE_SPECS_)

#undef IMPL_CREATE_TYPE_SPECS_


} // cuda


#endif // TRAITS_H
