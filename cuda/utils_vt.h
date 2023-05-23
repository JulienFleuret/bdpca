#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <vector_functions.h>

#include "traits.h"

#include <cmath>

namespace cuda
{

//#undef SPEC_VECTOR_TYPE_TRAITS

///////////////////////////////////////   MAKE_TYPE   ///////////////////////////////////////////////////////////////////////

template<class VT, class T>
__device__ VT make_type(const T& v);

#define IMPL_MAKE_TYPE_SPECS_(type)\
    template<> __device__ __forceinline__ type ## 1 make_type<type ## 1>(const type& v){ return make_ ## type ## 1(v);}\
    template<> __device__ __forceinline__ type ## 2 make_type<type ## 2>(const type& v){ return make_ ## type ## 2(v, v);}\
    template<> __device__ __forceinline__ type ## 3 make_type<type ## 3>(const type& v){ return make_ ## type ## 3(v, v, v);}\
    template<> __device__ __forceinline__ type ## 4 make_type<type ## 4>(const type& v){ return make_ ## type ## 4(v, v, v, v);}

DECL_ALL_VEC_TYPES(IMPL_MAKE_TYPE_SPECS_)

#undef IMPL_MAKE_TYPE_SPECS_

///////////////////////////////////////   V_REDUCE_SUM, V_LOAD, V_STORE   ///////////////////////////////////////////////////////

template<class T> __device__   typename vector_type_traits<T>::element_type v_reduce_sum(const T& a);
template<class VT, class T> __device__   VT v_load(const T* ptr);
template<class T, class VT> __device__   void v_store(T* ptr, const VT& v);

#define SPEC_V_REDUCE_SUM_IMPL_(type)\
    template<> __device__  __forceinline__ type v_reduce_sum(const type ## 1& a){ return a.x;}\
    template<> __device__  __forceinline__ type v_reduce_sum(const type ## 2& a){ return a.x + a.y;} \
    template<> __device__  __forceinline__ type v_reduce_sum(const type ## 3& a){ return a.x + a.y + a.z;} \
    template<> __device__  __forceinline__ type v_reduce_sum(const type ## 4& a){ return a.x + a.y + a.z + a.w;}

DECL_ALL_VEC_TYPES(SPEC_V_REDUCE_SUM_IMPL_)

#undef SPEC_V_REDUCE_SUM_IMPL_


#define SPEC_V_LOAD_STORE_IMPL_(type)\
    template<> __device__  __forceinline__ type ## 1 v_load<type ## 1, type> (const type* ptr){ return *reinterpret_cast<const type ## 1*>(ptr);} \
    template<> __device__  __forceinline__ type ## 2 v_load<type ## 2, type> (const type* ptr){ return *reinterpret_cast<const type ## 2*>(ptr);} \
    template<> __device__  __forceinline__ type ## 3 v_load<type ## 3, type> (const type* ptr){ return *reinterpret_cast<const type ## 3*>(ptr);} \
    template<> __device__  __forceinline__ type ## 4 v_load<type ## 4, type> (const type* ptr){ return *reinterpret_cast<const type ## 4*>(ptr);} \
    template<> __device__  __forceinline__ void v_store<type, type ## 1> (type* ptr, const type ## 1& v){ *reinterpret_cast<type ## 1*>(ptr) = v;} \
    template<> __device__  __forceinline__ void v_store<type, type ## 2> (type* ptr, const type ## 2& v){ *reinterpret_cast<type ## 2*>(ptr) = v;} \
    template<> __device__  __forceinline__ void v_store<type, type ## 3> (type* ptr, const type ## 3& v){ *reinterpret_cast<type ## 3*>(ptr) = v;} \
    template<> __device__  __forceinline__ void v_store<type, type ## 4> (type* ptr, const type ## 4& v){ *reinterpret_cast<type ## 4*>(ptr) = v;}

DECL_ALL_VEC_TYPES(SPEC_V_LOAD_STORE_IMPL_)


#undef SPEC_V_LOAD_STORE_IMPL_

///////////////////////////////////////   OPERATOR+ | OPERATOR- | OPERATOR* | OPERATOR/   ////////////////////////////////////////////////////////////////////

#define IMPL_VEC_TYPE_OP_(type, op)\
    __device__  __forceinline__ type ## 4 operator op(const type ## 4 & left, const type ## 4 & right) \
{\
    return make_ ## type ## 4(left.x op right.x, left.y op right.y, left.z op right.z, left.w op right.w);\
}\
    __device__  __forceinline__ type ## 3 operator op(const type ## 3 & left, const type ## 3 & right) \
{\
    return make_ ## type ## 3(left.x op right.x, left.y op right.y, left.z op right.z);\
}\
    __device__  __forceinline__ type ## 2 operator op(const type ## 2 & left, const type ## 2 & right) \
{\
    return make_ ## type ## 2(left.x op right.x, left.y op right.y);\
}\
    __device__  __forceinline__ type ## 1 operator op(const type ## 1 & left, const type ## 1 & right) \
{\
    return make_ ## type ## 1(left.x op right.x);\
} \
    __device__  __forceinline__ type ## 4& operator op ## =(type ## 4 & left, const type ## 4 & right) \
{\
    left = left op right; \
    return left;\
} \
    __device__  __forceinline__ type ## 3 operator op ## =(type ## 3 & left, const type ## 3 & right) \
{\
    left = left op right; \
    return left;\
}\
    __device__  __forceinline__ type ## 2 operator op ## =(type ## 2 & left, const type ## 2 & right) \
{\
    left = left op right; \
    return left;\
}\
    __device__  __forceinline__ type ## 1 operator op ## =(type ## 1 & left, const type ## 1 & right) \
{\
    left = left op right; \
    return left;\
}

#define IMPL_VEC_TYPE_PLUS_(type) IMPL_VEC_TYPE_OP_(type, +)
#define IMPL_VEC_TYPE_MINUS_(type) IMPL_VEC_TYPE_OP_(type, -)
#define IMPL_VEC_TYPE_TIMES_(type) IMPL_VEC_TYPE_OP_(type, *)
#define IMPL_VEC_TYPE_DIVIDES_(type) IMPL_VEC_TYPE_OP_(type, /)

DECL_ALL_VEC_TYPES(IMPL_VEC_TYPE_PLUS_)
DECL_ALL_VEC_TYPES(IMPL_VEC_TYPE_MINUS_)
DECL_ALL_VEC_TYPES(IMPL_VEC_TYPE_TIMES_)
DECL_ALL_VEC_TYPES(IMPL_VEC_TYPE_DIVIDES_)

#undef IMPL_VEC_TYPE_PLUS_
#undef IMPL_VEC_TYPE_MINUS_
#undef IMPL_VEC_TYPE_TIMES_
#undef IMPL_VEC_TYPE_DIVIDES_

#undef IMPL_VEC_TYPE_OP

///////////////////////////////////////   V_SETZEROS   ///////////////////////////////////////////////////////////////////////

template<class T>
__device__  T v_setzeros();

#define IMPL_V_SET_ZEROS_(type)\
    template<> __device__ __forceinline__ type ## 1 v_setzeros<type ## 1>(){ return make_ ## type ## 1(static_cast<type>(0));} \
    template<> __device__  __forceinline__ type ## 2 v_setzeros<type ## 2>(){ return make_ ## type ## 2(static_cast<type>(0), static_cast<type>(0));} \
    template<> __device__  __forceinline__ type ## 3 v_setzeros<type ## 3>(){ return make_ ## type ## 3(static_cast<type>(0), static_cast<type>(0), static_cast<type>(0));} \
    template<> __device__  __forceinline__ type ## 4 v_setzeros<type ## 4>(){ return make_ ## type ## 4(static_cast<type>(0), static_cast<type>(0), static_cast<type>(0), static_cast<type>(0));}

DECL_ALL_VEC_TYPES(IMPL_V_SET_ZEROS_)

#undef DECL_ALL_VEC_TYPES

///////////////////////////////////////   FMA_OP   ///////////////////////////////////////////////////////////////////////

#define IMPL_FMA_OP(type, fun)\
    __device__   __forceinline__ type fma_op(const type& a, const type& b, const type& c){ return fun(a,b,c);} \
    __device__   __forceinline__ type ## 1 fma_op(const type ## 1& a, const type ## 1& b, const type ## 1& c){ return make_ ## type ## 1(fun(a.x,b.x,c.x) );} \
    __device__   __forceinline__ type ## 2 fma_op(const type ## 2& a, const type ## 2& b, const type ## 2& c){ return make_ ## type ## 2(fun(a.x,b.x,c.x), fun(a.y,b.y,c.y) );} \
    __device__   __forceinline__ type ## 3 fma_op(const type ## 3& a, const type ## 3& b, const type ## 3& c){ return make_ ## type ## 3(fun(a.x,b.x,c.x), fun(a.y,b.y,c.y), fun(a.z,b.z,c.z) );} \
    __device__   __forceinline__ type ## 4 fma_op(const type ## 4& a, const type ## 4& b, const type ## 4& c){ return make_ ## type ## 4(fun(a.x,b.x,c.x), fun(a.y,b.y,c.y), fun(a.z,b.z,c.z), fun(a.w,b.w,c.w) );}

IMPL_FMA_OP(float, std::fmaf)
IMPL_FMA_OP(double, std::fma)

#undef IMPL_FMA_OP

} // cuda

#endif // UTILS_CUH
