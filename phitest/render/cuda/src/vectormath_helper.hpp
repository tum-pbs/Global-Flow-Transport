#pragma once

#ifndef VMATH_HELPER
#define VMATH_HELPER

#define EXPAND_VECTOR_XYZ(v) v.x, v.y, v.z
#define EXPAND_VECTOR_YZW(v) v.y, v.z, v.w
#define EXPAND_VECTOR_ZYX(v) v.z, v.y, v.x

#include "cuda-samples/Common/helper_math.h" //operators for cuda vector types

#ifndef FLOAT_MIN
#define FLOAT_MIN 1.17549e-38
#endif

#ifndef FLOAT_MAX
#define FLOAT_MAX 3.40282e+38
#endif

#ifndef FLOAT_LOWEST
#define FLOAT_LOWEST - FLOAT_MAX
#endif

//--- Missing cuda math helpers ---

inline __host__ __device__ float1 operator+(const float1 a, const float1 b)
{
	return make_float1(a.x + b.x);
}
inline __host__ __device__ float1 operator-(const float1 a, const float1 b)
{
	return make_float1(a.x - b.x);
}
inline __host__ __device__ float1 operator*(const float1 a, const float1 b)
{
	return make_float1(a.x * b.x);
}
inline __host__ __device__ float1 operator*(const float1 a, const float b)
{
	return make_float1(a.x * b);
}
inline __host__ __device__ float1 operator*(const float a, const float1 b)
{
	return make_float1(a * b.x);
}
inline __host__ __device__ void operator+=(float1 &a, const float1 b)
{
	a.x += b.x;
}
inline __host__ __device__ void operator+=(float1 &a, const float b)
{
	a.x += b;
}
inline __host__ __device__ void operator/=(float1 &a, const float b)
{
	a.x /= b;
}

inline  __host__ __device__ float1 fminf(float1 a, float1 b)
{
	return make_float1(fminf(a.x,b.x));
}
inline __host__ __device__ float1 fmaxf(float1 a, float1 b)
{
	return make_float1(fmaxf(a.x,b.x));
}

// cuda - glm compatibility

inline __host__ __device__ float4 make_float4(const glm::vec3 a, const float b){
	return make_float4(a.x, a.y, a.z, b);
}
inline __host__ __device__ float4 make_float4(const glm::vec4 a){
	return make_float4(a.x, a.y, a.z, a.w);
}

// vec4

inline __host__ __device__ void assign(float4 &a, const glm::vec4 b){
	a.x = b.x;
	a.y = b.y;
	a.z = b.z;
	a.w = b.w;
}
inline __host__ __device__ void assign(glm::vec4 &a, const float4 b){
	a.x = b.x;
	a.y = b.y;
	a.z = b.z;
	a.w = b.w;
}

inline __host__ __device__ void assign(float4 &a, const glm::vec3 v, const float b){
	a.x = v.x;
	a.y = v.y;
	a.z = v.z;
	a.w = b;
}
inline __host__ __device__ void assign(float2 &a, const float v, const float b){
	a.x = v;
	a.y = b;
}

inline __host__ __device__ float4 operator+(const float4 a, const glm::vec4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ glm::vec4 operator+(const glm::vec4 a, const float4 b)
{
	return glm::vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, const glm::vec4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ void operator+=(glm::vec4 &a, const float4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}

inline __host__ __device__ float4 operator-(const float4 a, const glm::vec4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ glm::vec4 operator-(const glm::vec4 a, const float4 b)
{
	return glm::vec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, const glm::vec4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ void operator-=(glm::vec4 &a, const float4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}

inline __host__ __device__ float4 operator*(const float4 a, const glm::vec4 b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ glm::vec4 operator*(const glm::vec4 a, const float4 b)
{
	return glm::vec4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(float4 &a, const glm::vec4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ void operator*=(glm::vec4 &a, const float4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}

inline __host__ __device__ float4 operator/(const float4 a, const glm::vec4 b)
{
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ glm::vec4 operator/(const glm::vec4 a, const float4 b)
{
	return glm::vec4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ void operator/=(float4 &a, const glm::vec4 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
}
inline __host__ __device__ void operator/=(glm::vec4 &a, const float4 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
}

inline __host__ __device__ glm::vec3 operator<(const glm::vec3 v, const float b)
{
	return glm::vec3(v.x<b, v.y<b, v.z<b);
}
inline __host__ __device__ glm::vec3 operator<(const glm::vec3 v, const glm::ivec3 b)
{
	return glm::vec3(v.x<b.x, v.y<b.y, v.z<b.z);
}
inline __host__ __device__ glm::vec3 operator>(const glm::vec3 v, const float b)
{
	return glm::vec3(v.x>b, v.y>b, v.z>b);
}
inline __host__ __device__ glm::ivec3 operator<(const glm::ivec3 v, const int32_t b)
{
	return glm::ivec3(v.x<b, v.y<b, v.z<b);
}
inline __host__ __device__ glm::ivec3 operator<(const int32_t b, const glm::ivec3 v)
{
	return glm::ivec3(b<v.x, b<v.y, b<v.z);
}
inline __host__ __device__ glm::ivec3 operator<(const glm::ivec3 v, const glm::ivec3 b)
{
	return glm::vec3(v.x<b.x, v.y<b.y, v.z<b.z);
}

//ivec3 (for indexing)

inline __host__ __device__ uint3 operator+(const dim3 a, const glm::ivec3 b)
{
	return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ glm::ivec3 operator+(const glm::ivec3 a, const dim3 b)
{
	return glm::ivec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

//some math functions

namespace vmath{
	template<typename T>
	__device__ __host__ inline T make_cudaFloat(const float a);
	template<>
	__device__ __host__ inline float1 make_cudaFloat(const float a){return make_float1(a);}
	template<>
	__device__ __host__ inline float2 make_cudaFloat(const float a){return make_float2(a);}
	template<>
	__device__ __host__ inline float3 make_cudaFloat(const float a){return make_float3(a);}
	template<>
	__device__ __host__ inline float4 make_cudaFloat(const float a){return make_float4(a);}
	
	template<typename T, typename V>
	__device__ __host__ inline V lerp(const V a, const V b, const T t){
		return a + t*(b-a); //(T(1) - t)*a + t*b;
	}
	__device__ inline float flerp(const float a, const float b, const float t){
		return __fmaf_rn(__fsub_rn(b,a),t,a);
	}
	
	constexpr __host__ __device__ float sum(const float4 v){
		return v.x+v.y+v.z+v.w;
	}
	constexpr __host__ __device__ float sum(const float3 v){
		return v.x+v.y+v.z;
	}
	constexpr __host__ __device__ float sum(const float2 v){
		return v.x+v.y;
	}
	constexpr __host__ __device__ float sum(const float1 v){
		return v.x;
	}
	constexpr __host__ __device__ float sum(const glm::vec4 v){
		return v.x+v.y+v.z+v.w;
	}
	constexpr __host__ __device__ float sum(const glm::vec3 v){
		return v.x+v.y+v.z;
	}
	
	constexpr __host__ __device__ float prod(const glm::vec3 v){
		return v.x*v.y*v.z;
	}
	constexpr __host__ __device__ float prod(const float3 v){
		return v.x*v.y*v.z;
	}
	constexpr __host__ __device__ float prod(const float2 v){
		return v.x*v.y;
	}
	constexpr __host__ __device__ int32_t prod(const glm::ivec3 v){
		return v.x*v.y*v.z;
	}
	constexpr __host__ __device__ uint32_t prod(const uint3 v){
		return v.x*v.y*v.z;
	}
	inline __host__ __device__ uint32_t prod(const dim3 v){
		return v.x*v.y*v.z;
	}
	
	constexpr __host__ __device__ int32_t exp2(uint32_t e){
		return 1<<e;
	}
	//https://stackoverflow.com/questions/14997165/fastest-way-to-get-a-positive-modulo-in-c-c
	template<typename T, typename M>
	constexpr __host__ __device__ T positivemod(const T v, const M m){
	//	T mod = v%m;
	//	T isNegative = v<0;
	//	return mod + m*isNegative;
		return (v%m) + m*(v<0);
	}
}

#endif //VMATH_HELPER