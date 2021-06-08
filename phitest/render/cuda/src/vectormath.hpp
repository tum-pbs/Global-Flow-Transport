#pragma once

#ifndef _INCLUDE_VECTORMATH
#define _INCLUDE_VECTORMATH

#include "cuda-samples/Common/helper_math.h" //math functions for cuda vector types, from cuda samples

#ifndef FLOAT_MIN
#define FLOAT_MIN 1.17549e-38
#endif

#ifndef FLOAT_MAX
#define FLOAT_MAX 3.40282e+38
#endif

#ifndef FLOAT_LOWEST
#define FLOAT_LOWEST - FLOAT_MAX
#endif

/* Extensions and missing functions of helper_math*/
inline __host__ __device__ float1 make_float1(int1 a)
{
    return make_float1(float(a.x));
}
inline __host__ __device__ float1 make_float1(uint1 a)
{
    return make_float1(float(a.x));
}

inline __host__ __device__ float3 operator<(const float3 v, const float b)
{
	return make_float3(v.x<b, v.y<b, v.z<b);
}
inline __host__ __device__ float3 operator<(const float b, const float3 v)
{
	return make_float3(b<v.x, b<v.y, b<v.z);
}

inline __host__ __device__ float1 operator<(const float1 v, const float1 b)
{
	return make_float1(v.x<b.x);
}
inline __host__ __device__ float2 operator<(const float2 v, const float2 b)
{
	return make_float2(v.x<b.x, v.y<b.y);
}
inline __host__ __device__ float3 operator<(const float3 v, const float3 b)
{
	return make_float3(v.x<b.x, v.y<b.y, v.z<b.z);
}
inline __host__ __device__ float4 operator<(const float4 v, const float4 b)
{
	return make_float4(v.x<b.x, v.y<b.y, v.z<b.z, v.w<b.w);
}

inline __host__ __device__ int3 operator<(const int3 v, const int32_t b)
{
	return make_int3(v.x<b, v.y<b, v.z<b);
}
inline __host__ __device__ int3 operator<(const int32_t b, const int3 v)
{
	return make_int3(b<v.x, b<v.y, b<v.z);
}
inline __host__ __device__ int3 operator<(const int3 v, const int3 b)
{
	return make_int3(v.x<b.x, v.y<b.y, v.z<b.z);
}

inline __host__ __device__ int3 operator%(const int3 v, const int3 m){
	return make_int3(v.x%m.x, v.y%m.y, v.z%m.z);
}
inline __host__ __device__ int3 operator%(const int3 v, const int32_t m){
	return make_int3(v.x%m, v.y%m, v.z%m);
}

//missing functions for 1-element vectors

inline __host__ __device__ float1 operator+(const float1 a, const float1 b)
{
	return make_float1(a.x + b.x);
}
inline __host__ __device__ float1 operator-(const float1 a, const float1 b)
{
	return make_float1(a.x - b.x);
}
inline __host__ __device__ float4 operator-(float b, float4 a)
{
    return make_float4(b - a.x, b - a.y, b - a.z, b - a.w);
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

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float1 floorf(float1 v)
{
	return make_float1(floorf(v.x));
}

////////////////////////////////////////////////////////////////////////////////
// ceil
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float1 ceilf(float1 v)
{
    return make_float1(ceilf(v.x));
}
inline __host__ __device__ float2 ceilf(float2 v)
{
    return make_float2(ceilf(v.x), ceilf(v.y));
}
inline __host__ __device__ float3 ceilf(float3 v)
{
    return make_float3(ceilf(v.x), ceilf(v.y), ceilf(v.z));
}
inline __host__ __device__ float4 ceilf(float4 v)
{
    return make_float4(ceilf(v.x), ceilf(v.y), ceilf(v.z), ceilf(v.w));
}

inline __host__ __device__ float1 fracf(float1 v)
{
	return make_float1(fracf(v.x));
}

inline __device__ __host__ float1 lerp(float1 a, float1 b, float t)
{
	return a + t*(b-a);
}


namespace vmath{

template<typename T, typename A>
__device__ __host__ inline T make_cudaFloat(const A a);
template<>
__device__ __host__ inline float1 make_cudaFloat(const float a){return make_float1(a);}
template<>
__device__ __host__ inline float2 make_cudaFloat(const float a){return make_float2(a);}
template<>
__device__ __host__ inline float3 make_cudaFloat(const float a){return make_float3(a);}
template<>
__device__ __host__ inline float4 make_cudaFloat(const float a){return make_float4(a);}
template<>
__device__ __host__ inline float1 make_cudaFloat(const int1 a){return make_float1(a);}
template<>
__device__ __host__ inline float2 make_cudaFloat(const int2 a){return make_float2(a);}
template<>
__device__ __host__ inline float3 make_cudaFloat(const int3 a){return make_float3(a);}
template<>
__device__ __host__ inline float4 make_cudaFloat(const int4 a){return make_float4(a);}

/*Vector functions*/

constexpr __host__ __device__ float sum(const float1 v){
	return v.x;
}
constexpr __host__ __device__ size_t sum(const int1 v){
	return v.x;
}
constexpr __host__ __device__ float sum(const float2 v){
	return v.x + v.y;
}
constexpr __host__ __device__ size_t sum(const int2 v){
	return v.x + v.y;
}
constexpr __host__ __device__ float sum(const float3 v){
	return v.x + v.y + v.z;
}
constexpr __host__ __device__ size_t sum(const int3 v){
	return v.x + v.y + v.z;
}
constexpr __host__ __device__ float sum(const float4 v){
	return v.x + v.y + v.z + v.w;
}
constexpr __host__ __device__ size_t sum(const int4 v){
	return v.x + v.y + v.z + v.w;
}

constexpr __host__ __device__ float prod(const float1 v){
	return v.x;
}
constexpr __host__ __device__ size_t prod(const int1 v){
	return v.x;
}
constexpr __host__ __device__ float prod(const float2 v){
	return v.x * v.y;
}
constexpr __host__ __device__ size_t prod(const int2 v){
	return v.x * v.y;
}
constexpr __host__ __device__ float prod(const float3 v){
	return v.x * v.y * v.z;
}
constexpr __host__ __device__ size_t prod(const int3 v){
	return v.x * v.y * v.z;
}
constexpr __host__ __device__ float prod(const float4 v){
	return v.x * v.y * v.z * v.w;
}
constexpr __host__ __device__ size_t prod(const int4 v){
	return v.x * v.y * v.z * v.w;
}

inline __host__ __device__ float step(const float x, const float y){
	return(float)x<y;
}
inline __host__ __device__ float1 step(const float1 x, const float1 y){
	return x<y;
}
inline __host__ __device__ float2 step(const float2 x, const float2 y){
	return x<y;
}
inline __host__ __device__ float3 step(const float3 x, const float3 y){
	return x<y;
}
inline __host__ __device__ float4 step(const float4 x, const float4 y){
	return x<y;
}

template<typename T>
inline __device__ __host__ T lerp(T a, T b, T t)
{
	return a + t*(b-a);
}

//https://stackoverflow.com/questions/14997165/fastest-way-to-get-a-positive-modulo-in-c-c
template<typename T, typename M>
constexpr __host__ __device__ T positivemod(const T v, const M m){
//	T mod = v%m;
//	T isNegative = v<0;
//	return mod + m*isNegative;
	return (v%m) + m*(v<0);
}

inline __host__ __device__ int3 pmod(const int3 v, const int3 m){
	return make_int3(
		(v.x%m.x) + m.x*(v.x<0),
		(v.y%m.y) + m.y*(v.y<0),
		(v.z%m.z) + m.z*(v.z<0)
	);
}
inline __host__ __device__ int3 pmod(const int3 v, const int32_t m){
	return make_int3(
		(v.x%m) + m*(v.x<0),
		(v.y%m) + m*(v.y<0),
		(v.z%m) + m*(v.z<0)
	);
}


/* Matix types, column-major*/
typedef struct float3x3{
	float3 c0,c1,c2;
} float3x3;

typedef struct __align__(16) float4x4{
	float4 c0,c1,c2,c3;
} float4x4;

inline __host__ __device__ float4x4 make_float4x4(
	const float m00, const float m01, const float m02, const float m03,
	const float m10, const float m11, const float m12, const float m13,
	const float m20, const float m21, const float m22, const float m23,
	const float m30, const float m31, const float m32, const float m33){
	return (float4x4){
		.c0 = make_float4(m00, m01, m02, m03),
		.c1 = make_float4(m10, m11, m12, m13),
		.c2 = make_float4(m20, m21, m22, m23),
		.c3 = make_float4(m30, m31, m32, m33)
	};
}
inline __host__ __device__ float4x4 make_float4x4(const float *arr){
	return make_float4x4(
		arr[0], arr[1], arr[2], arr[3],
		arr[4], arr[5], arr[6], arr[7],
		arr[8], arr[9], arr[10], arr[11],
		arr[12], arr[13], arr[14], arr[15]
	);
}
inline __host__ __device__ float4x4 make_float4x4(const float4 c0,const float4 c1, const float4 c2, const float4 c3){
	return (float4x4){.c0 = c0,.c1 = c1, .c2 = c2, .c3 = c3};
}
inline __host__ __device__ float4x4 make_float4x4(const float4 diag){
	return make_float4x4(
		diag.x, 0.f, 0.f, 0.f,
		0.f, diag.y, 0.f, 0.f,
		0.f, 0.f, diag.z, 0.f,
		0.f, 0.f, 0.f, diag.w
	);
}
inline __host__ __device__ float4x4 make_float4x4(const float diag){
	return make_float4x4(
		diag, 0.f, 0.f, 0.f,
		0.f, diag, 0.f, 0.f,
		0.f, 0.f, diag, 0.f,
		0.f, 0.f, 0.f, diag
	);
}

/* Matix functions*/
inline __host__ __device__ float4x4 transpose(const float4x4 m){
	return make_float4x4(
		m.c0.x, m.c1.x, m.c2.x, m.c3.x,
		m.c0.y, m.c1.y, m.c2.y, m.c3.y,
		m.c0.z, m.c1.z, m.c2.z, m.c3.z,
		m.c0.w, m.c1.w, m.c2.w, m.c3.w
	);
}

inline __host__ __device__ float4x4 operator*(const float4x4 m, const float s){
	return make_float4x4(
		m.c0 * s,
		m.c1 * s,
		m.c2 * s,
		m.c3 * s
	);
}

inline __host__ float4x4 inverse(const float4x4 m){
	//from GLM
	float Coef00 = m.c2.z * m.c3.w - m.c3.z * m.c2.w; //m[2][2] * m[3][3] - m[3][2] * m[2][3];
	float Coef02 = m.c1.z * m.c3.w - m.c3.z * m.c1.w; //m[1][2] * m[3][3] - m[3][2] * m[1][3];
	float Coef03 = m.c1.z * m.c2.w - m.c2.z * m.c1.w; //m[1][2] * m[2][3] - m[2][2] * m[1][3];

	float Coef04 = m.c2.y * m.c3.w - m.c3.y * m.c2.w; //m[2][1] * m[3][3] - m[3][1] * m[2][3];
	float Coef06 = m.c1.y * m.c3.w - m.c3.y * m.c1.w; //m[1][1] * m[3][3] - m[3][1] * m[1][3];
	float Coef07 = m.c1.y * m.c2.w - m.c2.y * m.c1.w; //m[1][1] * m[2][3] - m[2][1] * m[1][3];

	float Coef08 = m.c2.y * m.c3.z - m.c3.y * m.c2.z; //m[2][1] * m[3][2] - m[3][1] * m[2][2];
	float Coef10 = m.c1.y * m.c3.z - m.c3.y * m.c1.z; //m[1][1] * m[3][2] - m[3][1] * m[1][2];
	float Coef11 = m.c1.y * m.c2.z - m.c2.y * m.c1.z; //m[1][1] * m[2][2] - m[2][1] * m[1][2];

	float Coef12 = m.c2.x * m.c3.w - m.c3.x * m.c2.w; //m[2][0] * m[3][3] - m[3][0] * m[2][3];
	float Coef14 = m.c1.x * m.c3.w - m.c3.x * m.c1.w; //m[1][0] * m[3][3] - m[3][0] * m[1][3];
	float Coef15 = m.c1.x * m.c2.w - m.c2.x * m.c1.w; //m[1][0] * m[2][3] - m[2][0] * m[1][3];

	float Coef16 = m.c2.x * m.c3.z - m.c3.x * m.c2.z; //m[2][0] * m[3][2] - m[3][0] * m[2][2];
	float Coef18 = m.c1.x * m.c3.z - m.c3.x * m.c1.z; //m[1][0] * m[3][2] - m[3][0] * m[1][2];
	float Coef19 = m.c1.x * m.c2.z - m.c2.x * m.c1.z; //m[1][0] * m[2][2] - m[2][0] * m[1][2];

	float Coef20 = m.c2.x * m.c3.y - m.c3.x * m.c2.y; //m[2][0] * m[3][1] - m[3][0] * m[2][1];
	float Coef22 = m.c1.x * m.c3.y - m.c3.x * m.c1.y; //m[1][0] * m[3][1] - m[3][0] * m[1][1];
	float Coef23 = m.c1.x * m.c2.y - m.c2.x * m.c1.y; //m[1][0] * m[2][1] - m[2][0] * m[1][1];

	float4 Fac0 = make_float4(Coef00, Coef00, Coef02, Coef03);
	float4 Fac1 = make_float4(Coef04, Coef04, Coef06, Coef07);
	float4 Fac2 = make_float4(Coef08, Coef08, Coef10, Coef11);
	float4 Fac3 = make_float4(Coef12, Coef12, Coef14, Coef15);
	float4 Fac4 = make_float4(Coef16, Coef16, Coef18, Coef19);
	float4 Fac5 = make_float4(Coef20, Coef20, Coef22, Coef23);

	float4 Vec0 = make_float4(m.c1.x, m.c0.x, m.c0.x, m.c0.x); //m[1][0], m[0][0], m[0][0], m[0][0]);
	float4 Vec1 = make_float4(m.c1.y, m.c0.y, m.c0.y, m.c0.y); //m[1][1], m[0][1], m[0][1], m[0][1]);
	float4 Vec2 = make_float4(m.c1.z, m.c0.z, m.c0.z, m.c0.z); //m[1][2], m[0][2], m[0][2], m[0][2]);
	float4 Vec3 = make_float4(m.c1.w, m.c0.w, m.c0.w, m.c0.w); //m[1][3], m[0][3], m[0][3], m[0][3]);

	float4 Inv0 = Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2;
	float4 Inv1 = Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4;
	float4 Inv2 = Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5;
	float4 Inv3 = Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5;

	float4 SignA = make_float4(+1, -1, +1, -1);
	float4 SignB = make_float4(-1, +1, -1, +1);
	float4x4 Inverse = make_float4x4(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB);

	float4 Row0 = make_float4(Inverse.c0.x, Inverse.c1.x, Inverse.c2.x, Inverse.c3.x);

	float Dot1 = dot(m.c0, Row0); //(m[0] * Row0);
	//T Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

	float OneOverDeterminant = 1.f / Dot1;

	return Inverse * OneOverDeterminant;
}


inline __host__ __device__ float4 matmul(const float4x4 m, const float4 v){
	return m.c0*v.x + m.c1*v.y + m.c2*v.z + m.c3*v.w;
}
inline __host__ __device__ float4 matmul(const float4 v, const float4x4 m){
	return make_float4(dot(m.c0, v), dot(m.c1, v), dot(m.c2, v), dot(m.c3, v));
}
inline __host__ __device__ float4x4 matmul(const float4x4 m1, const float4x4 m2){
	return make_float4x4(
		matmul(m1, m2.c0),
		matmul(m1, m2.c1),
		matmul(m1, m2.c2),
		matmul(m1, m2.c3)
	);
}

/* Vector swizzle */

#define SW2(v,a,b, type) make_##type##2(v.a, v.b)
#define SW2_F(v,a,b) SW2(v,a,b,float)
#define SW3(v,a,b,c, type) make_##type##3(v.a, v.b, v.c)
#define SW3_F(v,a,b,c) SW3(v,a,b,c,float)
#define SW4(v,a,b,c,d, type) make_##type##4(v.a, v.b, v.c, v.d)
#define SW4_F(v,a,b,c,d) SW4(v,a,b,c,d,float)


#define SWIZZLE2(a,b,T,type) inline __host__ __device__ type##2 a##b(const T v){ return make_##type##2(v.a, v.b);}
#define SWIZZLE3(a,b,c,T,type) inline __host__ __device__ type##3 a##b##c(const T v){ return make_##type##3(v.a, v.b, v.c);}
#define SWIZZLE4(a,b,c,d,T,type) inline __host__ __device__ type##4 a##b##c##d(const T v){ return make_##type##4(v.a, v.b, v.c, v.d);}

#define SWIZZLE2_TYPE(T) \
	SWIZZLE2(x,x,T##1,T) \
	SWIZZLE2(x,x,T##2,T) \
	SWIZZLE2(x,x,T##3,T) \
	SWIZZLE2(x,x,T##4,T) \
	SWIZZLE2(x,y,T##2,T) \
	SWIZZLE2(x,y,T##3,T) \
	SWIZZLE2(x,y,T##4,T) \
	SWIZZLE2(x,z,T##3,T) \
	SWIZZLE2(x,z,T##4,T) \
	SWIZZLE2(x,w,T##4,T) \
	\
	SWIZZLE2(y,x,T##2,T) \
	SWIZZLE2(y,x,T##3,T) \
	SWIZZLE2(y,x,T##4,T) \
	SWIZZLE2(y,y,T##2,T) \
	SWIZZLE2(y,y,T##3,T) \
	SWIZZLE2(y,y,T##4,T) \
	SWIZZLE2(y,z,T##3,T) \
	SWIZZLE2(y,z,T##4,T) \
	SWIZZLE2(y,w,T##4,T) \
	\
	SWIZZLE2(z,x,T##3,T) \
	SWIZZLE2(z,x,T##4,T) \
	SWIZZLE2(z,y,T##3,T) \
	SWIZZLE2(z,y,T##4,T) \
	SWIZZLE2(z,z,T##3,T) \
	SWIZZLE2(z,z,T##4,T) \
	SWIZZLE2(z,w,T##4,T) \
	\
	SWIZZLE2(w,x,T##4,T) \
	SWIZZLE2(w,y,T##4,T) \
	SWIZZLE2(w,z,T##4,T) \
	SWIZZLE2(w,w,T##4,T)

#define SWIZZLE3_TYPE(T) \
	SWIZZLE3(x,x,x,T##1,T) \
	SWIZZLE3(x,x,x,T##2,T) \
	SWIZZLE3(x,x,x,T##3,T) \
	SWIZZLE3(x,x,x,T##4,T) \
	SWIZZLE3(x,x,y,T##2,T) \
	SWIZZLE3(x,x,y,T##3,T) \
	SWIZZLE3(x,x,y,T##4,T) \
	SWIZZLE3(x,x,z,T##3,T) \
	SWIZZLE3(x,x,z,T##4,T) \
	SWIZZLE3(x,x,w,T##4,T) \
	\
	SWIZZLE3(x,y,x,T##2,T) \
	SWIZZLE3(x,y,x,T##3,T) \
	SWIZZLE3(x,y,x,T##4,T) \
	SWIZZLE3(x,y,y,T##2,T) \
	SWIZZLE3(x,y,y,T##3,T) \
	SWIZZLE3(x,y,y,T##4,T) \
	SWIZZLE3(x,y,z,T##3,T) \
	SWIZZLE3(x,y,z,T##4,T) \
	SWIZZLE3(x,y,w,T##4,T) \
	\
	SWIZZLE3(x,z,x,T##3,T) \
	SWIZZLE3(x,z,x,T##4,T) \
	SWIZZLE3(x,z,y,T##3,T) \
	SWIZZLE3(x,z,y,T##4,T) \
	SWIZZLE3(x,z,z,T##3,T) \
	SWIZZLE3(x,z,z,T##4,T) \
	SWIZZLE3(x,z,w,T##4,T) \
	\
	SWIZZLE3(x,w,x,T##4,T) \
	SWIZZLE3(x,w,y,T##4,T) \
	SWIZZLE3(x,w,z,T##4,T) \
	SWIZZLE3(x,w,w,T##4,T) \
	\
	\
	SWIZZLE3(y,x,y,T##2,T) \
	SWIZZLE3(y,x,y,T##3,T) \
	SWIZZLE3(y,x,y,T##4,T) \
	SWIZZLE3(y,x,z,T##3,T) \
	SWIZZLE3(y,x,z,T##4,T) \
	SWIZZLE3(y,x,w,T##4,T) \
	\
	SWIZZLE3(y,y,x,T##2,T) \
	SWIZZLE3(y,y,x,T##3,T) \
	SWIZZLE3(y,y,x,T##4,T) \
	SWIZZLE3(y,y,y,T##2,T) \
	SWIZZLE3(y,y,y,T##3,T) \
	SWIZZLE3(y,y,y,T##4,T) \
	SWIZZLE3(y,y,z,T##3,T) \
	SWIZZLE3(y,y,z,T##4,T) \
	SWIZZLE3(y,y,w,T##4,T) \
	\
	SWIZZLE3(y,z,x,T##3,T) \
	SWIZZLE3(y,z,x,T##4,T) \
	SWIZZLE3(y,z,y,T##3,T) \
	SWIZZLE3(y,z,y,T##4,T) \
	SWIZZLE3(y,z,z,T##3,T) \
	SWIZZLE3(y,z,z,T##4,T) \
	SWIZZLE3(y,z,w,T##4,T) \
	\
	SWIZZLE3(y,w,x,T##4,T) \
	SWIZZLE3(y,w,y,T##4,T) \
	SWIZZLE3(y,w,z,T##4,T) \
	SWIZZLE3(y,w,w,T##4,T) \
	\
	\
	SWIZZLE3(z,x,y,T##3,T) \
	SWIZZLE3(z,x,y,T##4,T) \
	SWIZZLE3(z,x,z,T##3,T) \
	SWIZZLE3(z,x,z,T##4,T) \
	SWIZZLE3(z,x,w,T##4,T) \
	\
	SWIZZLE3(z,y,x,T##3,T) \
	SWIZZLE3(z,y,x,T##4,T) \
	SWIZZLE3(z,y,y,T##3,T) \
	SWIZZLE3(z,y,y,T##4,T) \
	SWIZZLE3(z,y,z,T##3,T) \
	SWIZZLE3(z,y,z,T##4,T) \
	SWIZZLE3(z,y,w,T##4,T) \
	\
	SWIZZLE3(z,z,x,T##3,T) \
	SWIZZLE3(z,z,x,T##4,T) \
	SWIZZLE3(z,z,y,T##3,T) \
	SWIZZLE3(z,z,y,T##4,T) \
	SWIZZLE3(z,z,z,T##3,T) \
	SWIZZLE3(z,z,z,T##4,T) \
	SWIZZLE3(z,z,w,T##4,T) \
	\
	SWIZZLE3(z,w,x,T##4,T) \
	SWIZZLE3(z,w,y,T##4,T) \
	SWIZZLE3(z,w,z,T##4,T) \
	SWIZZLE3(z,w,w,T##4,T) \
	\
	\
	SWIZZLE3(w,x,y,T##4,T) \
	SWIZZLE3(w,x,z,T##4,T) \
	SWIZZLE3(w,x,w,T##4,T) \
	\
	SWIZZLE3(w,y,x,T##4,T) \
	SWIZZLE3(w,y,y,T##4,T) \
	SWIZZLE3(w,y,z,T##4,T) \
	SWIZZLE3(w,y,w,T##4,T) \
	\
	SWIZZLE3(w,z,x,T##4,T) \
	SWIZZLE3(w,z,y,T##4,T) \
	SWIZZLE3(w,z,z,T##4,T) \
	SWIZZLE3(w,z,w,T##4,T) \
	\
	SWIZZLE3(w,w,x,T##4,T) \
	SWIZZLE3(w,w,y,T##4,T) \
	SWIZZLE3(w,w,z,T##4,T) \
	SWIZZLE3(w,w,w,T##4,T)

/*
#define SWIZZLE4_TYPE(T) \
	SWIZZLE4(x,x,x,x,T##1,T) \
	SWIZZLE4(x,x,x,x,T##2,T) \
	SWIZZLE4(x,x,x,x,T##3,T) \
	SWIZZLE4(x,x,x,x,T##4,T) 
...
*/

#undef SWIZZLE2_TYPE
#undef SWIZZLE3_TYPE

#undef SWIZZLE2
#undef SWIZZLE3
#undef SWIZZLE4

} //END vmath

#endif //_INCLUDE_VECTORMATH