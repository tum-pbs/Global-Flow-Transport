
#pragma once

#ifndef KERNEL_SETUP
#define KERNEL_SETUP

#define UG_PTR __restrict__ //Unique Global pointer
//#define UG_PTR

//total maximum block size (x*y*z) is 512 (1024, depending on architecture)
//these are NOT reversed when using REVERSE_THREAD_AXIS_ORDER
#ifndef BLOCK_SIZE_X
#error "BLOCK_SIZE_X needs to be defined"
#endif

#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 1
#endif

#ifndef BLOCK_SIZE_Z
#define BLOCK_SIZE_Z 1
#endif

#define BLOCK_SIZE BLOCK_SIZE_X*BLOCK_SIZE_Y*BLOCK_SIZE_Z
#define BLOCK_DIMS BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z
#define MAKE_BLOCK_SIZE glm::ivec3 blockSize = glm::ivec3(BLOCK_DIMS)#


//https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
inline int32_t ceil_div(int32_t x,int32_t y){return x/y + (x%y!=0);} //(x+y-1)/y ?

//#define REVERSE_THREAD_AXIS_ORDER
//it might be faster to reverse the thread axis order, depending on memory access patterns
#ifdef REVERSE_THREAD_AXIS_ORDER
#define GRID_DIMS(size) ceil_div(size.z, BLOCK_SIZE_X),ceil_div(size.y, BLOCK_SIZE_Y),ceil_div(size.x, BLOCK_SIZE_Z)
#define GRID_DIMS_BLOCK(size, blockSize) ceil_div(size.z, blockSize.x),ceil_div(size.y, blockSize.y),ceil_div(size.x, blockSize.z)
#else
#define GRID_DIMS(size) ceil_div(size.x, BLOCK_SIZE_X),ceil_div(size.y, BLOCK_SIZE_Y),ceil_div(size.z, BLOCK_SIZE_Z)
#define GRID_DIMS_BLOCK(size, blockSize) ceil_div(size.x, blockSize.x),ceil_div(size.y, blockSize.y),ceil_div(size.z, blockSize.z)
#endif

//#include "render_errors.hpp"

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
  if (err == cudaSuccess) return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;
  exit(10);
}
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
//#define CUDA_CHECK_RETURN_EXIT(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

//--- Logging and Profiling ---

#define LOG_V3_XYZ(v) "(" << v.x << "," << v.y << "," << v.z << ")"
#define LOG_V4_XYZW(v) "(" << v.x << "," << v.y << "," << v.z  << "," << v.w << ")"
#define LOG_M44_COL(m) "[" << m[0][0] << "," << m[1][0] << "," << m[2][0] << "," << m[3][0] << ";\n" \
						   << m[0][1] << "," << m[1][1] << "," << m[2][1] << "," << m[3][1] << ";\n" \
						   << m[0][2] << "," << m[1][2] << "," << m[2][2] << "," << m[3][2] << ";\n" \
						   << m[0][3] << "," << m[1][3] << "," << m[2][3] << "," << m[3][3] << "]"

#ifdef LOG
#undef LOG
#endif
#ifdef LOGGING
#define LOG(msg) std::cout << __FILE__ << "[" << __LINE__ << "]: " << msg << std::endl
#else
#define LOG(msg)
#endif

#ifdef PROFILING
#include <chrono>
//no support for nesting for now.
auto start = std::chrono::high_resolution_clock::now();
__host__ void beginSample(){start = std::chrono::high_resolution_clock::now();}
__host__ void endSample(std::string name){
	const auto end = std::chrono::high_resolution_clock::now();
	std::cout << "\'" << name << "\': " << (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() * 1e-6) << "ms" << std::endl;
}
#define BEGIN_SAMPLE beginSample()
#define END_SAMPLE(name) endSample(name)
#else
#define BEGIN_SAMPLE
#define END_SAMPLE(name)
#endif

//--- Vector and Matrix helpers

#define MAT4_FROM_ARRAY(name, arr) glm::mat4 name(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14], arr[15])
#define IVEC3_FROM_ARRAY(name, arr) glm::ivec3 name((int32_t) (arr)[0], (int32_t) (arr)[1], (int32_t) (arr)[2])
#define IVEC4_FROM_ARRAY(name, arr) glm::ivec4 name((int32_t) (arr)[0], (int32_t) (arr)[1], (int32_t) (arr)[2], (int32_t) (arr)[3])
#define IVEC3_INVERT(vec) glm::ivec3(vec.z,vec.y,vec.x) //glm::swizzle<glm::Z,gml::Y,glm::X>(vec)

#define EXPAND_VECTOR3(v) v.x, v.y, v.z
#define EXPAND_VECTOR3_REVERSE(v) v.z, v.y, v.x


//--- Index Calculations ---

//returns the global 3D index of the current thread as vector.
__device__ inline glm::ivec3 globalThreadIdx3D(){
#ifdef REVERSE_THREAD_AXIS_ORDER
	return glm::ivec3(blockIdx.z*blockDim.z + threadIdx.z, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.x*blockDim.x + threadIdx.x);
#else
	return glm::ivec3(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z);
#endif
}
#define MAKE_GLOBAL_INDEX const glm::ivec3 globalIdx = globalThreadIdx3D()

__device__ inline glm::ivec3 globalThreadIdx3DOverlapped(const glm::ivec3 overlap){
#ifdef REVERSE_THREAD_AXIS_ORDER
	return glm::ivec3(blockIdx.z*(blockDim.z-overlap.z) + threadIdx.z, blockIdx.y*(blockDim.y-overlap.y) + threadIdx.y, blockIdx.x*(blockDim.x-overlap.x) + threadIdx.x);
#else
	return glm::ivec3(blockIdx.x*(blockDim.x-overlap.x) + threadIdx.x, blockIdx.y*(blockDim.y-overlap.y) + threadIdx.y, blockIdx.z*(blockDim.z-overlap.z) + threadIdx.z);
#endif
}

//for bounds checking
#define CHECK_BOUNDS_SV3S(l, c1, v, c2, u) l c1 v.x && v.x c2 u && l c1 v.y && v.y c2 u && l c1 v.z && v.z c2 u
#define CHECK_BOUNDS_SV3V3(l, c1, v, c2, u) l c1 v.x && v.x c2 u.x && l c1 v.y && v.y c2 u.y && l c1 v.z && v.z c2 u.z
#define CHECK_BOUNDS_V3V3V3(l, c1, v, c2, u) l.x c1 v.x && v.x c2 u.x && l.x c1 v.y && v.y c2 u.y && l.x c1 v.z && v.z c2 u.z
#define CHECK_BOUND_SV3(v1, c, v2) v1 c v2.x && v1 c v2.y && v1 c v2.z
#define CHECK_BOUND_V3S(v1, c, v2) v1.x c v2 && v1.y c v2 && v1.z c v2
#define CHECK_BOUND_V3V3(v1, c, v2) v1.x c v2.x && v1.y c v2.y && v1 c v2.z
template<typename T, typename D>
__device__ inline bool isInDimensions(const T position, const D dimensions){
	return (position.x < dimensions.x && position.y < dimensions.y && position.z < dimensions.z);
}
template<typename T, typename D>
__device__ inline bool isInDimensions(const T x, const T y, const T z, const D dimensions){
	return (x < dimensions.x && y < dimensions.y && z < dimensions.z);
}
template<typename V3>
__device__ inline bool isNonNegative(const V3 position){
	//return (position.x >=0 && position.y >=0 && position.z >=0);
	return CHECK_BOUND_SV3(0, <=, position);
}


//--- Common Constant Buffer Types

#ifdef CBUF_DIMENSIONS
struct Dimensions{
	glm::ivec3 input;
#ifdef CBUF_DIMENSIONS_INVERSE
	glm::vec3 input_inv;
#endif
	glm::ivec3 output;
#ifdef CBUF_DIMENSIONS_INVERSE
	glm::vec3 output_inv;
#endif
#ifdef CBUF_DIMENSIONS_BATCH
	int32_t batch;
#endif
#ifdef CBUF_DIMENSIONS_CHANNEL
	int32_t channel;
#endif
};
__constant__ Dimensions c_dimensions;

inline glm::ivec3 dimensionsFromGridShape(const long long int* shape, uint32_t offset=1){
	return glm::ivec3(shape[offset+2], shape[offset+1], shape[offset]); //default offset 1: NDHWC (zyx) -> WHD (xyz)
}

__host__ inline void setDimensions(Dimensions& dims, const long long int* input_shape, const long long int* output_shape){
	memset(&dims, 0, sizeof(Dimensions));
	dims.input = dimensionsFromGridShape(input_shape);//swizzle from z,y,x to x,y,z
	dims.output = dimensionsFromGridShape(output_shape);
#ifdef CBUF_DIMENSIONS_INVERSE
	dims.input_inv = 1.f/glm::vec3(dims.input);
	dims.output_inv = 1.f/glm::vec3(dims.output);
#endif
#ifdef CBUF_DIMENSIONS_BATCH
	dims.batch = input_shape[0];
#endif
#ifdef CBUF_DIMENSIONS_CHANNEL
	dims.channel = input_shape[4];
#endif
	cudaError_t err = cudaMemcpyToSymbol(c_dimensions, &dims, sizeof(Dimensions));
	if(err!=cudaSuccess){
		std::cerr << "Error " << cudaGetErrorString(err) << " ("<< err << ") while setting c_dimensions constant buffer." << std::endl;
		exit(1);
	}
}
#endif //CBUF_DIMENSIONS


#endif //KERNEL_SETUP