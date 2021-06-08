
#pragma once

#ifndef _INCLUDE_DIMENSIONS_2
#define _INCLUDE_DIMENSIONS_2

#include"vectormath.hpp"

struct Dimensions{
	int3 input;
#ifdef CBUF_DIMENSIONS_INVERSE
	float3 input_inv;
#endif
	int3 output;
#ifdef CBUF_DIMENSIONS_INVERSE
	float3 output_inv;
#endif
#ifdef CBUF_DIMENSIONS_BATCH
	int32_t batch;
#endif
#ifdef CBUF_DIMENSIONS_CHANNEL
	int32_t channel;
#endif
};
__constant__ Dimensions c_dimensions;

inline int3 dimensionsFromGridShape(const long long int* shape, uint32_t offset=1){
	return make_int3(shape[offset+2], shape[offset+1], shape[offset]); //default offset 1: NDHWC (zyx) -> WHD (xyz)
}

__host__ inline void setDimensions(Dimensions& dims, const long long int* input_shape, const long long int* output_shape){
	memset(&dims, 0, sizeof(Dimensions));
	dims.input = dimensionsFromGridShape(input_shape);//swizzle from z,y,x to x,y,z
	dims.output = dimensionsFromGridShape(output_shape);
#ifdef CBUF_DIMENSIONS_INVERSE
	dims.input_inv = 1.f/make_float3(dims.input);
	dims.output_inv = 1.f/make_float3(dims.output);
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


#endif //_INCLUDE_DIMENSIONS_2