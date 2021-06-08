
#include <cuda_runtime.h>
//#include "cuda-samples/Common/helper_cuda.h"
#include <iostream>
#include <string>

#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

#include "vectormath_helper.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/vec_swizzle.hpp>

#include "vector_io.hpp"

//defines for kernel setup
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8

#define CBUF_DIMENSIONS
//#define CBUF_DIMENSIONS_INVERSE

//#define LOGGING
//#define PROFILING

#include "kernel_setup.hpp"
#include "render_errors.hpp"

#include "blending.hpp"



template<typename T, typename BLEND, bool KeepDims>
//2D groups moving along z of 3D input and blending to 2D output, writing the output after each step
__global__ void kBlend(const T* input, T* output){
	MAKE_GLOBAL_INDEX;
	if(isInDimensions(globalIdx, c_dimensions.output) && globalIdx.z==0){
		glm::ivec3 position = globalIdx;
		const int32_t depth = c_dimensions.input.z;
		
		T acc = {0};
		for(int32_t i=0;i<depth;++i){
			position.z = i;
			const T data = vectorIO::readVectorType3D<T>(position, c_dimensions.input, input);
			acc = BLEND::blend(acc, data); //blend<T,BM>(acc, data);
			if(KeepDims) vectorIO::writeVectorType3D<T>(acc, position, c_dimensions.output, output);
		}
		if(!KeepDims) vectorIO::writeVectorType3D<T>(acc, globalIdx, c_dimensions.output, output);
	}
}


template<typename T, typename BLEND, bool KeepDims>
//2D groups moving along z of 3D input and blending to 2D output, writing the output after each step
__global__ void kBlendGrad(const T* input, T* input_grads, const T* output, const T* output_grads){
	MAKE_GLOBAL_INDEX;
	if(isInDimensions(globalIdx, c_dimensions.output) && globalIdx.z==0){
		glm::ivec3 position = globalIdx;
		const int32_t depth = c_dimensions.input.z;
		
		T grads = {0};
		T prev_out = {0};
		if(!KeepDims){
			grads = vectorIO::readVectorType3D<T>(position, c_dimensions.output, output_grads);
			prev_out = vectorIO::readVectorType3D<T>(position, c_dimensions.output, output);
		}
		for(int32_t i=depth-1;i>=0;--i){
			position.z = i;
			if(KeepDims){
				grads += vectorIO::readVectorType3D<T>(position, c_dimensions.output, output_grads);
				prev_out = vectorIO::readVectorType3D<T>(position, c_dimensions.output, output);
			}
			const T data = vectorIO::readVectorType3D<T>(position, c_dimensions.input, input);
			const T d_cellOut = BLEND::blendGradients(grads, data, prev_out); //blendGrad<T,BM>(grads, data, prev_out);
			vectorIO::writeVectorType3D<T>(d_cellOut, position, c_dimensions.input, input_grads);
		}
	}
}

template<typename T>
void ReduceGridBlendKernelLauncher(const float* _input, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		float* _output, const long long int* output_shape){
	const T* input = reinterpret_cast<const T*>(_input);
	T* output = reinterpret_cast<T*>(_output);
	
	LOG("Set dimensions");
	const size_t batchSize = input_shape[0];
	Dimensions dims;
	setDimensions(dims, input_shape, output_shape);
	
	const size_t inputSliceSizeElements = vmath::prod(dims.input);
	const size_t outputSliceSizeElements = vmath::prod(dims.output);
	
	glm::ivec3 compute_dims = dims.output;
	compute_dims.z = 1;
	const dim3 grid(GRID_DIMS(compute_dims));
	const dim3 block(BLOCK_DIMS);
	LOG("Sample " << batchSize << " grids");
	for(size_t batch=0; batch<batchSize; ++batch){
		LOG("Grid " << batch);
		BEGIN_SAMPLE;
		{
			switch(blend_mode){
				case Blending::BLEND_BEERLAMBERT:
					if(keep_dims){
						kBlend<T, Blending::BlendState<T, Blending::BLEND_BEERLAMBERT>, true><<<grid, block>>>(input, output);
					}else{
						kBlend<T, Blending::BlendState<T, Blending::BLEND_BEERLAMBERT>, false><<<grid, block>>>(input, output);
					}
					break;
				case Blending::BLEND_ALPHA:
					if(keep_dims){
						kBlend<T, Blending::BlendState<T, Blending::BLEND_ALPHA>, true><<<grid, block>>>(input, output);
					}else{
						kBlend<T, Blending::BlendState<T, Blending::BLEND_ALPHA>, false><<<grid, block>>>(input, output);
					}
					break;
				case Blending::BLEND_ALPHAADDITIVE:
					if(keep_dims){
						kBlend<T, Blending::BlendState<T, Blending::BLEND_ALPHAADDITIVE>, true><<<grid, block>>>(input, output);
					}else{
						kBlend<T, Blending::BlendState<T, Blending::BLEND_ALPHAADDITIVE>, false><<<grid, block>>>(input, output);
					}
					break;
				case Blending::BLEND_ADDITIVE:
					if(keep_dims){
						kBlend<T, Blending::BlendState<T, Blending::BLEND_ADDITIVE>, true><<<grid, block>>>(input, output);
					}else{
						kBlend<T, Blending::BlendState<T, Blending::BLEND_ADDITIVE>, false><<<grid, block>>>(input, output);
					}
					break;
				default:
					throw RenderError::RenderError("Unkown blend_mode");
			}
			//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}
		END_SAMPLE("blend kernel");
		input += inputSliceSizeElements;
		output += outputSliceSizeElements;
	}
	cudaError_t err = cudaDeviceSynchronize();
	if(err!=cudaSuccess){
		throw RenderError::CudaError(RenderError::Formatter() << 
			__FILE__ << "[" << __LINE__ << "]: Cuda error '" << cudaGetErrorString(err) << "' (" << err << ") in ReduceGridBlendKernelLauncher(). " <<
			"input shape " << LOG_V3_XYZ(dims.input) << ", output shape " << LOG_V3_XYZ(dims.output)
			);
	}
}

void GridBlendRKernelLauncher(const float* _input, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		float* _output, const long long int* output_shape){
	ReduceGridBlendKernelLauncher<float1>(_input, input_shape, blend_mode, keep_dims, _output, output_shape);
}
void GridBlendRGKernelLauncher(const float* _input, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		float* _output, const long long int* output_shape){
	ReduceGridBlendKernelLauncher<float2>(_input, input_shape, blend_mode, keep_dims, _output, output_shape);
}
void GridBlendRGBAKernelLauncher(const float* _input, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		float* _output, const long long int* output_shape){
	ReduceGridBlendKernelLauncher<float4>(_input, input_shape, blend_mode, keep_dims, _output, output_shape);
}

template<typename T>
void ReduceGridBlendGradKernelLauncher(const float* _input, float* _input_grads, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		const float* _output, const float* _output_grads, const long long int* output_shape){
	const T* input = reinterpret_cast<const T*>(_input);
	T* input_grads = reinterpret_cast<T*>(_input_grads);
	const T* output = reinterpret_cast<const T*>(_output);
	const T* output_grads = reinterpret_cast<const T*>(_output_grads);
	
	LOG("Set dimensions");
	const size_t batchSize = input_shape[0];
	Dimensions dims;
	setDimensions(dims, input_shape, output_shape);
	
	const size_t inputSliceSizeElements = vmath::prod(dims.input);
	const size_t outputSliceSizeElements = vmath::prod(dims.output);
	
	glm::ivec3 compute_dims = dims.output;
	compute_dims.z = 1;
	const dim3 grid(GRID_DIMS(compute_dims));
	const dim3 block(BLOCK_DIMS);
	LOG("Sample " << batchSize << " grids");
	for(size_t batch=0; batch<batchSize; ++batch){
		LOG("Grid " << batch);
		BEGIN_SAMPLE;
		{
			switch(blend_mode){
#define BLEND_CASE(blend) case Blending::blend: \
	if(keep_dims){kBlendGrad<T, Blending::BlendState<T, Blending::blend>, true><<<grid, block>>>(input, input_grads, output, output_grads);} \
	else{kBlendGrad<T, Blending::BlendState<T, Blending::blend>, false><<<grid, block>>>(input, input_grads, output, output_grads);} \
break;
				case Blending::BLEND_BEERLAMBERT:
					if(keep_dims){
						kBlendGrad<T, Blending::BlendState<T, Blending::BLEND_BEERLAMBERT>, true><<<grid, block>>>(input, input_grads, output, output_grads);
					}else{
						kBlendGrad<T, Blending::BlendState<T, Blending::BLEND_BEERLAMBERT>, false><<<grid, block>>>(input, input_grads, output, output_grads);
					}
					break;
				BLEND_CASE(BLEND_ALPHA)
				BLEND_CASE(BLEND_ADDITIVE)
#undef BLEND_CASE
				default:
					throw RenderError::RenderError("Unkown blend_mode");
			}
			//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}
		END_SAMPLE("blend grads kernel");
		input += inputSliceSizeElements;
		input_grads += inputSliceSizeElements;
		output += outputSliceSizeElements;
		output_grads += outputSliceSizeElements;
	}
	cudaError_t err = cudaDeviceSynchronize();
	if(err!=cudaSuccess){
		throw RenderError::CudaError(RenderError::Formatter() << 
			__FILE__ << "[" << __LINE__ << "]: Cuda error '" << cudaGetErrorString(err) << "' (" << err << ") in ReduceGridBlendGradKernelLauncher(). " <<
			"input shape " << LOG_V3_XYZ(dims.input) << ", output shape " << LOG_V3_XYZ(dims.output)
			);
	}
}

void ReduceGridBlendRGradKernelLauncher(const float* _input, float* _input_grads, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		const float* _output, const float* _output_grads, const long long int* output_shape){
	ReduceGridBlendGradKernelLauncher<float1>(_input, _input_grads, input_shape, blend_mode, keep_dims, _output, _output_grads, output_shape);
}
void ReduceGridBlendRGGradKernelLauncher(const float* _input, float* _input_grads, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		const float* _output, const float* _output_grads, const long long int* output_shape){
	ReduceGridBlendGradKernelLauncher<float2>(_input, _input_grads, input_shape, blend_mode, keep_dims, _output, _output_grads, output_shape);
}
void ReduceGridBlendRGBAGradKernelLauncher(const float* _input, float* _input_grads, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		const float* _output, const float* _output_grads, const long long int* output_shape){
	ReduceGridBlendGradKernelLauncher<float4>(_input, _input_grads, input_shape, blend_mode, keep_dims, _output, _output_grads, output_shape);
}

