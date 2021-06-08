#pragma once

#ifndef _INCLUDE_RAYMARCH_GRID
#define _INCLUDE_RAYMARCH_GRID

#include "tensorflow/core/framework/tensor_types.h"
#include "sampling_settings_v2.hpp"
#include "blending_settings.hpp"

using GPUDevice = Eigen::GpuDevice;

template<typename Device, typename T, int32_t C>
struct RaymarchGridKernel{
	void operator()(const Device& d,
		const T* input,const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* frustum, int32_t numCameras,
		const Sampling::FilterMode filterMode, const Sampling::BoundaryMode boundaryMode,
		const Blending::BlendMode blendMode, const bool globalSampling,
		T* output, const long long int* output_shape);
};

#define NORM_GRADS
#define NORM_BY_WEIGHT
enum GradSampleNorm : int32_t {NORMALIZE_GRADIENT_NONE=0, NORMALIZE_GRADIENT_BY_COUNT=1, NORMALIZE_GRADIENT_BY_WEIGHT=2};

#ifdef NORM_GRADS
	#ifdef NORM_BY_WEIGHT
		const GradSampleNorm NORMALIZE_GRADIENTS = NORMALIZE_GRADIENT_BY_WEIGHT;
		using sampleCount_t = float; 
		#define TFsampleCount_t DT_FLOAT
	#undef NORM_BY_WEIGHT
	#else //by count
		const GradSampleNorm NORMALIZE_GRADIENTS = NORMALIZE_GRADIENT_BY_COUNT;
		using sampleCount_t = uint32_t;
		#define TFsampleCount_t DT_UINT32
	#endif
#undef NORM_GRADS
#else
	using sampleCount_t = void;
	#define TFsampleCount_t DT_UINT8
	const GradSampleNorm NORMALIZE_GRADIENTS = NORMALIZE_GRADIENT_NONE
#endif
	

template<typename Device, typename T, int32_t C>
struct RaymarchGridGradKernel{
	void operator()(const Device& d,
		const T* input, T* inputGrads, T* sampleBuffer, sampleCount_t* sampleCounter, const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* frustum, int32_t numCameras,
		const Sampling::FilterMode filterMode, const Sampling::BoundaryMode boundaryMode,
		const Blending::BlendMode blendMode, const bool globalSampling,
		const T* output, const T* outputGrads, const long long int* output_shape);
};


#endif //_INCLUDE_RAYMARCH_GRID