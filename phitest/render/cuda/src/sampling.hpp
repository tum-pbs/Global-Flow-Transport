#pragma once

#ifndef _INCLUDE_SAMPLING
#define _INCLUDE_SAMPLING

#include<limits>
#include "sampling_settings.hpp"
#include "transformations.hpp"
#include "vectormath_helper.hpp"

namespace Sampling{

// --- Sampling position generation ---
//wrapper for transformations
//LoD computation
//input: index coordinats: float grids indices with +0.5 offset applied (handeled by sampling kernel)

template<CoordinateMode CM>
__device__ inline glm::vec4 calcLoD(const glm::vec3 idxCoords);

template<>
__device__ inline glm::vec4 calcLoD<TransformLinDepth>(const glm::vec3 idxCoords){
	const glm::vec3 d_x = (glm::xyz(IDXtoOSlinearDepth(idxCoords + glm::vec3(-1,0,0), c_dimensions.output_inv)) - \
							glm::xyz(IDXtoOSlinearDepth(idxCoords + glm::vec3(1,0,0), c_dimensions.output_inv))) *0.5f;
	const glm::vec3 d_y = (glm::xyz(IDXtoOSlinearDepth(idxCoords + glm::vec3(0,-1,0), c_dimensions.output_inv)) - \
							glm::xyz(IDXtoOSlinearDepth(idxCoords + glm::vec3(0,1,0), c_dimensions.output_inv))) *0.5f;
	const glm::vec3 d_z = (glm::xyz(IDXtoOSlinearDepth(idxCoords + glm::vec3(0,0,-1), c_dimensions.output_inv)) - \
							glm::xyz(IDXtoOSlinearDepth(idxCoords + glm::vec3(0,0,1), c_dimensions.output_inv))) *0.5f;
							
	const float px = glm::length(glm::vec3(d_x.x, d_y.x, d_z.x));
	const float py = glm::length(glm::vec3(d_x.y, d_y.y, d_z.y));
	const float pz = glm::length(glm::vec3(d_x.z, d_y.z, d_z.z));
	const float d = max(px, max(py, pz));
	const float lod = max(0.f, log2f(d));
	
	//return cell size (x,y,z) and LoD (w)
	return glm::vec4(glm::length(d_x), glm::length(d_y), glm::length(d_z), lod);
}
template<>
__device__ inline glm::vec4 calcLoD<TransformLinDepthReverse>(const glm::vec3 idxCoords){
	const glm::vec3 d_x = (OStoIDXlinearDepth(glm::vec4(idxCoords + glm::vec3(-1,0,0), 1.f), c_dimensions.input) - \
							OStoIDXlinearDepth(glm::vec4(idxCoords + glm::vec3(1,0,0), 1.f), c_dimensions.input)) *0.5f;
	const glm::vec3 d_y = (OStoIDXlinearDepth(glm::vec4(idxCoords + glm::vec3(0,-1,0), 1.f), c_dimensions.input) - \
							OStoIDXlinearDepth(glm::vec4(idxCoords + glm::vec3(0,1,0), 1.f), c_dimensions.input)) *0.5f;
	const glm::vec3 d_z = (OStoIDXlinearDepth(glm::vec4(idxCoords + glm::vec3(0,0,-1), 1.f), c_dimensions.input) - \
							OStoIDXlinearDepth(glm::vec4(idxCoords + glm::vec3(0,0,1), 1.f), c_dimensions.input)) *0.5f;
							
	const float px = glm::length(glm::vec3(d_x.x, d_y.x, d_z.x));
	const float py = glm::length(glm::vec3(d_x.y, d_y.y, d_z.y));
	const float pz = glm::length(glm::vec3(d_x.z, d_y.z, d_z.z));
	const float d = max(px, max(py, pz));
	const float lod = max(0.f, log2f(d));
	
	//return cell size (x,y,z) and LoD (w)
	return glm::vec4(glm::length(d_x), glm::length(d_y), glm::length(d_z), lod);
}


template<CoordinateMode CM, bool MIP>
__device__ inline glm::vec4 samplePos3D(const glm::vec3 idxCoords);
// Sampling position for forward transformation
template<>
__device__ inline glm::vec4 samplePos3D<Transform, false>(const glm::vec3 idxCoords){
	const glm::vec3 posOS = IDXtoOS(idxCoords, c_dimensions.output_inv);
	return glm::vec4(posOS, 0.f);
}
template<>
__device__ inline glm::vec4 samplePos3D<Transform, true>(const glm::vec3 idxCoords){
	const glm::vec3 posOS = IDXtoOS(idxCoords, c_dimensions.output_inv);
	const glm::vec3 d_x = posOS - glm::xyz(IDXtoOS(idxCoords+glm::vec3(1,0,0), c_dimensions.output_inv));
	const glm::vec3 d_y = posOS - glm::xyz(IDXtoOS(idxCoords+glm::vec3(0,1,0), c_dimensions.output_inv));
	const glm::vec3 d_z = posOS - glm::xyz(IDXtoOS(idxCoords+glm::vec3(0,0,1), c_dimensions.output_inv));
	const float px = glm::length(glm::vec3(d_x.x, d_y.x, d_z.x));
	const float py = glm::length(glm::vec3(d_x.y, d_y.y, d_z.y));
	const float pz = glm::length(glm::vec3(d_x.z, d_y.z, d_z.z));
	const float d = max(px, max(py, pz));
	const float lod = max(0.f, log2f(d));
	return glm::vec4(posOS, lod);
}
// Sampling position for backward transformation
template<>
__device__ inline glm::vec4 samplePos3D<TransformReverse, false>(const glm::vec3 idxCoords){
	const glm::vec3 posIDX = OStoIDX(glm::vec4(idxCoords, 1.f), c_dimensions.input);
	return glm::vec4(posIDX, 0.f);
}
template<>
__device__ inline glm::vec4 samplePos3D<TransformReverse, true>(const glm::vec3 idxCoords){
	const glm::vec3 posIDX = OStoIDX(glm::vec4(idxCoords, 1.f), c_dimensions.input);
	const glm::vec3 d_x = posIDX - OStoIDX(glm::vec4(idxCoords+glm::vec3(1,0,0), 1.f), c_dimensions.input);
	const glm::vec3 d_y = posIDX - OStoIDX(glm::vec4(idxCoords+glm::vec3(0,1,0), 1.f), c_dimensions.input);
	const glm::vec3 d_z = posIDX - OStoIDX(glm::vec4(idxCoords+glm::vec3(0,0,1), 1.f), c_dimensions.input);
	const float px = glm::length(d_x);
	const float py = glm::length(d_y);
	const float pz = glm::length(d_z);
	const float d = max(px, max(py, pz));
	const float lod = max(0.f, log2f(d));
	return glm::vec4(posIDX, lod);
}
// Sampling position for forward transformation with linearized depth
template<>
__device__ inline glm::vec4 samplePos3D<TransformLinDepth, false>(const glm::vec3 idxCoords){
	const glm::vec3 posOS = IDXtoOSlinearDepth(idxCoords, c_dimensions.output_inv);
	return glm::vec4(posOS, 0.f);
}
template<>
__device__ inline glm::vec4 samplePos3D<TransformLinDepth, true>(const glm::vec3 idxCoords){
	const glm::vec3 posOS = IDXtoOSlinearDepth(idxCoords, c_dimensions.output_inv);
	const glm::vec3 d_x = posOS - glm::xyz(IDXtoOSlinearDepth(idxCoords+glm::vec3(1,0,0), c_dimensions.output_inv));
	const glm::vec3 d_y = posOS - glm::xyz(IDXtoOSlinearDepth(idxCoords+glm::vec3(0,1,0), c_dimensions.output_inv));
	const glm::vec3 d_z = posOS - glm::xyz(IDXtoOSlinearDepth(idxCoords+glm::vec3(0,0,1), c_dimensions.output_inv));
	const float px = glm::length(glm::vec3(d_x.x, d_y.x, d_z.x));
	const float py = glm::length(glm::vec3(d_x.y, d_y.y, d_z.y));
	const float pz = glm::length(glm::vec3(d_x.z, d_y.z, d_z.z));
	const float d = max(px, max(py, pz));
	const float lod = max(0.f, log2f(d));
	return glm::vec4(posOS, lod);
}
// Sampling position for backward transformation with linearized depth
template<>
__device__ inline glm::vec4 samplePos3D<TransformLinDepthReverse, false>(const glm::vec3 idxCoords){
	const glm::vec3 posIDX = OStoIDXlinearDepth(glm::vec4(idxCoords, 1.f), c_dimensions.input);
	return glm::vec4(posIDX, 0.f);
}
template<>
__device__ inline glm::vec4 samplePos3D<TransformLinDepthReverse, true>(const glm::vec3 idxCoords){
	const glm::vec3 posIDX = OStoIDXlinearDepth(glm::vec4(idxCoords, 1.f), c_dimensions.input);
	const glm::vec3 d_x = posIDX - OStoIDXlinearDepth(glm::vec4(idxCoords+glm::vec3(1,0,0), 1.f), c_dimensions.input);
	const glm::vec3 d_y = posIDX - OStoIDXlinearDepth(glm::vec4(idxCoords+glm::vec3(0,1,0), 1.f), c_dimensions.input);
	const glm::vec3 d_z = posIDX - OStoIDXlinearDepth(glm::vec4(idxCoords+glm::vec3(0,0,1), 1.f), c_dimensions.input);
	const float px = glm::length(d_x);//glm::vec3(d_x.x, d_y.x, d_z.x));
	const float py = glm::length(d_y);//glm::vec3(d_x.y, d_y.y, d_z.y));
	const float pz = glm::length(d_z);//glm::vec3(d_x.z, d_y.z, d_z.z));
	const float d = max(px, max(py, pz));
	const float lod = max(0.f, log2f(d));
	return glm::vec4(posIDX, lod);
}

//--- Sampling ---
__device__ inline glm::vec3 trilinearWeightsRegular(const glm::vec3 position){
	return glm::fract(position);
}

/*
//recursive interpolation template
template<typename T, uint8_t DIM, typename VF, typename VI>
//DIM: dimension of grid/interpolation, VF: vector type for position matching DIM, VI: vector type for dimensions
__device__ inline T readInterpolated(const VF weights, const VI ceil, const VI floor, const T* buf, const VI buf_dims){
	
}template<typename T, typename VF, typename VI>
//DIM: dimension of grid/interpolation, VF: vector type for position matching DIM, VI: vector type for dimensions
__device__ inline T readInterpolated<T,1,VF,VI>(const VF weights, const VI ceil, const VI floor, const T* buf, const VI buf_dims){
}
*/

/*
* Cell centers at +0.0 offset, spacing is uniformly 1.0. The fractional part of the sampling coordinate is the interpolation weight.
*/
template<typename T, SamplerSettings::BoundaryMode BM>
__device__ inline T read3DInterpolated(const glm::vec3 position, const T* buf, const glm::ivec3 buf_dims){
	
	//weights for ceil, floor weights are (1-weights)
	const glm::vec3 weights = trilinearWeightsRegular(position);
	
	//calculate corner indices
	glm::ivec3  ceilIdx = glm::ivec3(glm::ceil(position));
	glm::ivec3 floorIdx = glm::ivec3(glm::floor(position));
	
	if (BM == SamplerSettings::BORDER){// const 0 outside domain
		if(!isInDimensions<glm::vec3, glm::ivec3>(position+0.5f, buf_dims) || !isNonNegative(position+0.5f)){
			return vmath::make_cudaFloat<T>(0.f);
		}
		glm::ivec3  ceilValid = (-1 <  ceilIdx)*( ceilIdx < buf_dims);
		glm::ivec3 floorValid = (-1 < floorIdx)*(floorIdx < buf_dims);
		T data = {0};
		//read and interpolate along x
		const T v00 = vmath::lerp(
							(floorValid.x*floorValid.y*floorValid.z)>0.f
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf)
								: data,
							( ceilValid.x*floorValid.y*floorValid.z)>0.f
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf)
								: data,
							weights.x);
		const T v01 = vmath::lerp(
							(floorValid.x* ceilValid.y*floorValid.z)>0.f
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf)
								: data,
							( ceilValid.x* ceilValid.y*floorValid.z)>0.f
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf)
								: data,
							weights.x);
		//interpolate along y
		const T v0 = vmath::lerp(v00, v01, weights.y);
		
		
		const T v10 = vmath::lerp(
							(floorValid.x*floorValid.y* ceilValid.z)>0.f
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf)
								: data,
							( ceilValid.x*floorValid.y* ceilValid.z)>0.f
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf)
								: data,
							weights.x);
		const T v11 = vmath::lerp(
							(floorValid.x* ceilValid.y* ceilValid.z)>0.f
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf)
								: data,
							( ceilValid.x* ceilValid.y* ceilValid.z)>0.f
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf)
								: data,
							weights.x);
		//interpolate along y
		const T v1 = vmath::lerp(v10, v11, weights.y);
		
		
		//interpolate along z
		return vmath::lerp(v0, v1, weights.z); //(1-weights.z)*yValue.x + weights.z*yValue.y;
		
	}else{//here the indices are always valid, so no special handling needed
		if(BM == SamplerSettings::CLAMP){
			ceilIdx = glm::clamp(ceilIdx, glm::ivec3(0), buf_dims -1);
			floorIdx = glm::clamp(floorIdx, glm::ivec3(0), buf_dims -1);
		}
		else if (BM == SamplerSettings::WRAP){//periodic
			ceilIdx = vmath::positivemod<glm::ivec3, glm::ivec3>(ceilIdx, buf_dims);
			floorIdx = vmath::positivemod<glm::ivec3, glm::ivec3>(floorIdx, buf_dims);
		}
		//read and interpolate along x
		const T v00 = vmath::lerp( vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf),
							vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf),
							weights.x);
		const T v01 = vmath::lerp( vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf),
							vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf),
							weights.x);
		//interpolate along y
		const T v0 = vmath::lerp(v00, v01, weights.y);
		
		
		const T v10 = vmath::lerp( vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf),
							vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf),
							weights.x);
		const T v11 = vmath::lerp( vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf),
							vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf),
							weights.x);
		//interpolate along y
		const T v1 = vmath::lerp(v10, v11, weights.y);
		
		
		//interpolate along z
		return vmath::lerp(v0, v1, weights.z); //(1-weights.z)*yValue.x + weights.z*yValue.y;
	}
}
template<typename T>
struct DataGrad3D{
	T dx,dz,dy;
};
template<typename T, SamplerSettings::BoundaryMode BM>
__device__ inline DataGrad3D<T> read3DGrad(const glm::vec3 position, const T* buf, const glm::ivec3 buf_dims){
	
	//weights for ceil, floor weights are (1-weights)
	const glm::vec3 weights = trilinearWeightsRegular(position);
	
	//calculate corner indices
	glm::ivec3  ceilIdx = glm::ivec3(glm::ceil(position));
	glm::ivec3 floorIdx = glm::ivec3(glm::floor(position));
	
	if (BM == SamplerSettings::BORDER){// const 0 outside domain
		if(!isInDimensions<glm::vec3, glm::ivec3>(position+0.5f, buf_dims) || !isNonNegative(position+0.5f)){
			DataGrad3D<T> zero = {0};
			return zero;
		}
		glm::ivec3  ceilValid = (-1 <  ceilIdx)*( ceilIdx < buf_dims);
		glm::ivec3 floorValid = (-1 < floorIdx)*(floorIdx < buf_dims);
		T data = {0};
		//read
		const T fxfyfz = (floorValid.x*floorValid.y*floorValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf)
								: data;
		const T cxfyfz = ( ceilValid.x*floorValid.y*floorValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf)
								: data;
		const T fxcyfz = (floorValid.x* ceilValid.y*floorValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf)
								: data;
		const T cxcyfz = ( ceilValid.x* ceilValid.y*floorValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf)
								: data;
		const T fxfycz = (floorValid.x*floorValid.y* ceilValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf)
								: data;
		const T cxfycz = ( ceilValid.x*floorValid.y* ceilValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf)
								: data;
		const T fxcycz = (floorValid.x* ceilValid.y* ceilValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf)
								: data;
		const T cxcycz = ( ceilValid.x* ceilValid.y* ceilValid.z)
								? vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf)
								: data;
		//interpolate differences
		DataGrad3D<T> dataGrad;
		dataGrad.dx = vmath::lerp(
						vmath::lerp((cxfyfz-fxfyfz),(cxcyfz-fxcyfz), weights.y),
						vmath::lerp((cxfycz-fxfycz),(cxcycz-fxcycz), weights.y),
					weights.z);
		dataGrad.dy = vmath::lerp(
						vmath::lerp((fxcyfz-fxfyfz),(cxcyfz-cxfyfz), weights.x),
						vmath::lerp((fxcycz-fxfycz),(cxcycz-cxfycz), weights.x),
					weights.z);
		dataGrad.dz = vmath::lerp(
						vmath::lerp((fxfycz-fxfyfz),(cxfycz-cxfyfz), weights.x),
						vmath::lerp((fxcycz-fxcyfz),(cxcycz-cxcyfz), weights.x),
					weights.y);
		
		return dataGrad;
		
	}else{//here the indices are always valid, so no special handling needed
		if(BM == SamplerSettings::CLAMP){
			ceilIdx = glm::clamp(ceilIdx, glm::ivec3(0), buf_dims -1);
			floorIdx = glm::clamp(floorIdx, glm::ivec3(0), buf_dims -1);
		}
		else if (BM == SamplerSettings::WRAP){//periodic
			ceilIdx = vmath::positivemod<glm::ivec3, glm::ivec3>(ceilIdx, buf_dims);
			floorIdx = vmath::positivemod<glm::ivec3, glm::ivec3>(floorIdx, buf_dims);
		}
		//read
		const T fxfyfz = vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf);
		const T cxfyfz = vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf);
		const T fxcyfz = vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf);
		const T cxcyfz = vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x,  ceilIdx.y, floorIdx.z, buf_dims, buf);
		const T fxfycz = vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf);
		const T cxfycz = vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x, floorIdx.y,  ceilIdx.z, buf_dims, buf);
		const T fxcycz = vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf);
		const T cxcycz = vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>( ceilIdx.x,  ceilIdx.y,  ceilIdx.z, buf_dims, buf);
		//interpolate differences
		DataGrad3D<T> dataGrad;
		dataGrad.dx = vmath::lerp(
						vmath::lerp((cxfyfz-fxfyfz),(cxcyfz-fxcyfz), weights.y),
						vmath::lerp((cxfycz-fxfycz),(cxcycz-fxcycz), weights.y),
					weights.z);
		dataGrad.dy = vmath::lerp(
						vmath::lerp((fxcyfz-fxfyfz),(cxcyfz-cxfyfz), weights.x),
						vmath::lerp((fxcycz-fxfycz),(cxcycz-cxfycz), weights.x),
					weights.z);
		dataGrad.dz = vmath::lerp(
						vmath::lerp((fxfycz-fxfyfz),(cxfycz-cxfyfz), weights.x),
						vmath::lerp((fxcycz-fxcyfz),(cxcycz-cxcyfz), weights.x),
					weights.y);
		
		return dataGrad;
	}
}
/*
* Cell centers at +0.0 offset, spacing is uniformly 1.0. The fractional part of the sampling coordinate is the interpolation weight.
*/
template<typename T, SamplerSettings::BoundaryMode BM>
__device__ inline T read3DNearest(const glm::vec3 position, const T* buf, const glm::ivec3 buf_dims){
	glm::ivec3 idx = glm::ivec3(position +.5f);
	if (BM == SamplerSettings::BORDER){
		if(!isInDimensions<glm::ivec3, glm::ivec3>(idx, buf_dims) || !isNonNegative(idx)){
			return vmath::make_cudaFloat<T>(0.f);
		}
	}
	else if(BM == SamplerSettings::CLAMP){
		idx = glm::clamp(idx, glm::ivec3(0), buf_dims -1);
	}
	else if (BM == SamplerSettings::WRAP){//periodic
		idx = vmath::positivemod<glm::ivec3, glm::ivec3>(idx, buf_dims);
	}
	return vectorIO::readVectorType3D<T, T, glm::ivec3>(idx, buf_dims, buf);
}
template<typename T, SamplerSettings::BoundaryMode BM>
__device__ inline T read3DMin(const glm::vec3 position, const T* buf, const glm::ivec3 buf_dims){
	
	
	//calculate corner indices
	glm::ivec3  ceilIdx = glm::ivec3(glm::ceil(position));
	glm::ivec3 floorIdx = glm::ivec3(glm::floor(position));
	
	if (BM == SamplerSettings::BORDER){// const 0 outside domain
		if(!isInDimensions<glm::vec3, glm::ivec3>(position+0.5f, buf_dims) || !isNonNegative(position+0.5f)){
			return vmath::make_cudaFloat<T>(0.f);
		}
		glm::ivec3  ceilValid = (-1 <  ceilIdx)*( ceilIdx < buf_dims);
		glm::ivec3 floorValid = (-1 < floorIdx)*(floorIdx < buf_dims);
		T data;
		if(!(vmath::prod(ceilValid)*vmath::prod(floorValid))){ //any cell invalid/out of bounds
			data = vmath::make_cudaFloat<T>(0.f);
		}else{
			data = vmath::make_cudaFloat<T>(FLOAT_MAX);
		}
		#define cmpMin(vx, vy, vz) if((vx##Valid.x* vy##Valid.y* vz##Valid.z)>0.f) data = fminf(data, vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(vx##Idx.x, vy##Idx.y, vz##Idx.z, buf_dims, buf))
		//if((floorValid.x*floorValid.y*floorValid.z)>0.f) data = vmath::min<T>(data, vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf));
		cmpMin(floor, floor, floor);
		cmpMin( ceil, floor, floor);
		cmpMin(floor,  ceil, floor);
		cmpMin( ceil,  ceil, floor);
		
		cmpMin(floor, floor,  ceil);
		cmpMin( ceil, floor,  ceil);
		cmpMin(floor,  ceil,  ceil);
		cmpMin( ceil,  ceil,  ceil);
		#undef cmpMin
		return data;
	}else{//here the indices are always valid, so no special handling needed
		if(BM == SamplerSettings::CLAMP){
			ceilIdx = glm::clamp(ceilIdx, glm::ivec3(0), buf_dims -1);
			floorIdx = glm::clamp(floorIdx, glm::ivec3(0), buf_dims -1);
		}
		else if (BM == SamplerSettings::WRAP){//periodic
			ceilIdx = vmath::positivemod<glm::ivec3, glm::ivec3>(ceilIdx, buf_dims);
			floorIdx = vmath::positivemod<glm::ivec3, glm::ivec3>(floorIdx, buf_dims);
		}
		T data = vmath::make_cudaFloat<T>(FLOAT_MAX);
		
		#define cmpMin(vx, vy, vz) data = fminf(data, vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(vx##Idx.x, vy##Idx.y, vz##Idx.z, buf_dims, buf))
		//data = vmath::min<T>(data, vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf));
		cmpMin(floor, floor, floor);
		cmpMin( ceil, floor, floor);
		cmpMin(floor,  ceil, floor);
		cmpMin( ceil,  ceil, floor);
		
		cmpMin(floor, floor,  ceil);
		cmpMin( ceil, floor,  ceil);
		cmpMin(floor,  ceil,  ceil);
		cmpMin( ceil,  ceil,  ceil);
		#undef cmpMin
		
		return data;
	}
}
template<typename T, SamplerSettings::BoundaryMode BM>
__device__ inline T read3DMax(const glm::vec3 position, const T* buf, const glm::ivec3 buf_dims){
	
	
	//calculate corner indices
	glm::ivec3  ceilIdx = glm::ivec3(glm::ceil(position));
	glm::ivec3 floorIdx = glm::ivec3(glm::floor(position));
	
	if (BM == SamplerSettings::BORDER){// const 0 outside domain
		if(!isInDimensions<glm::vec3, glm::ivec3>(position+0.5f, buf_dims) || !isNonNegative(position+0.5f)){
			return vmath::make_cudaFloat<T>(0.f);
		}
		glm::ivec3  ceilValid = (-1 <  ceilIdx)*( ceilIdx < buf_dims);
		glm::ivec3 floorValid = (-1 < floorIdx)*(floorIdx < buf_dims);
		T data;
		if(!(vmath::prod(ceilValid)*vmath::prod(floorValid))){ //any cell invalid/out of bounds
			data = vmath::make_cudaFloat<T>(0.f);
		}else{
			data = vmath::make_cudaFloat<T>(FLOAT_LOWEST);
		}
		#define cmpMax(vx, vy, vz) if((vx##Valid.x* vy##Valid.y* vz##Valid.z)>0.f) data = fmaxf(data, vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(vx##Idx.x, vy##Idx.y, vz##Idx.z, buf_dims, buf))
		//if((floorValid.x*floorValid.y*floorValid.z)>0.f) data = vmath::max<T>(data, vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf));
		cmpMax(floor, floor, floor);
		cmpMax( ceil, floor, floor);
		cmpMax(floor,  ceil, floor);
		cmpMax( ceil,  ceil, floor);
		
		cmpMax(floor, floor,  ceil);
		cmpMax( ceil, floor,  ceil);
		cmpMax(floor,  ceil,  ceil);
		cmpMax( ceil,  ceil,  ceil);
		#undef cmpMax
		return data;
	}else{//here the indices are always valid, so no special handling needed
		if(BM == SamplerSettings::CLAMP){
			ceilIdx = glm::clamp(ceilIdx, glm::ivec3(0), buf_dims -1);
			floorIdx = glm::clamp(floorIdx, glm::ivec3(0), buf_dims -1);
		}
		else if (BM == SamplerSettings::WRAP){//periodic
			ceilIdx = vmath::positivemod<glm::ivec3, glm::ivec3>(ceilIdx, buf_dims);
			floorIdx = vmath::positivemod<glm::ivec3, glm::ivec3>(floorIdx, buf_dims);
		}
		T data = vmath::make_cudaFloat<T>(FLOAT_LOWEST);
		
		#define cmpMax(vx, vy, vz) data = fmaxf(data, vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(vx##Idx.x, vy##Idx.y, vz##Idx.z, buf_dims, buf))
		//data = vmath::min<T>(data, vectorIO::readVectorType3D<T, T, int32_t, glm::ivec3>(floorIdx.x, floorIdx.y, floorIdx.z, buf_dims, buf));
		cmpMax(floor, floor, floor);
		cmpMax( ceil, floor, floor);
		cmpMax(floor,  ceil, floor);
		cmpMax( ceil,  ceil, floor);
		
		cmpMax(floor, floor,  ceil);
		cmpMax( ceil, floor,  ceil);
		cmpMax(floor,  ceil,  ceil);
		cmpMax( ceil,  ceil,  ceil);
		#undef cmpMax
		
		return data;
	}
}


/*
//recursive interpolation template
template<typename T, uint8_t DIM, typename VF, typename VI>
//DIM: dimension of grid/interpolation, VF: vector type for position matching DIM, VI: vector type for dimensions
__device__ inline T readInterpolatedChannel(const glm::vec3 position, const int32_t channel, const T* buf, const glm::ivec3 buf_dims, const int32_t buf_channel_dim);
*/
template<typename T>
__device__ inline T read3DInterpolatedChannel(const glm::vec3 position, const int32_t channel, const T* buf, const glm::ivec3 buf_dims, const int32_t buf_channel_dim){
	//weights for ceil, floor weights are (1-weights)
	const glm::vec3 weights = trilinearWeightsRegular(position);
	
	//calculate corner indices
	const glm::ivec3 ceilIdx = glm::ceil(position);
	const glm::ivec3 floorIdx = glm::floor(position);
	
	//read and interpolate along x
	//channel_dim*(dims.x*(dims.y*pos.z + pos.y) + pos.x) + channel
	size_t idxTmp = buf_dims.x*(buf_dims.y*floorIdx.z + floorIdx.y);
	const T v00 = vmath::lerp( buf[buf_channel_dim*(idxTmp + floorIdx.x) + channel],
						buf[buf_channel_dim*(idxTmp +  ceilIdx.x) + channel],
						weights.x);
	idxTmp = buf_dims.x*(buf_dims.y*floorIdx.z + ceilIdx.y);
	const T v01 = vmath::lerp( buf[buf_channel_dim*(idxTmp + floorIdx.x) + channel],
						buf[buf_channel_dim*(idxTmp +  ceilIdx.x) + channel],
						weights.x);
	//interpolate along y
	const T v0 = vmath::lerp(v00, v01, weights.y);
	
	idxTmp = buf_dims.x*(buf_dims.y*floorIdx.z + ceilIdx.y);
	const T v10 = vmath::lerp( buf[buf_channel_dim*(idxTmp + floorIdx.x) + channel],
						buf[buf_channel_dim*(idxTmp +  ceilIdx.x) + channel],
						weights.x);
	idxTmp = buf_dims.x*(buf_dims.y*ceilIdx.z + ceilIdx.y);
	const T v11 = vmath::lerp( buf[buf_channel_dim*(idxTmp + floorIdx.x) + channel],
						buf[buf_channel_dim*(idxTmp +  ceilIdx.x) + channel],
						weights.x);
	//interpolate along y
	const T v1 = vmath::lerp(v10, v11, weights.y);
	
	//interpolate along z
	return vmath::lerp(v0, v1, weights.z); //(1-weights.z)*yValue.x + weights.z*yValue.y;
}

// cell centers are at +0.5 offset
//template<typename T_DATA, SamplerSettings::MipMode MIP, SamplerSettings::SamplingMode SM>
template<typename T_DATA>
struct Sampler2{
	SamplerSettings settings;
	glm::ivec4 dimensions;
	//float cellCenterOffset;
	
	union{
		const T_DATA UG_PTR *d_input; //device pointer
		const T_DATA UG_PTR *const *d_mips; //device pointer
		//cudaTextureObject_t texture;
	};
	
	__device__ constexpr float getLoD(const glm::vec4 position) const {
		return glm::clamp(position.w + settings.mipBias, settings.mipClampMin, settings.mipClampMax);
	}
	
	__device__ constexpr glm::vec3 getSamplingPosition(const glm::vec3 position) const {
		return position - settings.cellCenterOffset;
	}
	
	__device__ constexpr glm::vec3 getCeilWeights(const glm::vec3 position) const {
		return glm::fract(position - settings.cellCenterOffset);
	}
	
	__device__ inline int32_t getMipLevelFloor(const float lod) const {
		return (settings.mipMode==SamplerSettings::MIPMODE_NONE)?
			0 :
			min(static_cast<int32_t>(lod+.5f*(settings.mipMode==SamplerSettings::MIPMODE_NEAREST)), settings.mipLevel);
	}
	//should only be called if settings.mipMode==SamplerSettings::MipMode::LINEAR
	__device__ inline int32_t getMipLevelCeil(const float lod) const {
		return min(static_cast<int32_t>(ceil(lod)), settings.mipLevel);
	}
	
	// --- Sampling ---
	
	__device__ inline T_DATA _internal_fetch3D(const glm::vec3 position, const glm::ivec3 buf_dimensions, const T_DATA *buf) const {
		T_DATA data = {0};
		// undo center offset to +0.0 as needed by the interpolation
		if(settings.filterMode==SamplerSettings::FILTERMODE_LINEAR){
			if(settings.boundaryMode==SamplerSettings::BORDER){
				data = read3DInterpolated<T_DATA, SamplerSettings::BORDER>(position - settings.cellCenterOffset, buf, buf_dimensions);
			}else if(settings.boundaryMode==SamplerSettings::CLAMP){
				data = read3DInterpolated<T_DATA, SamplerSettings::CLAMP>(position - settings.cellCenterOffset, buf, buf_dimensions);
			}
		}
		else{ //NEAREST
			if(settings.boundaryMode==SamplerSettings::BORDER){
				data = read3DNearest<T_DATA, SamplerSettings::BORDER>(position - settings.cellCenterOffset, buf, buf_dimensions);
			}else if(settings.boundaryMode==SamplerSettings::CLAMP){
				data = read3DNearest<T_DATA, SamplerSettings::CLAMP>(position - settings.cellCenterOffset, buf, buf_dimensions);
			}
		}
		return data;
	}
	__device__ inline T_DATA read3D(const glm::vec3 position) const {
		return _internal_fetch3D(position, dimensions, d_input);
	}
	__device__ inline T_DATA read3DLevel(const glm::vec3 position, const int32_t mipLevel) const {
		const glm::ivec3 mipDims = dimensions>>mipLevel;
		const glm::vec3 mipPos = position/static_cast<float>(1<<mipLevel);
		return _internal_fetch3D(mipPos, mipDims, d_mips[mipLevel]);
	}
	
	__device__ inline T_DATA sample3D(const glm::vec4 position) const {
		if(settings.mipMode==SamplerSettings::MIPMODE_NONE){
			return read3D(position);
		}else{
			const float lod = getLoD(position);
			//if NONE: input, if NEAREST: mip[int(mip+0.5)], if LINEAR: mip[int(mip)]
			const int32_t mipLevelFloor = getMipLevelFloor(lod);
			T_DATA data = read3DLevel(position, mipLevelFloor);
			
			if(settings.mipMode==SamplerSettings::MIPMODE_LINEAR){
				const int32_t mipLevelCeil = getMipLevelCeil(lod);
				if(mipLevelCeil>mipLevelFloor){
					data = vmath::lerp<float, T_DATA>(data, read3DLevel(position, mipLevelCeil), glm::fract(lod));
				}
			}
			return data;
		}
	}
	__device__ inline T_DATA sample3Dnormalized(const glm::vec4 normalizedPosition) const {
		glm::vec4 position = normalizedPosition;
		position.x *= dimensions.x;
		position.y *= dimensions.y;
		position.z *= dimensions.z;
		return sample3D(position);
	}
	
	// --- Data Gradients ---
	
	__device__ inline DataGrad3D<T_DATA> _internal_fetchGrad3D(const glm::vec3 position, const glm::ivec3 buf_dimensions, const T_DATA *buf) const {
		DataGrad3D<T_DATA> data = {0};
		// undo center offset to +0.0 as needed by the interpolation
		if(settings.filterMode==SamplerSettings::FILTERMODE_LINEAR){
			if(settings.boundaryMode==SamplerSettings::BORDER){
				data = read3DGrad<T_DATA, SamplerSettings::BORDER>(position - settings.cellCenterOffset, buf, buf_dimensions);
			}else if(settings.boundaryMode==SamplerSettings::CLAMP){
				data = read3DGrad<T_DATA, SamplerSettings::CLAMP>(position - settings.cellCenterOffset, buf, buf_dimensions);
			}
		}
		else{ //NEAREST
			// return 0 as data gradient for NN sampling
		}
		return data;
	}
	__device__ inline DataGrad3D<T_DATA> readGrad3D(const glm::vec3 position) const {
		return _internal_fetchGrad3D(position, dimensions, d_input);
	}
	__device__ inline DataGrad3D<T_DATA> readGrad3DLevel(const glm::vec3 position, int32_t mipLevel) const {
		const glm::ivec3 mipDims = dimensions>>mipLevel;
		const glm::vec3 mipPos = position/static_cast<float>(1<<mipLevel);
		return _internal_fetchGrad3D(mipPos, mipDims, d_mips[mipLevel]);
	}
	
	//return x, y, z data gradient at the sample location (0 for NN)
	__device__ inline DataGrad3D<T_DATA> sampleGrad3D(const glm::vec4 position) const {
		if(settings.mipMode==SamplerSettings::MIPMODE_NONE){
			return readGrad3D(position);
		}else{
			const float lod = getLoD(position);
			//if NONE: input, if NEAREST: mip[int(mip+0.5)], if LINEAR: mip[int(mip)]
			const int32_t mipLevelFloor = getMipLevelFloor(lod);
			DataGrad3D<T_DATA> data = readGrad3DLevel(position, mipLevelFloor);
			
			if(settings.mipMode==SamplerSettings::MIPMODE_LINEAR){
				const int32_t mipLevelCeil = getMipLevelCeil(lod);
				if(mipLevelCeil>mipLevelFloor){
					//const T_DATA *mipCeil = mips[mipLevelCeil];
					float lodWeight = glm::fract(lod);
					DataGrad3D<T_DATA> data2 = readGrad3DLevel(position, mipLevelCeil);
					data.dx = vmath::lerp(data.dx,data2.dx, lodWeight);
					data.dy = vmath::lerp(data.dy,data2.dy, lodWeight);
					data.dz = vmath::lerp(data.dz,data2.dz, lodWeight);
				}
			}
			return data;
		}
	}
};

/*
* Sampling templates
*/

template<typename T_DATA>
struct Grid3D{
	glm::ivec4 dimensions; //x,y,z,channel
	glm::vec3 dimensionsInverse;
	int32_t mipLevel;
	union{
		T_DATA UG_PTR *d_data; //device pointer
		T_DATA UG_PTR *const *d_mips; //device pointer
	};
};

__device__ inline float getLoD(const SamplerSettings settings, const glm::vec4 position){
	return glm::clamp(position.w + settings.mipBias, settings.mipClampMin, settings.mipClampMax);
}

template<SamplerSettings::MipMode MM>
__device__ inline int32_t getMipLevelFloor(const float lod, const int32_t maxLod){
	return (MM==SamplerSettings::MIPMODE_NONE)?
		0 :
		min(static_cast<int32_t>(lod+.5f*(MM==SamplerSettings::MIPMODE_NEAREST)), maxLod);
}
//should only be called if settings.mipMode==SamplerSettings::MipMode::LINEAR
__device__ inline int32_t getMipLevelCeil(const float lod, const int32_t maxLod){
	return min(static_cast<int32_t>(ceil(lod)), maxLod);
}

// --- Sampling ---

template<typename T_DATA, SamplerSettings::FilterMode FM, SamplerSettings::BoundaryMode BM>
__device__ inline T_DATA _internal_fetch3D(const SamplerSettings settings, const glm::vec3 position, const glm::ivec3 buf_dimensions, const T_DATA *buf){
	T_DATA data = {0};
	// undo center offset to +0.0 as needed by the interpolation
	//position -= settings.cellCenterOffset;
	if(FM==SamplerSettings::FILTERMODE_LINEAR){
		data = read3DInterpolated<T_DATA, BM>(position - settings.cellCenterOffset, buf, buf_dimensions);
	}
	else if(FM==SamplerSettings::FILTERMODE_NEAREST){ //NEAREST
		data = read3DNearest<T_DATA, BM>(position - settings.cellCenterOffset, buf, buf_dimensions);
	}
	else if(FM==SamplerSettings::FILTERMODE_MIN){
		data = read3DMin<T_DATA, BM>(position - settings.cellCenterOffset, buf, buf_dimensions);
	}
	else if(FM==SamplerSettings::FILTERMODE_MAX){
		data = read3DMax<T_DATA, BM>(position - settings.cellCenterOffset, buf, buf_dimensions);
	}
	return data;
}

template<typename T_DATA, SamplerSettings::FilterMode FM, SamplerSettings::BoundaryMode BM>
__device__ inline T_DATA read3D(const SamplerSettings settings, const Grid3D<const T_DATA> input, const glm::vec3 position){
	return _internal_fetch3D<T_DATA, FM, BM>(settings, position, input.dimensions, input.d_data);
}

template<typename T_DATA, SamplerSettings::FilterMode FM, SamplerSettings::BoundaryMode BM>
__device__ inline T_DATA read3DLevel(const SamplerSettings settings, const Grid3D<const T_DATA> input, const glm::vec3 position, const int32_t mipLevel){
	const glm::ivec3 mipDims = input.dimensions>>mipLevel;
	const glm::vec3 mipPos = position/static_cast<float>(1<<mipLevel);
	return _internal_fetch3D<T_DATA, FM, BM>(settings, mipPos, mipDims, input.d_mips[mipLevel]);
}

template<typename T_DATA, SamplerSettings::FilterMode FM, SamplerSettings::MipMode MM, SamplerSettings::BoundaryMode BM>
__device__ inline T_DATA sample3D(const SamplerSettings settings, const Grid3D<const T_DATA> input, const glm::vec4 position){
	if(MM==SamplerSettings::MIPMODE_NONE){
		return read3D<T_DATA, FM, BM>(settings, input, position);
	}else{
		const float lod = getLoD(settings, position);
		//if NONE: input, if NEAREST: mip[int(mip+0.5)], if LINEAR: mip[int(mip)]
		const int32_t mipLevelFloor = getMipLevelFloor<MM>(lod, input.mipLevel);
		T_DATA data = read3DLevel<T_DATA, FM, BM>(settings, input, position, mipLevelFloor);
		
		if(MM==SamplerSettings::MIPMODE_LINEAR){
			const int32_t mipLevelCeil = getMipLevelCeil(lod, input.mipLevel);
			if(mipLevelCeil>mipLevelFloor){
				//const T_DATA *mipCeil = mips[mipLevelCeil];
				data = vmath::lerp<float, T_DATA>(data, read3DLevel<T_DATA, FM, BM>(settings, input, position, mipLevelCeil), glm::fract(lod));
			}
		}
		return data;
	}
}

// --- Data Gradients ---

template<typename T_DATA, SamplerSettings::FilterMode FM, SamplerSettings::BoundaryMode BM>
__device__ inline DataGrad3D<T_DATA> _internal_fetchGrad3D(const SamplerSettings settings, const glm::vec3 position, const glm::ivec3 buf_dimensions, const T_DATA *buf){
	DataGrad3D<T_DATA> data = {0};
	// undo center offset to +0.0 as needed by the interpolation
	if(FM==SamplerSettings::FILTERMODE_LINEAR){
		data = read3DGrad<T_DATA, BM>(position - settings.cellCenterOffset, buf, buf_dimensions);
	}
	else{ //NEAREST, MIN, MAX
		// return 0 as data gradient for step functions. A linear data gradient might be useful for NN sampling?
	}
	return data;
}
template<typename T_DATA, SamplerSettings::FilterMode FM, SamplerSettings::BoundaryMode BM>
__device__ inline DataGrad3D<T_DATA> readGrad3D(const SamplerSettings settings, const Grid3D<const T_DATA> input, const glm::vec3 position){
	return _internal_fetchGrad3D<T_DATA, FM, BM>(settings, position, input.dimensions, input.d_data);
}
template<typename T_DATA, SamplerSettings::FilterMode FM, SamplerSettings::BoundaryMode BM>
__device__ inline DataGrad3D<T_DATA> readGrad3DLevel(const SamplerSettings settings, const Grid3D<const T_DATA> input, const glm::vec3 position, int32_t mipLevel){
	const glm::ivec3 mipDims = input.dimensions>>mipLevel;
	const glm::vec3 mipPos = position/static_cast<float>(1<<mipLevel);
	return _internal_fetchGrad3D<T_DATA, FM, BM>(settings, mipPos, mipDims, input.d_mips[mipLevel]);
}

//return x, y, z data gradient at the sample location (0 for NN)
template<typename T_DATA, SamplerSettings::FilterMode FM, SamplerSettings::MipMode MM, SamplerSettings::BoundaryMode BM>
__device__ inline DataGrad3D<T_DATA> sampleGrad3D(const SamplerSettings settings, const Grid3D<const T_DATA> input, const glm::vec4 position){
	if(settings.mipMode==SamplerSettings::MIPMODE_NONE){
		return readGrad3D<T_DATA, FM, BM>(settings, input, position);
	}else{
		const float lod = getLoD(settings, position);
		//if NONE: input, if NEAREST: mip[int(mip+0.5)], if LINEAR: mip[int(mip)]
		const int32_t mipLevelFloor = getMipLevelFloor<MM>(lod, input.mipLevel);
		DataGrad3D<T_DATA> data = readGrad3DLevel<T_DATA, FM, BM>(settings, input, position, mipLevelFloor);
		
		if(settings.mipMode==SamplerSettings::MIPMODE_LINEAR){
			const int32_t mipLevelCeil = getMipLevelCeil(lod, input.mipLevel);
			if(mipLevelCeil>mipLevelFloor){
				//const T_DATA *mipCeil = mips[mipLevelCeil];
				float lodWeight = glm::fract(lod);
				DataGrad3D<T_DATA> data2 = readGrad3DLevel<T_DATA, FM, BM>(settings, input, position, mipLevelCeil);
				data.dx = vmath::lerp(data.dx,data2.dx, lodWeight);
				data.dy = vmath::lerp(data.dy,data2.dy, lodWeight);
				data.dz = vmath::lerp(data.dz,data2.dz, lodWeight);
			}
		}
		return data;
	}
}

} //Sampling
#endif //_INCLUDE_SAMPLING