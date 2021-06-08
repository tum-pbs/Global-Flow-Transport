#pragma once

#ifndef _INCLUDE_SAMPLING_SETTIGNS
#define _INCLUDE_SAMPLING_SETTIGNS

#ifdef __CUDACC__
#define CUDA_QUALIFIER __host__ __device__
#else
#define CUDA_QUALIFIER
#endif

namespace Sampling{
	
struct SamplerSettings{
	enum MipMode{MIPMODE_NONE=0, MIPMODE_NEAREST=1, MIPMODE_LINEAR=2} mipMode;
	enum FilterMode{FILTERMODE_NEAREST=0, FILTERMODE_LINEAR=1, FILTERMODE_MIN=2, FILTERMODE_MAX=3} filterMode; //FILTERMODE_MIN FILTERMODE_MAX
	bool useTexture;
	enum BoundaryMode{BORDER=0, CLAMP=1, WRAP=2, MIRROR=3} boundaryMode;
	float cellCenterOffset;
	int32_t mipLevel;
	float mipClampMin, mipClampMax, mipBias;
};
CUDA_QUALIFIER constexpr bool usesMip(const SamplerSettings &settings){ return settings.mipMode != SamplerSettings::MIPMODE_NONE;}
CUDA_QUALIFIER constexpr bool usesTexture(const SamplerSettings &settings){ return settings.useTexture;}


enum SamplingMode : uint32_t{
	NEAREST = 0b0000,
	LINEAR = 0b0001,
	NEAREST_MIP_NEAREST = 0b0100,
	NEAREST_MIP_LINEAR = 0b0110,
	LINEAR_MIP_NEAREST = 0b0101,
	LINEAR_MIP_LINEAR = 0b0101,
	
	TEX_NEAREST = 0b1000,
	TEX_LINEAR = 0b1001,
	TEX_NEAREST_MIP_NEAREST = 0b1100,
	TEX_NEAREST_MIP_LINEAR = 0b1110,
	TEX_LINEAR_MIP_NEAREST = 0b1101,
	TEX_LINEAR_MIP_LINEAR = 0b1101,
	
	//helpers
	_MIP = 0b0100,
	_MIP_LINEAR = 0b0010,
	_TEX = 0b1000,
	_TEX_MIP = 0b1100,
};

CUDA_QUALIFIER constexpr uint32_t mipFlag(const SamplingMode flags){ return (_MIP & flags);}
CUDA_QUALIFIER constexpr bool usesMip(const SamplingMode flags){ return mipFlag(flags) !=0;}

CUDA_QUALIFIER constexpr uint32_t textureFlag(const SamplingMode flags){ return (_TEX & flags);}
CUDA_QUALIFIER constexpr bool usesTexture(const SamplingMode flags){ return textureFlag(flags) !=0;}


enum CoordinateMode{
	TransformLinDepth,
	TransformLinDepthReverse,
	TransformLinDepthDoubleReverse,
	Transform,
	TransformReverse,
	PixelRaysView,
	PixelRaysWorld,
	LuT,
};


} //Sampling
#endif //_INCLUDE_SAMPLING_SETTIGNS