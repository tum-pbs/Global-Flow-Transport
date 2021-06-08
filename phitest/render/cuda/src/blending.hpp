#pragma once

#ifndef _INCLUDE_BLENDING
#define _INCLUDE_BLENDING

#include "blending_settings.hpp"

namespace Blending{
template<typename T_DATA, BlendMode BM>
struct BlendState{
	//returns updated accumulator values
	__device__ static inline T_DATA blend(const T_DATA accumulator, const T_DATA cell_in){
		return accumulator + cell_in;
	}
	//returns grads_cell and updates grads_out to hold grads_in
	__device__ static inline T_DATA blendGradients(T_DATA& grads_out, const T_DATA cell_in, T_DATA& cell_out){
		return grads_out;
	}
};

template<>
struct BlendState<float4, BLEND_BEERLAMBERT>{
	__device__ static inline float4 blend(const float4 acc, const float4 cell_in){
		const float d = max(acc.w + cell_in.w, 0.f);
		const float t = exp(-d); //transmissivity
		return make_float4(acc.x + t*cell_in.x, acc.y + t*cell_in.y, acc.z + t*cell_in.z, //color
			d); //opacity
	}
	__device__ static inline float4 blendGradients(float4& grads_out, const float4 cell_in, float4& cell_out){
		const float e = exp(-cell_out.w);
		cell_out.w = max(cell_out.w - cell_in.w, 0.f);
		const float3 cell_light = make_float3(cell_in);//glm::xyz(cell);
		const float3 grads_light = make_float3(grads_out);//glm::xyz(grads_out);
		
		const float d_dacc = grads_out.w - e*vmath::sum(cell_light*grads_light);
		const float d_d = d_dacc;//grads_out.w + e*vmath::sum(cell_light*grads_light);
		
		const float3 d_lacc = grads_light;
		const float3 d_l = e*grads_light;
		
		grads_out = make_float4(d_lacc, d_dacc);
		return make_float4(d_l, d_d);
	}
};
template<>
struct BlendState<float2, BLEND_BEERLAMBERT>{
	__device__ static inline float2 blend(const float2 acc, const float2 cell_in){
		const float d = max(acc.y + cell_in.y, 0.f);
		const float t = exp(-d); //transmissivity
		return make_float2(acc.x + t*cell_in.x, //color
			d); //opacity
	}
	__device__ static inline float2 blendGradients(float2& grads_out, const float2 cell_in, float2& cell_out){
		const float e = exp(-cell_out.y);
		cell_out.y = max(cell_out.y - cell_in.y, 0.f);
		const float cell_light = cell_in.x;//glm::xyz(cell);
		const float grads_light = grads_out.x;//glm::xyz(grads_out);
		
		const float d_dacc = grads_out.y - e*cell_light*grads_light;
		const float d_d = d_dacc;//grads_out.y + e*vmath::sum(cell_light*grads_light);
		
		const float d_lacc = grads_light;
		const float d_l = e*grads_light;
		
		grads_out = make_float2(d_lacc, d_dacc);
		return make_float2(d_l, d_d);
	}
};
template struct BlendState<float1, BLEND_BEERLAMBERT>;

//Beer-Lambert without cell self-attenuation
template<>
struct BlendState<float4, BLEND_BEERLAMBERT_EXCLUSIVE>{
	__device__ static inline float4 blend(const float4 acc, const float4 cell_in){
		const float t = exp(-acc.w); //transmissivity
		return make_float4(acc.x + t*cell_in.x, acc.y + t*cell_in.y, acc.z + t*cell_in.z, //color
			acc.w + cell_in.w); //opacity
	}
	__device__ static inline float4 blendGradients(float4& grads_out, const float4 cell_in, float4& cell_out){
		cell_out.w = max(cell_out.w - cell_in.w, 0.f);
		const float e = exp(-cell_out.w);
		const float3 cell_light = make_float3(cell_in);//glm::xyz(cell);
		const float3 grads_light = make_float3(grads_out);//glm::xyz(grads_out);
		
		const float d_dacc = grads_out.w - e*vmath::sum(cell_light*grads_light);
		const float d_d = grads_out.w;// + vmath::sum(e*cell_light*grads_light);
		
		const float3 d_lacc = grads_light;
		const float3 d_l = e*grads_light;
		
		grads_out = make_float4(d_lacc, d_dacc);
		return make_float4(d_l, d_d);
	}
};
template<>
struct BlendState<float2, BLEND_BEERLAMBERT_EXCLUSIVE>{
	__device__ static inline float2 blend(const float2 acc, const float2 cell_in){
		const float t = exp(-acc.y); //transmissivity
		return make_float2(acc.x + t*cell_in.x, //color
			acc.y + cell_in.y); //opacity
	}
	__device__ static inline float2 blendGradients(float2& grads_out, const float2 cell_in, float2& cell_out){
		cell_out.y = max(cell_out.y - cell_in.y, 0.f);
		const float e = exp(-cell_out.y);
		const float cell_light = cell_in.x;//glm::xyz(cell);
		const float grads_light = grads_out.x;//glm::xyz(grads_out);
		
		const float d_dacc = grads_out.y - e*cell_light*grads_light;
		const float d_d = grads_out.y;// + vmath::sum(e*cell_light*grads_light);
		
		const float d_lacc = grads_light;
		const float d_l = e*grads_light;
		
		grads_out = make_float2(d_lacc, d_dacc);
		return make_float2(d_l, d_d);
	}
};
template struct BlendState<float1, BLEND_BEERLAMBERT_EXCLUSIVE>;

template<>
struct BlendState<float4, BLEND_ALPHA>{
	__device__ static inline float4 blend(const float4 acc, const float4 cell_in){
		const float t = (1.f-acc.w); //transmissivity
		return make_float4(acc.x + t*cell_in.x, acc.y + t*cell_in.y, acc.z + t*cell_in.z, //vmath::lerp(lm::rgb(cell)*cell.a, glm::rgb(acc),acc.a),
			//acc.a + (1.f-acc.a)*cell.a); //accumulatedDensity*accumulatedLight + (1 - accumulatedDensity)*cellLight;
			acc.w + t*cell_in.w); //1.f - (1.f - acc.w) * (1.f - cell_in.w));
	}
	__device__ static inline float4 blendGradients(float4& grads_out, const float4 cell_in, float4& cell_out){
		cell_out.w = (1.f-cell_out.w)/(1.f-cell_in.w);
		const float t = (1.f-cell_out.w); //transmissivity
		const float3 cell_light = make_float3(cell_in);
		const float3 grads_light = make_float3(grads_out);
		
		const float d_dacc = (1 - cell_in.w)*grads_out.w - vmath::sum(cell_light*grads_light);
		const float d_d = t*grads_out.w;
		
		const float3 d_lacc = grads_light;
		const float3 d_l = t*grads_light;
		
		grads_out = make_float4(d_lacc, d_dacc);
		return make_float4(d_l, d_d);
	}
};
template<>
struct BlendState<float2, BLEND_ALPHA>{
	__device__ static inline float2 blend(const float2 acc, const float2 cell_in){
		const float t = (1.f-acc.y); //transmissivity
		return make_float2(acc.x + t*cell_in.x, //vmath::lerp(lm::rgb(cell)*cell.a, glm::rgb(acc),acc.a),
			//acc.a + (1.f-acc.a)*cell.a); //accumulatedDensity*accumulatedLight + (1 - accumulatedDensity)*cellLight;
			acc.y + t*cell_in.y); //1.f - (1.f - acc.w) * (1.f - cell_in.w));
	}
	__device__ static inline float2 blendGradients(float2& grads_out, const float2 cell_in, float2& cell_out){
		cell_out.y = (1.f-cell_out.y)/(1.f-cell_in.y);
		const float t = (1.f-cell_out.y); //transmissivity
		const float cell_light = cell_in.x;
		const float grads_light = grads_out.x;
		
		const float d_dacc = (1 - cell_in.y)*grads_out.y - cell_light*grads_light;
		const float d_d = t*grads_out.y;
		
		const float d_lacc = grads_light;
		const float d_l = t*grads_light;
		
		grads_out = make_float2(d_lacc, d_dacc);
		return make_float2(d_l, d_d);
	}
};
template struct BlendState<float1, BLEND_ALPHA>;

template<>
struct BlendState<float4, BLEND_ALPHAADDITIVE>{
	__device__ static inline float4 blend(const float4 acc, const float4 cell_in){
		const float t = (1.f-cell_in.w); //transmissivity
		return make_float4(acc.x + t*cell_in.x, acc.y + t*cell_in.y, acc.z + t*cell_in.z, //vmath::lerp(lm::rgb(cell)*cell.a, glm::rgb(acc),acc.a),
			acc.w + cell_in.w); //unused
	}
	__device__ static inline float4 blendGradients(float4& grads_out, const float4 cell_in, float4& cell_out){
		cell_out.w = cell_out.w-cell_in.w;
		const float t = (1.f-cell_in.w); //transmissivity
		const float3 cell_light = make_float3(cell_in);
		const float3 grads_light = make_float3(grads_out);
		
		const float d_dacc = grads_out.w;
		const float d_d = grads_out.w  - vmath::sum(cell_light*grads_light) ;
		
		const float3 d_lacc = grads_light;
		const float3 d_l = t*grads_light;
		
		grads_out = make_float4(d_lacc, d_dacc);
		return make_float4(d_l, d_d);
	}
};
template<>
struct BlendState<float2, BLEND_ALPHAADDITIVE>{
	__device__ static inline float2 blend(const float2 acc, const float2 cell_in){
		const float t = (1.f-cell_in.y); //transmissivity
		return make_float2(acc.x + t*cell_in.x, //vmath::lerp(lm::rgb(cell)*cell.a, glm::rgb(acc),acc.a),
			acc.y + cell_in.y); //unused
	}
	__device__ static inline float2 blendGradients(float2& grads_out, const float2 cell_in, float2& cell_out){
		cell_out.y = cell_out.y-cell_in.y;
		const float t = (1.f-cell_in.y); //transmissivity
		const float cell_light = cell_in.x;
		const float grads_light = grads_out.x;
		
		const float d_dacc = grads_out.y;
		const float d_d = grads_out.y - cell_light*grads_light;
		
		const float d_lacc = grads_light;
		const float d_l = t*grads_light;
		
		grads_out = make_float2(d_lacc, d_dacc);
		return make_float2(d_l, d_d);
	}
};
template struct BlendState<float1, BLEND_ALPHAADDITIVE>;


template struct BlendState<float4, BLEND_ADDITIVE>;
template struct BlendState<float2, BLEND_ADDITIVE>;
template struct BlendState<float1, BLEND_ADDITIVE>;

}




#endif //_INCLUDE_BLENDING