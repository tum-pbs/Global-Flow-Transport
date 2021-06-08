#pragma once

#ifndef _INCLUDE_TRANFORMATIONS_2
#define _INCLUDE_TRANFORMATIONS_2

#include "vectormath.hpp"

#ifdef CBUF_FRUSTUM
struct FrustumParams{
	float near,far,left,right,top,bottom;
};
__constant__ FrustumParams c_frustum;

__host__ inline void setFrustumParams(FrustumParams &frustum, const float* params){
	checkCudaErrors(cudaMemcpyToSymbol(c_frustum, params, sizeof(FrustumParams)));
}
#endif //CBUF_FRUSTUM

using mat4 = vmath::float4x4;

struct Transformations{
	mat4 M_model;
	mat4 M_view;
	mat4 M_modelView;
	mat4 M_projection;
#ifdef CBUF_TRANSFORM_INVERSE
	mat4 M_model_inv;
	mat4 M_view_inv;
	mat4 M_modelView_inv;
	mat4 M_projection_inv;
#endif //CBUF_TRANSFORM_INVERSE
};
__constant__ Transformations c_transform;

__host__ inline void setTransformations(Transformations& transforms, const float* M, const float* V, const float* P){
	memset(&transforms, 0, sizeof(Transformations));
	transforms.M_model = (M==nullptr)? vmath::make_float4x4(1.f) : vmath::make_float4x4(M);
	transforms.M_view = (V==nullptr)? vmath::make_float4x4(1.f) : vmath::make_float4x4(V);
	transforms.M_projection = vmath::make_float4x4(P);
	transforms.M_modelView = vmath::matmul(transforms.M_view, transforms.M_model);
	/*
	LOG("Model: " << LOG_M44_COL(transforms.M_model) << std::endl);
	LOG("View: " << LOG_M44_COL(transforms.M_view) << std::endl);
	LOG("Proj: " << LOG_M44_COL(transforms.M_projection) << std::endl);
	LOG("MV: " << LOG_M44_COL(transforms.M_modelView) << std::endl);//*/
	
#ifdef CBUF_TRANSFORM_INVERSE
	transforms.M_model_inv = vmath::inverse(transforms.M_model);
	transforms.M_view_inv = vmath::inverse(transforms.M_view);
	transforms.M_projection_inv = vmath::inverse(transforms.M_projection);
	transforms.M_modelView_inv = vmath::inverse(transforms.M_modelView);
#endif
	
	checkCudaErrors(cudaMemcpyToSymbol(c_transform, &transforms, sizeof(Transformations)));
}

/* --- TRANSFORMATIONS ---
*  Coordinate Spaces:
*    (globalIdx: integer array index in the camera grid, 0<=IDX<dimensions. handeled by sampling kernel)
*    IDX: intermediate object-space coordiante of camera grid, float in [0,dimensons]. like OS.
*    (IDX-normalized: IDX-OS/dimensions in [0,1]) 
*    NDC: IDX-OS in [-1,1], as needed for projection
*    VS: view-space coordinates, before projection. camera at (0,0,0) looking along -z.
*    (WS: common world-space coordinates not used here)
*    OS: object-space coordinates of the volume grid, float in [0,dimensons], cell size=1.0
*    (sampling-idx/OS-IDX: shifted object-space coordinates s.t. cell centers are at integer positions. handeled by sampler cell offset)
*/

template<typename T>
__device__ constexpr T indexToCoords(const T idx){ return idx + 0.5f;}
template<typename T>
__device__ constexpr T coordsToIndex(const T coords){ return coords - 0.5f;}

//--- Transform Forward ---

//convert between view- and object-space given the (inverse) model-view matrix
__device__ inline float4 OStoVS(const float4 positionOS){
	return vmath::matmul(c_transform.M_modelView, positionOS);
}
//convert between NDC [-1,1] and view-space given an (inverse) projection matrix
__device__ inline float4 VStoNDC(const float4 positionVS){
	float4 positionNDC =  vmath::matmul(c_transform.M_projection, positionVS);
	return positionNDC/positionNDC.w;
}
//global 3D thread index to normalized device coordinates (NDC) given the 3D dimentions of the dispatch
//(z is in [-1,1] after perspective divide in forward pass, not in [0,1] as stored in the depth buffer)
__device__ inline float3 NDCtoIDX(const float4 ndc, const float3 dimensions){
	float3 position_normalized = (make_float3(ndc) * 0.5f) + 0.5f;
	return position_normalized * dimensions;
}
//wrapper to convert thread index directly to object space

__device__ inline float3 OStoIDX(const float4 positionOS, const float3 dimensions){
	return NDCtoIDX(VStoNDC(OStoVS(positionOS)), dimensions);
}
//change depth to be linear in VS for better sample coverage
__device__ inline float3 OStoIDXlinearDepth(const float4 positionOS, const float3 dimensions){
	float4 ndc = VStoNDC(OStoVS(positionOS));
	float3 idx = NDCtoIDX(ndc, dimensions);
	
	//correct mapping of ndc to idx s.t. idx has uniform depth/distance in VS
	float z = -c_transform.M_projection.c3.z/(ndc.z + c_transform.M_projection.c2.z);
	float t = -(z+c_frustum.near)/(c_frustum.far-c_frustum.near); //z=-(n+t*(f-n)) -> t = -(z+n)/(f-n)
	idx.z = t*dimensions.z;
	
	return idx;
	//*/
}

//--- Transform Backward
__device__ inline float4 IDXtoNDC(const float3 idx, const float3 dimensions_inv){
	const float3 position_normalized = idx * dimensions_inv;
	return make_float4((position_normalized * 2.0f) - 1.0f, 1.0f);
}
#ifdef CBUF_TRANSFORM_INVERSE
__device__ inline float4 NDCtoVS(const float4 positionNDC){
	float4 positionVS =  vmath::matmul(c_transform.M_projection_inv, positionNDC);
	return positionVS/positionVS.w;
}
__device__ inline float4 VStoOS(const float4 positionVS){
	return  vmath::matmul(c_transform.M_modelView_inv, positionVS);
}
__device__ inline float4 IDXtoOS(const float3 idx, const float3 dimensions_inv){
	return VStoOS(NDCtoVS(IDXtoNDC(idx, dimensions_inv)));
}
__device__ inline float4 IDXtoOSlinearDepth(const float3 idx, const float3 dimensions_inv){
	float4 ndc = IDXtoNDC(idx, dimensions_inv);
	
	//correct mapping of idx to ndc s.t. idx has uniform depth/distance in VS
	float t = idx.z * dimensions_inv.z; //float(idx.z)
	float z = - lerp(c_frustum.near, c_frustum.far, t);
	ndc.z = -c_transform.M_projection.c2.z -c_transform.M_projection.c3.z/z;//VStoNDC(float4(0.f,0.f,z,1.f)).z; //
	
	return VStoOS(NDCtoVS(ndc));
	//*/
}
#endif //CBUF_TRANSFORM_INVERSE


#endif //_INCLUDE_TRANFORMATIONS_2