#pragma once

#ifndef _INCLUDE_TRANFORMATIONS
#define _INCLUDE_TRANFORMATIONS

#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include <glm/gtc/matrix_inverse.hpp>

#ifdef CBUF_FRUSTUM
struct FrustumParams{
	float near,far,left,right,top,bottom;
};
__constant__ FrustumParams c_frustum;

__host__ inline void setFrustumParams(FrustumParams &frustum, const float* params){
	checkCudaErrors(cudaMemcpyToSymbol(c_frustum, params, sizeof(FrustumParams)));
}
#endif //CBUF_FRUSTUM

using mat4 = glm::mat4;

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

inline glm::mat4 mat4FromArray(const float* arr){
	return glm::mat4(arr[0], arr[1], arr[2], arr[3],
					 arr[4], arr[5], arr[6], arr[7],
					 arr[8], arr[9], arr[10], arr[11],
					 arr[12], arr[13], arr[14], arr[15]);
}
__host__ inline void setTransformations(Transformations& transforms, const float* M, const float* V, const float* P){
	memset(&transforms, 0, sizeof(Transformations));
	transforms.M_model = (M==nullptr)? glm::mat4(1.f) : mat4FromArray(M);
	transforms.M_view = (V==nullptr)? glm::mat4(1.f) : mat4FromArray(V);
	transforms.M_projection = mat4FromArray(P);
	transforms.M_modelView = transforms.M_view * transforms.M_model;
	/*
	LOG("Model: " << LOG_M44_COL(transforms.M_model) << std::endl);
	LOG("View: " << LOG_M44_COL(transforms.M_view) << std::endl);
	LOG("Proj: " << LOG_M44_COL(transforms.M_projection) << std::endl);
	LOG("MV: " << LOG_M44_COL(transforms.M_modelView) << std::endl);//*/
	
#ifdef CBUF_TRANSFORM_INVERSE
	transforms.M_model_inv = glm::inverse(transforms.M_model);
	transforms.M_view_inv = glm::inverse(transforms.M_view);
	transforms.M_projection_inv = glm::inverse(transforms.M_projection);
	transforms.M_modelView_inv = glm::inverse(transforms.M_modelView);
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
__device__ inline glm::vec4 OStoVS(const glm::vec4 positionOS){
	return c_transform.M_modelView * positionOS;
}
//convert between NDC [-1,1] and view-space given an (inverse) projection matrix
__device__ inline glm::vec4 VStoNDC(const glm::vec4 positionVS){
	glm::vec4 positionNDC = c_transform.M_projection * positionVS;
	return positionNDC/positionNDC.w;
}
//global 3D thread index to normalized device coordinates (NDC) given the 3D dimentions of the dispatch
//(z is in [-1,1] after perspective divide in forward pass, not in [0,1] as stored in the depth buffer)
__device__ inline glm::vec3 NDCtoIDX(const glm::vec4 ndc, const glm::vec3 dimensions){
	glm::vec3 position_normalized = glm::fma(glm::xyz(ndc), glm::vec3(0.5f), glm::vec3(0.5f));
	return position_normalized * dimensions;
}
//wrapper to convert thread index directly to object space

__device__ inline glm::vec3 OStoIDX(const glm::vec4 positionOS, const glm::vec3 dimensions){
	return NDCtoIDX(VStoNDC(OStoVS(positionOS)), dimensions);
}
//change depth to be linear in VS for better sample coverage
__device__ inline glm::vec3 OStoIDXlinearDepth(const glm::vec4 positionOS, const glm::vec3 dimensions){
	glm::vec4 ndc = VStoNDC(OStoVS(positionOS));
	glm::vec3 idx = NDCtoIDX(ndc, dimensions);
	
	//correct mapping of ndc to idx s.t. idx has uniform depth/distance in VS
	float z = -c_transform.M_projection[3].z/(ndc.z + c_transform.M_projection[2].z);
	float t = -(z+c_frustum.near)/(c_frustum.far-c_frustum.near); //z=-(n+t*(f-n)) -> t = -(z+n)/(f-n)
	idx.z = t*dimensions.z;
	
	return idx;
	//*/
}

//--- Transform Backward
__device__ inline glm::vec4 IDXtoNDC(const glm::vec3 idx, const glm::vec3 dimensions_inv){
	const glm::vec3 position_normalized = idx * dimensions_inv;
	return glm::vec4(glm::fma(position_normalized, glm::vec3(2.0f), glm::vec3(-1.0f)), 1.0f);
}
#ifdef CBUF_TRANSFORM_INVERSE
__device__ inline glm::vec4 NDCtoVS(const glm::vec4 positionNDC){
	glm::vec4 positionVS = c_transform.M_projection_inv * positionNDC;
	return positionVS/positionVS.w;
}
__device__ inline glm::vec4 VStoOS(const glm::vec4 positionVS){
	return c_transform.M_modelView_inv * positionVS;
}
__device__ inline glm::vec4 IDXtoOS(const glm::vec3 idx, const glm::vec3 dimensions_inv){
	return VStoOS(NDCtoVS(IDXtoNDC(idx, dimensions_inv)));
}
__device__ inline glm::vec4 IDXtoOSlinearDepth(const glm::vec3 idx, const glm::vec3 dimensions_inv){
	glm::vec4 ndc = IDXtoNDC(idx, dimensions_inv);
	
	//correct mapping of idx to ndc s.t. idx has uniform depth/distance in VS
	float t = idx.z * dimensions_inv.z; //float(idx.z)
	float z = - vmath::flerp(c_frustum.near, c_frustum.far, t);
	ndc.z = -c_transform.M_projection[2].z -c_transform.M_projection[3].z/z;//VStoNDC(glm::vec4(0.f,0.f,z,1.f)).z; //
	
	return VStoOS(NDCtoVS(ndc));
	//*/
}
#endif //CBUF_TRANSFORM_INVERSE


#endif //_INCLUDE_TRANFORMATIONS