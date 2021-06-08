#pragma once

#ifndef VECTOR_IO
#define VECTOR_IO

namespace vectorIO{
	
	
	template<typename C, typename D>
	__device__ constexpr size_t flatIdx4D(const C pos, const D dims);
	template<typename C, typename D>
	__device__ constexpr size_t flatIdx3D(const C pos, const D dims);
	template<typename C, typename D, typename I>
	__device__ constexpr size_t flatIdx3DChannel(const C pos, const I channel, const D dims, const I channel_dim);
	template<typename C, typename D>
	__device__ constexpr size_t flatIdx2D(const C pos, const D dims);
	
	template<typename C, typename D>
	__device__ constexpr size_t flatIdx3D(const C x, const C y, const C z, const D dims);
	template<typename C, typename D>
	__device__ constexpr size_t flatIdx2D(const C x, const C y, const D dims);
	
	template<typename C, typename D>
	__device__ constexpr C unflatIdx3D(const size_t flatIdx, const D dims);
	
	
	//vector conversion cuda <-> glm
	template<typename I, typename O>
	__device__ inline O toVector(const I v);
	
	//--- Buffer IO for Vector Types ---
	//Read
	template<typename V, int VSIZE, typename T>
	__device__ inline V readVectorTypeAbs(const size_t idx, const T* buf);
	
	template<typename V, typename T>
	__device__ inline V readVectorType(const size_t idx, const T* buf);
	
	template<typename V, typename T, typename I>
	__device__ inline V readVectorType3D(const I pos, const I dims, const T* buf);
	
	template<typename V, typename T, typename C, typename I>
	__device__ inline V readVectorType3D(const C x, const C y, const C z, const I dims, const T* buf);
	
	template<typename V, typename T, typename I>
	__device__ inline V readVectorType3DBounds(const I pos, const I dims, const T* buf);
	
	template<typename V, typename T, typename C, typename I>
	__device__ inline V readVectorType3DBounds(const C x, const C y, const C z, const I dims, const T* buf);
	
	//Write

	//vector type V and single element type T with vector size VSIZE. T must be compatible with V.
	//write the first VSIZE elements of vector 'v' to flat index 'idx'. idx is relative to T.
	template<typename V, int VSIZE, typename T>
	__device__ inline void writeVectorTypeAbs(const V v, const size_t idx, T* buf);
	
	template<typename V, typename T>
	__device__ inline void writeVectorType(const V v, const size_t idx, T* buf);
	
	template<typename V, typename T, typename I>
	__device__ inline void writeVectorType3D(const V v, const I pos, const I dims, T* buf);
	
}

#include "vector_io.inl"

#endif //VECTOR_IO