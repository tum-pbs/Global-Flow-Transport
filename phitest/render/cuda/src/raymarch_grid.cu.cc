/* helpers */

#include <cuda_runtime.h>
#include "cuda-samples/Common/helper_cuda.h"
#include <iostream>
#include <stdexcept>

static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
  if (err == cudaSuccess) return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;
  exit(10);
}
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

//--- Logging and Profiling ---
//#define LOGGING
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

//total maximum block size (x*y*z) is 512 (1024, depending on architecture)
//these are NOT reversed when using REVERSE_THREAD_AXIS_ORDER
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 1

#define BLOCK_SIZE BLOCK_SIZE_X*BLOCK_SIZE_Y*BLOCK_SIZE_Z
#define BLOCK_DIMS BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z

#include"raymarch_grid.hpp"
#include"vectormath.hpp"

#define CBUF_FRUSTUM
#define CBUF_TRANSFORM_INVERSE
#include"transformations_v2.hpp"
#define CBUF_DIMENSIONS_INVERSE
#include"dimensions_v2.hpp"
#include"sampling_v2.hpp"
//#include"sampling_settings_v2.hpp"
#include"blending.hpp"
//#include"blending_settings.hpp"
#include"vector_io.hpp"

//https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
inline int32_t ceil_div(int32_t x,int32_t y){return (x+y-1)/y;}
__host__ inline dim3 gridDims3D(const int3 dimensions){
	return dim3(ceil_div(dimensions.x, BLOCK_SIZE_X),ceil_div(dimensions.y, BLOCK_SIZE_Y),ceil_div(dimensions.z, BLOCK_SIZE_Z));
}
__host__ inline dim3 blockDims3D(){
	return dim3(BLOCK_SIZE_X,BLOCK_SIZE_Y,BLOCK_SIZE_Z);
}

//returns the global 3D index of the current thread as vector.
__device__ inline int3 globalThreadIdx3D(){
	return make_int3(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z);
}

template<typename T, typename D>
__device__ inline bool isInDimensions(const T position, const D dimensions){
	return (position.x < dimensions.x && position.y < dimensions.y && position.z < dimensions.z);
}
template<typename T, typename D>
__device__ inline bool isInDimensions(const T x, const T y, const T z, const D dimensions){
	return (x < dimensions.x && y < dimensions.y && z < dimensions.z);
}

/* to render position as color for debugging
template<typename T>
__device__ inline T getColorFromPos(const float3 pos);
template<>
__device__ inline float1 getColorFromPos(const float3 pos){return make_float1(pos.z);}
template<>
__device__ inline float2 getColorFromPos(const float3 pos){return make_float2(pos);}
template<>
__device__ inline float4 getColorFromPos(const float3 pos){return make_float4(pos,1.f);}
*/

template<typename T, typename BLEND, Sampling::FilterMode FM, Sampling::BoundaryMode BM>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kRaymarchGridTransformLindepth(const T* input, T* output){
	/* front to back. get first and last sample point and interpolate for linear depth.
	* output: 3D array DHW with D=1
	* outputDimensions: dimensions of output with z number of steps
	*/
	const int3 globalIdx = globalThreadIdx3D();
#ifdef LOGGING
	if(globalIdx.x==0&& globalIdx.y==0&& globalIdx.z==0){
		printf("--- Kernel running! ---\n");
		printf("K: zMax = %f\n", static_cast<float>(max(c_dimensions.output.z-1, 1)));
	}
#endif
	if(globalIdx.x<c_dimensions.output.x && globalIdx.y<c_dimensions.output.y && globalIdx.z==0){
		const float zMax = static_cast<float>(max(c_dimensions.output.z-1, 1));
		const float3 nearPos = make_float3(IDXtoOS(indexToCoords(make_float3(globalIdx.x, globalIdx.y, 0.f)), c_dimensions.output_inv));
		const float3 farPos = make_float3(IDXtoOS(indexToCoords(make_float3(globalIdx.x, globalIdx.y, zMax)), c_dimensions.output_inv));
		const float3 step = (farPos - nearPos) / zMax;
		float3 samplePos = nearPos;
		/*
		T acc = getColorFromPos<T>(make_float3(
			globalIdx.x * c_dimensions.output_inv.x,
			globalIdx.y * c_dimensions.output_inv.y,
			zMax
		));*/
		T acc = vmath::make_cudaFloat<T>(0.f);
		for(int32_t i=0;i<c_dimensions.output.z;++i){
			const T sample = Sampling::sample3D<T, FM, BM>(samplePos, input, c_dimensions.input);
			acc = BLEND::blend(acc, sample);
			samplePos += step;
		}
		vectorIO::writeVectorType3D<T, T, int3>(acc, make_int3(globalIdx.x, globalIdx.y, 0), c_dimensions.output, output);
	}
	//*/
}


template<typename T, Sampling::FilterMode FM, Sampling::BoundaryMode BM>
__host__ inline void LauchRaymarchGridTransformLindepth_SwitchBlend(const Blending::BlendMode blendMode, \
		const dim3 grid, const dim3 block, const GPUDevice& d, \
		const T* input, T* output){
	switch(blendMode){
		case Blending::BLEND_BEERLAMBERT:
			LOG("BEERLAMBERT blending mode");
			kRaymarchGridTransformLindepth<T, Blending::BlendState<T, Blending::BLEND_BEERLAMBERT>, FM, BM><<<grid, block>>>(input, output);
			break;
		case Blending::BLEND_ALPHA:
			LOG("ALPHA blending mode");
			kRaymarchGridTransformLindepth<T, Blending::BlendState<T, Blending::BLEND_ALPHA>, FM, BM><<<grid, block>>>(input, output);
			break;
		case Blending::BLEND_ADDITIVE:
			LOG("ADDITIVE blending mode");
			kRaymarchGridTransformLindepth<T, Blending::BlendState<T, Blending::BLEND_ADDITIVE>, FM, BM><<<grid, block>>>(input, output);
			break;
		case Blending::BLEND_ALPHAADDITIVE:
			LOG("ALPHAADDITIVE blending mode");
			kRaymarchGridTransformLindepth<T, Blending::BlendState<T, Blending::BLEND_ALPHAADDITIVE>, FM, BM><<<grid, block>>>(input, output);
			break;
		default:
			LOG("Unknown blending mode");
			throw std::runtime_error("Unknown blending mode");
	}
}
template<typename T, Sampling::FilterMode FM>
__host__ inline void LauchRaymarchGridTransformLindepth_SwitchBoundaryBlend(const Sampling::BoundaryMode boundaryMode, \
		const Blending::BlendMode blendMode, const dim3 grid, const dim3 block, const GPUDevice& d, \
		const T* input, T* output){
	switch(boundaryMode){
		case Sampling::BOUNDARY_BORDER:
			LOG("BORDER boundary mode");
			LauchRaymarchGridTransformLindepth_SwitchBlend<T, FM, Sampling::BOUNDARY_BORDER>(blendMode, grid, block, d, input, output);
			break;
		case Sampling::BOUNDARY_CLAMP:
			LOG("CLAMP boundary mode");
			LauchRaymarchGridTransformLindepth_SwitchBlend<T, FM, Sampling::BOUNDARY_CLAMP>(blendMode, grid, block, d, input, output);
			break;
		case Sampling::BOUNDARY_WRAP:
			LOG("WRAP boundary mode");
			LauchRaymarchGridTransformLindepth_SwitchBlend<T, FM, Sampling::BOUNDARY_WRAP>(blendMode, grid, block, d, input, output);
			break;
		default:
			LOG("Unknown boundary mode");
			throw std::runtime_error("Unknown boundary mode");
	}
}
template<typename T>
__host__ inline void LauchRaymarchGridTransformLindepth_SwitchFilterBoundaryBlend(const Sampling::FilterMode filterMode, const Sampling::BoundaryMode boundaryMode, \
		const Blending::BlendMode blendMode, const dim3 grid, const dim3 block, const GPUDevice& d, \
		const T* input, T* output){
	switch(filterMode){
		case Sampling::FILTERMODE_LINEAR:
			LOG("LINEAR filter mode");
			LauchRaymarchGridTransformLindepth_SwitchBoundaryBlend<T, Sampling::FILTERMODE_LINEAR>(boundaryMode, blendMode, grid, block, d, input, output);
			break;
		case Sampling::FILTERMODE_NEAREST:
			LOG("NEAREST filter mode");
			LauchRaymarchGridTransformLindepth_SwitchBoundaryBlend<T, Sampling::FILTERMODE_NEAREST>(boundaryMode, blendMode, grid, block, d, input, output);
			break;
		case Sampling::FILTERMODE_MIN:
			LOG("MIN filter mode");
			LauchRaymarchGridTransformLindepth_SwitchBoundaryBlend<T, Sampling::FILTERMODE_MIN>(boundaryMode, blendMode, grid, block, d, input, output);
			break;
		case Sampling::FILTERMODE_MAX:
			LOG("MAX filter mode");
			LauchRaymarchGridTransformLindepth_SwitchBoundaryBlend<T, Sampling::FILTERMODE_MAX>(boundaryMode, blendMode, grid, block, d, input, output);
			break;
		default:
			LOG("Unknown filter mode");
			throw std::runtime_error("Unknown filter mode");
	}
	
}
//*/
template<typename T>
void RaymarchGridKernelLauncher(const GPUDevice& d,
		const T* input, const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* _frustum, int32_t numCameras,
		const Sampling::FilterMode filterMode, const Sampling::BoundaryMode boundaryMode,
		const Blending::BlendMode blendMode, const bool globalSampling,
		T* output, const long long int* output_shape){
	
	LOG("Start RaymarchGridKernelLauncher");
#ifdef PROFILING
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
	
	//precompute globals
	BEGIN_SAMPLE;
	LOG("Set dimensions");
	const size_t batchSize = input_shape[0];
	Dimensions dims;
	setDimensions(dims, input_shape, output_shape+1);
	LOG("Dimensions in: " << LOG_V3_XYZ(dims.input) << ", pitch: " << dims.input.x*sizeof(T));
	LOG("Dimensions out: " << LOG_V3_XYZ(dims.output) << ", pitch: " << dims.output.x*sizeof(T));
	const int3 kernelDims = make_int3(dims.output.x, dims.output.y, 1);
	LOG("Dimensions set");
	
	const size_t inputSliceSizeElements = vmath::prod(dims.input);
	const size_t outputSliceSizeElements = vmath::prod(kernelDims);
	
	//Camera params are set per camera
	Transformations transforms;
	FrustumParams frustum;
	int32_t lastCamera=-1;
	
	END_SAMPLE("Precompute and copy global constants");
	
	const dim3 grid = gridDims3D(kernelDims);
	const dim3 block = blockDims3D();
	LOG("Sample " << batchSize << " grids with " << numCameras << " cameras");
	LOG("Dimensions grid: " << LOG_V3_XYZ(grid));
	LOG("Dimensions block: " << LOG_V3_XYZ(block));
	
	for(size_t batch=0; batch<batchSize; ++batch){
		LOG("Grid/batch " << batch);
		const T *currInput = input+batch*inputSliceSizeElements;
		
		size_t camera = globalSampling? 0 : batch;
		size_t endCamera = globalSampling? numCameras : camera+1;
		for(; camera<endCamera; ++camera){ //TODO make cameras async/parallel
			LOG("Camera " << camera);
			T* currOutput = output+(globalSampling? batch*numCameras + camera : batch)*outputSliceSizeElements;
			BEGIN_SAMPLE;
			{
				//only set new camera if there are multiple
				setTransformations(transforms, M + batch*16, V + camera*16, P + camera*16);
				if(lastCamera!=camera){
					setFrustumParams(frustum, _frustum + camera*6);
					lastCamera=camera;
				}
				
			}
			END_SAMPLE("Set transformation CBuffer");
			BEGIN_SAMPLE;
			{
				LauchRaymarchGridTransformLindepth_SwitchFilterBoundaryBlend<T>(filterMode, boundaryMode, blendMode, grid, block, d, currInput, currOutput);
#ifdef PROFILING
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
			}
			END_SAMPLE("Sample kernel");
		}
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	LOG("End RaymarchGridKernelLauncher");
}
#define DEFINE_GPU_SPECS(T, C, VEC) \
	template<> \
	void RaymarchGridKernel<GPUDevice, T, C>::operator()(const GPUDevice& d, \
		const T* input, const long long int* input_shape, \
		const float* M, const float* V, const float* P, const float* frustum, int32_t numCameras, \
		const Sampling::FilterMode filterMode, const Sampling::BoundaryMode boundaryMode, \
		const Blending::BlendMode blendMode, const bool globalSampling, \
		T* output, const long long int* output_shape){ \
	RaymarchGridKernelLauncher<VEC>(d, \
		reinterpret_cast<const VEC*>(input), input_shape, \
		M, V, P, frustum, numCameras, \
		filterMode, boundaryMode, blendMode, globalSampling, \
		reinterpret_cast<VEC*>(output), output_shape); \
	} \
	template struct RaymarchGridKernel<GPUDevice, T, C>;
DEFINE_GPU_SPECS(float, 1, float1);
DEFINE_GPU_SPECS(float, 2, float2);
DEFINE_GPU_SPECS(float, 4, float4);

#undef DEFINE_GPU_SPECS

/* --- Gradient Pass --- */


template<typename T, bool SetZero, bool AddGrad>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kNormalize3DGradients(const T* gradIn, sampleCount_t* num_samples, T* gradOut){
	//N.B. gradIn and gradOut may be the same (so don't make them __restrict__)
	const int3 globalIdx = globalThreadIdx3D();
	if(isInDimensions<int3,int3>(globalIdx, c_dimensions.input)){
		const size_t flatIdx = vectorIO::flatIdx3D(globalIdx.x, globalIdx.y, globalIdx.z, c_dimensions.input);
		const sampleCount_t n = num_samples[flatIdx];
		if(n > sampleCount_t(0)){
			const float weight = 1.0f / static_cast<float>(n);
			T data = vectorIO::readVectorType<T, T>(flatIdx, gradIn) * weight;
			if(AddGrad){
				data = data + vectorIO::readVectorType<T, T>(flatIdx, gradOut);
			}
			vectorIO::writeVectorType<T, T>(data, flatIdx, gradOut);
			
			if(SetZero){
				num_samples[flatIdx] = sampleCount_t(0);
			}
		}
	}
}

template<typename T,  typename BLEND, Sampling::FilterMode FM, Sampling::BoundaryMode BM>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kRaymarchGridTransformLindepthGrad(const T* input, T* inputGrad, sampleCount_t* sampleCounter, const T* output, const T* outputGrad){
	//back to front
	const int3 globalIdx = globalThreadIdx3D();
#ifdef LOGGING
	if(globalIdx.x==0&& globalIdx.y==0&& globalIdx.z==0){
		printf("--- Gradient Kernel running! ---\n");
		printf("K: zMax = %f\n", static_cast<float>(max(c_dimensions.output.z-1, 1)));
	}
#endif
	if(globalIdx.x<c_dimensions.output.x && globalIdx.y<c_dimensions.output.y && globalIdx.z==0){
		const float zMax = static_cast<float>(max(c_dimensions.output.z-1, 1));
		const float3 nearPos = make_float3(IDXtoOS(indexToCoords(make_float3(globalIdx.x, globalIdx.y, 0.f)), c_dimensions.output_inv));
		const float3 farPos = make_float3(IDXtoOS(indexToCoords(make_float3(globalIdx.x, globalIdx.y, zMax)), c_dimensions.output_inv));
		const float3 step = (farPos - nearPos) / zMax;
		float3 samplePos = farPos;
		
		T acc = vectorIO::readVectorType3D<T, T, int32_t, int3>(globalIdx.x, globalIdx.y, 0, c_dimensions.output, output);
		T gradOut = vectorIO::readVectorType3D<T, T, int32_t, int3>(globalIdx.x, globalIdx.y, 0, c_dimensions.output, outputGrad);
		for(int32_t i=(c_dimensions.output.z-1);i>=0;--i){
			const T sample = Sampling::sample3D<T, FM, BM>(samplePos, input, c_dimensions.input);
			const T grad = BLEND::blendGradients(gradOut, sample, acc);
			 //TODO this is for linear only, implement other filter modes
			Sampling::scatter3D<T, FM, BM, sampleCount_t>(grad, samplePos, inputGrad, c_dimensions.input, sampleCounter);
			samplePos -= step;
		}
	}
}

template<typename T, Sampling::FilterMode FM, Sampling::BoundaryMode BM>
__host__ inline void LauchRaymarchGridTransformLindepthGrad_SwitchBlend(const Blending::BlendMode blendMode, \
		const dim3 grid, const dim3 block, const GPUDevice& d, \
		const T* input, T* inputGrad, sampleCount_t* sampleCounter, const T* output, const T* outputGrad){
	switch(blendMode){
		case Blending::BLEND_BEERLAMBERT:
			LOG("BEERLAMBERT blending mode");
			kRaymarchGridTransformLindepthGrad<T, Blending::BlendState<T, Blending::BLEND_BEERLAMBERT>, FM, BM><<<grid, block>>>(input, inputGrad, sampleCounter, output, outputGrad);
			break;
		case Blending::BLEND_ALPHA:
			LOG("ALPHA blending mode");
			kRaymarchGridTransformLindepthGrad<T, Blending::BlendState<T, Blending::BLEND_ALPHA>, FM, BM><<<grid, block>>>(input, inputGrad, sampleCounter, output, outputGrad);
			break;
		case Blending::BLEND_ADDITIVE:
			LOG("ADDITIVE blending mode");
			kRaymarchGridTransformLindepthGrad<T, Blending::BlendState<T, Blending::BLEND_ADDITIVE>, FM, BM><<<grid, block>>>(input, inputGrad, sampleCounter, output, outputGrad);
			break;
		case Blending::BLEND_ALPHAADDITIVE:
			LOG("ALPHAADDITIVE blending mode");
			kRaymarchGridTransformLindepthGrad<T, Blending::BlendState<T, Blending::BLEND_ALPHAADDITIVE>, FM, BM><<<grid, block>>>(input, inputGrad, sampleCounter, output, outputGrad);
			break;
		default:
			LOG("Unknown blending mode");
			throw std::runtime_error("Unknown blending mode");
	}
}
template<typename T, Sampling::FilterMode FM>
__host__ inline void LauchRaymarchGridTransformLindepthGrad_SwitchBoundaryBlend(const Sampling::BoundaryMode boundaryMode, \
		const Blending::BlendMode blendMode, const dim3 grid, const dim3 block, const GPUDevice& d, \
		const T* input, T* inputGrad, sampleCount_t* sampleCounter, const T* output, const T* outputGrad){
	switch(boundaryMode){
		case Sampling::BOUNDARY_BORDER:
			LOG("BORDER boundary mode");
			LauchRaymarchGridTransformLindepthGrad_SwitchBlend<T, FM, Sampling::BOUNDARY_BORDER>(blendMode, grid, block, d, input, inputGrad, sampleCounter, output, outputGrad);
			break;
		case Sampling::BOUNDARY_CLAMP:
			LOG("CLAMP boundary mode");
			LauchRaymarchGridTransformLindepthGrad_SwitchBlend<T, FM, Sampling::BOUNDARY_CLAMP>(blendMode, grid, block, d, input, inputGrad, sampleCounter, output, outputGrad);
			break;
		case Sampling::BOUNDARY_WRAP:
			LOG("WRAP boundary mode");
			LauchRaymarchGridTransformLindepthGrad_SwitchBlend<T, FM, Sampling::BOUNDARY_WRAP>(blendMode, grid, block, d, input, inputGrad, sampleCounter, output, outputGrad);
			break;
		default:
			LOG("Unknown boundary mode");
			throw std::runtime_error("Unknown boundary mode");
	}
}
template<typename T>
__host__ inline void LauchRaymarchGridTransformLindepthGrad_SwitchFilterBoundaryBlend(const Sampling::FilterMode filterMode, const Sampling::BoundaryMode boundaryMode, \
		const Blending::BlendMode blendMode, const dim3 grid, const dim3 block, const GPUDevice& d, \
		const T* input, T* inputGrad, sampleCount_t* sampleCounter, const T* output, const T* outputGrad){
	switch(filterMode){
		case Sampling::FILTERMODE_LINEAR:
			LOG("LINEAR filter mode");
			LauchRaymarchGridTransformLindepthGrad_SwitchBoundaryBlend<T, Sampling::FILTERMODE_LINEAR>(boundaryMode, blendMode, grid, block, d, input, inputGrad, sampleCounter, output, outputGrad);
			break;
		case Sampling::FILTERMODE_NEAREST:
			LOG("NEAREST filter mode");
			LauchRaymarchGridTransformLindepthGrad_SwitchBoundaryBlend<T, Sampling::FILTERMODE_NEAREST>(boundaryMode, blendMode, grid, block, d, input, inputGrad, sampleCounter, output, outputGrad);
			break;
		case Sampling::FILTERMODE_MIN:
			LOG("MIN filter mode");
			LauchRaymarchGridTransformLindepthGrad_SwitchBoundaryBlend<T, Sampling::FILTERMODE_MIN>(boundaryMode, blendMode, grid, block, d, input, inputGrad, sampleCounter, output, outputGrad);
			break;
		case Sampling::FILTERMODE_MAX:
			LOG("MAX filter mode");
			LauchRaymarchGridTransformLindepthGrad_SwitchBoundaryBlend<T, Sampling::FILTERMODE_MAX>(boundaryMode, blendMode, grid, block, d, input, inputGrad, sampleCounter, output, outputGrad);
			break;
		default:
			LOG("Unknown filter mode");
			throw std::runtime_error("Unknown filter mode");
	}
	
}


template<typename T>
void RaymarchGridKernelLauncherGrad(const GPUDevice& d,
		const T* input, T* inputGrad, T* sampleBuffer, sampleCount_t* sampleCounter, const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* _frustum, int32_t numCameras,
		const Sampling::FilterMode filterMode, const Sampling::BoundaryMode boundaryMode,
		const Blending::BlendMode blendMode, const bool globalSampling,
		const T* output, const T* outputGrad, const long long int* output_shape){
	
	LOG("Start RaymarchGridKernelLauncherGrad");
#ifdef PROFILING
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
	
	//precompute globals
	BEGIN_SAMPLE;
	LOG("Set dimensions");
	const size_t batchSize = input_shape[0];
	Dimensions dims;
	setDimensions(dims, input_shape, output_shape+1);
	LOG("Dimensions in: " << LOG_V3_XYZ(dims.input) << ", pitch: " << dims.input.x*sizeof(T));
	LOG("Dimensions out: " << LOG_V3_XYZ(dims.output) << ", pitch: " << dims.output.x*sizeof(T));
	const int3 kernelDims = make_int3(dims.output.x, dims.output.y, 1);
	LOG("Dimensions set");
	
	const size_t inputSliceSizeElements = vmath::prod(dims.input);
	const size_t outputSliceSizeElements = vmath::prod(kernelDims);
	
	//Camera params are set per camera
	Transformations transforms;
	FrustumParams frustum;
	int32_t lastCamera=-1;
	
	END_SAMPLE("Precompute and copy global constants");
	
	const dim3 grid = gridDims3D(kernelDims);
	const dim3 block = blockDims3D();
	LOG("Sample " << batchSize << " grids with " << numCameras << " cameras");
	LOG("Dimensions grid: " << LOG_V3_XYZ(grid));
	LOG("Dimensions block: " << LOG_V3_XYZ(block));
	
	// zero gradient buffers
	BEGIN_SAMPLE;
	{
		checkCudaErrors(cudaMemset(inputGrad, 0, inputSliceSizeElements*sizeof(T)*batchSize));
		if(sampleCounter!=nullptr) checkCudaErrors(cudaMemset(sampleCounter, 0, inputSliceSizeElements*sizeof(sampleCount_t)));
		if(sampleBuffer!=nullptr) checkCudaErrors(cudaMemset(sampleBuffer, 0, inputSliceSizeElements*sizeof(T)));
#ifdef PROFILING
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
	}
	END_SAMPLE("Set gradient buffers zero");
	
	
	for(size_t batch=0; batch<batchSize; ++batch){
		LOG("Grid/batch " << batch);
		LOG("Dimensions in: " << LOG_V3_XYZ(dims.input) << ", pitch: " << dims.input.x*sizeof(T));
		const size_t inputOffset = batch*inputSliceSizeElements;
		const T* currInput = input + inputOffset;
		T* currInputGrad = inputGrad + inputOffset;
		
	
		size_t camera = globalSampling? 0 : batch;
		size_t endCamera = globalSampling? numCameras : camera+1;
		for(; camera<endCamera; ++camera){ //TODO make cameras async/parallel?
			LOG("Camera " << camera);
			const size_t outputOffset = (globalSampling? batch*numCameras + camera : batch)*outputSliceSizeElements;
			const T* currOutput = output + outputOffset;
			const T* currOutputGrad = outputGrad + outputOffset;
			
			//normalization is per-camera/view
			const bool bufferGradientsForNormalization = (NORMALIZE_GRADIENTS!=NORMALIZE_GRADIENT_NONE) && globalSampling && (camera>0);
			T* inputGradBuffer;
			if(bufferGradientsForNormalization){
				inputGradBuffer = sampleBuffer;
			}else{
				inputGradBuffer = currInputGrad;
			}
			
			BEGIN_SAMPLE;
			{
				//only set new camera if there are multiple
				setTransformations(transforms, M + batch*16, V + camera*16, P + camera*16);
				if(lastCamera!=camera){
					setFrustumParams(frustum, _frustum + camera*6);
					lastCamera=camera;
				}
				
			}
			END_SAMPLE("Set transformation CBuffer");
			
			BEGIN_SAMPLE;
			{
				LauchRaymarchGridTransformLindepthGrad_SwitchFilterBoundaryBlend<T>(filterMode, boundaryMode, blendMode, grid, block, d, \
					currInput, inputGradBuffer, sampleCounter, currOutput, currOutputGrad);
#ifdef PROFILING
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
			}
			END_SAMPLE("Sample kernel");
		
			if(NORMALIZE_GRADIENTS!=NORMALIZE_GRADIENT_NONE){
				LOG("Normalize gradients");
				BEGIN_SAMPLE;
				{
					const dim3 grad_grid = gridDims3D(dims.input);
					if(bufferGradientsForNormalization){
						kNormalize3DGradients<T, true, true><<<grad_grid, block>>>(inputGradBuffer, sampleCounter, currInputGrad);
						checkCudaErrors(cudaMemset(sampleBuffer, 0, inputSliceSizeElements*sizeof(T)));
					}else{
						kNormalize3DGradients<T, true, false><<<grad_grid, block>>>(inputGradBuffer, sampleCounter, currInputGrad);
					}
					//checkCudaErrors(cudaMemset(sampleCounter, 0, inputSliceSizeElements*sizeof(uint32_t)));
#ifdef PROFILING
					CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
				}
				END_SAMPLE("Grad normalize");
			}
		
		}
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	LOG("End RaymarchGridKernelLauncherGrad");
}
#define DEFINE_GPU_SPECS(T, C, VEC) \
	template<> \
	void RaymarchGridGradKernel<GPUDevice, T, C>::operator()( \
		const GPUDevice& d, \
		const T* input, T* inputGrads, T* sampleBuffer, sampleCount_t* sampleCounter, const long long int* input_shape, \
		const float* M, const float* V, const float* P, const float* frustum, int32_t numCameras, \
		const Sampling::FilterMode filterMode, const Sampling::BoundaryMode boundaryMode, \
		const Blending::BlendMode blendMode, const bool globalSampling, \
		const T* output, const T* outputGrads, const long long int* output_shape \
		){ \
	RaymarchGridKernelLauncherGrad<VEC>(d, \
		reinterpret_cast<const VEC*>(input), reinterpret_cast<VEC*>(inputGrads), reinterpret_cast<VEC*>(sampleBuffer), sampleCounter, input_shape, \
		M, V, P, frustum, numCameras, \
		filterMode, boundaryMode, blendMode, globalSampling, \
		reinterpret_cast<const VEC*>(output), reinterpret_cast<const VEC*>(outputGrads), output_shape); \
	} \
	template struct RaymarchGridGradKernel<GPUDevice, T, C>;
DEFINE_GPU_SPECS(float, 1, float1);
DEFINE_GPU_SPECS(float, 2, float2);
DEFINE_GPU_SPECS(float, 4, float4);


#undef DEFINE_GPU_SPECS
