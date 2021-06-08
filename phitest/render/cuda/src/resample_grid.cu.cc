
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include "cuda-samples/Common/helper_cuda.h"
#include <iostream>
#include <string>
#include "resample_grid.hpp"
#include "render_errors.hpp"
//#define LOGGING

#ifdef LOGGING
#define PROFILING
#endif

#ifdef PROFILING
#include <chrono>
#endif

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/vec_swizzle.hpp>

//#include <helper_math.h> //operators for cuda vector types
#include "vectormath_helper.hpp"
#include "vector_io.hpp"

//kernel_setup params
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 4
#define BLOCK_SIZE_Z 4
#define CBUF_DIMENSIONS
#define CBUF_DIMENSIONS_INVERSE
#include "kernel_setup.hpp"

#define CBUF_TRANSFORM_INVERSE
#define CBUF_FRUSTUM
#include "transformations.hpp"
#include "sampling.hpp"

#define MIPGEN_BLOCK_X 8
#define MIPGEN_BLOCK_Y 2
#define MIPGEN_BLOCK_Z 2


template<typename T>
__global__ void 
__launch_bounds__((MIPGEN_BLOCK_X*MIPGEN_BLOCK_Y*MIPGEN_BLOCK_Z))
kGenerateMip3D(const T* UG_PTR input, T* UG_PTR output, glm::ivec3 dimensions){
	__shared__ T voxelBlock[MIPGEN_BLOCK_Z][MIPGEN_BLOCK_Y][MIPGEN_BLOCK_X];
	MAKE_GLOBAL_INDEX;
	if(isInDimensions<glm::ivec3,glm::ivec3>(globalIdx, dimensions)){
		voxelBlock[threadIdx.z][threadIdx.y][threadIdx.x] = vectorIO::readVectorType3D<T, T, glm::ivec3>(globalIdx, dimensions, input);
	}else{
		T def = {0};
		voxelBlock[threadIdx.z][threadIdx.y][threadIdx.x] = def;
	}
	__syncthreads();
	if((threadIdx.z&1)==0){ //(threadIdx.z%2)==0 even index
		voxelBlock[threadIdx.z][threadIdx.y][threadIdx.x] = (voxelBlock[threadIdx.z][threadIdx.y][threadIdx.x] + voxelBlock[threadIdx.z+1][threadIdx.y][threadIdx.x])*0.5f;
	}
	__syncthreads();
	if((threadIdx.z&1)==0 && (threadIdx.y&1)==0){
		voxelBlock[threadIdx.z][threadIdx.y][threadIdx.x] = (voxelBlock[threadIdx.z][threadIdx.y][threadIdx.x] + voxelBlock[threadIdx.z][threadIdx.y+1][threadIdx.x])*0.5f;
	}
	__syncthreads();
	glm::ivec3 outIdx = globalIdx>>1;
	glm::ivec3 outDims = dimensions>>1;
	if((threadIdx.z&1)==0 && (threadIdx.y&1)==0 && (threadIdx.x&1)==0 && isInDimensions<glm::ivec3,glm::ivec3>(outIdx, outDims)){
		vectorIO::writeVectorType3D<T,T, glm::ivec3>((voxelBlock[threadIdx.z][threadIdx.y][threadIdx.x] + voxelBlock[threadIdx.z][threadIdx.y][threadIdx.x+1])*0.5f, outIdx, outDims, output);
	}
}
//run on the (lower) input res and ADD to the 8 corresponding output elements
template<typename T>
__global__ void 
__launch_bounds__((MIPGEN_BLOCK_X*MIPGEN_BLOCK_Y*MIPGEN_BLOCK_Z))
kCollapseGradMip3D(const T* UG_PTR input, T* UG_PTR output, glm::ivec3 dimensions){
	__shared__ T voxelBlock[MIPGEN_BLOCK_Z][MIPGEN_BLOCK_Y][MIPGEN_BLOCK_X];
	MAKE_GLOBAL_INDEX;
	glm::ivec3 inDims = dimensions>>1;
	T data = vectorIO::readVectorType3D<T, T, glm::ivec3>(globalIdx, inDims, input);
	glm::ivec3 outIdx = globalIdx<<1;
	
	glm::ivec3 offset = glm::ivec3(0);
	#pragma unroll
	for(; offset.z<2; ++offset.z){
		#pragma unroll
		for(; offset.y<2; ++offset.y){
			#pragma unroll
			for(; offset.x<2; ++offset.x){
				size_t idx = vectorIO::flatIdx3D(outIdx + offset, dimensions);
				output[idx] += data;
			}
		}
	}
}

//use ALLOCATE_MIPS to have MipAtlas3D allocate memory for mip maps using cudaMalloc, otherwise provide your own buffer (e.g. from tensorflow temp allocation)
//using this might cause OOM issues when using tensorflow with its default greedy memory allocation
//#define ALLOCATE_MIPS

#define ALIGN_UP(addr, align) (((addr) + (align-1) ) & ~(align-1))
#define ALIGN_ADDR_UP(addr, align) ((reinterpret_cast<uintptr_t>(addr) + static_cast<uintptr_t>(align-1) ) & ~static_cast<uintptr_t>(align-1))
template<typename T>
class MipAtlas3D{
public:
	__host__ MipAtlas3D(const size_t level, const glm::ivec3 baseDims, const size_t textureAlignment=128)
			: m_level(level), m_allocated(false), m_initialized(false), m_generated(false), m_baseDimensions(baseDims), m_baseLevel(nullptr), m_ptr(nullptr), m_alignment(textureAlignment){
		LOG("Init mip atlas with "<<m_level<<" levels and "<<m_alignment<<" byte alignment");
		m_dimensions = new glm::ivec3[m_level+1];
		m_mipOffsetsBytes = new size_t[m_level];
		
		m_dimensions[0] = m_baseDimensions;
		m_mipAtlasSizeBytes = (m_level+1)*sizeof(const T*);
		
		m_mipsSizeBytes=0;
		for(int32_t m=1; m<(m_level+1); ++m){
			m_dimensions[m] = m_baseDimensions>>m;
			m_mipOffsetsBytes[m-1] = m_mipsSizeBytes;
			m_mipsSizeBytes += ALIGN_UP(vmath::prod(m_dimensions[m])*sizeof(T), m_alignment);
		}
		m_totalSize = m_mipsSizeBytes + m_mipAtlasSizeBytes + sizeof(const T*);
		
		m_mips = new T*[m_level];
		checkCudaErrors(cudaMallocHost(&m_maps_raw, m_mipAtlasSizeBytes + sizeof(const T*)));
		m_maps = reinterpret_cast<const T**>(ALIGN_ADDR_UP(m_maps_raw, sizeof(const T*)));
		LOG("Mip atlas size "<<m_mipAtlasSizeBytes<<"B at host "<<m_maps<<", mips size "<<m_mipsSizeBytes<<"B, total "<<m_totalSize<<"B.");
	}
	__host__ ~MipAtlas3D(){
		//free memory
		delete[] m_dimensions;
		cudaFreeHost(m_maps_raw);
		delete[] m_mips;
		delete[] m_mipOffsetsBytes;
		if(m_allocated){
			cudaFree(m_ptr);
		}
	}
	__host__ void initialize(void* buffer, const bool allocate=false, const bool setZero=false){
		if(m_allocated and allocate){
			throw std::logic_error("Mips already allocated.");
		}
		
		uint8_t* tmp_ptr;
		if(allocate){
			LOG("Allocate "<<m_totalSize<<"B of memory for "<<m_level<<" mips and atlas");
			checkCudaErrors(cudaMalloc(&m_ptr, m_totalSize));
			m_allocated = true;
			tmp_ptr = static_cast<uint8_t*>(m_ptr);
		}else{
			tmp_ptr = reinterpret_cast<uint8_t*>(ALIGN_ADDR_UP(buffer, m_alignment));
			m_ptr = buffer;
		}
		
		if(setZero){
			checkCudaErrors(cudaMemset(tmp_ptr, 0, m_totalSize));
		}
		
		for(int32_t m=0; m<m_level; ++m){
			m_mips[m] = reinterpret_cast<T*>(tmp_ptr + m_mipOffsetsBytes[m]);
			LOG("Mip "<<m+1<<" at device "<<m_mips[m]<<" (offset "<<m_mipOffsetsBytes[m]<<"B)");
		}
		//mip data offset table (atlas) at the end
		m_dd_maps = reinterpret_cast<const T * *>(ALIGN_ADDR_UP(tmp_ptr + m_mipsSizeBytes, sizeof(const T*)));
		LOG("device mip atlas at "<<m_dd_maps);
		m_initialized = true;
	}
	
	__host__ void generate(const T* baseLevel, const cudaStream_t& stream, const bool async=false){
		if(!m_initialized){
			LOG("Initialize memory for mips before generation.");
			throw std::logic_error("MipAtlas not initialized.");
		}
		m_baseLevel = baseLevel;
		m_maps[0] = baseLevel;
		LOG("Generate "<<m_level<<" mips for input of "<<LOG_V3_XYZ(m_baseDimensions)<<" at "<<m_maps[0]);
		const dim3 mipBlock(MIPGEN_BLOCK_X,MIPGEN_BLOCK_Y,MIPGEN_BLOCK_Z);
		for(int32_t m=0; m<m_level; ++m){
			const dim3 mipGrid = dim3(ceil_div( m_dimensions[m].x, MIPGEN_BLOCK_X),
										ceil_div( m_dimensions[m].y, MIPGEN_BLOCK_Y),
										ceil_div( m_dimensions[m].z, MIPGEN_BLOCK_Z));
			kGenerateMip3D<T><<<mipGrid, mipBlock, 0, stream>>>(m_maps[m], m_mips[m], m_dimensions[m]);
#ifdef PROFILING
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
			m_maps[m+1] = m_mips[m];
		}
		//copy here as currInput canges for each batch
		if(async){
			LOG("Async copy "<<m_mipAtlasSizeBytes<<"B mip atlas from host "<<m_maps<< " to device "<<m_dd_maps);
			checkCudaErrors(cudaMemcpyAsync(m_dd_maps, m_maps, m_mipAtlasSizeBytes, cudaMemcpyHostToDevice, stream));
		}else{
			LOG("Copy "<<m_mipAtlasSizeBytes<<"B mip atlas from host "<<m_maps<< " to device "<<m_dd_maps);
			checkCudaErrors(cudaMemcpy(m_dd_maps, m_maps, m_mipAtlasSizeBytes, cudaMemcpyHostToDevice));
		}
		m_generated = true;
	}
	
	__host__ const T *const * getAtlas() const{
		if(!m_generated){
			throw std::logic_error("Mips have not been generated before request.");
		}
		return m_dd_maps;
	}
	__host__ const T *const * getAtlas(const T* baseLevel, const cudaStream_t& stream){
		if(!m_generated || m_baseLevel!=baseLevel){
			generate(baseLevel, stream);
		}
		return m_dd_maps;
	}
	__device__ const T *const * getAtlasDevice() const{
		return m_initialized ? m_dd_maps : nullptr;
	}
	
private:
	const size_t m_alignment;
	bool m_allocated;
	bool m_initialized;
	bool m_generated;
	size_t m_level; //without the finest level
	glm::ivec3 m_baseDimensions;
	const T* m_baseLevel;
	
	glm::ivec3 *m_dimensions; //level+1 elements
	size_t *m_mipOffsetsBytes;
	
	void *m_maps_raw;
	const T * *m_maps; // array on host with level+1 ptrs to mips on device (host ptr of device ptrs)
	const T * *m_dd_maps; //array on device with level+1 ptrs to mips on device (device ptr of device ptrs)
	
	T * *m_mips; //array on host with level ptrs to mips on device. does not include const base level, used to modify when generating mips.
	void* m_ptr; //device ptr to start of allocated device memory
	size_t m_mipAtlasSizeBytes;
	size_t m_mipsSizeBytes;
	size_t m_totalSize;
};
#undef ALIGN_UP
#undef ALIGN_ADDR_UP

template<typename T, bool LOD, Sampling::CoordinateMode CM>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kSample3D(Sampling::Sampler2<T> sampler, T* output){
	MAKE_GLOBAL_INDEX;
	if(isInDimensions<glm::ivec3,glm::ivec3>(globalIdx, c_dimensions.output)){
		glm::vec3 idxCoords = indexToCoords(glm::vec3(globalIdx));
		glm::vec4 samplePos = Sampling::samplePos3D<CM, LOD>(idxCoords);
		T data = sampler.sample3D(samplePos);
		//TODO improved depth correction (with binary depthCorrectionMask as parameter)?
		vectorIO::writeVectorType3D<T, T, glm::ivec3>(data, globalIdx, c_dimensions.output, output);
	}
}
/*version to loop arbitrary channel and batches
* only works on simple data types, no vectors. no texture memory used
* shift thread index coordinates to channel, loop over z and batch:
*   loop Db, loop Dz, Tz->Dy, Ty->Dx, Tx->Dc; for [b][z][y][x][c] data D and [z][y][x] threads T.
*/ //TODO test
/*
template<typename T, Sampling::SamplingMode SM, Sampling::CoordinateMode CM>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kSample3DChannel(const T* input, T* output, const int32_t channel_dim, const int32_t batch_size){
	MAKE_GLOBAL_INDEX;
	glm::ivec3 globalPos = glm::ivec3(globalIdx.y, globalIdx.z, 0);
	
	if(isInDimensions<glm::ivec3,glm::ivec3>(globalPos, c_dimensions.output) && globalIdx.x<channel_dim){
		for(int32_t batch=0; batch<batch_size; ++batch){ //assume same sample positions for each batch
			for(; globalPos.z<c_dimensions.output.z ; ++(globalPos.z)){ //iterate over outer z dim (==1 for 2D? tri->bilinear interpolation!)
				glm::vec3 samplePos = Sampling::samplePos3D<CM, Sampling::usesMip(SM)>(globalPos);
				T data = Sampling::Sampler<T, const T*, SM>::sample3DChannel(input, samplePos, globalIdx.x, c_dimensions.input, channel_dim);
				output[vectorIO::flatIdx3DChannel(globalPos, globalIdx.x, c_dimensions.output, channel_dim)] = data;
			}
		}
	}
}*/

/*
* input: grid data to sample from, arr[b][z][y][x][c] layout, prod(input_shape) elements
* input_shape: shape of the input data, 4 elements (b,z,y,x) (more are ignored)
* MV: 4x4 model-view matrix; column major; x,y,z,w layout
* P: 4x4 projection matrix
* frustum: OpenGL view frustum parameters [near, far, left, right, top, bottom]
* lookup: grid with absolute positions to sample from, same dimensionality as output
* useLookup: wether to use the lookup grid or the (perspective) transformation as sample position
* globalSampling: wether sampling positions are the same for every batch
* output: grid to store output, arr[b][z][y][x][c] layout, prod(output_shape) elements
* output_shape: shape of the output data, 4 elements (b,z,y,x)
*/
template<typename T>
void SampleGridKernelLauncher(const GPUDevice& d,
		const void* _input, const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* _frustum, int32_t numCameras,
	//	const float* _lookup,
		uint8_t* _mipAtlas,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling,
		void* _output, const long long int* output_shape){
	
	LOG("Start SampleGridKernelLauncher");
#ifdef PROFILING
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
	
	const T* input = reinterpret_cast<const T*>(_input);
	T* output = reinterpret_cast<T*>(_output);
	/*
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	LOG("Device properties: " << prop.name);
	LOG("\tCompute: " << prop.major << "." << prop.minor);
	LOG("\tShared memory /block. " << prop.sharedMemPerBlock);
	LOG("\tMax threads /block: " << prop.maxThreadsPerBlock << " (" << prop.maxThreadsDim[0] << "," << prop.maxThreadsDim[2] << "," << prop.maxThreadsDim[2] << ")");
	LOG("\tWarp size: " << prop.warpSize);
	LOG("\tMem pitch: " << prop.memPitch);
	//*/
	
	
	//precompute globals
	BEGIN_SAMPLE;
	LOG("Set dimensions");
	const size_t batchSize = input_shape[0];
	Dimensions dims;
	setDimensions(dims, input_shape, output_shape+1);
	LOG("Dimensions set");
	
	
	const size_t inputSliceSizeElements = vmath::prod(dims.input);
	const size_t outputSliceSizeElements = vmath::prod(dims.output);//dims.output.x*dims.output.y*dims.output.z;
	
	const bool useMipmap=samplingSettings.mipMode != Sampling::SamplerSettings::MIPMODE_NONE;//Sampling::usesMip(samplingMode);
	MipAtlas3D<T> inputMips(samplingSettings.mipLevel, dims.input);
	
	//LOG("Set transformations");
	Transformations transforms;
	FrustumParams frustum;
	int32_t lastCamera=-1;
	
	END_SAMPLE("Precompute and copy global constants");
	
	
	Sampling::Sampler2<T> sampler;
	memset(&sampler, 0, sizeof(Sampling::Sampler2<T>));
	//tmp setup from old settings
	sampler.settings = samplingSettings;
	sampler.settings.mipClampMin = 0.f;
	sampler.settings.mipClampMax = static_cast<float>(samplingSettings.mipLevel);
	//sampler.settings.mipLevel +=1;
	sampler.dimensions = glm::ivec4(dims.input, sizeof(T)/sizeof(float)); //x,y,z,channel
	//sampler.cellCenterOffset = coordinateMode!=Sampling::LuT? 0.5f : 0.0f;
	
	//LOG("filter mode: "<<sampler.settings.filterMode);
	if(useMipmap){
		BEGIN_SAMPLE;
		{
#ifdef ALLOCATE_MIPS
			inputMips.initialize(nullptr, true);
#else //using tensorflow allocator instead
			inputMips.initialize(_mipAtlas);
#endif
#ifdef PROFILING
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
		}
		END_SAMPLE("Mipmap allocation");
	}
	
	
	const dim3 grid(GRID_DIMS(dims.output));
	const dim3 block(BLOCK_DIMS);
	LOG("Sample " << batchSize << " grids with " << numCameras << " cameras");
	
	for(size_t batch=0; batch<batchSize; ++batch){
		LOG("Grid/batch " << batch);
		LOG("Dimensions in: " << LOG_V3_XYZ(dims.input) << ", pitch: " << dims.input.x*sizeof(T));
		const T *currInput = input+batch*inputSliceSizeElements;
		if(useMipmap){
			BEGIN_SAMPLE;
			{
				sampler.d_mips = inputMips.getAtlas(currInput, d.stream());
#ifdef PROFILING
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
			}
			END_SAMPLE("Mipmap generation");
		}else{
			sampler.d_input = currInput;
		}
		
		size_t camera = globalSampling? 0 : batch;
		size_t endCamera = globalSampling? numCameras : camera+1;
		for(; camera<endCamera; ++camera){ //TODO make cameras async/parallel?
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
			
			LOG("Dipatch CUDA sampling kernel: from grid with dims " << LOG_V3_XYZ(dims.input) << " to grid with dims " << LOG_V3_XYZ(dims.output) << " with " << LOG_V3_XYZ(grid) << " tread groups of " << LOG_V3_XYZ(block) << " threads");
			BEGIN_SAMPLE;
			{
				switch(coordinateMode){
					case Sampling::TransformReverse:
						if(useMipmap) kSample3D<T, true, Sampling::TransformReverse><<<grid, block, 0, d.stream()>>>(sampler, currOutput);
						else kSample3D<T, false, Sampling::TransformReverse><<<grid, block, 0, d.stream()>>>(sampler, currOutput);
				break;
					case Sampling::Transform: 
						if(useMipmap) kSample3D<T, true, Sampling::Transform><<<grid, block, 0, d.stream()>>>(sampler, currOutput);
						else kSample3D<T, false, Sampling::Transform><<<grid, block, 0, d.stream()>>>(sampler, currOutput);
				break;
					case Sampling::TransformLinDepthReverse:
						if(useMipmap) kSample3D<T, true, Sampling::TransformLinDepthReverse><<<grid, block, 0, d.stream()>>>(sampler, currOutput);
						else kSample3D<T, false, Sampling::TransformLinDepthReverse><<<grid, block, 0, d.stream()>>>(sampler, currOutput);
				break;
					case Sampling::TransformLinDepth: 
						if(useMipmap) kSample3D<T, true, Sampling::TransformLinDepth><<<grid, block, 0, d.stream()>>>(sampler, currOutput);
						else kSample3D<T, false, Sampling::TransformLinDepth><<<grid, block, 0, d.stream()>>>(sampler, currOutput);
				break;
					default: throw std::invalid_argument("Unsupported coordinate mode.");
				}
#ifdef PROFILING
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
			}
			END_SAMPLE("Sample kernel");
			
		}
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	LOG("End SampleGridKernelLauncher");
}

#define DEFINE_GPU_SPECS(T, C, VEC) \
	template<> \
	void SampleGridKernel<GPUDevice, T, C>::operator()(const GPUDevice& d, \
		const void* input, const long long int* input_shape, \
		const float* M, const float* V, const float* P, const float* frustum, int32_t numCameras, \
		uint8_t* mipAtlas, \
		const Sampling::CoordinateMode coordinateMode, \
		const Sampling::SamplerSettings samplingMode, const bool globalSampling, \
		void* output, const long long int* output_shape){ \
	SampleGridKernelLauncher<VEC>(d, \
		input, input_shape, \
		M, V, P, frustum, numCameras, mipAtlas, \
		coordinateMode, \
		samplingMode, globalSampling, \
		output, output_shape); \
	} \
	template struct SampleGridKernel<GPUDevice, T, C>;
DEFINE_GPU_SPECS(float, 1, float1);
DEFINE_GPU_SPECS(float, 2, float2);
DEFINE_GPU_SPECS(float, 4, float4);


#undef DEFINE_GPU_SPECS


template<typename T, Sampling::SamplerSettings::FilterMode FM, Sampling::SamplerSettings::MipMode MM, Sampling::SamplerSettings::BoundaryMode BM>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kSampleGrid3DLuT(const Sampling::SamplerSettings settings, const Sampling::Grid3D<const T> input, T* output, const float4* lookup, const bool relative, const bool normalized){
	MAKE_GLOBAL_INDEX;
	if(isInDimensions<glm::ivec3,glm::ivec3>(globalIdx, c_dimensions.output)){
		glm::vec4 samplePos = vectorIO::readVectorType3D<glm::vec4,float4,glm::ivec3>(globalIdx, c_dimensions.output, lookup);
		//support for normalized and relative coordinates
		if(normalized){//sampling coordinates are normalized to [0,1], this de-normalization assumes 0.5 center offset.
			//There could be a center offset in sampler.settings.cellCenterOffset?
			samplePos *= glm::vec4(c_dimensions.output, 1.0);
		}
		if(relative){//sampling coordines are relative to the output position, center offset does not matter
			samplePos += glm::vec4(globalIdx, 0.0);
		}
		T data = Sampling::sample3D<T, FM, MM, BM>(settings, input, samplePos); //sampler.sample3D(samplePos);
		//TODO improved depth correction (with binary depthCorrectionMask as parameter)?
		vectorIO::writeVectorType3D<T, T, glm::ivec3>(data, globalIdx, c_dimensions.output, output);
	}//*/
}

/*
* input: grid data to sample from, arr[b][z][y][x][c] layout, prod(input_shape) elements
* input_shape: shape of the input data, 4 elements (b,z,y,x) (more are ignored)
* MV: 4x4 model-view matrix; column major; x,y,z,w layout
* P: 4x4 projection matrix
* frustum: OpenGL view frustum parameters [near, far, left, right, top, bottom]
* lookup: grid with absolute positions to sample from, same dimensionality as output
* useLookup: wether to use the lookup grid or the (perspective) transformation as sample position
* globalSampling: wether sampling positions are the same for every batch
* output: grid to store output, arr[b][z][y][x][c] layout, prod(output_shape) elements
* output_shape: shape of the output data, 4 elements (b,z,y,x)
*/
template<typename T>
void SampleGridLuTKernelLauncher(const void* _input, const long long int* input_shape,
		int32_t numCameras,
		const float* _lookup, uint8_t* _mipAtlas,
		const Sampling::CoordinateMode coordinateMode,
		Sampling::SamplerSettings samplingSettings, const bool globalSampling,
		const bool relative, const bool normalized,
		void* _output, const long long int* output_shape){
	
	LOG("Start SampleGridLuTKernelLauncher");
	
	const float4* lookup = reinterpret_cast<const float4*>(_lookup);
	const T* input = reinterpret_cast<const T*>(_input);
	T* output = reinterpret_cast<T*>(_output);
	
	//const bool useTextures=samplingSettings.useTexture;//Sampling::usesTexture(samplingMode);
	const bool useMipmap=samplingSettings.mipMode != Sampling::SamplerSettings::MIPMODE_NONE;//Sampling::usesMip(samplingMode);
	
	//precompute globals
	BEGIN_SAMPLE;
	LOG("Set dimensions");
	const size_t batchSize = input_shape[0];
	Dimensions dims;
	setDimensions(dims, input_shape, output_shape+1);
	LOG("Dimensions set");
	
	
	const size_t inputSliceSizeElements = vmath::prod(dims.input);
	const size_t outputSliceSizeElements = vmath::prod(dims.output);
	
	
	END_SAMPLE("Precompute and copy global constants");
	
	MipAtlas3D<T> inputMips(samplingSettings.mipLevel, dims.input);
	//tmp setup from old settings
	samplingSettings.mipClampMin = 0.f;
	samplingSettings.mipClampMax = static_cast<float>(samplingSettings.mipLevel);
	//samplingSettings.mipLevel +=1;
	//sampler.dimensions = glm::ivec4(dims.input, sizeof(T)/sizeof(float)); //x,y,z,channel
	//sampler.cellCenterOffset = coordinateMode!=Sampling::LuT? 0.5f : 0.0f;
	
	//LOG("filter mode: "<<sampler.settings.filterMode);
	Sampling::Grid3D<const T> inputGrid;
	memset(&inputGrid, 0, sizeof(Sampling::Grid3D<const T>));
	inputGrid.dimensions = glm::ivec4(dims.input, sizeof(T)/sizeof(float));
	inputGrid.dimensionsInverse = 1.0f/glm::vec3(dims.input);
	inputGrid.mipLevel = 0;
	
	if(useMipmap){
		inputGrid.mipLevel = samplingSettings.mipLevel;
		BEGIN_SAMPLE;
		{
#ifdef ALLOCATE_MIPS
			inputMips.initialize(nullptr, true);
#else
			inputMips.initialize(_mipAtlas);
#endif
#ifdef PROFILING
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
		}
		END_SAMPLE("Mipmap allocation");
	}
	
	
	int32_t lastCamera=-1;
	const float4* currLookup = lookup;
	
	
	const dim3 grid(GRID_DIMS(dims.output));
	const dim3 block(BLOCK_DIMS);
	LOG("Sample " << batchSize << " grids with " << numCameras << " cameras");
	
	for(size_t batch=0; batch<batchSize; ++batch){
		LOG("Grid/batch " << batch);
		LOG("Dimensions in: " << LOG_V3_XYZ(dims.input) << ", pitch: " << dims.input.x*sizeof(T));
		T* currInput = const_cast<T*>(input+batch*inputSliceSizeElements);
		if(useMipmap){
			BEGIN_SAMPLE;
			{
				inputMips.generate(currInput, 0);
				inputGrid.d_mips = inputMips.getAtlas();
#ifdef PROFILING
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
			}
			END_SAMPLE("Mipmap generation");
		}else{
			inputGrid.d_data = currInput;
		}
		
		size_t camera = globalSampling? 0 : batch;
		size_t endCamera = globalSampling? numCameras : camera+1;
		for(; camera<endCamera; ++camera){ //TODO make cameras async/parallel
			LOG("Camera " << camera);
			T* currOutput = output+(globalSampling? batch*numCameras + camera : batch)*outputSliceSizeElements;
			//only set new camera if there are multiple
			if(lastCamera!=camera){
				currLookup = lookup + camera*outputSliceSizeElements;
				lastCamera=camera;
			}
			
			LOG("Dipatch CUDA sampling kernel: from grid with dims " << LOG_V3_XYZ(dims.input) << " to grid with dims " << LOG_V3_XYZ(dims.output) << " with " << LOG_V3_XYZ(grid) << " thread groups of " << LOG_V3_XYZ(block) << " threads");
			BEGIN_SAMPLE;
			{
				#define KERNEL_SWITCH(FM, MM, BM) if((samplingSettings.filterMode==Sampling::SamplerSettings::FILTERMODE_##FM) \
					&&(samplingSettings.mipMode==Sampling::SamplerSettings::MIPMODE_##MM) \
					&&(samplingSettings.boundaryMode==Sampling::SamplerSettings::BM)) \
						kSampleGrid3DLuT<T, Sampling::SamplerSettings::FILTERMODE_##FM, Sampling::SamplerSettings::MIPMODE_##MM, Sampling::SamplerSettings::BM> \
						<<<grid, block>>>(samplingSettings, inputGrid, currOutput, currLookup, relative, normalized)
				KERNEL_SWITCH(NEAREST, NONE, BORDER);
				else KERNEL_SWITCH(LINEAR, NONE, BORDER);
				else KERNEL_SWITCH(MIN, NONE, BORDER);
				else KERNEL_SWITCH(MAX, NONE, BORDER);
				else KERNEL_SWITCH(NEAREST, NEAREST, BORDER);
				else KERNEL_SWITCH(LINEAR, NEAREST, BORDER);
				else KERNEL_SWITCH(MIN, NEAREST, BORDER);
				else KERNEL_SWITCH(MAX, NEAREST, BORDER);
				else KERNEL_SWITCH(NEAREST, LINEAR, BORDER);
				else KERNEL_SWITCH(LINEAR, LINEAR, BORDER);
				else KERNEL_SWITCH(MIN, LINEAR, BORDER);
				else KERNEL_SWITCH(MAX, LINEAR, BORDER);
				
				else KERNEL_SWITCH(NEAREST, NONE, CLAMP);
				else KERNEL_SWITCH(LINEAR, NONE, CLAMP);
				else KERNEL_SWITCH(MIN, NONE, CLAMP);
				else KERNEL_SWITCH(MAX, NONE, CLAMP);
				else KERNEL_SWITCH(NEAREST, NEAREST, CLAMP);
				else KERNEL_SWITCH(LINEAR, NEAREST, CLAMP);
				else KERNEL_SWITCH(MIN, NEAREST, CLAMP);
				else KERNEL_SWITCH(MAX, NEAREST, CLAMP);
				else KERNEL_SWITCH(NEAREST, LINEAR, CLAMP);
				else KERNEL_SWITCH(LINEAR, LINEAR, CLAMP);
				else KERNEL_SWITCH(MIN, LINEAR, CLAMP);
				else KERNEL_SWITCH(MAX, LINEAR, CLAMP);
				
				else KERNEL_SWITCH(NEAREST, NONE, WRAP);
				else KERNEL_SWITCH(LINEAR, NONE, WRAP);
				else KERNEL_SWITCH(MIN, NONE, WRAP);
				else KERNEL_SWITCH(MAX, NONE, WRAP);
				else KERNEL_SWITCH(NEAREST, NEAREST, WRAP);
				else KERNEL_SWITCH(LINEAR, NEAREST, WRAP);
				else KERNEL_SWITCH(MIN, NEAREST, WRAP);
				else KERNEL_SWITCH(MAX, NEAREST, WRAP);
				else KERNEL_SWITCH(NEAREST, LINEAR, WRAP);
				else KERNEL_SWITCH(LINEAR, LINEAR, WRAP);
				else KERNEL_SWITCH(MIN, LINEAR, WRAP);
				else KERNEL_SWITCH(MAX, LINEAR, WRAP);
				
				else throw std::invalid_argument("Unsupported sampling configuration.");
				#undef KERNEL_SWITCH
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			}
			END_SAMPLE("Sample kernel");
		}
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	{
	}
	END_SAMPLE("Free memory");
	LOG("End SampleGridLuTKernelLauncher");
}

void SampleRGridLuTKernelLauncher(
		const void* input, const long long int* input_shape,
		int32_t numCameras,
		const float* lookup, uint8_t* mipAtlas,
		const Sampling::CoordinateMode coordinateMode,
		Sampling::SamplerSettings samplingMode, const bool globalSampling, const bool relative, const bool normalized,
		void* output, const long long int* output_shape){
	SampleGridLuTKernelLauncher<float1>(input, input_shape,
		 numCameras, lookup, mipAtlas,
		coordinateMode,
		samplingMode, globalSampling, relative, normalized,
		output, output_shape);
}
void SampleRGGridLuTKernelLauncher(
		const void* input, const long long int* input_shape,
		int32_t numCameras,
		const float* lookup, uint8_t* mipAtlas,
		const Sampling::CoordinateMode coordinateMode,
		Sampling::SamplerSettings samplingMode, const bool globalSampling, const bool relative, const bool normalized,
		void* output, const long long int* output_shape){
	SampleGridLuTKernelLauncher<float2>(input, input_shape,
		 numCameras, lookup, mipAtlas,
		coordinateMode,
		samplingMode, globalSampling, relative, normalized,
		output, output_shape);
}
void SampleRGBAGridLuTKernelLauncher(
		const void* input, const long long int* input_shape,
		int32_t numCameras,
		const float* lookup, uint8_t* mipAtlas,
		const Sampling::CoordinateMode coordinateMode,
		Sampling::SamplerSettings samplingMode, const bool globalSampling, const bool relative, const bool normalized,
		void* output, const long long int* output_shape){
	SampleGridLuTKernelLauncher<float4>(input, input_shape,
		 numCameras, lookup, mipAtlas,
		coordinateMode,
		samplingMode, globalSampling, relative, normalized,
		output, output_shape);
}

/********************************************
*	--- Backwards / Gradient Pass ---
********************************************/

//input and output retain their forward meaning (i.e. we compute output->input)


/*
template<typename T>
__device__ inline void atomicAddVectorType3D(const T v, const glm::ivec3 pos, const glm::ivec3 dims, T* buf);
template<>
__device__ inline void atomicAddVectorType3D(const float1 v, const glm::ivec3 pos, const glm::ivec3 dims, float1* buf){
	const size_t offset = vectorIO::flatIdx3D(pos, dims);
	float * buf_raw = reinterpret_cast<float*>(buf + offset);
	atomicAdd(buf_raw, v.x);
}
template<>
__device__ inline void atomicAddVectorType3D(const float2 v, const glm::ivec3 pos, const glm::ivec3 dims, float2* buf){
	const size_t offset = vectorIO::flatIdx3D(pos, dims);
	float * buf_raw = reinterpret_cast<float*>(buf + offset);
	atomicAdd(buf_raw, v.x);
	atomicAdd(buf_raw +1, v.y);
}
template<>
__device__ inline void atomicAddVectorType3D(const float4 v, const glm::ivec3 pos, const glm::ivec3 dims, float4* buf){
	const size_t offset = vectorIO::flatIdx3D(pos, dims);
	float * buf_raw = reinterpret_cast<float*>(buf + offset);
	atomicAdd(buf_raw, v.x);
	atomicAdd(buf_raw +1, v.y);
	atomicAdd(buf_raw +2, v.z);
	atomicAdd(buf_raw +3, v.w);
}
*/
template<typename T>
__device__ inline void atomicAddVectorType3D(const T v, const int32_t x, const int32_t y, const int32_t z, const glm::ivec3 dims, T* buf);
template<>
__device__ inline void atomicAddVectorType3D(const float1 v, const int32_t x, const int32_t y, const int32_t z, const glm::ivec3 dims, float1* buf){
	const size_t offset = vectorIO::flatIdx3D(x,y,z, dims);
	float * buf_raw = reinterpret_cast<float*>(buf + offset);
	atomicAdd(buf_raw, v.x);
}
template<>
__device__ inline void atomicAddVectorType3D(const float2 v, const int32_t x, const int32_t y, const int32_t z, const glm::ivec3 dims, float2* buf){
	const size_t offset = vectorIO::flatIdx3D(x,y,z, dims);
	float * buf_raw = reinterpret_cast<float*>(buf + offset);
	atomicAdd(buf_raw, v.x);
	atomicAdd(buf_raw +1, v.y);
}
template<>
__device__ inline void atomicAddVectorType3D(const float4 v, const int32_t x, const int32_t y, const int32_t z, const glm::ivec3 dims, float4* buf){
	const size_t offset = vectorIO::flatIdx3D(x,y,z, dims);
	float * buf_raw = reinterpret_cast<float*>(buf + offset);
	atomicAdd(buf_raw, v.x);
	atomicAdd(buf_raw +1, v.y);
	atomicAdd(buf_raw +2, v.z);
	atomicAdd(buf_raw +3, v.w);
}

// position is without offset (+0.0)
template<typename T, Sampling::SamplerSettings::BoundaryMode BM, bool CountSamples>
__device__ void scatterGradInterpolated(const T out_grad, const glm::vec3 position, T* UG_PTR input_grad, uint32_t* UG_PTR num_samples){
	if(BM!=Sampling::SamplerSettings::BORDER || (CHECK_BOUNDS_SV3V3(-1.f, <, position, <, c_dimensions.input))){
		const glm::vec3 cw = glm::fract(position);
		const glm::vec3 fw = 1.f - cw;
		glm::ivec3 ceilIdx = glm::ivec3(glm::ceil(position));
		glm::ivec3 floorIdx = glm::ivec3(glm::floor(position));
		
		if(BM==Sampling::SamplerSettings::CLAMP){
			ceilIdx = glm::clamp(ceilIdx, glm::ivec3(0), c_dimensions.input -1);
			floorIdx = glm::clamp(floorIdx, glm::ivec3(0), c_dimensions.input -1);
		}else if(BM==Sampling::SamplerSettings::WRAP){
			ceilIdx = vmath::positivemod<glm::ivec3, glm::ivec3>(ceilIdx, c_dimensions.input);
			floorIdx = vmath::positivemod<glm::ivec3, glm::ivec3>(floorIdx, c_dimensions.input);
		}
		
		//accumulate weighted gradients
		if(BM!=Sampling::SamplerSettings::BORDER || (floorIdx.x>=0 && floorIdx.y>=0 && floorIdx.z>=0)){
			atomicAddVectorType3D<T>(out_grad*(fw.x*fw.y*fw.z), floorIdx.x, floorIdx.y, floorIdx.z, c_dimensions.input, input_grad);
			if(CountSamples){ atomicInc(num_samples + vectorIO::flatIdx3D(floorIdx.x, floorIdx.y, floorIdx.z, c_dimensions.input), 0xffffffff); }
		}
		if(BM!=Sampling::SamplerSettings::BORDER || (ceilIdx.x<c_dimensions.input.x && floorIdx.y>=0 && floorIdx.z>=0)){
			atomicAddVectorType3D<T>(out_grad*(cw.x*fw.y*fw.z), ceilIdx.x, floorIdx.y, floorIdx.z, c_dimensions.input, input_grad);
			if(CountSamples){ atomicInc(num_samples + vectorIO::flatIdx3D(ceilIdx.x, floorIdx.y, floorIdx.z, c_dimensions.input), 0xffffffff); }
		}
		if(BM!=Sampling::SamplerSettings::BORDER || (floorIdx.x>=0 && ceilIdx.y<c_dimensions.input.y && floorIdx.z>=0)){
			atomicAddVectorType3D<T>(out_grad*(fw.x*cw.y*fw.z), floorIdx.x, ceilIdx.y, floorIdx.z, c_dimensions.input, input_grad);
			if(CountSamples){ atomicInc(num_samples + vectorIO::flatIdx3D(floorIdx.x, ceilIdx.y, floorIdx.z, c_dimensions.input), 0xffffffff); }
		}
		if(BM!=Sampling::SamplerSettings::BORDER || (ceilIdx.x<c_dimensions.input.x && ceilIdx.y<c_dimensions.input.y && floorIdx.z>=0)){
			atomicAddVectorType3D<T>(out_grad*(cw.x*cw.y*fw.z), ceilIdx.x, ceilIdx.y, floorIdx.z, c_dimensions.input, input_grad);
			if(CountSamples){ atomicInc(num_samples + vectorIO::flatIdx3D(ceilIdx.x, ceilIdx.y, floorIdx.z, c_dimensions.input), 0xffffffff); }
		}
		
		if(BM!=Sampling::SamplerSettings::BORDER || (floorIdx.x>=0 && floorIdx.y>=0 && ceilIdx.z<c_dimensions.input.z)){
			atomicAddVectorType3D<T>(out_grad*(fw.x*fw.y*cw.z), floorIdx.x, floorIdx.y, ceilIdx.z, c_dimensions.input, input_grad);
			if(CountSamples){ atomicInc(num_samples + vectorIO::flatIdx3D(floorIdx.x, floorIdx.y, ceilIdx.z, c_dimensions.input), 0xffffffff); }
		}
		if(BM!=Sampling::SamplerSettings::BORDER || (ceilIdx.x<c_dimensions.input.x && floorIdx.y>=0 && ceilIdx.z<c_dimensions.input.z)){
			atomicAddVectorType3D<T>(out_grad*(cw.x*fw.y*cw.z), ceilIdx.x, floorIdx.y, ceilIdx.z, c_dimensions.input, input_grad);
			if(CountSamples){ atomicInc(num_samples + vectorIO::flatIdx3D(ceilIdx.x, floorIdx.y, ceilIdx.z, c_dimensions.input), 0xffffffff); }
		}
		if(BM!=Sampling::SamplerSettings::BORDER || (floorIdx.x>=0 && ceilIdx.y<c_dimensions.input.y && ceilIdx.z<c_dimensions.input.z)){
			atomicAddVectorType3D<T>(out_grad*(fw.x*cw.y*cw.z), floorIdx.x, ceilIdx.y, ceilIdx.z, c_dimensions.input, input_grad);
			if(CountSamples){ atomicInc(num_samples + vectorIO::flatIdx3D(floorIdx.x, ceilIdx.y, ceilIdx.z, c_dimensions.input), 0xffffffff); }
		}
		if(BM!=Sampling::SamplerSettings::BORDER || (ceilIdx.x<c_dimensions.input.x && ceilIdx.y<c_dimensions.input.y && ceilIdx.z<c_dimensions.input.z)){
			atomicAddVectorType3D<T>(out_grad*(cw.x*cw.y*cw.z), ceilIdx.x, ceilIdx.y, ceilIdx.z, c_dimensions.input, input_grad);
			if(CountSamples){ atomicInc(num_samples + vectorIO::flatIdx3D(ceilIdx.x, ceilIdx.y, ceilIdx.z, c_dimensions.input), 0xffffffff); }
		}
	}
}

template<typename T, bool SetZero>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kNormalize3DGradients(T* UG_PTR grad, uint32_t* UG_PTR num_samples){
	MAKE_GLOBAL_INDEX;
	if(isInDimensions<glm::ivec3,glm::ivec3>(globalIdx, c_dimensions.input)){
		const size_t flatIdx = vectorIO::flatIdx3D(globalIdx.x, globalIdx.y, globalIdx.z, c_dimensions.input);
		const uint32_t n = num_samples[flatIdx];
		if(n>0){
			const float weight = 1.0f / static_cast<float>(n);
			T data = vectorIO::readVectorType<T, T>(flatIdx, grad);
			vectorIO::writeVectorType<T, T>(data * weight, flatIdx, grad);
			
			if(SetZero){
				num_samples[flatIdx] = 0;
			}
		}
	}
}


/*
 *scatter gradients to original inputs the output was interpolated from
 *using atomicAdd for floats
 *
 * TODO handling LoD/mipmapping: scatter gradients to correct mip(s) and combine them after
*/
template<typename T, bool LOD, Sampling::CoordinateMode CM, bool CountSamples>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kScatter3DGradients(Sampling::Sampler2<T> sampler,  const T* UG_PTR output_grad, T* UG_PTR input_grad, uint32_t* UG_PTR num_samples){
	MAKE_GLOBAL_INDEX;
	if(isInDimensions<glm::ivec3,glm::ivec3>(globalIdx, c_dimensions.output)){
		const T out_grad = vectorIO::readVectorType3D<T, T, glm::ivec3>(globalIdx, c_dimensions.output, output_grad);
		const glm::vec3 idxCoords = indexToCoords(glm::vec3(globalIdx));
		const glm::vec4 samplePos = Sampling::samplePos3D<CM, LOD>(idxCoords);
		const glm::vec3 sampleIdx = sampler.getSamplingPosition(samplePos);
		
		if(sampler.settings.boundaryMode==Sampling::SamplerSettings::BORDER){
			scatterGradInterpolated<T, Sampling::SamplerSettings::BORDER, CountSamples>(out_grad, sampleIdx, input_grad, num_samples);
		}else if(sampler.settings.boundaryMode==Sampling::SamplerSettings::CLAMP){
			scatterGradInterpolated<T, Sampling::SamplerSettings::CLAMP, CountSamples>(out_grad, sampleIdx, input_grad, num_samples);
		}else if(sampler.settings.boundaryMode==Sampling::SamplerSettings::WRAP){
			scatterGradInterpolated<T, Sampling::SamplerSettings::WRAP, CountSamples>(out_grad, sampleIdx, input_grad, num_samples);
		}
	}
	
}
//*/


template<typename T>
void SampleGridGradKernelLauncher(const void* _input, const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* _frustum, int32_t numCameras,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, uint32_t* sample_count_buffer){ //, void* _lookup_grad
	
	LOG("Start SampleGridGradKernelLauncher");
	
	const T* input = reinterpret_cast<const T*>(_input);
	T* input_grad = reinterpret_cast<T*>(_input_grad);
	
	
	const T* output_grad = reinterpret_cast<const T*>(_output_grad);
	
	//precompute globals
	
	BEGIN_SAMPLE;
	LOG("Set dimensions");
	const size_t batchSize = input_shape[0];
	Dimensions dims;
	setDimensions(dims, input_shape, output_shape+1);
	LOG("Dimensions set");
	
	
	const size_t inputSliceSizeElements = vmath::prod(dims.input);
	const size_t outputSliceSizeElements = vmath::prod(dims.output);
	
	//LOG("Set transformations");
	Transformations transforms;
	FrustumParams frustum;
	int32_t lastCamera=-1;
	
	END_SAMPLE("Precompute and copy global constants");
	
	const bool useMipmap=false;//samplingSettings.mipMode != Sampling::SamplerSettings::MIPMODE_NONE;//Sampling::usesMip(samplingMode);
	MipAtlas3D<T> inputMips(samplingSettings.mipLevel, dims.input);
	
	Sampling::Sampler2<T> sampler;
	memset(&sampler, 0, sizeof(Sampling::Sampler2<T>));
	//tmp setup from old settings
	sampler.settings = samplingSettings;
	//gradients for mipmapped sampling currently not supported
	sampler.settings.mipMode = Sampling::SamplerSettings::MIPMODE_NONE;
	//sampler.settings.mipClampMin = 0.f;
	//sampler.settings.mipClampMax = static_cast<float>(mipLevels);
	//sampler.settings.mipLevel +=1;
	sampler.dimensions = glm::ivec4(dims.input, sizeof(T)/sizeof(float));
	
	// zero gradient buffers
	BEGIN_SAMPLE;
	{
		checkCudaErrors(cudaMemset(input_grad, 0, inputSliceSizeElements*sizeof(T)*batchSize));
		checkCudaErrors(cudaMemset(sample_count_buffer, 0, inputSliceSizeElements*sizeof(uint32_t)));
#ifdef PROFILING
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
	}
	END_SAMPLE("Set gradient buffers zero");
	
	
	const dim3 grid(GRID_DIMS(dims.output));
	const dim3 block(BLOCK_DIMS);
	LOG("Sample " << batchSize << " grids with " << numCameras << " cameras");
	
	for(size_t batch=0; batch<batchSize; ++batch){
		LOG("Grid/batch " << batch);
		LOG("Dimensions in: " << LOG_V3_XYZ(dims.input) << ", pitch: " << dims.input.x*sizeof(T));
		T* currInput = const_cast<T*>(input+batch*inputSliceSizeElements);
		T* currInput_grad = input_grad+batch*inputSliceSizeElements;
		sampler.d_input = currInput;
		
	
		size_t camera = globalSampling? 0 : batch;
		size_t endCamera = globalSampling? numCameras : camera+1;
		for(; camera<endCamera; ++camera){ //TODO make cameras async/parallel
			LOG("Camera " << camera);
			const T* currOutput_grad = output_grad+(globalSampling? batch*numCameras + camera : batch)*outputSliceSizeElements;
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
			
			LOG("Dipatch CUDA sampling kernel: from grid with dims " << LOG_V3_XYZ(dims.input) << " to grid with dims " << LOG_V3_XYZ(dims.output) << " with " << LOG_V3_XYZ(grid) << " tread groups of " << LOG_V3_XYZ(block) << " threads");
			BEGIN_SAMPLE;
			{
				switch(coordinateMode){
					case Sampling::TransformReverse:
						if(useMipmap) kScatter3DGradients<T, true, Sampling::TransformReverse, NORMALIZE_GRADIENTS><<<grid, block>>>(
							sampler, currOutput_grad, currInput_grad, sample_count_buffer);
						else kScatter3DGradients<T, false, Sampling::TransformReverse, NORMALIZE_GRADIENTS><<<grid, block>>>(
							sampler, currOutput_grad, currInput_grad, sample_count_buffer);
				break;
					case Sampling::Transform: 
						if(useMipmap) kScatter3DGradients<T, true, Sampling::Transform, NORMALIZE_GRADIENTS><<<grid, block>>>(
							sampler, currOutput_grad, currInput_grad, sample_count_buffer);
						else kScatter3DGradients<T, false, Sampling::Transform, NORMALIZE_GRADIENTS><<<grid, block>>>(
							sampler, currOutput_grad, currInput_grad, sample_count_buffer);
				break;
					case Sampling::TransformLinDepthReverse:
						if(useMipmap) kScatter3DGradients<T, true, Sampling::TransformLinDepthReverse, NORMALIZE_GRADIENTS><<<grid, block>>>(
							sampler, currOutput_grad, currInput_grad, sample_count_buffer);
						else kScatter3DGradients<T, false, Sampling::TransformLinDepthReverse, NORMALIZE_GRADIENTS><<<grid, block>>>(
							sampler, currOutput_grad, currInput_grad, sample_count_buffer);
				break;
					case Sampling::TransformLinDepth: 
						if(useMipmap) kScatter3DGradients<T, true, Sampling::TransformLinDepth, NORMALIZE_GRADIENTS><<<grid, block>>>(
							sampler, currOutput_grad, currInput_grad, sample_count_buffer);
						else kScatter3DGradients<T, false, Sampling::TransformLinDepth, NORMALIZE_GRADIENTS><<<grid, block>>>(
							sampler, currOutput_grad, currInput_grad, sample_count_buffer);
				break;
					default: throw std::invalid_argument("Unsupported coordinate mode.");
				}
				
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			}
			END_SAMPLE("Sample kernel gradients");
			
		}
		if(useMipmap){
			//TODO: fuse mips into output
			//kCollapseGradMip3D<T><<<>>>(mip[m+1], mip[m], dimensions[m]);
		}
		if(NORMALIZE_GRADIENTS){
			BEGIN_SAMPLE;
			{
				const dim3 grad_grid(GRID_DIMS(dims.input));
				kNormalize3DGradients<T, true><<<grad_grid, block>>>(currInput_grad, sample_count_buffer);
				//checkCudaErrors(cudaMemset(sample_count_buffer, 0, inputSliceSizeElements*sizeof(uint32_t)));
			}
			END_SAMPLE("Grad normalize");
		}
	}
	
	
	BEGIN_SAMPLE;
	{
//		if(useMipmap){
			//freeMipAtlas3D(inputMips);
			//freeMipAtlas3D(input_gradMips);
//		}
	}
	END_SAMPLE("Free memory");
	
	LOG("End SampleGridGradKernelLauncher");
}

void SampleRGridGradKernelLauncher(const void* _input, const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* _frustum, int32_t numCameras,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, uint32_t* sample_count_buffer){
	SampleGridGradKernelLauncher<float1>(_input, input_shape, M, V, P, _frustum, numCameras, coordinateMode, samplingSettings, globalSampling, _output_grad, output_shape, _input_grad, sample_count_buffer);
}
void SampleRGGridGradKernelLauncher(const void* _input, const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* _frustum, int32_t numCameras,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, uint32_t* sample_count_buffer){
	SampleGridGradKernelLauncher<float2>(_input, input_shape, M, V, P, _frustum, numCameras, coordinateMode, samplingSettings, globalSampling, _output_grad, output_shape, _input_grad, sample_count_buffer);
}
void SampleRGBAGridGradKernelLauncher(const void* _input, const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* _frustum, int32_t numCameras,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, uint32_t* sample_count_buffer){
	SampleGridGradKernelLauncher<float4>(_input, input_shape, M, V, P, _frustum, numCameras, coordinateMode, samplingSettings, globalSampling, _output_grad, output_shape, _input_grad, sample_count_buffer);
}


/*
 * input (grad): [depth, height, width, channel (1-4)]
 * lut (grad): [depth, height, width, channel (4)], channel: (abs_x, abs_y, abs_z, LoD)
https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/resampler/kernels/resampler_ops_gpu.cu.cc
*/
template<typename T, Sampling::SamplerSettings::FilterMode FM, Sampling::SamplerSettings::MipMode MM, Sampling::SamplerSettings::BoundaryMode BM, bool CountSamples> //const T* UG_PTR input, const float4* UG_PTR lookup
__global__ void kScatterLuT3DGradients(const Sampling::SamplerSettings settings, const Sampling::Grid3D<const T> input, const float4* UG_PTR lookup, const T* UG_PTR output_grad, T* input_grad, float4* lookup_grad, const bool relative, const bool normalized, uint32_t* UG_PTR num_samples){
	MAKE_GLOBAL_INDEX;
	if(isInDimensions<glm::ivec3,glm::ivec3>(globalIdx, c_dimensions.output)){
		glm::vec4 samplePos = vectorIO::readVectorType3D<glm::vec4,float4,glm::ivec3>(globalIdx, c_dimensions.output, lookup);
		//TODO support for normalized and relative coordinates
		if(normalized){//sampling coordinates are normalized to [0,1], this de-normalization assumes 0.5 center offset.
			//There could be a center offset in sampler.settings.cellCenterOffset?
			samplePos *= glm::vec4(c_dimensions.output, 1.0);
		}
		if(relative){//sampling coordines are relative to the output position, center offset does not matter
			samplePos += glm::vec4(globalIdx, 0.0);
		}
		
		//TODO improved depth correction (with binary depthCorrectionMask as parameter)?
		const T out_grad = vectorIO::readVectorType3D<T, T, glm::ivec3>(globalIdx, c_dimensions.output, output_grad);
		
		if(FM==Sampling::SamplerSettings::FILTERMODE_LINEAR){
			Sampling::DataGrad3D<T> dataGrad = Sampling::sampleGrad3D<T,FM,MM,BM>(settings, input, samplePos);
			// TODO make LoD grads?
			const float4 lut_grad = make_float4(vmath::sum(out_grad*dataGrad.dx), vmath::sum(out_grad*dataGrad.dy), vmath::sum(out_grad*dataGrad.dz), 0.f);
			//atomicAddVectorType3D<float4>(lut_grad, globalIdx, c_dimensions.output, lookup_grad);
			vectorIO::writeVectorType3D<float4, float4, glm::ivec3>(lut_grad, globalIdx, c_dimensions.output, lookup_grad);
		}//else gradients are 0, so keep 0 (memset outside)
		
		//TODO: scattering should respect the filter mode.
		const glm::vec3 sampleIdx = (samplePos - settings.cellCenterOffset);//sampler.getSamplingPosition(samplePos);
		scatterGradInterpolated<T,BM, CountSamples>(out_grad, sampleIdx, input_grad, num_samples);
		//NEAREST, MIN, MAX: send gradients only to the cell the fwd value was taken from
	}
}

template<typename T>
void SampleGridLuTGradKernelLauncher(const void* _input, const long long int* input_shape,
		int32_t numCameras,
		const float* _lookup,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling, const bool relative, const bool normalized,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, void* _lookup_grad, uint32_t* sample_count_buffer){
	
	LOG("Start SampleGridLuTGradKernelLauncher");
	
	const T* input = reinterpret_cast<const T*>(_input);
	T* input_grad = reinterpret_cast<T*>(_input_grad);
	
	const float4* lookup = reinterpret_cast<const float4*>(_lookup);
	float4* lookup_grad = reinterpret_cast<float4*>(_lookup_grad);
	
	const T* output_grad = reinterpret_cast<const T*>(_output_grad);
	
	//precompute globals
	
	BEGIN_SAMPLE;
	LOG("Set dimensions");
	const size_t batchSize = input_shape[0];
	Dimensions dims;
	setDimensions(dims, input_shape, output_shape+1);
	LOG("Dimensions set");
	
	
	const size_t inputSliceSizeElements = vmath::prod(dims.input);
	const size_t outputSliceSizeElements = vmath::prod(dims.output);
	
	//LOG("Set transformations");
//	Transformations transforms;
//	FrustumParams frustum;
	int32_t lastCamera=-1;
	
	END_SAMPLE("Precompute and copy global constants");
	
	//LOG("filter mode: "<<sampler.settings.filterMode);
	Sampling::Grid3D<const T> inputGrid;
	memset(&inputGrid, 0, sizeof(Sampling::Grid3D<const T>));
	inputGrid.dimensions = glm::ivec4(dims.input, sizeof(T)/sizeof(float));
	inputGrid.dimensionsInverse = 1.0f/glm::vec3(dims.input);
	inputGrid.mipLevel = 0;
	
	// zero gradient buffers
	BEGIN_SAMPLE;
	{
		checkCudaErrors(cudaMemset(input_grad, 0, inputSliceSizeElements*sizeof(T)*batchSize));
		checkCudaErrors(cudaMemset(lookup_grad, 0, outputSliceSizeElements*sizeof(float4)*numCameras));
#ifdef PROFILING
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
#endif
	}
	END_SAMPLE("Set gradient buffers zero");
	
	const float4* currLookup = lookup;
	float4* currLookup_grad = lookup_grad;
	
	const dim3 grid(GRID_DIMS(dims.output));
	const dim3 block(BLOCK_DIMS);
	LOG("Sample " << batchSize << " grids with " << numCameras << " cameras");
	
	for(size_t batch=0; batch<batchSize; ++batch){
		LOG("Grid/batch " << batch);
		LOG("Dimensions in: " << LOG_V3_XYZ(dims.input) << ", pitch: " << dims.input.x*sizeof(T));
		T* currInput = const_cast<T*>(input+batch*inputSliceSizeElements);
		T* currInput_grad = input_grad+batch*inputSliceSizeElements;
		inputGrid.d_data = currInput;
		
	
		size_t camera = globalSampling? 0 : batch;
		size_t endCamera = globalSampling? numCameras : camera+1;
		for(; camera<endCamera; ++camera){ //TODO make cameras async/parallel
			LOG("Camera " << camera);
			const T* currOutput_grad = output_grad+(globalSampling? batch*numCameras + camera : batch)*outputSliceSizeElements;
			BEGIN_SAMPLE;
			{
				//only set new camera if there are multiple
				//setTransformations(transforms, M + batch*16, V + camera*16, P + camera*16);
				if(lastCamera!=camera){
					currLookup = lookup + camera*outputSliceSizeElements;
					currLookup_grad = lookup_grad + camera*outputSliceSizeElements;
					lastCamera=camera;
				}
				
			}
			END_SAMPLE("Set transformation CBuffer");
			
			LOG("Dipatch CUDA sampling kernel: from grid with dims " << LOG_V3_XYZ(dims.input) << " to grid with dims " << LOG_V3_XYZ(dims.output) << " with " << LOG_V3_XYZ(grid) << " tread groups of " << LOG_V3_XYZ(block) << " threads");
			BEGIN_SAMPLE;
			{
				#define KERNEL_SWITCH(FM, MM, BM) if((samplingSettings.filterMode==Sampling::SamplerSettings::FILTERMODE_##FM) \
					&&(samplingSettings.mipMode==Sampling::SamplerSettings::MIPMODE_##MM) \
					&&(samplingSettings.boundaryMode==Sampling::SamplerSettings::BM)) \
						kScatterLuT3DGradients<T, Sampling::SamplerSettings::FILTERMODE_##FM, Sampling::SamplerSettings::MIPMODE_##MM, Sampling::SamplerSettings::BM, NORMALIZE_GRADIENTS> \
						<<<grid, block>>>(samplingSettings, inputGrid, currLookup, currOutput_grad, currInput_grad, currLookup_grad, relative, normalized, sample_count_buffer)
				KERNEL_SWITCH(NEAREST, NONE, BORDER);
				else KERNEL_SWITCH(LINEAR, NONE, BORDER);
				else KERNEL_SWITCH(MIN, NONE, BORDER);
				else KERNEL_SWITCH(MAX, NONE, BORDER);
				else KERNEL_SWITCH(NEAREST, NEAREST, BORDER);
				else KERNEL_SWITCH(LINEAR, NEAREST, BORDER);
				else KERNEL_SWITCH(MIN, NEAREST, BORDER);
				else KERNEL_SWITCH(MAX, NEAREST, BORDER);
				else KERNEL_SWITCH(NEAREST, LINEAR, BORDER);
				else KERNEL_SWITCH(LINEAR, LINEAR, BORDER);
				else KERNEL_SWITCH(MIN, LINEAR, BORDER);
				else KERNEL_SWITCH(MAX, LINEAR, BORDER);
				
				else KERNEL_SWITCH(NEAREST, NONE, CLAMP);
				else KERNEL_SWITCH(LINEAR, NONE, CLAMP);
				else KERNEL_SWITCH(MIN, NONE, CLAMP);
				else KERNEL_SWITCH(MAX, NONE, CLAMP);
				else KERNEL_SWITCH(NEAREST, NEAREST, CLAMP);
				else KERNEL_SWITCH(LINEAR, NEAREST, CLAMP);
				else KERNEL_SWITCH(MIN, NEAREST, CLAMP);
				else KERNEL_SWITCH(MAX, NEAREST, CLAMP);
				else KERNEL_SWITCH(NEAREST, LINEAR, CLAMP);
				else KERNEL_SWITCH(LINEAR, LINEAR, CLAMP);
				else KERNEL_SWITCH(MIN, LINEAR, CLAMP);
				else KERNEL_SWITCH(MAX, LINEAR, CLAMP);
				
				else KERNEL_SWITCH(NEAREST, NONE, WRAP);
				else KERNEL_SWITCH(LINEAR, NONE, WRAP);
				else KERNEL_SWITCH(MIN, NONE, WRAP);
				else KERNEL_SWITCH(MAX, NONE, WRAP);
				else KERNEL_SWITCH(NEAREST, NEAREST, WRAP);
				else KERNEL_SWITCH(LINEAR, NEAREST, WRAP);
				else KERNEL_SWITCH(MIN, NEAREST, WRAP);
				else KERNEL_SWITCH(MAX, NEAREST, WRAP);
				else KERNEL_SWITCH(NEAREST, LINEAR, WRAP);
				else KERNEL_SWITCH(LINEAR, LINEAR, WRAP);
				else KERNEL_SWITCH(MIN, LINEAR, WRAP);
				else KERNEL_SWITCH(MAX, LINEAR, WRAP);
				
				else throw std::invalid_argument("Unsupported sampling configuration.");
				#undef KERNEL_SWITCH
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			}
			END_SAMPLE("Sample kernel gradients");
			
		}
//		if(useMipmap){
			//TODO: fuse mips into output
			//kCollapseGradMip3D<T><<<>>>(mip[m+1], mip[m], dimensions[m]);
//		}
		if(NORMALIZE_GRADIENTS){
			BEGIN_SAMPLE;
			{
				const dim3 grad_grid(GRID_DIMS(dims.input));
				kNormalize3DGradients<T, true><<<grad_grid, block>>>(currInput_grad, sample_count_buffer);
			}
			END_SAMPLE("Grad normalize");
		}
	}
	
	BEGIN_SAMPLE;
	{
//		if(useMipmap){
			//freeMipAtlas3D(inputMips);
			//freeMipAtlas3D(input_gradMips);
//		}
	}
	END_SAMPLE("Free memory");
	
	LOG("End SampleGridLuTGradKernelLauncher");
}

void SampleRGridLuTGradKernelLauncher(const void* _input, const long long int* input_shape,
		int32_t numCameras,
		const float* _lookup,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling, const bool relative, const bool normalized,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, void* _lookup_grad, uint32_t* sample_count_buffer){
	SampleGridLuTGradKernelLauncher<float1>(_input, input_shape, numCameras, _lookup, coordinateMode, samplingSettings, globalSampling, relative, normalized, 
		_output_grad, output_shape, _input_grad, _lookup_grad, sample_count_buffer);
}
void SampleRGGridLuTGradKernelLauncher(const void* _input, const long long int* input_shape,
		int32_t numCameras,
		const float* _lookup,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling, const bool relative, const bool normalized,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, void* _lookup_grad, uint32_t* sample_count_buffer){
	SampleGridLuTGradKernelLauncher<float2>(_input, input_shape, numCameras, _lookup, coordinateMode, samplingSettings, globalSampling, relative, normalized, 
		_output_grad, output_shape, _input_grad, _lookup_grad, sample_count_buffer);
}
void SampleRGBAGridLuTGradKernelLauncher(const void* _input, const long long int* input_shape,
		int32_t numCameras,
		const float* _lookup,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling, const bool relative, const bool normalized,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, void* _lookup_grad, uint32_t* sample_count_buffer){
	SampleGridLuTGradKernelLauncher<float4>(_input, input_shape, numCameras, _lookup, coordinateMode, samplingSettings, globalSampling, relative, normalized, 
		_output_grad, output_shape, _input_grad, _lookup_grad, sample_count_buffer);
}

/********************************************
*	Compute LoD
********************************************/

template<Sampling::CoordinateMode CM>
__global__ void 
__launch_bounds__(BLOCK_SIZE)
kComputeLoD3D(float4* output){
	MAKE_GLOBAL_INDEX;
	if(isInDimensions<glm::ivec3,glm::ivec3>(globalIdx, c_dimensions.output)){
		glm::vec3 idxCoords = indexToCoords(glm::vec3(globalIdx));
		glm::vec4 sizeLoD = Sampling::calcLoD<CM>(idxCoords);
		glm::vec4 pos = Sampling::samplePos3D<CM,false>(idxCoords);
		pos.w=sizeLoD.w;
		vectorIO::writeVectorType3D<float4,float4, glm::ivec3>(vectorIO::toVector<glm::vec4,float4>(pos), globalIdx, c_dimensions.output, output);
	}
}

void ComputeLoDKernelLauncher(const long long int* input_shape,
		const float* MV, const float* P, const float* _frustum,
		const Sampling::CoordinateMode coordinateMode,
		void* output, const long long int* output_shape){
	
	LOG("Start ComputeLoDKernelLauncher");
	
	//precompute globals
	BEGIN_SAMPLE;
	LOG("Set dimensions");
	const size_t batchSize = input_shape[0];
	Dimensions dims;
	setDimensions(dims, input_shape, output_shape);
	LOG("Dimensions set");
	
	const size_t outputSliceSizeElements = vmath::prod(dims.output);
	
	Transformations transforms;
	FrustumParams frustum;
	
	END_SAMPLE("Precompute and copy global constants");
	
	
	const dim3 grid(GRID_DIMS(dims.output));
	const dim3 block(BLOCK_DIMS);
	LOG("Generate sampling coords for " << batchSize << " grids");
	for(size_t batch=0; batch<batchSize; ++batch){
		LOG("Grid " << batch);
		LOG("Dimensions in: " << LOG_V3_XYZ(dims.input) << ", pitch: " << dims.input.x*sizeof(float4));
		float4* currOutput = reinterpret_cast<float4*>(output)+batch*outputSliceSizeElements;
		
		LOG("Set transformations");
		if(coordinateMode!=Sampling::LuT){
			setTransformations(transforms, nullptr, MV + batch*16, P + batch*16);
			LOG("Transformations set");
			setFrustumParams(frustum, _frustum + batch*6);
			LOG("FrustumParams set");
		}
		
		LOG("Dipatch CUDA kernel: " << dims.output.x << " x " << dims.output.y << " x " << dims.output.z << " threads");
		BEGIN_SAMPLE;
		{
			switch(coordinateMode){
				case Sampling::TransformLinDepthReverse: kComputeLoD3D<Sampling::TransformLinDepthReverse><<<grid, block>>>(currOutput);
			break;
				case Sampling::TransformLinDepth: kComputeLoD3D<Sampling::TransformLinDepth><<<grid, block>>>(currOutput);
			break;
				default: throw std::invalid_argument("Unsupported coordinate mode.");
			}
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}
		END_SAMPLE("compute LoD kernel");
		
	}
	
	BEGIN_SAMPLE;
	{
		
	}
	END_SAMPLE("Free memory");
	LOG("End ComputeLoDKernelLauncher");
}

#endif //GOOGLE_CUDA