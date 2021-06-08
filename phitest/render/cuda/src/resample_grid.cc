#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/framework/array_ops.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <iostream>
#include <string>
//#define LOGGING
#include "resample_grid.hpp"
#include "render_errors.hpp"

#ifdef LOGGING
#define MYLOG(msg) std::cout << __FILE__ << "[" << __LINE__ << "]: " << msg << std::endl
#define LOG_PRINTF(msg) printf(msg)
#else
#define MYLOG(msg)
#define LOG_PRINTF(msg)
#endif

using namespace tensorflow;

// Sample at transformed 3D grid positions from an input grid
REGISTER_OP("SampleGridTransform")
	.Attr("T: {float}")
	.Input("input: T") // NDHWC
	.Input("matrix_m: float32") // 4x4 matrix or batch N thereof
	.Input("matrix_v: float32") // 4x4 matrix or batch V thereof
	.Input("matrix_p: float32") // 4x4 matrix or batch V thereof
	.Input("frustum_params: float32") // 6 elemnts or batch V thereof
	.Input("output_shape: int32") // DHW
	.Attr("interpolation: {'NEAREST', 'LINEAR'} = 'LINEAR'")
	.Attr("boundary: {'BORDER', 'CLAMP'} = 'BORDER'")
	.Attr("mipmapping: {'NONE', 'NEAREST', 'LINEAR'} = 'NONE'")
	.Attr("num_mipmaps: int = 0")
	.Attr("mip_bias: float = 0.0")
	.Attr("coordinate_mode: {'TRANSFORM_LINDEPTH', 'TRANSFORM_LINDEPTH_REVERSE', 'TRANSFORM', 'TRANSFORM_REVERSE'} = 'TRANSFORM_LINDEPTH'") //, 'RAY'
	.Attr("cell_center_offset: float = 0.5") // offset of cell center coordinates from cell indices. should be 0.5 for all transform modes and if mipmapping is used
	.Attr("separate_camera_batch: bool = true")
	.Output("output: T") // NVDHWC
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
	{
		::tensorflow::shape_inference::ShapeHandle inShape = c->input(0);
		::tensorflow::shape_inference::ShapeHandle outShape;
		TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &outShape));
		TF_RETURN_IF_ERROR(c->Subshape(outShape, 0, 3, &outShape));
		
		//prepend camera batch
		bool globalSampling;
		TF_RETURN_IF_ERROR(c->GetAttr("separate_camera_batch", &globalSampling));
		if(globalSampling){
			TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(c->UnknownDim()), outShape, &outShape));
		}else{
			TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(1), outShape, &outShape));
		}
		//prepend data batch
		TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(c->Dim(inShape, 0)), outShape, &outShape));
		//append channel
		TF_RETURN_IF_ERROR(c->Concatenate(outShape, c->Vector(c->Dim(inShape, -1)), &outShape));
		c->set_output(0, outShape);
		return Status::OK();
	});

// the gradient op
REGISTER_OP("SampleGridTransformGrad")
	.Input("input: float32") // NDHWC
	.Input("output_grad: float32") // NVDHWC
	.Input("matrix_m: float32") // 4x4 matrix or batch N thereof
	.Input("matrix_v: float32") // 4x4 matrix or batch V thereof
	.Input("matrix_p: float32") // 4x4 matrix or batch V thereof
	.Input("frustum_params: float32")
	.Attr("interpolation: {'NEAREST', 'LINEAR'} = 'LINEAR'")
	.Attr("boundary: {'BORDER', 'CLAMP'} = 'BORDER'")
	.Attr("mipmapping: {'NONE'} = 'NONE'") //, 'NEAREST', 'LINEAR'. currently not supported
	.Attr("num_mipmaps: int = 0")
	.Attr("mip_bias: float = 0.0")
	.Attr("coordinate_mode: {'TRANSFORM_LINDEPTH', 'TRANSFORM_LINDEPTH_REVERSE', 'TRANSFORM', 'TRANSFORM_REVERSE'} = 'TRANSFORM_LINDEPTH'") //, 'RAY'
	.Attr("cell_center_offset: float = 0.5") // offset of cell center coordinates from cell indices. should be 0.5 for all transform modes and if mipmapping is used
	.Attr("separate_camera_batch: bool = true")
	.Output("input_grad: float32")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
	{
		c->set_output(0, c->input(0));
		return Status::OK();
	});

REGISTER_OP("SampleGridLut")
	.Input("input: float32") // NDHWC
	.Input("lookup: float32") //Lookup Table, defines output shape
	.Attr("interpolation: {'NEAREST', 'LINEAR', 'MIN', 'MAX'} = 'LINEAR'")
	.Attr("boundary: {'BORDER', 'CLAMP', 'WRAP'} = 'BORDER'")
	.Attr("mipmapping: {'NONE', 'NEAREST', 'LINEAR'} = 'NONE'")
	.Attr("num_mipmaps: int = 0")
	.Attr("mip_bias: float = 0.0")
	.Attr("coordinate_mode: {'LOOKUP'} = 'LOOKUP'") //, 'RAY'
	.Attr("cell_center_offset: float = 0.0") // offset of cell center coordinates from cell indices. should be 0.5 for all transform modes and if mipmapping is used
	.Attr("separate_camera_batch: bool = true")
	.Attr("relative_coords: bool = true")
	.Attr("normalized_coords: bool = false")
	.Output("output: float32") // NVDHWC
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
	{
		::tensorflow::shape_inference::ShapeHandle channel;
		TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 3, &channel));
		::tensorflow::shape_inference::ShapeHandle outShape;
		TF_RETURN_IF_ERROR(c->Subshape(c->input(1), 0, 3, &outShape));
		TF_RETURN_IF_ERROR(c->Concatenate(outShape, channel, &outShape));
		c->set_output(0, outShape);
		return Status::OK();
	});


// the gradient op
REGISTER_OP("SampleGridLutGrad")
	.Input("input: float32") // NDHWC
	.Input("output_grad: float32") // NVDHWC
	.Input("lookup: float32") //LuT if used
	.Attr("interpolation: {'NEAREST', 'LINEAR', 'MIN', 'MAX'} = 'LINEAR'")
	.Attr("boundary: {'BORDER', 'CLAMP', 'WRAP'} = 'BORDER'")
	.Attr("mipmapping: {'NONE'} = 'NONE'") //, 'NEAREST', 'LINEAR'. currently not supported
	.Attr("num_mipmaps: int = 0")
	.Attr("mip_bias: float = 0.0")
	.Attr("coordinate_mode: {'LOOKUP'} = 'LOOKUP'") //, 'RAY'
	.Attr("cell_center_offset: float = 0.0") // offset of cell center coordinates from cell indices. should be 0.5 for all transform modes and if mipmapping is used
	.Attr("separate_camera_batch: bool = true")
	.Attr("relative_coords: bool = true")
	.Attr("normalized_coords: bool = false")
	.Output("input_grad: float32")
	.Output("lookup_grad: float32") // None if lookup is not used
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
	{
		c->set_output(0, c->input(0));
		c->set_output(1, c->input(2));
		return Status::OK();
	});

REGISTER_OP("LodTransform")
	.Input("input_shape: int32") //DHW shape of the grid that is sampled FROM
	.Input("matrix_mv: float32")
	.Input("matrix_p: float32")
	.Input("frustum_params: float32")
	.Input("output_shape: int32") //DHW shape of the grid that is sampled TO
	.Attr("inverse_transform: bool = false") // if false: sample from object grid to frustum grid, else: vice versa
	.Output("output: float32") //VDHWC V-batch of sampling coordinates, C=4 (x,y,z,LoD)
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
	{
		::tensorflow::shape_inference::ShapeHandle channel;
		TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 3, &channel));
		::tensorflow::shape_inference::ShapeHandle outShape;
		TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &outShape));
		TF_RETURN_IF_ERROR(c->Subshape(outShape, 0, 3, &outShape));
		TF_RETURN_IF_ERROR(c->Merge(outShape, channel, &outShape));
		c->set_output(0, outShape);
		return Status::OK();
	});

Status makeShapeFromTensor(const Tensor& sizes, TensorShape* shape){
	MYLOG("Create shape from tensor");
	auto sizes_flat = sizes.flat<int32>();
	const int64 num_dims = sizes_flat.size();
	for(int64 i=0; i<num_dims; ++i){
		const int32 dim = sizes_flat(i);
		if(dim<1){
			return errors::InvalidArgument("All shape dimensions must be positive, dimension ", i, " is ", dim);
		}
		MYLOG("Add dimension " << dim);
		shape->AddDim(dim);
	}
	return Status::OK();
}

bool isValidTransformMatrix(const TensorShape& matrixShape, int32_t& numMatrices){
	if(matrixShape.dims()==2){
		numMatrices = 1;
		return matrixShape.dim_size(0)==matrixShape.dim_size(1) && matrixShape.dim_size(0)==4;
	} else if(matrixShape.dims()==3){
		numMatrices = matrixShape.dim_size(0);
		return matrixShape.dim_size(1)==matrixShape.dim_size(2) && matrixShape.dim_size(1)==4;
	} else{
		numMatrices = 0;
		return false;
	}
}

bool isValidFrustumParams(const TensorShape& shape, int32_t& numCameras){
	if(shape.dims()==1){
		numCameras = 1;
		return shape.dim_size(0)==6;
	}else if(shape.dims()==2){
		numCameras = shape.dim_size(0);
		return shape.dim_size(1)==6;
	}else{
		numCameras = 0;
		return false;
	}
}

/*
	calculate linear memory needed for mips and atlas
	set numMips to maximum possible level of mips with the given dimensions
*/
size_t setupMipAtlas(const TensorShape& base_shape, Sampling::SamplerSettings& samplingSettings, const size_t elementSizeBytes, const size_t prtSizeBytes, const size_t alignment=128){
	
	#define ALIGN_ADDR_UP(addr) (((addr) + (alignment-1) ) & ~(alignment-1))
	size_t mipsSizeBytes = ALIGN_ADDR_UP((samplingSettings.mipLevel+1) * prtSizeBytes) + alignment;
	
	for(int32_t m=1; m<(samplingSettings.mipLevel+1); ++m){
		size_t elements = (base_shape.dim_size(1)>>m) * (base_shape.dim_size(2)>>m) * (base_shape.dim_size(3)>>m);
		if(elements<=0){ //too small, mip would be empty
			samplingSettings.mipLevel = m-1;
			break;
		}
		mipsSizeBytes += ALIGN_ADDR_UP(elements * elementSizeBytes);
	}
	//no mips, so turn off the mipmapping system
	if(samplingSettings.mipLevel==0){
		samplingSettings.mipMode==Sampling::SamplerSettings::MIPMODE_NONE;
	}
	
	return mipsSizeBytes;
}

#define COND_SET_SM(SM) (samplingMode.compare(#SM)==0) mode=Sampling::SM
bool setSamplingMode(Sampling::SamplingMode &mode, const std::string samplingMode){
	if COND_SET_SM(LINEAR);
	else if COND_SET_SM(LINEAR_MIP_LINEAR);
	else if COND_SET_SM(TEX_LINEAR);
	else if COND_SET_SM(NEAREST);
	else if COND_SET_SM(TEX_NEAREST);
	else return false;
	return true;
}
#undef COND_SET_SM

#define COND_SET_CM(s,CM) (coordinateMode.compare(#s)==0) mode=Sampling::CM
bool setCoordinateMode(Sampling::CoordinateMode &mode, const std::string coordinateMode){
	if COND_SET_CM(TRANSFORM_LINDEPTH,TransformLinDepth);
	else if COND_SET_CM(TRANSFORM_LINDEPTH_REVERSE,TransformLinDepthReverse);
	else if COND_SET_CM(TRANSFORM,Transform);
	else if COND_SET_CM(TRANSFORM_REVERSE,TransformReverse);
	else if COND_SET_CM(LOOKUP,LuT);
	else return false;
	return true;
}
#undef COND_SET_CM


#if GOOGLE_CUDA
#define DECLARE_GPU_SPEC(T, C) \
	template<> \
	void SampleGridKernel<GPUDevice, T, C>::operator()( \
		const GPUDevice& d, \
		const void* input, const long long int* input_shape, \
		const float* M, const float* V, const float* P, const float* frustum, int32_t numCameras, \
		uint8_t* mipAtlas, \
		const Sampling::CoordinateMode coordinateMode, \
		const Sampling::SamplerSettings, const bool globalSampling, \
		void* output, const long long int* output_shape \
		); \
	extern template struct SampleGridKernel<GPUDevice, T, C>;
DECLARE_GPU_SPEC(float, 1)
DECLARE_GPU_SPEC(float, 2)
DECLARE_GPU_SPEC(float, 4)
#undef DECLARE_GPU_SPEC
#endif

//only for 3D grids with up to 4 channel
//no support for batches yet?
template<typename Device, typename T>
class SampleGridTransformOp : public OpKernel{
public:
	explicit SampleGridTransformOp(OpKernelConstruction *context) : OpKernel(context){
		
		memset(&m_samplingSettings, 0, sizeof(Sampling::SamplerSettings));
		std::string s_interpolation;
		OP_REQUIRES_OK(context, context->GetAttr("interpolation", &s_interpolation));
		if(s_interpolation.compare("LINEAR")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_LINEAR;
		else if(s_interpolation.compare("MIN")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_MIN;
		else if(s_interpolation.compare("MAX")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_MAX;
		
		std::string s_boundary;
		OP_REQUIRES_OK(context, context->GetAttr("boundary", &s_boundary));
		if(s_boundary.compare("CLAMP")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::CLAMP;
		if(s_boundary.compare("WRAP")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::WRAP;
		if(s_boundary.compare("MIRROR")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::MIRROR;
		
		std::string s_mipmapping;
		OP_REQUIRES_OK(context, context->GetAttr("mipmapping", &s_mipmapping));
		if(s_mipmapping.compare("NEAREST")==0) m_samplingSettings.mipMode = Sampling::SamplerSettings::MIPMODE_NEAREST;
		else if(s_mipmapping.compare("LINEAR")==0) m_samplingSettings.mipMode = Sampling::SamplerSettings::MIPMODE_LINEAR;
		
		OP_REQUIRES_OK(context, context->GetAttr("num_mipmaps", &m_samplingSettings.mipLevel));
		OP_REQUIRES(context, m_samplingSettings.mipLevel>0 || m_samplingSettings.mipMode==Sampling::SamplerSettings::MIPMODE_NONE,
			errors::InvalidArgument("when using mipmaps num_mipmaps must be larger than 0."));
			
		OP_REQUIRES_OK(context, context->GetAttr("mip_bias", &m_samplingSettings.mipBias));
		
		OP_REQUIRES_OK(context, context->GetAttr("separate_camera_batch", &m_globalSampling));
		
		std::string s_coordmode;
		OP_REQUIRES_OK(context, context->GetAttr("coordinate_mode", &s_coordmode));
		OP_REQUIRES(context, setCoordinateMode(m_coordinateMode, s_coordmode),
			errors::InvalidArgument("invalid coordinate_mode."));
		OP_REQUIRES_OK(context, context->GetAttr("cell_center_offset", &m_samplingSettings.cellCenterOffset));
		if(m_coordinateMode!=Sampling::LuT){
			OP_REQUIRES(context, m_samplingSettings.cellCenterOffset==0.5f,
			errors::InvalidArgument("Invalid cell center offset for given coordinate mode."));
		}
	}
	
	void Compute(OpKernelContext *context) override{
		
		MYLOG("Sample transform op kernel start");
		
		const Tensor& input_grid = context->input(0);
		const Tensor& tensor_M = context->input(1);
		const Tensor& tensor_V = context->input(2);
		const Tensor& tensor_P = context->input(3);
		const Tensor& frustum = context->input(4);
		const Tensor& output_shape_tensor = context->input(5);
		
		//check input
		MYLOG("Check input");
		TensorShape input_shape = input_grid.shape();
		OP_REQUIRES(context, input_grid.dims()==5 && input_shape.dim_size(4)<=4,
			errors::InvalidArgument("Invalid input shape (NDHWC):", input_shape.DebugString()));
		OP_REQUIRES(context, 0<(input_shape.dim_size(0)*input_shape.dim_size(1)*input_shape.dim_size(2)*input_shape.dim_size(3)*input_shape.dim_size(4)),
			errors::InvalidArgument("Empty input (NDHWC):", input_shape.DebugString()));
		const int64 batch = input_shape.dim_size(0);
		const int64 channel = input_shape.dim_size(4);
		OP_REQUIRES(context, !(channel==3||channel>4),
				errors::Unimplemented("Only 1,2 and 4 Channel supported."));
		
		//check output shape
		MYLOG("Check output shape tensor");
		OP_REQUIRES(context, output_shape_tensor.dims()==1 && output_shape_tensor.dim_size(0)==3, errors::InvalidArgument("Invalid output_shape"));
		MYLOG("Create output shape");
		TensorShape output_shape;
		OP_REQUIRES_OK(context, makeShapeFromTensor(output_shape_tensor, &output_shape));
		MYLOG("Check output shape");
		OP_REQUIRES(context, output_shape.dims()==3,
			errors::InvalidArgument("Invalid output_shape"));
		OP_REQUIRES(context, 0<(output_shape.dim_size(0)*output_shape.dim_size(1)*output_shape.dim_size(2)),
			errors::InvalidArgument("Empty output (NDHWC):", output_shape.DebugString()));
		output_shape.InsertDim(0,batch);
		output_shape.AddDim(channel);
		MYLOG("output_shape: " << output_shape.dim_size(0) << ", " << output_shape.dim_size(1) << ", " << output_shape.dim_size(2) << ", " << output_shape.dim_size(3));
		
		//check transform matrics
		int32_t numCameras=0;
		OP_REQUIRES(context, m_coordinateMode==Sampling::TransformLinDepth || m_coordinateMode==Sampling::TransformLinDepthReverse
			|| m_coordinateMode==Sampling::Transform || m_coordinateMode==Sampling::TransformReverse,
			errors::InvalidArgument("Invalid coordinate_mode"));
		MYLOG("Check transform");
		int32 numMatrix_M=0, numMatrix_V=0, numMatrix_P=0, numFrustum=0;
		OP_REQUIRES(context, isValidTransformMatrix(tensor_M.shape(), numMatrix_M),
			errors::InvalidArgument("transformation matrix_m must be a 4x4 square matrix or a batch of 4x4 square matrices:", tensor_M.shape().DebugString()));
		OP_REQUIRES(context, numMatrix_M==batch, errors::InvalidArgument("model matrix batch size mismatch"));
		
		OP_REQUIRES(context, isValidTransformMatrix(tensor_V.shape(), numMatrix_V),
			errors::InvalidArgument("transformation matrix_v must be a 4x4 square matrix or a batch of 4x4 square matrices:", tensor_V.shape().DebugString()));
		OP_REQUIRES(context, isValidTransformMatrix(tensor_P.shape(), numMatrix_P),
			errors::InvalidArgument("transformation matrix_p must be a 4x4 square matrix or a batch of 4x4 square matrices:", tensor_P.shape().DebugString()));
		OP_REQUIRES(context, isValidFrustumParams(frustum.shape(), numFrustum),
			errors::InvalidArgument("frustum must be a 1D tensor of 6 elements or a batch thereof:", frustum.shape().DebugString()));
		//TODO
		OP_REQUIRES(context, numMatrix_V==numMatrix_P && numMatrix_V==numFrustum, errors::InvalidArgument("camera batch size mismatch"));
		numCameras = numMatrix_V;
		
		if(!m_globalSampling){
			OP_REQUIRES(context, numCameras==batch, errors::InvalidArgument("camera batch must match data batch when not using global sampling."));
			output_shape.InsertDim(1,1);
		}else{
			output_shape.InsertDim(1,numCameras);
		}
		
		
		
		//allocate outout
		MYLOG("Allocate output");
		//OP_REQUIRES(context, output_shape.IsValid(), errors::InvalidArgument("Broken output shape"));
		Tensor* output_grid = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_grid));
		MYLOG("Check allocated output\n");
		auto out = output_grid->flat<float>();
		MYLOG("Allocated output size: " << out.size() << " - " << output_grid->NumElements());
		
		//allocate temporary mip atlas
		uint8_t* mipAtlas = nullptr;
		Tensor mip_atlas;
		size_t mipAtlasSize = setupMipAtlas(input_shape, m_samplingSettings, sizeof(float)*channel, sizeof(float*));
		if(m_samplingSettings.mipMode!=Sampling::SamplerSettings::MIPMODE_NONE){
			MYLOG("Allocate mips");
			TensorShape mip_atlas_shape;
			mip_atlas_shape.AddDim(mipAtlasSize);
			OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT8, mip_atlas_shape, &mip_atlas));
			mipAtlas = mip_atlas.flat<uint8_t>().data();
		}
		
		
		//TODO handle arbitrary amount of channel
		// - move channel dimension outwards (to batch) and handle only 1-channel case internally. would also handle batches
		//   this would benefit from NCHW layout, otherwise have to transpose
		//   or just require NCHW as input format (or NHWC with up to 4 channel, the rest has to be packed in N) and let tensorflow/user handle the conversion...
		// - split into up-to-4-channel partitions. might be faster? but is harder to handle
		
		MYLOG("Resample");
		switch(channel){
		case 1:
			SampleGridKernel<Device, T, 1>()(context->eigen_device<Device>(),
				input_grid.flat<T>().data(), input_shape.dim_sizes().data(),
				tensor_M.flat<float>().data(), tensor_V.flat<float>().data(), tensor_P.flat<float>().data(), frustum.flat<float>().data(), numCameras,
				mipAtlas,
				m_coordinateMode,
				m_samplingSettings, m_globalSampling,
				output_grid->flat<T>().data(), output_shape.dim_sizes().data());
			break;
		case 2:
			SampleGridKernel<Device, T, 2>()(context->eigen_device<Device>(),
				input_grid.flat<T>().data(), input_shape.dim_sizes().data(),
				tensor_M.flat<float>().data(), tensor_V.flat<float>().data(), tensor_P.flat<float>().data(), frustum.flat<float>().data(), numCameras,
				mipAtlas,
				m_coordinateMode,
				m_samplingSettings, m_globalSampling,
				output_grid->flat<T>().data(), output_shape.dim_sizes().data());
			break;
		case 4:
			SampleGridKernel<Device, T, 4>()(context->eigen_device<Device>(),
				input_grid.flat<T>().data(), input_shape.dim_sizes().data(),
				tensor_M.flat<float>().data(), tensor_V.flat<float>().data(), tensor_P.flat<float>().data(), frustum.flat<float>().data(), numCameras,
				mipAtlas,
				m_coordinateMode,
				m_samplingSettings, m_globalSampling,
				output_grid->flat<T>().data(), output_shape.dim_sizes().data());
			break;
		default:
			OP_REQUIRES(context, false,
				errors::Unimplemented("Only 1,2 and 4 Channel supported."));
		}
		
		MYLOG("Kernel done");
	}
private:
	bool m_inverseTransform;
	bool m_globalSampling;
	Sampling::SamplerSettings m_samplingSettings;
	Sampling::CoordinateMode m_coordinateMode;
};


void SampleRGridGradKernelLauncher(const void* _input, const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* _frustum, int32_t numCameras,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, uint32_t* sample_count_buffer);
void SampleRGGridGradKernelLauncher(const void* _input, const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* _frustum, int32_t numCameras,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, uint32_t* sample_count_buffer);
void SampleRGBAGridGradKernelLauncher(const void* _input, const long long int* input_shape,
		const float* M, const float* V, const float* P, const float* _frustum, int32_t numCameras,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, uint32_t* sample_count_buffer);

class SampleGridTransformGradOp : public OpKernel{
public:
	explicit SampleGridTransformGradOp(OpKernelConstruction *context) : OpKernel(context){
		
		memset(&m_samplingSettings, 0, sizeof(Sampling::SamplerSettings));
		std::string s_interpolation;
		OP_REQUIRES_OK(context, context->GetAttr("interpolation", &s_interpolation));
		if(s_interpolation.compare("LINEAR")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_LINEAR;
		else if(s_interpolation.compare("MIN")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_MIN;
		else if(s_interpolation.compare("MAX")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_MAX;
		std::string s_boundary;
		OP_REQUIRES_OK(context, context->GetAttr("boundary", &s_boundary));
		if(s_boundary.compare("CLAMP")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::CLAMP;
		if(s_boundary.compare("WRAP")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::WRAP;
		if(s_boundary.compare("MIRROR")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::MIRROR;
		
		std::string s_mipmapping;
		OP_REQUIRES_OK(context, context->GetAttr("mipmapping", &s_mipmapping));
		if(s_mipmapping.compare("NEAREST")==0) m_samplingSettings.mipMode = Sampling::SamplerSettings::MIPMODE_NEAREST;
		else if(s_mipmapping.compare("LINEAR")==0) m_samplingSettings.mipMode = Sampling::SamplerSettings::MIPMODE_LINEAR;
		
		OP_REQUIRES_OK(context, context->GetAttr("num_mipmaps", &m_samplingSettings.mipLevel));
		OP_REQUIRES(context, m_samplingSettings.mipLevel>0 || m_samplingSettings.mipMode==Sampling::SamplerSettings::MIPMODE_NONE,
			errors::InvalidArgument("when using mipmaps num_mipmaps must be larger than 0."));
			
		OP_REQUIRES_OK(context, context->GetAttr("mip_bias", &m_samplingSettings.mipBias));
		
		OP_REQUIRES_OK(context, context->GetAttr("separate_camera_batch", &m_globalSampling));
		std::string s_coordmode;
		OP_REQUIRES_OK(context, context->GetAttr("coordinate_mode", &s_coordmode));
		OP_REQUIRES(context, setCoordinateMode(m_coordinateMode, s_coordmode),
			errors::InvalidArgument("invalid coordinate_mode."));
		OP_REQUIRES_OK(context, context->GetAttr("cell_center_offset", &m_samplingSettings.cellCenterOffset));
		if(m_coordinateMode!=Sampling::LuT){
			OP_REQUIRES(context, m_samplingSettings.cellCenterOffset==0.5f,
			errors::InvalidArgument("Invalid cell center offset for given coordinate mode."));
		}
	}
	
	void Compute(OpKernelContext *context) override{
		
		MYLOG("Gradient kernel start");
		
		const Tensor& input_grid = context->input(0);
		const Tensor& output_grad_grid = context->input(1);
		const Tensor& tensor_M = context->input(2);
		const Tensor& tensor_V = context->input(3);
		const Tensor& tensor_P = context->input(4);
		const Tensor& frustum = context->input(5);
		const Tensor& tensor_lookup = context->input(6);
		
		//check input
		MYLOG("Check input");
		TensorShape input_shape = input_grid.shape();
		OP_REQUIRES(context, input_grid.dims()==5 && input_shape.dim_size(4)<=4,
			errors::InvalidArgument("Invalid input shape (NDHWC):", input_shape.DebugString()));
		const int64 batch = input_shape.dim_size(0);
		const int64 channel = input_shape.dim_size(4);
		
		//check output gradients
		MYLOG("Check output_grads");
		TensorShape output_shape = output_grad_grid.shape();
		OP_REQUIRES(context, output_grad_grid.dims()==6 && output_shape.dim_size(5)<=4,
			errors::InvalidArgument("Invalid output_grads shape (NVDHWC):", output_shape.DebugString()));
		OP_REQUIRES(context, output_shape.dim_size(0)==batch,
			errors::InvalidArgument("output_grads batch size does not match input batch size."));
		
		
		//check transform matrics
		int32_t numCameras=0;
		OP_REQUIRES(context, m_coordinateMode==Sampling::TransformLinDepth || m_coordinateMode==Sampling::TransformLinDepthReverse
			|| m_coordinateMode==Sampling::Transform || m_coordinateMode==Sampling::TransformReverse,
			errors::InvalidArgument("Invalid coordinate_mode"));
		MYLOG("Check transform");
		int32 numMatrix_M=0, numMatrix_V=0, numMatrix_P=0, numFrustum=0;
		OP_REQUIRES(context, isValidTransformMatrix(tensor_M.shape(), numMatrix_M),
			errors::InvalidArgument("transformation matrix_m must be a 4x4 square matrix or a batch of 4x4 square matrices:", tensor_M.shape().DebugString()));
		OP_REQUIRES(context, numMatrix_M==batch, errors::InvalidArgument("model matrix batch size mismatch"));
		
		OP_REQUIRES(context, isValidTransformMatrix(tensor_V.shape(), numMatrix_V),
			errors::InvalidArgument("transformation matrix_v must be a 4x4 square matrix or a batch of 4x4 square matrices:", tensor_V.shape().DebugString()));
		OP_REQUIRES(context, isValidTransformMatrix(tensor_P.shape(), numMatrix_P),
			errors::InvalidArgument("transformation matrix_p must be a 4x4 square matrix or a batch of 4x4 square matrices:", tensor_P.shape().DebugString()));
		OP_REQUIRES(context, isValidFrustumParams(frustum.shape(), numFrustum),
			errors::InvalidArgument("frustum must be a 1D tensor of 6 elements or a batch thereof:", frustum.shape().DebugString()));
		//TODO
		OP_REQUIRES(context, numMatrix_V==numMatrix_P && numMatrix_V==numFrustum, errors::InvalidArgument("camera batch size mismatch"));
		numCameras = numMatrix_V;
		
		if(!m_globalSampling){
			OP_REQUIRES(context, numCameras==batch, errors::InvalidArgument("camera batch must match data batch when not using global sampling."));
			
			OP_REQUIRES(context, output_shape.dim_size(1)==1,
				errors::InvalidArgument("output_grads views size must be 1 if not using global sampling."));
		}else{
			OP_REQUIRES(context, output_shape.dim_size(1)==numCameras,
				errors::InvalidArgument("output_grads views size does not match cameras."));
		}
		
		uint32_t* sample_count_buffer = nullptr;
		Tensor sample_count_tensor;
		if(NORMALIZE_GRADIENTS){
			MYLOG("Allocate gradient counter");
			TensorShape sample_count_tensor_shape;
			sample_count_tensor_shape.AddDim(input_shape.dim_size(1));
			sample_count_tensor_shape.AddDim(input_shape.dim_size(2));
			sample_count_tensor_shape.AddDim(input_shape.dim_size(3));
			OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT32, sample_count_tensor_shape, &sample_count_tensor));
			sample_count_buffer = sample_count_tensor.flat<uint32_t>().data();
		}
		
		//allocate outout
		MYLOG("Allocate input gradients");
		//OP_REQUIRES(context, output_shape.IsValid(), errors::InvalidArgument("Broken output shape"));
		Tensor* input_grads = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &input_grads));
		
		
		
		MYLOG("Resample\n");
		switch(channel){
		case 1:
			SampleRGridGradKernelLauncher(input_grid.flat<float>().data(), input_shape.dim_sizes().data(),
				tensor_M.flat<float>().data(), tensor_V.flat<float>().data(), tensor_P.flat<float>().data(), frustum.flat<float>().data(), numCameras,
				m_coordinateMode,
				m_samplingSettings, m_globalSampling,
				output_grad_grid.flat<float>().data(), output_shape.dim_sizes().data(),
				input_grads->flat<float>().data(), sample_count_buffer);
			break;
		case 2:
			SampleRGGridGradKernelLauncher(input_grid.flat<float>().data(), input_shape.dim_sizes().data(),
				tensor_M.flat<float>().data(), tensor_V.flat<float>().data(), tensor_P.flat<float>().data(), frustum.flat<float>().data(), numCameras,
				m_coordinateMode,
				m_samplingSettings, m_globalSampling,
				output_grad_grid.flat<float>().data(), output_shape.dim_sizes().data(),
				input_grads->flat<float>().data(), sample_count_buffer);
			break;
		case 4:
			SampleRGBAGridGradKernelLauncher(input_grid.flat<float>().data(), input_shape.dim_sizes().data(),
				tensor_M.flat<float>().data(), tensor_V.flat<float>().data(), tensor_P.flat<float>().data(), frustum.flat<float>().data(), numCameras,
				m_coordinateMode,
				m_samplingSettings, m_globalSampling,
				output_grad_grid.flat<float>().data(), output_shape.dim_sizes().data(),
				input_grads->flat<float>().data(), sample_count_buffer);
			break;
		default:
			OP_REQUIRES(context, false,
				errors::Unimplemented("Only 1,2 and 4 Channel supported."));
		}
		//*/
		MYLOG("Gradient kernel done");
	}
private:
	bool m_globalSampling;
	Sampling::SamplerSettings m_samplingSettings;
	Sampling::CoordinateMode m_coordinateMode;
};


void SampleRGridLuTKernelLauncher(
		const void* input, const long long int* input_shape,
		int32_t numCameras,
		const float* lookup, uint8_t* mipAtlas,
		const Sampling::CoordinateMode coordinateMode,
		Sampling::SamplerSettings, const bool globalSampling, const bool relative, const bool normalized,
		void* output, const long long int* output_shape);
void SampleRGGridLuTKernelLauncher(
		const void* input, const long long int* input_shape,
		int32_t numCameras,
		const float* lookup, uint8_t* mipAtlas,
		const Sampling::CoordinateMode coordinateMode,
		Sampling::SamplerSettings, const bool globalSampling, const bool relative, const bool normalized,
		void* output, const long long int* output_shape);
void SampleRGBAGridLuTKernelLauncher(
		const void* input, const long long int* input_shape,
		int32_t numCameras,
		const float* lookup, uint8_t* mipAtlas,
		const Sampling::CoordinateMode coordinateMode,
		Sampling::SamplerSettings, const bool globalSampling, const bool relative, const bool normalized,
		void* output, const long long int* output_shape);

class SampleGridLuTOp : public OpKernel{
public:
	explicit SampleGridLuTOp(OpKernelConstruction *context) : OpKernel(context){
		
		memset(&m_samplingSettings, 0, sizeof(Sampling::SamplerSettings));
		std::string s_interpolation;
		OP_REQUIRES_OK(context, context->GetAttr("interpolation", &s_interpolation));
		if(s_interpolation.compare("LINEAR")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_LINEAR;
		else if(s_interpolation.compare("MIN")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_MIN;
		else if(s_interpolation.compare("MAX")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_MAX;
		
		std::string s_boundary;
		OP_REQUIRES_OK(context, context->GetAttr("boundary", &s_boundary));
		if(s_boundary.compare("CLAMP")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::CLAMP;
		if(s_boundary.compare("WRAP")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::WRAP;
		if(s_boundary.compare("MIRROR")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::MIRROR;
		
		std::string s_mipmapping;
		OP_REQUIRES_OK(context, context->GetAttr("mipmapping", &s_mipmapping));
		if(s_mipmapping.compare("NEAREST")==0) m_samplingSettings.mipMode = Sampling::SamplerSettings::MIPMODE_NEAREST;
		else if(s_mipmapping.compare("LINEAR")==0) m_samplingSettings.mipMode = Sampling::SamplerSettings::MIPMODE_LINEAR;
		
		OP_REQUIRES_OK(context, context->GetAttr("num_mipmaps", &m_samplingSettings.mipLevel));
		OP_REQUIRES(context, m_samplingSettings.mipLevel>0 || m_samplingSettings.mipMode==Sampling::SamplerSettings::MIPMODE_NONE,
			errors::InvalidArgument("when using mipmaps num_mipmaps must be larger than 0."));
			
		OP_REQUIRES_OK(context, context->GetAttr("mip_bias", &m_samplingSettings.mipBias));
		
		OP_REQUIRES_OK(context, context->GetAttr("separate_camera_batch", &m_globalSampling));
		OP_REQUIRES_OK(context, context->GetAttr("relative_coords", &m_relativeCoords));
		OP_REQUIRES_OK(context, context->GetAttr("normalized_coords", &m_normalizedCoords));
		
		std::string s_coordmode;
		OP_REQUIRES_OK(context, context->GetAttr("coordinate_mode", &s_coordmode));
		OP_REQUIRES(context, setCoordinateMode(m_coordinateMode, s_coordmode),
			errors::InvalidArgument("invalid coordinate_mode."));
		OP_REQUIRES_OK(context, context->GetAttr("cell_center_offset", &m_samplingSettings.cellCenterOffset));
		if(m_normalizedCoords){
			OP_REQUIRES(context, m_samplingSettings.cellCenterOffset==0.5f,
			errors::InvalidArgument("Invalid cell center offset must be 0.5 when using normalized coordinates."));
		}
	}
	
	void Compute(OpKernelContext *context) override{
		
		MYLOG("Sample transform op kernel start");
		
		const Tensor& input_grid = context->input(0);
		const Tensor& tensor_lookup = context->input(1);
		
		//check input
		MYLOG("Check input");
		TensorShape input_shape = input_grid.shape();
		OP_REQUIRES(context, input_grid.dims()==5 && input_shape.dim_size(4)<=4,
			errors::InvalidArgument("Invalid input shape (NDHWC):", input_shape.DebugString()));
		const int64 batch = input_shape.dim_size(0);
		const int64 channel = input_shape.dim_size(4);
		
		MYLOG("Check LuT");
		OP_REQUIRES(context, m_coordinateMode==Sampling::LuT,
			errors::InvalidArgument("Invalid coorindate_mode"));
		TensorShape lut_shape = tensor_lookup.shape();
		OP_REQUIRES(context, lut_shape.dims()==5 && lut_shape.dim_size(4)==4,
			errors::InvalidArgument("Invalid lut shape (NDHWC) with C=4:", lut_shape.DebugString()));
		int32_t numCameras = lut_shape.dim_size(0);
		
		//check output shape
		MYLOG("Create output shape");
		TensorShape output_shape;
		output_shape.AddDim(lut_shape.dim_size(1));
		output_shape.AddDim(lut_shape.dim_size(2));
		output_shape.AddDim(lut_shape.dim_size(3));
		output_shape.InsertDim(0,batch);
		output_shape.AddDim(channel);
		MYLOG("output_shape: " << output_shape.dim_size(0) << ", " << output_shape.dim_size(1) << ", " << output_shape.dim_size(2) << ", " << output_shape.dim_size(3));
		
		if(!m_globalSampling){
			OP_REQUIRES(context, numCameras==batch, errors::InvalidArgument("camera batch must match data batch when not using global sampling."));
			output_shape.InsertDim(1,1);
		}else{
			output_shape.InsertDim(1,numCameras);
		}
		
		
		
		//allocate outout
		MYLOG("Allocate output");
		//OP_REQUIRES(context, output_shape.IsValid(), errors::InvalidArgument("Broken output shape"));
		Tensor* output_grid = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_grid));
		MYLOG("Check allocated output\n");
		auto out = output_grid->flat<float>();
		MYLOG("Allocated output size: " << out.size() << " - " << output_grid->NumElements());
		
		//allocate temporary mip atlas
		uint8_t* mipAtlas = nullptr;
		Tensor mip_atlas;
		size_t mipAtlasSize = setupMipAtlas(input_shape, m_samplingSettings, sizeof(float)*channel, sizeof(float*));
		if(m_samplingSettings.mipMode!=Sampling::SamplerSettings::MIPMODE_NONE){
			MYLOG("Allocate mips");
			TensorShape mip_atlas_shape;
			mip_atlas_shape.AddDim(mipAtlasSize);
			OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT8, mip_atlas_shape, &mip_atlas));
			mipAtlas = mip_atlas.flat<uint8_t>().data();
		}
		
		
		//TODO handle arbitrary amount of channel
		// - move channel dimension outwards (to batch) and handle only 1-channel case internally. would also handle batches
		//   this would benefit from NCHW layout, otherwise have to transpose
		//   or just require NCHW as input format (or NHWC with up to 4 channel, the rest has to be packed in N) and let tensorflow/user handle the conversion...
		// - split into up-to-4-channel partitions. might be faster? but is harder to handle
		
		MYLOG("Resample");
		switch(channel){
		case 1:
			SampleRGridLuTKernelLauncher(input_grid.flat<float>().data(), input_shape.dim_sizes().data(),
					numCameras,
					tensor_lookup.flat<float>().data(), mipAtlas,
					m_coordinateMode,
					m_samplingSettings, m_globalSampling, m_relativeCoords, m_normalizedCoords,
					output_grid->flat<float>().data(), output_shape.dim_sizes().data());
			break;
		case 2:
			SampleRGGridLuTKernelLauncher(input_grid.flat<float>().data(), input_shape.dim_sizes().data(),
					numCameras,
					tensor_lookup.flat<float>().data(), mipAtlas,
					m_coordinateMode,
					m_samplingSettings, m_globalSampling, m_relativeCoords, m_normalizedCoords,
					output_grid->flat<float>().data(), output_shape.dim_sizes().data());
			break;
		case 4:
			SampleRGBAGridLuTKernelLauncher(input_grid.flat<float>().data(), input_shape.dim_sizes().data(),
					numCameras,
					tensor_lookup.flat<float>().data(), mipAtlas,
					m_coordinateMode,
					m_samplingSettings, m_globalSampling, m_relativeCoords, m_normalizedCoords,
					output_grid->flat<float>().data(), output_shape.dim_sizes().data());
			break;
		default:
			OP_REQUIRES(context, false,
				errors::Unimplemented("Only 1,2 and 4 Channel supported."));
		}
		
		MYLOG("Kernel done");
	}
private:
	bool m_globalSampling;
	bool m_relativeCoords;
	bool m_normalizedCoords;
	Sampling::SamplerSettings m_samplingSettings;
	Sampling::CoordinateMode m_coordinateMode;
};



void SampleRGridLuTGradKernelLauncher(const void* _input, const long long int* input_shape,
		int32_t numCameras,
		const float* _lookup,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling, const bool relative, const bool normalized,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, void* _lookup_grad, uint32_t* sample_count_buffer);
void SampleRGGridLuTGradKernelLauncher(const void* _input, const long long int* input_shape,
		int32_t numCameras,
		const float* _lookup,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling, const bool relative, const bool normalized,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, void* _lookup_grad, uint32_t* sample_count_buffer);
void SampleRGBAGridLuTGradKernelLauncher(const void* _input, const long long int* input_shape,
		int32_t numCameras,
		const float* _lookup,
		const Sampling::CoordinateMode coordinateMode,
		const Sampling::SamplerSettings samplingSettings, const bool globalSampling, const bool relative, const bool normalized,
		const void* _output_grad, const long long int* output_shape,
		void* _input_grad, void* _lookup_grad, uint32_t* sample_count_buffer);

class SampleGridLuTGradOp : public OpKernel{
public:
	explicit SampleGridLuTGradOp(OpKernelConstruction *context) : OpKernel(context){
		
		memset(&m_samplingSettings, 0, sizeof(Sampling::SamplerSettings));
		std::string s_interpolation;
		OP_REQUIRES_OK(context, context->GetAttr("interpolation", &s_interpolation));
		if(s_interpolation.compare("LINEAR")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_LINEAR;
		else if(s_interpolation.compare("MIN")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_MIN;
		else if(s_interpolation.compare("MAX")==0) m_samplingSettings.filterMode = Sampling::SamplerSettings::FILTERMODE_MAX;
		std::string s_boundary;
		OP_REQUIRES_OK(context, context->GetAttr("boundary", &s_boundary));
		if(s_boundary.compare("CLAMP")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::CLAMP;
		if(s_boundary.compare("WRAP")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::WRAP;
		if(s_boundary.compare("MIRROR")==0) m_samplingSettings.boundaryMode = Sampling::SamplerSettings::MIRROR;
		
		std::string s_mipmapping;
		OP_REQUIRES_OK(context, context->GetAttr("mipmapping", &s_mipmapping));
		if(s_mipmapping.compare("NEAREST")==0) m_samplingSettings.mipMode = Sampling::SamplerSettings::MIPMODE_NEAREST;
		else if(s_mipmapping.compare("LINEAR")==0) m_samplingSettings.mipMode = Sampling::SamplerSettings::MIPMODE_LINEAR;
		
		OP_REQUIRES_OK(context, context->GetAttr("num_mipmaps", &m_samplingSettings.mipLevel));
		OP_REQUIRES(context, m_samplingSettings.mipLevel>0 || m_samplingSettings.mipMode==Sampling::SamplerSettings::MIPMODE_NONE,
			errors::InvalidArgument("when using mipmaps num_mipmaps must be larger than 0."));
			
		OP_REQUIRES_OK(context, context->GetAttr("mip_bias", &m_samplingSettings.mipBias));
		
		OP_REQUIRES_OK(context, context->GetAttr("separate_camera_batch", &m_globalSampling));
		OP_REQUIRES_OK(context, context->GetAttr("relative_coords", &m_relativeCoords));
		OP_REQUIRES_OK(context, context->GetAttr("normalized_coords", &m_normalizedCoords));
		std::string s_coordmode;
		OP_REQUIRES_OK(context, context->GetAttr("coordinate_mode", &s_coordmode));
		OP_REQUIRES(context, setCoordinateMode(m_coordinateMode, s_coordmode),
			errors::InvalidArgument("invalid coordinate_mode."));
		OP_REQUIRES_OK(context, context->GetAttr("cell_center_offset", &m_samplingSettings.cellCenterOffset));
		if(m_normalizedCoords){
			OP_REQUIRES(context, m_samplingSettings.cellCenterOffset==0.5f,
			errors::InvalidArgument("Invalid cell center offset must be 0.5 when using normalized coordinates."));
		}
	}
	
	void Compute(OpKernelContext *context) override{
		
		MYLOG("Gradient kernel start");
		
		const Tensor& input_grid = context->input(0);
		const Tensor& output_grad_grid = context->input(1);
		const Tensor& tensor_lookup = context->input(2);
		
		//check input
		MYLOG("Check input");
		TensorShape input_shape = input_grid.shape();
		OP_REQUIRES(context, input_grid.dims()==5 && input_shape.dim_size(4)<=4,
			errors::InvalidArgument("Invalid input shape (NDHWC):", input_shape.DebugString()));
		const int64 batch = input_shape.dim_size(0);
		const int64 channel = input_shape.dim_size(4);
		
		//check output gradients
		MYLOG("Check output_grads");
		TensorShape output_shape = output_grad_grid.shape();
		OP_REQUIRES(context, output_grad_grid.dims()==6 && output_shape.dim_size(5)<=4,
			errors::InvalidArgument("Invalid output_grads shape (NVDHWC):", output_shape.DebugString()));
		OP_REQUIRES(context, output_shape.dim_size(0)==batch,
			errors::InvalidArgument("output_grads batch size does not match input batch size."));
		
		
		//check transform matricies
		int32_t numCameras=0;
			MYLOG("Check LuT");
			TensorShape lut_shape = tensor_lookup.shape();
			OP_REQUIRES(context, lut_shape.dims()==5 && lut_shape.dim_size(4)==4
				&& lut_shape.dim_size(1)==output_shape.dim_size(2)
				&& lut_shape.dim_size(2)==output_shape.dim_size(3)
				&& lut_shape.dim_size(3)==output_shape.dim_size(4),
				errors::InvalidArgument("Invalid lut shape (VDHWC):", lut_shape.DebugString(), ", DHW must macht output shape:", output_shape.DebugString()));
			numCameras = lut_shape.dim_size(0);
		//}
		if(!m_globalSampling){
			OP_REQUIRES(context, numCameras==batch, errors::InvalidArgument("camera batch must match data batch when not using global sampling."));
			
			OP_REQUIRES(context, output_shape.dim_size(1)==1,
				errors::InvalidArgument("output_grads views size must be 1 if not using global sampling."));
		}else{
			OP_REQUIRES(context, output_shape.dim_size(1)==numCameras,
				errors::InvalidArgument("output_grads views size does not match cameras."));
		}
		
		uint32_t* sample_count_buffer = nullptr;
		Tensor sample_count_tensor;
		if(NORMALIZE_GRADIENTS){
			MYLOG("Allocate gradient counter");
			TensorShape sample_count_tensor_shape;
			sample_count_tensor_shape.AddDim(input_shape.dim_size(1));
			sample_count_tensor_shape.AddDim(input_shape.dim_size(2));
			sample_count_tensor_shape.AddDim(input_shape.dim_size(3));
			OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT32, sample_count_tensor_shape, &sample_count_tensor));
			sample_count_buffer = sample_count_tensor.flat<uint32_t>().data();
		}
		
		//allocate outout
		MYLOG("Allocate input gradients");
		//OP_REQUIRES(context, output_shape.IsValid(), errors::InvalidArgument("Broken output shape"));
		Tensor* input_grads = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &input_grads));
		
		Tensor* lookup_grads = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(1, lut_shape, &lookup_grads));
		
		
		MYLOG("Resample\n");
		switch(channel){
		case 1:
			SampleRGridLuTGradKernelLauncher(input_grid.flat<float>().data(), input_shape.dim_sizes().data(),
				numCameras,
				tensor_lookup.flat<float>().data(),
				m_coordinateMode,
				m_samplingSettings, m_globalSampling, m_relativeCoords, m_normalizedCoords,
				output_grad_grid.flat<float>().data(), output_shape.dim_sizes().data(),
				input_grads->flat<float>().data(), lookup_grads->flat<float>().data(), sample_count_buffer);
			break;
		case 2:
			SampleRGGridLuTGradKernelLauncher(input_grid.flat<float>().data(), input_shape.dim_sizes().data(),
				numCameras,
				tensor_lookup.flat<float>().data(),
				m_coordinateMode,
				m_samplingSettings, m_globalSampling, m_relativeCoords, m_normalizedCoords,
				output_grad_grid.flat<float>().data(), output_shape.dim_sizes().data(),
				input_grads->flat<float>().data(), lookup_grads->flat<float>().data(), sample_count_buffer);
			break;
		case 4:
			SampleRGBAGridLuTGradKernelLauncher(input_grid.flat<float>().data(), input_shape.dim_sizes().data(),
				numCameras,
				tensor_lookup.flat<float>().data(),
				m_coordinateMode,
				m_samplingSettings, m_globalSampling, m_relativeCoords, m_normalizedCoords,
				output_grad_grid.flat<float>().data(), output_shape.dim_sizes().data(),
				input_grads->flat<float>().data(), lookup_grads->flat<float>().data(), sample_count_buffer);
			break;
		default:
			OP_REQUIRES(context, false,
				errors::Unimplemented("Only 1,2 and 4 Channel supported."));
		}
		//*/
		MYLOG("Gradient kernel done");
	}
private:
	bool m_globalSampling;
	bool m_relativeCoords;
	bool m_normalizedCoords;
	Sampling::SamplerSettings m_samplingSettings;
	Sampling::CoordinateMode m_coordinateMode;
};

void ComputeLoDKernelLauncher(const long long int* input_shape,
	const float* MV, const float* P, const float* _frustum,
	const Sampling::CoordinateMode coordinateMode,
	void* output, const long long int* output_shape);

class LoDTransformOp : public OpKernel{
public:
	explicit LoDTransformOp(OpKernelConstruction *context) : OpKernel(context){
		
		OP_REQUIRES_OK(context, context->GetAttr("inverse_transform", &m_inverseTransform));
	}
	
	void Compute(OpKernelContext *context) override{
		
		MYLOG("Compute LoD transform op kernel start");
		
		const Tensor& input_shape_tensor = context->input(0);
		const Tensor& tensor_MV = context->input(1);
		const Tensor& tensor_P = context->input(2);
		const Tensor& frustum = context->input(3);
		const Tensor& output_shape_tensor = context->input(4);
		
		const int64 channel = 4;
		
		//check transform matrics
		MYLOG("Check transform");
		int32 numMatrix_MV=0, numMatrix_P=0, numFrustum=0;
		OP_REQUIRES(context, isValidTransformMatrix(tensor_MV.shape(), numMatrix_MV),
			errors::InvalidArgument("transformation matrix_mv must be a 4x4 square matrix or a batch of 4x4 square matrices:", tensor_MV.shape().DebugString()));
		OP_REQUIRES(context, isValidTransformMatrix(tensor_P.shape(), numMatrix_P),
			errors::InvalidArgument("transformation matrix_p must be a 4x4 square matrix or a batch of 4x4 square matrices:", tensor_P.shape().DebugString()));
		OP_REQUIRES(context, isValidFrustumParams(frustum.shape(), numFrustum),
			errors::InvalidArgument("frustum must be a 1D tensor of 6 elements or a batch thereof:", frustum.shape().DebugString()));
		//TODO
		OP_REQUIRES(context, numMatrix_MV==numMatrix_P && numMatrix_MV==numFrustum, errors::InvalidArgument("camera batch size mismatch"));
		const int64 batch = numMatrix_MV;
		
		//check input shape
		MYLOG("Check input shape tensor");
		OP_REQUIRES(context, input_shape_tensor.dims()==1 && input_shape_tensor.dim_size(0)==3, errors::InvalidArgument("Invalid input_shape"));
		MYLOG("Create input shape");
		TensorShape input_shape;
		OP_REQUIRES_OK(context, makeShapeFromTensor(input_shape_tensor, &input_shape));
		MYLOG("Check input shape");
		OP_REQUIRES(context, input_shape.dims()==3,
			errors::InvalidArgument("Invalid input_shape"));
		input_shape.InsertDim(0,batch);
		input_shape.AddDim(channel);
		MYLOG("input_shape: " << input_shape.dim_size(0) << ", " << input_shape.dim_size(1) << ", " << input_shape.dim_size(2) << ", " << input_shape.dim_size(3));
		
		//check output shape
		MYLOG("Check output shape tensor");
		OP_REQUIRES(context, output_shape_tensor.dims()==1 && output_shape_tensor.dim_size(0)==3, errors::InvalidArgument("Invalid output_shape"));
		MYLOG("Create output shape");
		TensorShape output_shape;
		OP_REQUIRES_OK(context, makeShapeFromTensor(output_shape_tensor, &output_shape));
		MYLOG("Check output shape");
		OP_REQUIRES(context, output_shape.dims()==3,
			errors::InvalidArgument("Invalid output_shape"));
		output_shape.InsertDim(0,batch);
		output_shape.AddDim(channel);
		MYLOG("output_shape: " << output_shape.dim_size(0) << ", " << output_shape.dim_size(1) << ", " << output_shape.dim_size(2) << ", " << output_shape.dim_size(3));
		
		
		//allocate outout
		MYLOG("Allocate output");
		//OP_REQUIRES(context, output_shape.IsValid(), errors::InvalidArgument("Broken output shape"));
		Tensor* output_grid = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_grid));
		MYLOG("Check allocated output\n");
		auto out = output_grid->flat<float>();
		MYLOG("Allocated output size: " << out.size() << " - " << output_grid->NumElements());
		
		MYLOG("Resample");
		ComputeLoDKernelLauncher(input_shape.dim_sizes().data(),
			tensor_MV.flat<float>().data(), tensor_P.flat<float>().data(), frustum.flat<float>().data(),
			m_inverseTransform? Sampling::TransformLinDepthReverse : Sampling::TransformLinDepth,
			output_grid->flat<float>().data(), output_shape.dim_sizes().data());
		
		MYLOG("Kernel done");
	}
private:
	bool m_inverseTransform;
};
#define REGISTER__CPU(T)

#undef REGISTER__CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
	REGISTER_KERNEL_BUILDER(Name("SampleGridTransform") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<T>("T") \
		.HostMemory("matrix_m") \
		.HostMemory("matrix_v") \
		.HostMemory("matrix_p") \
		.HostMemory("frustum_params") \
		.HostMemory("output_shape") \
		, SampleGridTransformOp<GPUDevice, T>);
REGISTER_GPU(float);
#undef REGISTER_GPU

REGISTER_KERNEL_BUILDER(Name("SampleGridTransformGrad") \
	.Device(DEVICE_GPU) \
	.HostMemory("matrix_m") \
	.HostMemory("matrix_v") \
	.HostMemory("matrix_p") \
	.HostMemory("frustum_params") \
	, SampleGridTransformGradOp);

REGISTER_KERNEL_BUILDER(Name("SampleGridLut") \
	.Device(DEVICE_GPU) \
	, SampleGridLuTOp);

REGISTER_KERNEL_BUILDER(Name("SampleGridLutGrad") \
	.Device(DEVICE_GPU) \
	, SampleGridLuTGradOp);

REGISTER_KERNEL_BUILDER(Name("LodTransform") \
	.Device(DEVICE_GPU) \
	.HostMemory("input_shape") \
	.HostMemory("matrix_mv") \
	.HostMemory("matrix_p") \
	.HostMemory("output_shape") \
	, LoDTransformOp);

#endif //GOOGLE_CUDA