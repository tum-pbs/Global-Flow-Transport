/*
* Fused version of resample_grid and reduce_blend, no memory needed for intermediate frustum-grid
* no mip-mapping
*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <iostream>
#include <string>
//#define LOGGING //uncomment to enable logging

#include "raymarch_grid.hpp"

#ifdef LOGGING
#define MYLOG(msg) std::cout << msg << std::endl
#define LOG_PRINTF(msg) printf(msg)
#else
#define MYLOG(msg)
#define LOG_PRINTF(msg)
#endif

using namespace tensorflow;


// Sample at transformed 3D grid positions from an input grid
REGISTER_OP("RaymarchGridTransform")
	.Attr("T: {float}")
	.Input("input: T") // NDHWC
	.Input("matrix_m: float32") // 4x4 matrix or batch N thereof
	.Input("matrix_v: float32") // 4x4 matrix or batch V thereof
	.Input("matrix_p: float32") // 4x4 matrix or batch V thereof
	.Input("frustum_params: float32") // 6 elemnts or batch V thereof
	.Input("output_shape: int32") // DHW
	.Attr("interpolation: {'NEAREST', 'LINEAR', 'MIN', 'MAX'} = 'LINEAR'")
	.Attr("boundary: {'BORDER', 'CLAMP', 'WRAP'} = 'BORDER'")
	.Attr("blending_mode: {'BEER_LAMBERT', 'ALPHA', 'ALPHA_ADDITIVE', 'ADDITIVE'} = 'BEER_LAMBERT'")
	.Attr("keep_dims: bool = false")
	.Attr("separate_camera_batch: bool = true")
	.Output("output: T") // NVDHWC
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
	{
		::tensorflow::shape_inference::ShapeHandle channel;
		TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 4, &channel));
		::tensorflow::shape_inference::ShapeHandle outShape;
		TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &outShape));
		TF_RETURN_IF_ERROR(c->Subshape(outShape, 0, 3, &outShape));
		TF_RETURN_IF_ERROR(c->Concatenate(outShape, channel, &outShape));
		c->set_output(0, outShape);
		return Status::OK();
	});


// the gradient op
REGISTER_OP("RaymarchGridTransformGrad")
	.Attr("T: {float}")
	.Input("input: T") // NDHWC
	.Input("output: T") // NVHWC
	.Input("output_grad: T") // NVHWC
	.Input("matrix_m: float32") // 4x4 matrix or batch N thereof
	.Input("matrix_v: float32") // 4x4 matrix or batch V thereof
	.Input("matrix_p: float32") // 4x4 matrix or batch V thereof
	.Input("frustum_params: float32")
	.Input("output_shape: int32") // DHW
	.Attr("interpolation: {'LINEAR'} = 'LINEAR'") //TODO 'NEAREST', 'MIN', 'MAX'
	.Attr("boundary: {'BORDER', 'CLAMP', 'WRAP'} = 'BORDER'")
	.Attr("blending_mode: {'BEER_LAMBERT', 'ALPHA', 'ALPHA_ADDITIVE', 'ADDITIVE'} = 'BEER_LAMBERT'")
	.Attr("keep_dims: bool = false")
	.Attr("separate_camera_batch: bool = true")
	.Output("input_grad: T")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
	{
		c->set_output(0, c->input(0));
		return Status::OK();
	});

//TODO: this fused version could support raymarching from a 2D pixel grid of starting positions, directions and step size



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

bool parseFilterMode(OpKernelConstruction *context, const std::string& filterModeStr, Sampling::FilterMode& filterMode){
	if(filterModeStr.compare("NEAREST")==0) filterMode = Sampling::FILTERMODE_NEAREST;
	else if(filterModeStr.compare("LINEAR")==0) filterMode = Sampling::FILTERMODE_LINEAR;
	else if(filterModeStr.compare("MIN")==0) filterMode = Sampling::FILTERMODE_MIN;
	else if(filterModeStr.compare("MAX")==0) filterMode = Sampling::FILTERMODE_MAX;
	else return false;
	return true;
}

bool parseBoundaryMode(OpKernelConstruction *context, const std::string& boundaryModeStr, Sampling::BoundaryMode& boundaryMode){
	if(boundaryModeStr.compare("BORDER")==0) boundaryMode = Sampling::BOUNDARY_BORDER;
	else if(boundaryModeStr.compare("CLAMP")==0) boundaryMode = Sampling::BOUNDARY_CLAMP;
	else if(boundaryModeStr.compare("WRAP")==0) boundaryMode = Sampling::BOUNDARY_WRAP;
	//else if(boundaryModeStr.compare("MIRROR")==0) boundaryMode = Sampling::BOUNDARY_MIRROR;
	else return false;
	return true;
}

bool parseBlendMode(OpKernelConstruction *context, const std::string& blendModeStr, Blending::BlendMode& blendMode){
	if(blendModeStr.compare("BEER_LAMBERT")==0) blendMode = Blending::BLEND_BEERLAMBERT;
	else if(blendModeStr.compare("ALPHA")==0) blendMode = Blending::BLEND_ALPHA;
	else if(blendModeStr.compare("ALPHA_ADDITIVE")==0) blendMode = Blending::BLEND_ALPHAADDITIVE;
	else if(blendModeStr.compare("ADDITIVE")==0) blendMode = Blending::BLEND_ADDITIVE;
	else return false;
	return true;
}

#if GOOGLE_CUDA
#define DECLARE_GPU_SPEC(T, C) \
	template<> \
	void RaymarchGridKernel<GPUDevice, T, C>::operator()( \
		const GPUDevice& d, \
		const T* input, const long long int* input_shape, \
		const float* M, const float* V, const float* P, const float* frustum, int32_t numCameras, \
		const Sampling::FilterMode filterMode, const Sampling::BoundaryMode boundaryMode, \
		const Blending::BlendMode blendMode, const bool globalSampling, \
		T* output, const long long int* output_shape \
		); \
	extern template struct RaymarchGridKernel<GPUDevice, T, C>;
DECLARE_GPU_SPEC(float, 1)
DECLARE_GPU_SPEC(float, 2)
DECLARE_GPU_SPEC(float, 4)
#undef DECLARE_GPU_SPEC
#endif

//only for 3D grids with up to 4 channel
//no support for batches yet?
template<typename Device, typename T>
class RaymarchGridTransformOp : public OpKernel{
public:
	explicit RaymarchGridTransformOp(OpKernelConstruction *context) : OpKernel(context){
		
		std::string s_interpolation;
		OP_REQUIRES_OK(context, context->GetAttr("interpolation", &s_interpolation));
		OP_REQUIRES(context, parseFilterMode(context, s_interpolation, m_filterMode), errors::InvalidArgument("invalid filter mode"));
		
		std::string s_boundary;
		OP_REQUIRES_OK(context, context->GetAttr("boundary", &s_boundary));
		OP_REQUIRES(context, parseBoundaryMode(context, s_boundary, m_boundaryMode), errors::InvalidArgument("invalid boundary mode"));
		
		OP_REQUIRES_OK(context, context->GetAttr("separate_camera_batch", &m_globalSampling));
		
		std::string blendingMode;
		OP_REQUIRES_OK(context, context->GetAttr("blending_mode", &blendingMode));
		OP_REQUIRES(context, parseBlendMode(context, blendingMode, m_blendMode), errors::InvalidArgument("invalid blend mode"));
		
		OP_REQUIRES_OK(context, context->GetAttr("keep_dims", &m_keepDims));
	}
	
	void Compute(OpKernelContext *context) override{
		
		MYLOG("RaymarchGridTransformOp kernel start");
		
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
		const int64 batch = input_shape.dim_size(0);
		const int64 channel = input_shape.dim_size(4);
		
		//check output shape
		MYLOG("Check output shape tensor");
		OP_REQUIRES(context, output_shape_tensor.dims()==1 && output_shape_tensor.dim_size(0)==3, errors::InvalidArgument("Invalid output_shape"));
		MYLOG("Create output shape");
		TensorShape output_shape;
		OP_REQUIRES_OK(context, makeShapeFromTensor(output_shape_tensor, &output_shape));
		MYLOG("Check output shape");
		OP_REQUIRES(context, output_shape.dims()==3,
			errors::InvalidArgument("Invalid output_shape"));
		//const int64 depthSamples = output_shape.dim_size(0);
		output_shape.InsertDim(0,batch);
		output_shape.AddDim(channel);
		MYLOG("output_shape: " << output_shape.dim_size(0) << ", " << output_shape.dim_size(1) << ", " << output_shape.dim_size(2) << ", " << output_shape.dim_size(3));
		
		//check transform matrics
		int32_t numCameras=0;
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
		if(!m_keepDims) output_shape.set_dim(0,1);
		TensorShape output_tensor_shape = output_shape;
		if(m_keepDims){ //return NVDHWC with D=1
			output_tensor_shape.set_dim(2,1);
		}else{ //return NVHWC
			output_tensor_shape.RemoveDim(2);
		}
		OP_REQUIRES_OK(context, context->allocate_output(0, output_tensor_shape, &output_grid));
		MYLOG("Check allocated output\n");
		
		
		
		MYLOG("Resample");
		switch(channel){
			case 1:
				RaymarchGridKernel<Device, T, 1>()(context->eigen_device<Device>(),
					input_grid.flat<T>().data(), input_shape.dim_sizes().data(),
					tensor_M.flat<float>().data(), tensor_V.flat<float>().data(), tensor_P.flat<float>().data(), frustum.flat<float>().data(), numCameras,
					m_filterMode, m_boundaryMode, m_blendMode, m_globalSampling,
					output_grid->flat<T>().data(), output_shape.dim_sizes().data());
				break;
			case 2:
				RaymarchGridKernel<Device, T, 2>()(context->eigen_device<Device>(),
					input_grid.flat<T>().data(), input_shape.dim_sizes().data(),
					tensor_M.flat<float>().data(), tensor_V.flat<float>().data(), tensor_P.flat<float>().data(), frustum.flat<float>().data(), numCameras,
					m_filterMode, m_boundaryMode, m_blendMode, m_globalSampling,
					output_grid->flat<T>().data(), output_shape.dim_sizes().data());
				break;
			case 4:
				RaymarchGridKernel<Device, T, 4>()(context->eigen_device<Device>(),
					input_grid.flat<T>().data(), input_shape.dim_sizes().data(),
					tensor_M.flat<float>().data(), tensor_V.flat<float>().data(), tensor_P.flat<float>().data(), frustum.flat<float>().data(), numCameras,
					m_filterMode, m_boundaryMode, m_blendMode, m_globalSampling,
					output_grid->flat<T>().data(), output_shape.dim_sizes().data());
				break;
			default:
				OP_REQUIRES(context, false,
					errors::Unimplemented("Only 1,2 and 4 Channel supported."));
		}
		
		MYLOG("RaymarchGridTransformOp kernel done");
	}
private:
	bool m_globalSampling;
	bool m_keepDims;
	Sampling::FilterMode m_filterMode;
	Sampling::BoundaryMode m_boundaryMode;
	Blending::BlendMode m_blendMode;
};


#if GOOGLE_CUDA
#define DECLARE_GPU_SPEC(T, C) \
	template<> \
	void RaymarchGridGradKernel<GPUDevice, T, C>::operator()( \
		const GPUDevice& d, \
		const T* input, T* inputGrads, T* sampleBuffer, sampleCount_t* sampleCounter, const long long int* input_shape, \
		const float* M, const float* V, const float* P, const float* frustum, int32_t numCameras, \
		const Sampling::FilterMode filterMode, const Sampling::BoundaryMode boundaryMode, \
		const Blending::BlendMode blendMode, const bool globalSampling, \
		const T* output, const T* outputGrads, const long long int* output_shape \
		); \
	extern template struct RaymarchGridGradKernel<GPUDevice, T, C>;
DECLARE_GPU_SPEC(float, 1)
DECLARE_GPU_SPEC(float, 2)
DECLARE_GPU_SPEC(float, 4)
#undef DECLARE_GPU_SPEC
#endif

template<typename Device, typename T>
class RaymarchGridTransformGradOp : public OpKernel{
public:
	explicit RaymarchGridTransformGradOp(OpKernelConstruction *context) : OpKernel(context){
		
		std::string s_interpolation;
		OP_REQUIRES_OK(context, context->GetAttr("interpolation", &s_interpolation));
		OP_REQUIRES(context, parseFilterMode(context, s_interpolation, m_filterMode), errors::InvalidArgument("invalid filter mode"));
		
		std::string s_boundary;
		OP_REQUIRES_OK(context, context->GetAttr("boundary", &s_boundary));
		OP_REQUIRES(context, parseBoundaryMode(context, s_boundary, m_boundaryMode), errors::InvalidArgument("invalid boundary mode"));
		
		OP_REQUIRES_OK(context, context->GetAttr("separate_camera_batch", &m_globalSampling));
		
		std::string blendingMode;
		OP_REQUIRES_OK(context, context->GetAttr("blending_mode", &blendingMode));
		OP_REQUIRES(context, parseBlendMode(context, blendingMode, m_blendMode), errors::InvalidArgument("invalid blend mode"));
		
		OP_REQUIRES_OK(context, context->GetAttr("keep_dims", &m_keepDims));
	}
	
	void Compute(OpKernelContext *context) override{
		
		MYLOG("RaymarchGridTransformGradOp kernel start");
		
		const Tensor& input_grid = context->input(0);
		const Tensor& output_grid = context->input(1);
		const Tensor& output_grad_grid = context->input(2);
		const Tensor& tensor_M = context->input(3);
		const Tensor& tensor_V = context->input(4);
		const Tensor& tensor_P = context->input(5);
		const Tensor& frustum = context->input(6);
		//still needed for number of samples in depth
		const Tensor& output_shape_tensor = context->input(7);
		
		//check input
		MYLOG("Check input");
		TensorShape input_shape = input_grid.shape();
		OP_REQUIRES(context, input_grid.dims()==5 && input_shape.dim_size(4)<=4,
			errors::InvalidArgument("Invalid input shape (NDHWC):", input_shape.DebugString()));
		const int64 batch = input_shape.dim_size(0);
		const int64 channel = input_shape.dim_size(4);
		
		//check output gradients
		MYLOG("Check output_grads");
		TensorShape output_shape = output_grid.shape();
		OP_REQUIRES(context, output_shape==output_grad_grid.shape(),
			errors::InvalidArgument("Shapes of output grid and output gradients must match:", output_shape.DebugString(), output_grad_grid.shape().DebugString()));
		
		MYLOG("output_shape: " << output_shape.dim_size(0) << ", " << output_shape.dim_size(1) << ", " << output_shape.dim_size(2) << ", " << output_shape.dim_size(3));
		if(m_keepDims){
			OP_REQUIRES(context, output_grad_grid.dims()==6 && output_shape.dim_size(5)<=4,
				errors::InvalidArgument("Invalid output_grads shape (NVDHWC):", output_shape.DebugString()));
		}else{
			OP_REQUIRES(context, output_grad_grid.dims()==5 && output_shape.dim_size(4)<=4,
				errors::InvalidArgument("Invalid output_grads shape (NVHWC):", output_shape.DebugString()));
			output_shape.InsertDim(2,1); //make NVDHWC for internal use
		}
		OP_REQUIRES(context, output_shape.dim_size(0)==batch,
			errors::InvalidArgument("output_grads batch size does not match input batch size."));
		
		
		
		//check transform matrics
		int32_t numCameras=0;
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
				errors::InvalidArgument("output_grads view dimension must be 1 if not using global sampling."));
		}else{
			OP_REQUIRES(context, output_shape.dim_size(1)==numCameras,
				errors::InvalidArgument("output_grads views dimension does not match number of cameras."));
		}
		
		TensorShape orig_output_shape; //NVDHWC matching output, but D is the number of depth samples from the original output_shape
		OP_REQUIRES_OK(context, makeShapeFromTensor(output_shape_tensor, &orig_output_shape));
		MYLOG("Check output shape");
		OP_REQUIRES(context, orig_output_shape.dims()==3 && output_shape.dim_size(3)==orig_output_shape.dim_size(1) && output_shape.dim_size(4)==orig_output_shape.dim_size(2),
			errors::InvalidArgument("output_shape must be rank 3 (DHW) with H and W matching those of output (NVDHWC):", orig_output_shape.DebugString(), output_shape.DebugString()));
		//const int64 depthSamples = output_shape.dim_size(0);
		orig_output_shape.InsertDim(0,batch);
		orig_output_shape.InsertDim(1, output_shape.dim_size(1));
		orig_output_shape.AddDim(channel);
		
		sampleCount_t* sample_count_buffer = nullptr;
		T* sample_buffer = nullptr;
		Tensor sample_count_tensor;
		Tensor sample_buffer_tensor;
		if(NORMALIZE_GRADIENTS!=NORMALIZE_GRADIENT_NONE){
			MYLOG("Allocate gradient counter");
			TensorShape sample_count_tensor_shape;
			sample_count_tensor_shape.AddDim(input_shape.dim_size(1));
			sample_count_tensor_shape.AddDim(input_shape.dim_size(2));
			sample_count_tensor_shape.AddDim(input_shape.dim_size(3));
			OP_REQUIRES_OK(context, context->allocate_temp(TFsampleCount_t, sample_count_tensor_shape, &sample_count_tensor));
			sample_count_buffer = sample_count_tensor.flat<sampleCount_t>().data();
			if(numCameras>1 && m_globalSampling){
				sample_count_tensor_shape.AddDim(channel);
				sample_count_tensor_shape.AddDim(sizeof(T));
				OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, sample_count_tensor_shape, &sample_buffer_tensor));
				sample_buffer = sample_buffer_tensor.flat<T>().data();
			}
		}
		
		//allocate outout
		MYLOG("Allocate input gradients");
		Tensor* input_grads = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &input_grads));
		
		
		MYLOG("Resample\n");
		switch(channel){
			case 1:
				RaymarchGridGradKernel<Device, T, 1>()(context->eigen_device<Device>(),
					input_grid.flat<T>().data(), input_grads->flat<T>().data(), sample_buffer, sample_count_buffer, input_shape.dim_sizes().data(),
					tensor_M.flat<float>().data(), tensor_V.flat<float>().data(), tensor_P.flat<float>().data(), frustum.flat<float>().data(), numCameras,
					m_filterMode, m_boundaryMode, m_blendMode, m_globalSampling,
					output_grid.flat<T>().data(), output_grad_grid.flat<T>().data(), orig_output_shape.dim_sizes().data());
				break;
			case 2:
				RaymarchGridGradKernel<Device, T, 2>()(context->eigen_device<Device>(),
					input_grid.flat<T>().data(), input_grads->flat<T>().data(), sample_buffer, sample_count_buffer, input_shape.dim_sizes().data(),
					tensor_M.flat<float>().data(), tensor_V.flat<float>().data(), tensor_P.flat<float>().data(), frustum.flat<float>().data(), numCameras,
					m_filterMode, m_boundaryMode, m_blendMode, m_globalSampling,
					output_grid.flat<T>().data(), output_grad_grid.flat<T>().data(), orig_output_shape.dim_sizes().data());
				break;
			case 4:
				RaymarchGridGradKernel<Device, T, 4>()(context->eigen_device<Device>(),
					input_grid.flat<T>().data(), input_grads->flat<T>().data(), sample_buffer, sample_count_buffer, input_shape.dim_sizes().data(),
					tensor_M.flat<float>().data(), tensor_V.flat<float>().data(), tensor_P.flat<float>().data(), frustum.flat<float>().data(), numCameras,
					m_filterMode, m_boundaryMode, m_blendMode, m_globalSampling,
					output_grid.flat<T>().data(), output_grad_grid.flat<T>().data(), orig_output_shape.dim_sizes().data());
				break;
			default:
				OP_REQUIRES(context, false,
					errors::Unimplemented("Only 1,2 and 4 Channel supported."));
		}
		//*/
		MYLOG("RaymarchGridTransformGradOp kernel done");
	}
private:
	bool m_globalSampling;
	bool m_keepDims;
	Sampling::FilterMode m_filterMode;
	Sampling::BoundaryMode m_boundaryMode;
	Blending::BlendMode m_blendMode;
};


#define REGISTER__CPU(T)

#undef REGISTER__CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
	REGISTER_KERNEL_BUILDER(Name("RaymarchGridTransform") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<T>("T") \
		.HostMemory("matrix_m") \
		.HostMemory("matrix_v") \
		.HostMemory("matrix_p") \
		.HostMemory("frustum_params") \
		.HostMemory("output_shape") \
		, RaymarchGridTransformOp<GPUDevice, T>);
REGISTER_GPU(float);
#undef REGISTER_GPU

#define REGISTER_GPU(T) \
	REGISTER_KERNEL_BUILDER(Name("RaymarchGridTransformGrad") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<T>("T") \
		.HostMemory("matrix_m") \
		.HostMemory("matrix_v") \
		.HostMemory("matrix_p") \
		.HostMemory("frustum_params") \
		.HostMemory("output_shape") \
		, RaymarchGridTransformGradOp<GPUDevice, T>);
REGISTER_GPU(float);
#undef REGISTER_GPU

#endif //GOOGLE_CUDA