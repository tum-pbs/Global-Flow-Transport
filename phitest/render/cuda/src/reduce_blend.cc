#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <iostream>
#include <string>
//#define LOGGING

#include "blending_settings.hpp"
#include "render_errors.hpp"

#ifdef LOGGING
#define MYLOG(msg) std::cout << msg << std::endl
#define LOG_PRINTF(msg) printf(msg)
#else
#define MYLOG(msg)
#define LOG_PRINTF(msg)
#endif

using namespace tensorflow;


REGISTER_OP("ReduceGridBlend")
	.Input("input: float32")
	.Attr("blending_mode: {'BEER_LAMBERT', 'ALPHA', 'ALPHA_ADDITIVE', 'ADDITIVE'} = 'BEER_LAMBERT'")
	.Attr("keep_dims: bool = false")
	.Output("output: float32")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
	{
		::tensorflow::shape_inference::ShapeHandle flatDim;
		TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 1, 4, &flatDim));
		c->set_output(0, flatDim);
		return Status::OK();
	});

// the gradient op
REGISTER_OP("ReduceGridBlendGrad")
	.Input("output_grad: float32")
	.Input("output: float32")
	.Input("input: float32")
	.Attr("blending_mode: {'BEER_LAMBERT', 'ALPHA', 'ADDITIVE'} = 'BEER_LAMBERT'") //, 'ALPHA_ADDITIVE'
	.Attr("keep_dims: bool = false")
	.Output("input_grad: float32")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
	{
		c->set_output(0, c->input(2));
		return Status::OK();
	});
	
void GridBlendRKernelLauncher(const float* _input, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		float* _output, const long long int* output_shape);
void GridBlendRGKernelLauncher(const float* _input, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		float* _output, const long long int* output_shape);
void GridBlendRGBAKernelLauncher(const float* _input, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		float* _output, const long long int* output_shape);

class ReduceGridBlendOp : public OpKernel{
public:
	explicit ReduceGridBlendOp(OpKernelConstruction *context) : OpKernel(context){
		std::string blendingMode;
		OP_REQUIRES_OK(context, context->GetAttr("blending_mode", &blendingMode));
		if(blendingMode.compare("BEER_LAMBERT")==0) m_blendMode=Blending::BLEND_BEERLAMBERT;
		else if(blendingMode.compare("ALPHA")==0) m_blendMode=Blending::BLEND_ALPHA;
		else if(blendingMode.compare("ALPHA_ADDITIVE")==0) m_blendMode=Blending::BLEND_ALPHAADDITIVE;
		else if(blendingMode.compare("ADDITIVE")==0) m_blendMode=Blending::BLEND_ADDITIVE;
		else {OP_REQUIRES(context, false, errors::InvalidArgument("invalid blending mode"));}
		
		OP_REQUIRES_OK(context, context->GetAttr("keep_dims", &m_keepDims));
	}
	
	void Compute(OpKernelContext *context) override{
		
		MYLOG("reduce blend op kernel start");
		
		const Tensor& input_grid = context->input(0);
		
		MYLOG("Check input");
		TensorShape input_shape = input_grid.shape();
		OP_REQUIRES(context, (input_grid.dims()==5 && input_shape.dim_size(4)<=4),
			errors::InvalidArgument("Invalid input shape", input_shape.DebugString()));
		const int64 batch = input_shape.dim_size(0);
		const int64 channel = input_shape.dim_size(4);
		
		MYLOG("Create output shape");
		TensorShape output_shape;
		output_shape.AddDim(batch);
		if(m_keepDims) output_shape.AddDim(input_shape.dim_size(1));
		output_shape.AddDim(input_shape.dim_size(2));
		output_shape.AddDim(input_shape.dim_size(3));
		output_shape.AddDim(channel);
		
		
		MYLOG("Allocate output");
		Tensor* output_grid = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_grid));
		MYLOG("Check allocated output");
	//	MYLOG("Allocated output size: " << out.size() << " - " << output_grid->NumElements());
		if(!m_keepDims) output_shape.InsertDim(1,1); //fill z dim, NDHWC needed for correct kernel dimensions handling
		MYLOG("output_shape: " << output_shape.dim_size(0) << ", " << output_shape.dim_size(1) << ", " << output_shape.dim_size(2) << ", " << output_shape.dim_size(3));
		
		MYLOG("Reduce");
		try{
			switch(channel){
				case 1:
					GridBlendRKernelLauncher(input_grid.flat<float>().data(), input_shape.dim_sizes().data(),
						m_blendMode, m_keepDims,
						output_grid->flat<float>().data(), output_shape.dim_sizes().data()
						);
					break;
				case 2:
					GridBlendRGKernelLauncher(input_grid.flat<float>().data(), input_shape.dim_sizes().data(),
						m_blendMode, m_keepDims,
						output_grid->flat<float>().data(), output_shape.dim_sizes().data()
						);
					break;
				case 4:
					GridBlendRGBAKernelLauncher(input_grid.flat<float>().data(), input_shape.dim_sizes().data(),
						m_blendMode, m_keepDims,
						output_grid->flat<float>().data(), output_shape.dim_sizes().data()
						);
					break;
				default:
					OP_REQUIRES(context, false,
						errors::Unimplemented("Only 1,2 and 4 Channel supported."));
			}
		} catch(RenderError::RenderError& e){
			OP_REQUIRES(context, false, errors::Internal(e.what()));
		}
		MYLOG("Kernel done");
	}
private:
	Blending::BlendMode m_blendMode;
	bool m_keepDims;
	
};


void ReduceGridBlendRGradKernelLauncher(const float* _input, float* _input_grads, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		const float* _output, const float* _output_grads, const long long int* output_shape);
void ReduceGridBlendRGGradKernelLauncher(const float* _input, float* _input_grads, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		const float* _output, const float* _output_grads, const long long int* output_shape);
void ReduceGridBlendRGBAGradKernelLauncher(const float* _input, float* _input_grads, const long long int* input_shape,
		const Blending::BlendMode blend_mode, const bool keep_dims,
		const float* _output, const float* _output_grads, const long long int* output_shape);

class ReduceGridBlendGradOp : public OpKernel{
public:
	explicit ReduceGridBlendGradOp(OpKernelConstruction *context) : OpKernel(context){
		std::string blendingMode;
		OP_REQUIRES_OK(context, context->GetAttr("blending_mode", &blendingMode));
		if(blendingMode.compare("BEER_LAMBERT")==0) m_blendMode=Blending::BLEND_BEERLAMBERT;
		else if(blendingMode.compare("ALPHA")==0) m_blendMode=Blending::BLEND_ALPHA;
		else if(blendingMode.compare("ALPHA_ADDITIVE")==0) m_blendMode=Blending::BLEND_ALPHAADDITIVE;
		else if(blendingMode.compare("ADDITIVE")==0) m_blendMode=Blending::BLEND_ADDITIVE;
		else {OP_REQUIRES(context, false, errors::InvalidArgument("invalid blending mode"));}
		
		OP_REQUIRES_OK(context, context->GetAttr("keep_dims", &m_keepDims));
	}
	
	void Compute(OpKernelContext *context) override{
		
		MYLOG("reduce blend gradient op kernel start");
		
		const Tensor& output_grads = context->input(0);
		const Tensor& output_grid = context->input(1);
		const Tensor& input_grid = context->input(2);
		
		MYLOG("Check input");
		TensorShape input_shape = input_grid.shape();
		OP_REQUIRES(context, (input_grid.dims()==5 && input_shape.dim_size(4)<=4),
			errors::InvalidArgument("Invalid input shape", input_shape.DebugString()));
		const int64 batch = input_shape.dim_size(0);
		const int64 channel = input_shape.dim_size(4);
		
		MYLOG("Check output (gradients)");
		OP_REQUIRES(context, output_grads.shape() == output_grid.shape(),
			errors::InvalidArgument("Mismatch between output gradients and output shape: ", output_grads.shape().DebugString(), output_grid.shape().DebugString()));
		TensorShape output_shape = output_grads.shape();
		if(m_keepDims){
			OP_REQUIRES(context, (output_grads.dims()==5 && output_shape.dim_size(4)==channel),
				errors::InvalidArgument("Invalid output gradients shape", output_shape.DebugString()));
			OP_REQUIRES(context, output_shape==input_shape,
				errors::InvalidArgument("output gradients shape does not match input shape", input_shape.DebugString()));
		}else{
			OP_REQUIRES(context, (output_grads.dims()==4 && output_shape.dim_size(3)==channel),
				errors::InvalidArgument("Invalid output gradients shape", output_shape.DebugString()));
			OP_REQUIRES(context, output_shape.dim_size(1)==input_shape.dim_size(2)&&
				output_shape.dim_size(2)==input_shape.dim_size(3),
				errors::InvalidArgument("output gradients shape does not match input shape (x,y)", input_shape.DebugString()));
			output_shape.InsertDim(1,1); //fill z dim
		}
		MYLOG("output_shape: " << output_shape.dim_size(0) << ", " << output_shape.dim_size(1) << ", " << output_shape.dim_size(2) << ", " << output_shape.dim_size(3));
		
		
		MYLOG("Allocate input gradients");
		Tensor* input_grads = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &input_grads));
		MYLOG("Check allocated input gradients");
		auto out = input_grads->flat<float>();
		MYLOG("Allocated input gradients size: " << out.size() << " - " << input_grads->NumElements());
		
		MYLOG("Reduce");
		try{
			switch(channel){
				case 1:
					ReduceGridBlendRGradKernelLauncher(input_grid.flat<float>().data(), input_grads->flat<float>().data(), input_shape.dim_sizes().data(),
						m_blendMode, m_keepDims,
						output_grid.flat<float>().data(), output_grads.flat<float>().data(), output_shape.dim_sizes().data()
						);
					break;
				case 2:
					ReduceGridBlendRGGradKernelLauncher(input_grid.flat<float>().data(), input_grads->flat<float>().data(), input_shape.dim_sizes().data(),
						m_blendMode, m_keepDims,
						output_grid.flat<float>().data(), output_grads.flat<float>().data(), output_shape.dim_sizes().data()
						);
					break;
				case 4:
					ReduceGridBlendRGBAGradKernelLauncher(input_grid.flat<float>().data(), input_grads->flat<float>().data(), input_shape.dim_sizes().data(),
						m_blendMode, m_keepDims,
						output_grid.flat<float>().data(), output_grads.flat<float>().data(), output_shape.dim_sizes().data()
						);
					break;
				default:
					OP_REQUIRES(context, false,
						errors::Unimplemented("Only 1,2 and 4 Channel supported."));
			}
		}catch(RenderError::RenderError& e){
			OP_REQUIRES(context, false, errors::Internal(e.what()));
		}
		
		MYLOG("Kernel done");
	}
private:
	Blending::BlendMode m_blendMode;
	bool m_keepDims;
	
};

REGISTER_KERNEL_BUILDER(Name("ReduceGridBlend") \
	.Device(DEVICE_GPU) \
	, ReduceGridBlendOp);

REGISTER_KERNEL_BUILDER(Name("ReduceGridBlendGrad") \
	.Device(DEVICE_GPU) \
	, ReduceGridBlendGradOp);

