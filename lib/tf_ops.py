import os
import tensorflow as tf
import numpy as np
#import numbers

import logging
log = logging.getLogger('TFops')
log.setLevel(logging.DEBUG)

from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
import scipy.signal

#https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-n-in-python
def next_pow_two(x):
	return 1<<(x-1).bit_length()

def shape_list(arr):
	if isinstance(arr, (tf.Tensor, tf.Variable)):
		return arr.get_shape().as_list()
	elif isinstance(arr, np.ndarray):
		return list(arr.shape)
	return None

def spatial_shape_list(arr):
	shape = shape_list(arr)
	assert len(shape)==5
	return shape[-4:-1]

spacial_shape_list = spatial_shape_list #compatibility...

def reshape_array_format(arr, in_fmt, out_fmt='NDHWC'):
	shape = shape_list(arr)
	if len(shape)!=len(in_fmt):
		raise ValueError("Array shape {} does not math input format '{}'".format(shape, in_fmt))
	if in_fmt==out_fmt:
		return arr
	squeeze = [in_fmt.index(_) for _ in in_fmt if _ not in out_fmt]
	if squeeze:
		if isinstance(arr, np.ndarray):
			arr = np.squeeze(arr, squeeze)
		elif isinstance(arr, tf.Tensor):
			arr = tf.squeeze(arr, squeeze)
	expand = [out_fmt.index(_) for _ in out_fmt if _ not in in_fmt]
	for axis in expand:
		if isinstance(arr, np.ndarray):
			arr = np.expand_dims(arr, axis)
		elif isinstance(arr, tf.Tensor):
			arr = tf.expand_dims(arr, axis)
	return arr

def tf_to_dict(obj):
	"""Convert a tf.Tensor or np.ndarray or any class to a JSON-serializable type."""
	if obj is None or isinstance(obj, (int, bool, float, str, list, tuple, dict)):
		return obj
	if isinstance(obj, (np.ndarray, np.number)):
		# alternative: save ndarray as .npz and put path here. might need base path
		return obj.tolist()
	if isinstance(obj, (tf.Tensor, tf.Variable)):
		# alternative: save ndarray as .npz and put path here. might need base path
		return obj.numpy().tolist()
	d = {
		"__class__":obj.__class__.__name__,
		"__module__":obj.__module__
	}
	if hasattr(obj, "to_dict"):
		d.update(obj.to_dict())
	else:
		d.update(obj.__dict__)
	return d

def tf_pad_to_shape(tensor, shape, alignment="CENTER", allow_larger_dims=False, **pad_args):
	"""pads 'tensor' to have at least size 'shape'
	
	Args:
		tensor (tf.Tensor): the tensor to pad
		shape (Iterable): the target shape, must have the same rank as the tensor shape. elements may be -1 to have no padding.
		alignment (str):
			"CENTER": equally pad before and after, pad less before in case of odd padding.
			"BEFORE": pad after the data (data comes before the padding).
			"AFTER" : pad before the data (data comes after the padding).
		allow_larger_dims (bool): ignore tensor dimensions that are larger than the target dimension (no padding). otherwise raise a ValueError
		**pad_args: kwargs passed to tf.pad (mode, name, constant_value)
	
	Returns:
		tf.Tensor: The padded input tensor
	
	Raises:
		ValueError:
			If shape is not compatible with the shape of tensor
			Or alignment is not one of 'CENTER', 'BEFORE', 'AFTER'
			Or any dimension of the input tensor is larger than its target (if allow_larger_dims==False)
	"""
	tensor_shape = shape_list(tensor)
	if len(tensor_shape) != len(shape): raise ValueError("Tensor shape %s is not compatible with target shape %s"%(tensor_shape, shape))
	alignment = alignment.upper()
	if not alignment in ["CENTER", "BEFORE", "AFTER"]: raise ValueError("Unknown alignment '%s', use 'CENTER', 'BEFORE', 'AFTER'"%alignment)
	
	diff = []
	for i, (ts, s) in enumerate(zip(tensor_shape, shape)):
		if ts>s and (not allow_larger_dims) and s!=-1:
			raise ValueError("Tensor size %d of dimension %d is larger than target size %d"%(ts, i, s))
		elif ts>=s or s==-1:
			diff.append(0)
		else: #ts<s
			diff.append(s-ts)
	
	paddings = []
	for i, d in enumerate(diff):
		if alignment=="CENTER":
			paddings.append((d//2, d-(d//2)))
		elif alignment=="BEFORE":
			paddings.append((0, d))
		elif alignment=="AFTER":
			paddings.append((d, 0))
			
	return tf.pad(tensor=tensor, paddings=paddings, **pad_args)

def tf_pad_to_next_pow_two(data, pad_axes=(0,1,2)):
	shape = data.get_shape().as_list()
	paddings = []
	for axis in range(len(shape)):
		if axis in pad_axes:
			dim=shape[axis]
			dif = next_pow_two(dim) - dim
			if dif%2==0:
				paddings.append([dif//2,dif//2])
			else:
				paddings.append([dif//2 +1,dif//2])
		else:
			paddings.append([0,0])
	return tf.pad(data, paddings)

#https://stackoverflow.com/questions/45254554/tensorflow-same-padding-calculation
def getSamePadding(shape, kernel, stride):
	pad = []
	for dim, k, s in zip(shape, kernel, stride):
		out_dim = int(np.ceil(dim/s))
		pad_total = max((out_dim - 1)*s + k - dim, 0)
		pad_before = int(pad_total//2)
		pad_after = pad_total - pad_before
		pad.append([pad_before, pad_after])
	return pad

def tf_color_gradient(data, c1, c2, vmin=0, vmax=1):
	data_norm = (data - vmin)/(vmax - vmin) # -> [0,1]
	return c1 + (c2-c1)*data_norm # lerp

def tf_element_transfer_func(data, grads):
	channel = shape_list(grads[0][1])[-1]
	r = tf.zeros(shape_list(data)[:-1] +[channel])
	for g1, g2 in zip(grads[:-1], grads[1:]):
		grad = tf_color_gradient(data, g1[1], g2[1], g1[0], g2[0])
		condition = tf.broadcast_to(tf.logical_and(tf.greater_equal(data, g1[0]), tf.less_equal(data,g2[0])), grad.get_shape())
		r = tf.where(condition, grad, r)
	return r

def tf_shift(tensor, shift, axis):
	tensor_shape = shape_list(tensor)
	tensor_rank = len(tensor_shape)
	if axis<0:
		axis += tensor_rank
	if axis<0 or tensor_rank<=axis:
		raise ValueError("Tensor axis out of bounds.")
	if shift==0:
		return tf.identity(tensor)
	elif shift<=-tensor_shape[axis] or shift>=tensor_shape[axis]:
		return tf.zeros_like(tensor)
	elif shift<0:
		shift = -shift
		pad = [(0,0)]*axis + [(0,shift)] + [(0,0)]*max(tensor_rank-axis-1, 0)
		return tf.pad(tf.split(tensor, [shift, tensor_shape[axis]-shift], axis=axis)[1], pad)
	elif shift>0:
		pad = [(0,0)]*axis + [(shift,0)] + [(0,0)]*max(tensor_rank-axis-1, 0)
		return tf.pad(tf.split(tensor, [tensor_shape[axis]-shift, shift], axis=axis)[0], pad)
		

def tf_reduce_dot(x,y, axis=None, keepdims=False, name=None):
	return tf.reduce_sum(tf.multiply(x,y), axis=axis, keepdims=keepdims, name=name)

def tf_reduce_var(input_tensor, axis=None, keepdims=False, name=None):
	m = tf.reduce_mean(input_tensor, axis=axis, keepdims=keepdims)
	return tf.reduce_mean(tf.square(tf.abs(input_tensor - m)), axis=axis, keepdims=keepdims)

def tf_reduce_std(input_tensor, axis=None, keepdims=False, name=None):
	return tf.sqrt(tf_reduce_var(input_tensor, axis=axis, keepdims=keepdims, name=name))

def tf_None_to_const(input_tensor, constant=0, dtype=tf.float32):
	return tf.constant(0, dtype=dtype) if input_tensor is None else input_tensor

@tf.custom_gradient
def tf_ReLU_lin_grad(input_tensor):
	y = tf.relu(input_tensor)
	def grad(dy):
		return tf.identity(dy)
	return y, grad

@tf.custom_gradient
def tf_grad_NaN_to_num(tensor):
	y = tf.identity(tensor)
	def grad(dy):
		return tf.where(tf.is_nan(dy), tf.zeros_like(dy), dy)
	return y, grad

@tf.custom_gradient
def tf_norm2(x, axis=None, keepdims=False, name=None):
	
	y = tf.norm(x, ord='euclidean', axis=axis, keepdims=keepdims, name=name)
	def grad(dy):
		if keepdims:
			return tf.div_no_nan(x, y)*dy
		else:
			return tf.div_no_nan(x, tf.expand_dims(y, axis))*tf.expand_dims(dy, axis)
		
	return y, grad

def tf_angle_between(x, y, axis, mode="RAD", keepdims=False, name=None, undefined="ORTHOGONAL", norm_eps=1e-12, constant=1):
	''' Angle between vectors (axis) in radians (mode=RAD), degrees (DEG) or as cosine similarity (COS).
		https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
		undefined:
			ORTHOGONAL: returns 90 degree if any of the two vectors has norm 0
			CONSTANT: returns constant if any of the two vector norms is smaller than norm_eps
	'''
#	norm_x = tf_norm2(x, axis=axis, keepdims=keepdims)
#	norm_y = tf_norm2(y, axis=axis, keepdims=keepdims)
#	c = tf.div_no_nan(tf_reduce_dot(x,y, axis=axis, keepdims=keepdims), norm_x * norm_y)
	x = tf.math.l2_normalize(x, axis=axis, epsilon=norm_eps)
	y = tf.math.l2_normalize(y, axis=axis, epsilon=norm_eps)
	c = tf_reduce_dot(x,y, axis=axis, keepdims=keepdims)
	
	if mode.upper()=="COS":
		result = c
	else:
		a = tf.acos(tf.clip_by_value(c, -1, 1))
		if mode.upper()=="RAD":
			result = a
		elif mode.upper()=="DEG":
			result = a * (180 / np.pi)
		else:
			raise ValueError("Unknown mode '%s' for angle between vectors."%mode)
	
	if undefined.upper()=="ORTHOGONAL":
		pass
	elif undefined.upper()=="CONSTANT":
		norm_x = tf_norm2(x, axis=axis, keepdims=keepdims)
		norm_y = tf_norm2(y, axis=axis, keepdims=keepdims)
		cond = tf.logical_or(tf.less(norm_x, norm_eps), tf.less(norm_y, norm_eps))
		constant = tf.cast(tf.broadcast_to(constant, result.get_shape()), dtype=result.dtype)
		result = tf.where(cond, constant, result)
	else:
		raise ValueError("Unknown mode '%s' for undefined angles."%undefined)
		
	return result

def tf_cosine_similarity(x, y, axis, keepdims=False):
	return tf_angle_between(x, y, axis, keepdims=keepdims, mode="COS", undefined="ORTHOGONAL")

def tf_PSNR(x, y, max_val=1.0, axes=[-3,-2,-1]):
	''' or use tf.image.psnr for 2D tensors'''
	rrmse = tf.rsqrt(tf.reduce_mean(tf.squared_difference(x,y)), axes)
	return tf.log(max_val*rrmse) * (20./tf.log(10.)) #tf.constant(8.685889638065035, dtype=tf.float32)

def tf_tensor_stats(data, scalar=False, as_dict=False):
	data_abs = tf.abs(data)
	
	d = {
		"min":tf.reduce_min(data),
		"max":tf.reduce_max(data),
		"mean":tf.reduce_mean(data),
		"std":tf_reduce_std(data),
		"abs":{
			"min":tf.reduce_min(data_abs),
			"max":tf.reduce_max(data_abs),
			"mean":tf.reduce_mean(data_abs),
			"std":tf_reduce_std(data_abs),
		},
	}
	
	if scalar:
		def to_scalar(inp):
			if isinstance(inp, tf.Tensor):
				return inp.numpy()
			if isinstance(inp, dict):
				for k in inp:
					inp[k] = to_scalar(inp[k])
			return inp
		d = to_scalar(d)
	
	if as_dict:
		return d
	else:
		return d['max'], d['min'], d['mean'], d['abs']['mean']

def tf_print_stats(data, name, log=None):
	max, min, mean, abs_mean = tf_tensor_stats(data)
	if log is None:
		print('{} stats: min {:.010e}, max {:.010e}, mean {:.010e}, abs-mean {:.010e}'.format(name, min, max, mean, abs_mean))
	else:
		log.info('{} stats: min {:.010e}, max {:.010e}, mean {:.010e}, abs-mean {:.010e}'.format(name, min, max, mean, abs_mean))
	return max, min, mean

def tf_log_barrier_ext(x, t):
	"""https://arxiv.org/abs/1904.04205
	Args:
		x (tf.Tensor): 
		t (scalar): scale or strength
	"""
	t_inv = 1./t
	t_inv_2 = t_inv*t_inv
	v1 = -t_inv*tf.log(-x)
	v2 = t*x - (t_inv*tf.log(t_inv_2) + t_inv)
	cond = (x<=(-t_inv2))
	return tf.where(cond, v1, v2)

def tf_log_barrier_ext_sq(x, t):
	"""https://arxiv.org/abs/1904.04205
	with quadratic extension
	
	Args:
		x (tf.Tensor): 
		t (scalar): scale or strength
	"""
	t_inv = 1./t
	t_inv_2 = t_inv*t_inv
	t_half = t*0.5
	
	v1 = -t_inv*tf.log(-x)
	
	v2_x = (x + (1 + t_inv_2))
	v2 = t_half*(v2_x*v2_x) - (t_inv*tf.log(t_inv_2) + t_half)
	
	cond = (x<=(-t_inv2))
	return tf.where(cond, v1, v2)

def tf_image_resize_mip(images, size, mip_bias=0.5, **resize_kwargs):
	'''Resize the image using nearest mip-mapping (if down-sampliing) and tf.image.resize_images
	
	N.B.: should switch to TF 2 image.resize(antialias=True) eventually
		https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
		https://github.com/tensorflow/tensorflow/issues/6720
	'''
	
	images_shape = np.asarray(shape_list(images), dtype=np.float32)
	target_shape = np.asarray(size, dtype=np.float32)
	relative_step = np.amax(images_shape[-3:-1]/target_shape)
	lod_raw = np.log2(relative_step)
	lod = np.floor(np.maximum(0., lod_raw + mip_bias)).astype(np.int32)
	if lod > 0.:
		window_size = np.left_shift(1, lod)
		window = (1,window_size,window_size,1)
		images = tf.nn.avg_pool(images, window, window, padding="SAME", data_format="NHWC")
	
	return tf.image.resize_images(images, size, **resize_kwargs)

def gaussian_1dkernel(size=5, sig=1.):
	"""
	Returns a 1D Gaussian kernel array with side length size and a sigma of sig
	"""
	gkern1d = tf.constant(scipy.signal.gaussian(size, std=sig), dtype=tf.float32)
	return (gkern1d/tf.reduce_sum(gkern1d))

def gaussian_2dkernel(size=5, sig=1.):
	"""
	Returns a 2D Gaussian kernel array with side length size and a sigma of sig
	"""
	gkern1d = tf.constant(scipy.signal.gaussian(size, std=sig), dtype=tf.float32)
	gkern2d = tf.einsum('i,j->ij',gkern1d, gkern1d)
	return (gkern2d/tf.reduce_sum(gkern2d))

def gaussian_3dkernel(size=5, sig=1.):
	"""
	Returns a 3D Gaussian kernel array with side length size and a sigma of sig
	"""
	gkern1d = tf.constant(scipy.signal.gaussian(size, std=sig), dtype=tf.float32)
	gkern3d = tf.einsum('i,j,k->ijk',gkern1d, gkern1d, gkern1d)
	return (gkern3d/tf.reduce_sum(gkern3d))
	
def tf_data_gaussDown2D(data, sigma = 1.5, stride=4, channel=3, padding='VALID'):
	"""
	tensorflow version of the 2D down-scaling by 4 with Gaussian blur
	sigma: the sigma used for Gaussian blur
	return: down-scaled data
	"""
	k_w = 1 + 2 * int(sigma * 3.0)
	gau_k = gaussian_2dkernel(k_w, sigma)
	gau_0 = tf.zeros_like(gau_k)
	gau_list = [[gau_k if i==o else gau_0 for i in range(channel)] for o in range(channel)]
	#	[gau_k, gau_0, gau_0],
	#	[gau_0, gau_k, gau_0],
	#	[gau_0, gau_0, gau_k]] # only works for RGB images!
	gau_wei = tf.transpose(gau_list, [2,3,0,1])
	
	fix_gkern = tf.constant( gau_wei, shape = [k_w, k_w, channel, channel], name='gauss_blurWeights', dtype=tf.float32)
	# shape [batch_size, crop_h, crop_w, 3]
	cur_data = tf.nn.conv2d(data, fix_gkern, strides=[1,stride,stride,1], padding=padding, name='gauss_downsample')

	return cur_data
	
def tf_data_gaussDown3D(data, sigma = 1.5, stride=4, channel=3, padding='VALID'):
	"""
	tensorflow version of the 3D down-scaling by 4 with Gaussian blur
	sigma: the sigma used for Gaussian blur
	return: down-scaled data
	"""
	k_w = 1 + 2 * int(sigma * 3.0)
	gau_k = gaussian_3dkernel(k_w, sigma)
	gau_0 = tf.zeros_like(gau_k)
	gau_list = [[gau_k if i==o else gau_0 for i in range(channel)] for o in range(channel)]
	gau_wei = tf.transpose(gau_list, [2,3,4,0,1])
	
	fix_gkern = tf.constant(gau_wei, shape = [k_w, k_w, k_w, channel, channel], name='gauss_blurWeights', dtype=tf.float32)
	# shape [batch_size, crop_h, crop_w, 3]
	cur_data = tf.nn.conv3d(data, fix_gkern, strides=[1,stride,stride,stride,1], padding=padding, name='gauss_downsample')

	return cur_data

def _tf_laplace_kernel_3d(neighbours=1):
	if neighbours==0:
		laplace_kernel = np.zeros((3,3,3), dtype=np.float32)
	elif neighbours==1:
		laplace_kernel = np.asarray(
		[	[[ 0, 0, 0],
			 [ 0,-1, 0],
			 [ 0, 0, 0]],
			[[ 0,-1, 0],
			 [-1, 6,-1],
			 [ 0,-1, 0]],
			[[ 0, 0, 0],
			 [ 0,-1, 0],
			 [ 0, 0, 0]],
		], dtype=np.float32)/6.
	elif neighbours==2:
		laplace_kernel = np.asarray(
		[	[[ 0,-1, 0],
			 [-1,-2,-1],
			 [ 0,-1, 0]],
			[[-1,-2,-1],
			 [-2,24,-2],
			 [-1,-2,-1]],
			[[ 0,-1, 0],
			 [-1,-2,-1],
			 [ 0,-1, 0]],
		], dtype=np.float32)/24.
	elif neighbours==3:
		# https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Image_Processing
		laplace_kernel = np.asarray(
		[	[[-2,-3,-2],
			 [-3,-6,-3],
			 [-2,-3,-2]],
			[[-3,-6,-3],
			 [-6,88,-6],
			 [-3,-6,-3]],
			[[-2,-3,-2],
			 [-3,-6,-3],
			 [-2,-3,-2]],
		], dtype=np.float32)/88.
	return tf.constant(laplace_kernel, dtype=tf.float32)


''' old n=3
laplace_kernel = np.asarray(
[	[[-1,-2,-1],
	 [-2,-4,-2],
	 [-1,-2,-1]],
	[[-2,-4,-2],
	 [-4,56,-4],
	 [-2,-4,-2]],
	[[-1,-2,-1],
	 [-2,-4,-2],
	 [-1,-2,-1]],
], dtype=np.float32)/56.
'''


def tf_laplace_filter_3d(inp, neighbours=1, padding='SAME', name='gauss_filter'):
	with tf.name_scope(name):
		channel = inp.get_shape().as_list()[-1]
		if channel != 1:
			raise ValueError('input channel must be 1')
		laplace_kernel = _tf_laplace_kernel_3d(neighbours)
		laplace_kernel = laplace_kernel[:,:,:,tf.newaxis, tf.newaxis]
		#laplace_kernel = tf.concat([gauss_kernel]*channel, axis=3)
		return tf.nn.conv3d(inp, laplace_kernel, strides=[1,1,1,1,1], padding=padding)


def tf_build_density_grid(sim_transform, density_scale=0.06, cube_thickness=8, sphere_radius=40, sphere_thickness=6):
	#density = 0.06
	#border = 1
	coords = tf.meshgrid(tf.range(sim_transform.grid_size[0], dtype=tf.float32), tf.range(sim_transform.grid_size[1], dtype=tf.float32), tf.range(sim_transform.grid_size[2], dtype=tf.float32), indexing='ij')
	coords = tf.transpose(coords, (1,2,3,0))
	coords_centerd = coords - np.asarray(sim_transform.grid_size)/2.0
	dist_center = tf.norm(coords_centerd, axis=-1, keepdims=True)
	density = tf.zeros(list(sim_transform.grid_size)+[1], dtype=tf.float32)
	ones = tf.ones(list(sim_transform.grid_size)+[1], dtype=tf.float32)
	
	is_in_sphere = tf.logical_and(dist_center<sphere_radius, dist_center>(sphere_radius-sphere_thickness))
	#print(is_in_sphere.get_shape().as_list())
	density = tf.where(is_in_sphere, ones, density)
	#print(tf.reduce_mean(density))
	
	is_in_cube = tf.reduce_sum(tf.cast(tf.logical_or(coords<cube_thickness, coords>(np.asarray(sim_transform.grid_size)-1 -cube_thickness)), dtype=tf.int8), axis=-1, keepdims=True)
	density = tf.where(is_in_cube>1, ones, density)
	#print(tf.reduce_mean(density))
	
	sim_data = np.expand_dims(np.expand_dims(density, 0),-1) #NDHWC
	sim_data *=density_scale
	sim_data = tf.constant(sim_data, dtype=tf.float32)
	sim_transform.set_data(sim_data)
	return sim_data

#https://stackoverflow.com/questions/49189496/can-symmetrically-paddding-be-done-in-convolution-layers-in-keras
# mirror (reflect) padding for convolutions, does not add a conv layer, instead use before valid-padding conv layer.
class MirrorPadND(tf.keras.layers.Layer):
	#def __init__(self, dim, kernel, stride, **kwargs):
	def __init__(self, dim=2, kernel=4, stride=1, **kwargs): #compatibility values, TODO: remove
		self.kernel = conv_utils.normalize_tuple(kernel, dim, 'kernel_size')
		self.stride = conv_utils.normalize_tuple(stride, dim, 'stride')
		self.dim = dim
		super(MirrorPadND, self).__init__(**kwargs)
	
	def build(self, input_shape):
		super(MirrorPadND, self).build(input_shape)
		
	def call(self, inputs):
		inputs_shape = inputs.get_shape().as_list()
		dim = len(inputs_shape)-2
		shape = inputs.get_shape().as_list()[-(dim+1):-1]
		pad = [[0,0]] + getSamePadding(shape, self.kernel, self.stride) + [[0,0]]
		padded = tf.pad(inputs, pad, 'REFLECT')
		return padded
	
	def compute_output_shape(self, input_shape):
		#print(input_shape)
		spatial_size = [int(_) for _ in input_shape[1:-1]]
		#print(spatial_size)
		shape = list(input_shape[:1]) + list(np.asarray(spatial_size) + np.sum(getSamePadding(spatial_size, self.kernel, self.stride), axis=-1)) + list(input_shape[-1:])
		#print(shape)
		return tensor_shape.TensorShape(shape)
	
	def get_config(self):
		config = {
			'dim':self.dim,
			'kernel':self.kernel,
			'stride':self.stride,
		}
		base_config = super(MirrorPadND, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

def ConvLayer(in_layer, dim, filters, kernel_size, stride=1, activation='none', alpha=0.2, padding='ZERO', **kwargs):
	x = in_layer
	if padding.upper()=='MIRROR':
		x = MirrorPadND(dim, kernel_size, stride)(x)
		padding = 'valid'
	elif padding.upper()=='ZERO':
		padding = 'same'
	elif padding.upper()=='NONE':
		padding = 'valid'
	else:
		raise ValueError('Unsupported padding: {}'.format(padding))
	
	if dim==1:
		x = tf.keras.layers.Conv1D(filters, kernel_size, stride, padding=padding, **kwargs)(x)
	elif dim==2:
		x = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding=padding, **kwargs)(x)
	elif dim==3:
		x = tf.keras.layers.Conv3D(filters, kernel_size, stride, padding=padding, **kwargs)(x)
	else:
		raise ValueError('Unsupported dimension: {}'.format(self.dim))
	
	if activation=='relu':
		x = tf.keras.layers.ReLU()(x)
	elif activation=='lrelu':
		x = tf.keras.layers.LeakyReLU(alpha=alpha)(x)
	
	return x

def discriminator(input_shape=[None,None,3], layers=[16]*4, kernel_size=3, strides=1, final_fc=False, activation='relu', alpha=0.2, noise_std=0.0, padding='ZERO'):
	dim = len(input_shape)-1
	num_layers = len(layers)
	if np.isscalar(strides):
		strides = [strides]*num_layers
	x = tf.keras.layers.Input(shape=input_shape, name='disc_input')
	inputs = x
	if noise_std>0:
		x = tf.keras.layers.GaussianNoise(stddev=noise_std)(x)
	x = ConvLayer(x, dim, 8, kernel_size, 1, activation, alpha, padding=padding, name='disc_in_conv')
	for filters, stride in zip(layers, strides):
		x = ConvLayer(x, dim, filters, kernel_size, stride, activation, alpha, padding=padding)
	if final_fc:
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(1, name='disc_output')(x)
	else:
		x = ConvLayer(x, dim, 1, kernel_size, padding=padding, name='disc_output')
	outputs = x
	return tf.keras.Model(inputs=[inputs], outputs=[outputs])
	'''
	keras_layers=[]
	if noise_std>0:
		keras_layers.append(tf.keras.layers.GaussianNoise(stddev=noise_std, input_shape=input_shape))
		keras_layers.append(tf.keras.layers.Conv2D(8, kernel_size, padding=padding))
	else:
		keras_layers.append(tf.keras.layers.Conv2D(8, kernel_size, padding=padding, input_shape=input_shape))
	if activation=='relu':
		keras_layers.append(tf.keras.layers.ReLU())
	elif activation=='lrelu':
		keras_layers.append(tf.keras.layers.LeakyReLU(alpha=alpha))
	for filters in layers:
		keras_layers.append(tf.keras.layers.Conv2D(filters, kernel_size, strides, padding=padding))
		if activation=='relu':
			keras_layers.append(tf.keras.layers.ReLU())
		elif activation=='lrelu':
			keras_layers.append(tf.keras.layers.LeakyReLU(alpha=alpha))
	if final_fc:
		keras_layers.append(tf.keras.layers.Flatten())
		keras_layers.append(tf.keras.layers.Dense(1))
	else:
		keras_layers.append(tf.keras.layers.Conv2D(1, kernel_size, padding=padding))
	return tf.keras.Sequential(keras_layers)
	'''


# save any keras model
def save_discriminator(disc_model, name, path):
	try:
		disc_model.save(os.path.join(path, '{}_model.h5'.format(name)))
		log.info('Saved discriminator keras model %s', name)
	except:
		log.error('Failed to save full discriminator model. Saving weights instead.', exc_info=True)
		try:
			disc_model.save_weights(os.path.join(path, '{}_weights.h5'.format(name)))
			log.info('Saved discriminator keras model weights %s', name)
		except:
			log.error('Failed to save discriminator weights. Saving as npz instead.', exc_info=True)
			try:
				weights = disc_model.get_weights()
				np.savez_compressed(os.path.join(path, '{}_weights.npz'.format(name)), weights)
				log.info('Saved discriminator keras model weights as npz %s', name)
			except:
				log.exception('Failed to save discriminator weights.')