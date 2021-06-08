import os
from collections.abc import Iterable

#from tensorflow.python.framework import ops
#from tensorflow.python.ops import array_ops

import numpy as np
import tensorflow as tf

import imageio
import logging

from .profiling import Profiler
from .camera import Camera
from .lighting import Light
from .transform import GridTransform
from lib.tf_ops import tf_data_gaussDown2D, tf_data_gaussDown3D, shape_list, spatial_shape_list

#experimental
from .vector import GridShape

from .cuda.ops_loader import sampling_ops, blending_ops, raymarching_ops #sample_grid_transform, sample_grid_lut, reduce_grid_blend, raymarch_grid_transform

def format_byte(size):
	units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'] #...
	size = float(size)
	i = 0
	while size > 1024. and i<len(units):
		size /=1024.
		i+=1
	return '{:7.02f} {}'.format(size, units[i])
	

def gammaCorrection(img, gamma=2.0):
	return np.power(img, 1.0/gamma)

class RenderingContext:
	def __init__(self, cameras, lights, dens_renderer, vel_renderer, monochrome):
		self.cameras = cameras
		self.lights = lights
		self.dens_renderer = dens_renderer
		self.vel_renderer = vel_renderer
		self.cam_x_range = [-30,30]
		self.cam_y_range = [0,360]
		self.cam_z_range = [0,360]
		self.monochrome = monochrome
	
	def randomize_camera_rotation(self, x_range=None, y_range=None, z_range=None):
		if x_range is None: x_range = self.cam_x_range
		if y_range is None: y_range = self.cam_y_range
		if z_range is None: z_range = self.cam_z_range
		num_cams = len(self.cameras)
		rand_y = np.random.uniform(*y_range, (num_cams))
		rand_x = np.random.uniform(*x_range, (num_cams))
		rand_z = np.random.uniform(*z_range, (num_cams))
		#print('disc cam rotation x:', rand_x, 'y:', rand_y)
		for i in range(num_cams):
			self.cameras[i].transform.parent.rotation_deg[0] = rand_x[i]
			self.cameras[i].transform.parent.rotation_deg[1] = rand_y[i]
			if z_range!=[0,0]:
				self.cameras[i].transform.rotation_deg[2] = rand_z[i]

class Renderer(object):
	def __init__(self, profiler=None, filter_mode ='LINEAR', boundary_mode='BORDER', \
			mipmapping='LINEAR', num_mips=2, mip_bias=0.0, \
			sample_gradients=False, fast_gradient_mip_bias_add=0.0, \
			blend_mode='BEER_LAMBERT', name='Renderer', **kwargs):
		self.log = logging.getLogger(name)
		self.cameras = []
		if profiler is not None:
			self.profiler = profiler
		else:
			self.profiler = Profiler()
		self.blending = blending_ops
		self.filter_mode = filter_mode
		self.boundary_mode = boundary_mode
		self.mip_mode = mipmapping
		self.num_mips = num_mips
		self.mip_bias = mip_bias
		self.gradient_mip_bias_add = fast_gradient_mip_bias_add
		self.sample_gradients = sample_gradients
		self.blend_mode = blend_mode
		if "luma" in kwargs:
			self.luma = kwargs["luma"]
		
		self.allow_fused = kwargs.get("fused", False)
		
		self.scattering_ratio = kwargs.get("scattering_ratio", 1.0)
	
	@property
	def can_render_fused(self):
		return self.allow_fused and (self.blend_mode in ["BEER_LAMBERT", "ADDITIVE", "ALPHA_ADDITIVE", "ALPHA"])
	# separate camera list by resolution and being static
	def _sort_cameras(self, cameras):
		def __split_by_size(cams):
			sizes = []
			for cam in cams:
				if cam.transform.grid_size not in sizes:
					sizes.append(cam.transform.grid_size)
			return [(size, [cam for cam in cams if cam.transform.grid_size==size]) for size in sizes]
		
		static_cams = __split_by_size([camera for camera in cameras if camera.static is not None])
		non_static_cams = __split_by_size([camera for camera in cameras if camera.static is None])
		
		return non_static_cams, static_cams
	
	
	def _get_camera_params_batch(self, cameras):
		M_V = [camera.view_matrix().transpose() for camera in cameras]
		M_P = [camera.projection_matrix().transpose() for camera in cameras]
		F = [camera.frustum() for camera in cameras]
		return M_V, M_P, F
	
	def get_camera_LuT(self, grid_transform, cam, inverse=False):
		M_model = grid_transform.get_transform_matrix().transpose()
		#for size, cams in sizes_cams:
		V,P,F = self._get_camera_params_batch([cam])
		MV = [M_model@v for v in V] #working with transpose matrices here
		if inverse:
			in_shape = cam.transform.grid_size
			out_shape = grid_transform.grid_size
		else:
			in_shape = grid_transform.grid_size
			out_shape = cam.transform.grid_size
		LuTs = sampling_ops.lod_transform(input_shape=in_shape, matrix_mv=MV, matrix_p=P, frustum_params=F, output_shape=out_shape, inverse_transform=inverse) #TODO
		return LuTs
	
	def _setup_static_camera_LuT(self, grid_transform, sizes_cams, inverse=False):
		size, cams = sizes_cams
		M_model = grid_transform.get_transform_matrix().transpose()
		V,P,F = self._get_camera_params_batch(cams)
		MV = [M_model@v for v in V] #working with transpose matrices here
		if inverse:
			in_shape = size
			out_shape = grid_transform.grid_size
		else:
			in_shape = grid_transform.grid_size
			out_shape = size
		LuTs = sampling_ops.lod_transform(input_shape=in_shape, matrix_mv=MV, matrix_p=P, frustum_params=F, output_shape=out_shape, inverse_transform=inverse) #TODO
		lut_shape = tf.shape(LuTs).numpy()
		self.log.info('generated LuTs: %s %s', lut_shape, format_byte(np.prod(lut_shape)*4))
		LuTs = tf.unstack(LuTs, axis=0)
		for lut, cam in zip(LuTs, cams):
			if inverse:
				cam.inverseLuT=tf.constant(lut)
			else:
				cam.LuT=tf.constant(lut)
	
	# on-demand lut generation from transform
	def __get_cam_luts(t, size, cams, inverse):
		if inverse: setup_cams = [cam for cam in cams if cam.inverseLuT is None]
		else: setup_cams = [cam for cam in cams if cam.LuT is None]
		
		if len(setup_cams)>0:
			self._setup_static_camera_LuT(transformation, (size, setup_cams), inverse)
		
		if inverse: luts = [cam.inverseLuT for cam in cams]
		else: luts = [cam.LuT for cam in cams]
		
		return luts
	
	def check_LoD(self, grid_transform, camera, check_inverse=True, name=None):
		_LuT_LoD = self.get_camera_LuT(grid_transform, camera)
		shape_list(_LuT_LoD)
		axes = [_ for _ in range(len(shape_list(_LuT_LoD))-1)]
		LoD_min = tf.reduce_min(_LuT_LoD, axis=axes)[-1].numpy()
		LoD_max = tf.reduce_max(_LuT_LoD, axis=axes)[-1].numpy()
		del _LuT_LoD
		if check_inverse:
			_LuT_LoD = self.get_camera_LuT(grid_transform, camera, inverse=True)
			shape_list(_LuT_LoD)
			axes = [_ for _ in range(len(shape_list(_LuT_LoD))-1)]
			LoD_grad_min = tf.reduce_min(_LuT_LoD, axis=axes)[-1].numpy()
			LoD_grad_max = tf.reduce_max(_LuT_LoD, axis=axes)[-1].numpy()
			del _LuT_LoD
			if name is not None:
				self.log.info("%s stats: shape: %s (%.2f Mi), step: %f, LoD: %f - %f (grad: %f - %f)", name, camera.transform.grid_size, np.prod(camera.transform.grid_size)/(1024*1024), camera.depth_step, LoD_min, LoD_max, LoD_grad_min, LoD_grad_max)
			stats = {"shape":camera.transform.grid_size, "step":camera.depth_step, "LoD_min":LoD_min, "LoD_max":LoD_max, "LoD_grad_min":LoD_grad_min, "LoD_grad_max":LoD_grad_max}
		else:
			if name is not None:
				self.log.info("%s stats: shape: %s (%.2f Mi), step: %f, LoD: %f - %f", name, camera.transform.grid_size, np.prod(camera.transform.grid_size)/(1024*1024), camera.depth_step, LoD_min, LoD_max)
			stats = {"shape":camera.transform.grid_size, "step":camera.depth_step, "LoD_min":LoD_min, "LoD_max":LoD_max}
		return stats
	
	# cameras must have the same resolution
	def _sample_transform(self, data, from_transformations, to_transformations, inverse=False, fix_scale_center=False):
		'''
			_sample_transform is currently experimental and assumes the output grid to be in a centered [-1,1] cube, so scale input accordingly or use fix_scale_center
		'''
		with self.profiler.sample('sample_transform'):
			CM = {False: 'TRANSFORM', True: 'TRANSFORM_REVERSE'}
			M = [t.get_transform_matrix().transpose() for t in from_transformations]
			if fix_scale_center:
				V = [(GridTransform(t.grid_size, scale=[2,2,2], center=True, normalize='ALL').get_transform_matrix()@t.get_inverse_transform()).transpose() for t in to_transformations]
			else:
				V = [t.get_inverse_transform().transpose() for t in to_transformations]
			P = [t.identity_matrix() for t in to_transformations]
			F = [[-1,1,-1,1,1,-1] for t in to_transformations]
			
			if self.sample_gradients:
				@tf.custom_gradient
				def __sample(x, m, v, p, f, out_shape): #NDHWC, N44, V44, V44, V6, 3(:DHW-out)
					with self.profiler.sample('Sampling kernel'):
						y = sampling_ops.sample_grid_transform(input=x, matrix_m=m, matrix_v=v, matrix_p=p, frustum_params=f, output_shape=out_shape, \
							interpolation = self.filter_mode, boundary=self.boundary_mode, mipmapping=self.mip_mode, num_mipmaps=self.num_mips, mip_bias=self.mip_bias, coordinate_mode=CM[inverse])
					in_shape = tf.shape(x)
					batch = len(m)
					views = len(v)
					
					def grad(dy): #NVDHWC
						with self.profiler.sample('Sampling gradients kernel'):
							#dy_batch = tf.unstack(dy) # N - VDHWC
							dx = []
							for i in range(batch): #iterate batch (N)
								m_views = tf.constant([m[i]]*views, dtype=tf.float32)
								dx_views = sampling_ops.sample_grid_transform(input=dy[i], matrix_m=m_views, matrix_v=v, matrix_p=p, frustum_params=f, output_shape=in_shape[-4:-1], \
									interpolation=self.filter_mode, boundary=self.boundary_mode, \
									mipmapping=self.mip_mode, num_mipmaps=self.num_mips, mip_bias=self.mip_bias + self.gradient_mip_bias_add, \
									coordinate_mode=CM[not inverse], separate_camera_batch=False) # V1DHWC
								dx.append(tf.reduce_sum(dx_views, axis=0))#1DHWC
							if batch>1:
								dx = tf.concat(dx, axis=0) #NDHWC
							else:
								dx = dx[0]
						return dx #[dx, None, None, None, None, None]
					
					return y, grad #NVDHWC, g(NVDHWC)->NDHWC
			
			with self.profiler.sample('Sampling kernel'):
				if self.sample_gradients:
					sampled = __sample(data, M, V, P, F, from_transformations[0].grid_size if inverse else to_transformations[0].grid_size)
				else:
					sampled = sampling_ops.sample_grid_transform(input=data, matrix_m=M, matrix_v=V, matrix_p=P, frustum_params=F, output_shape=from_transformations[0].grid_size if inverse else to_transformations[0].grid_size, \
						interpolation = self.filter_mode, boundary=self.boundary_mode, mipmapping=self.mip_mode, num_mipmaps=self.num_mips, mip_bias=self.mip_bias, \
						coordinate_mode=CM[inverse])
		return sampled #NVDHWC
		
	
	# cameras must have the same resolution
	def _sample_camera_transform(self, data, transformations, cameras, inverse=False):
		with self.profiler.sample('sample_camera_transform'):
			CM = {False: 'TRANSFORM_LINDEPTH', True: 'TRANSFORM_LINDEPTH_REVERSE'}
			M = [t.get_transform_matrix().transpose() for t in transformations]
			V, P, F = self._get_camera_params_batch(cameras)
			
			if self.sample_gradients:
				@tf.custom_gradient
				def __sample(x, m, v, p, f, out_shape): #NDHWC, N44, V44, V44, V6, 3(:DHW-out)
					with self.profiler.sample('Sampling kernel'):
						y = sampling_ops.sample_grid_transform(input=x, matrix_m=m, matrix_v=v, matrix_p=p, frustum_params=f, output_shape=out_shape, interpolation = self.filter_mode, boundary=self.boundary_mode, \
							mipmapping=self.mip_mode, num_mipmaps=self.num_mips, mip_bias=self.mip_bias, \
							coordinate_mode=CM[inverse])
					in_shape = tf.shape(x)
					batch = len(m)
					views = len(v)
					def grad(dy): #NVDHWC
						with self.profiler.sample('Sampling gradients kernel'):
							#dy_batch = tf.unstack(dy) # N - VDHWC
							dx = []
							for i in range(batch): #iterate batch (N)
								with self.profiler.sample('<kernel call>'):
									dx_views = sampling_ops.sample_grid_transform(input=dy[i], matrix_m=[m[i]]*views, matrix_v=v, matrix_p=p, frustum_params=f, output_shape=in_shape[-4:-1], interpolation = self.filter_mode, boundary=self.boundary_mode, \
										mipmapping=self.mip_mode, num_mipmaps=self.num_mips, mip_bias=self.mip_bias + self.gradient_mip_bias_add, \
										coordinate_mode=CM[not inverse], separate_camera_batch=False) # V1DHWC
								dx.append(tf.reduce_sum(dx_views, axis=0))#1DHWC
							if batch>1:
								dx = tf.concat(dx, axis=0) #NDHWC
							else:
								dx = dx[0]
						return dx #[dx, None, None, None, None, None]
					
					return y, grad #NVDHWC, g(NVDHWC)->NDHWC
			
			with self.profiler.sample('Sampling kernel'):
				if self.sample_gradients:
					sampled = __sample(data, M, V, P, F, transformations[0].grid_size if inverse else cameras[0].transform.grid_size)
				else:
					sampled = sampling_ops.sample_grid_transform(input=data, matrix_m=M, matrix_v=V, matrix_p=P, frustum_params=F, output_shape=transformations[0].grid_size if inverse else cameras[0].transform.grid_size, \
						interpolation = self.filter_mode, boundary=self.boundary_mode, mipmapping=self.mip_mode, num_mipmaps=self.num_mips, mip_bias=self.mip_bias, \
						coordinate_mode=CM[inverse])
		return sampled
	
	# sample transform cameras using cached lookup tables computed from the tranformation matrices
	# gives speedup for static MVP - grid setups (static scene and camera, as is common for training) at the cost of memory
	def _sample_camera_LuT(self, data, transformation, cameras, inverse=False):
		with self.profiler.sample('sample_camera_LuT'):
			
			if self.sample_gradients:
				@tf.custom_gradient
				def __sample(x, out_shape): #NDHWC, N44, V44, V44, V6, 3(:DHW-out)
					luts = self.__get_cam_luts(transformation, out_shape, cameras, inverse)
					with self.profiler.sample('Sampling kernel'):
						y = sampling_ops.sample_grid_lut(input=x, lookup=luts, interpolation = self.filter_mode, boundary=self.boundary_mode, \
							mipmapping=self.mip_mode, num_mipmaps=self.num_mips, mip_bias=self.mip_bias, \
							coordinate_mode='LOOKUP', relative_coords=False, normalized_coords=False)
					
					def grad(dy): #NVDHWC
						with self.profiler.sample('Sampling gradients kernel'):
							# no batch support here
							if tf.shape(dy)[0].numpy() != 1: raise NotImplementedError('camera lut rendering does not support data batches.', tf.shape(dy)[0].numpy())
							dy = dy[0] # N - VDHWC
							luts = self.__get_cam_luts(transformation, out_shape, cameras, not inverse)
							dx = sampling_ops.sample_grid_lut(input=dy, lookup=luts, interpolation = self.filter_mode, boundary=self.boundary_mode, \
									mipmapping=self.mip_mode, num_mipmaps=self.num_mips, mip_bias=self.mip_bias + self.gradient_mip_bias_add, \
									coordinate_mode='LOOKUP', separate_camera_batch=False, relative_coords=False, normalized_coords=False) # V1DHWC
							dx = tf.reduce_sum(dx, axis=0)#1DHWC
						return dx #[dx, None, None, None, None, None]
					
					return y, grad #NVDHWC, g(NVDHWC)->NDHWC
				
			with self.profiler.sample('Sampling kernel'):
				if self.sample_gradients:
					sampled = __sample(data, cameras[0].transform.grid_size)
				else:
					luts = self.__get_cam_luts(transformation, cameras[0].transform.grid_size, cameras, inverse)
					sampled = sampling_ops.sample_grid_lut(input=data, lookup=luts, \
						interpolation = self.filter_mode, boundary=self.boundary_mode, mipmapping=self.mip_mode, num_mipmaps=self.num_mips, mip_bias=self.mip_bias, \
						coordinate_mode='LOOKUP', relative_coords=False, normalized_coords=False)
		return sampled
	
	def _raymarch_camera_transform(self, data, transformations, cameras, **kwargs):
		with self.profiler.sample('raymarch_camera_transform'):
			M = [t.get_transform_matrix().transpose() for t in transformations]
			V, P, F = self._get_camera_params_batch(cameras)
			
			with self.profiler.sample('Sampling kernel'):
				sampled = raymarching_ops.raymarch_grid_transform(input=data, matrix_m=M, matrix_v=V, matrix_p=P, frustum_params=F, \
					output_shape=cameras[0].transform.grid_size, \
					interpolation=kwargs.get("filter_mode", self.filter_mode), \
					boundary=kwargs.get("boundary_mode", self.boundary_mode), \
					blending_mode=kwargs.get("blend_mode", self.blend_mode))
		#self.log.info("raymarch result shape: %s", shape_list(sampled))
		return sampled
		
	def sample_camera(self, data, transformations, cameras, inverse=False, allow_static=True, force_static=False, use_step_channel=None):
		#check data
		data_shape = tf.shape(data).numpy()
		if not len(data_shape)==5: raise ValueError('data must be 5D (NDHWC)')
		if not isinstance(transformations, Iterable):
			transformations =[transformations]
			if data_shape[0]!=1: raise ValueError('transformation and data batch size mismatch.')
			no_batch = True
		else:
			if len(transformations)!=data_shape[0]: raise ValueError('transformation and data batch size mismatch.')
			no_batch = False
		
		# check cameras
		if not isinstance(cameras, Iterable): cameras = [cameras] #compat
		if len(cameras)>1 and not all((cam.transform.grid_size==cameras[0].transform.grid_size for cam in cameras[1:])):
			raise ValueError('all cameras must have the same resolution (DHW). (use Renderer._sort_cameras() for batching.)')
		cam_size = cameras[0].transform.grid_size
		#check static rendering (precomputed LuT)
		sample_lut = False
		if allow_static and any((cam.static is not None for cam in cameras)):
			if not all((cam.static==transformations[0] for cam in cameras)):
				if force_static: raise ValueError('Camera static setup does not match transformation')
				else: self.log.warning('Incorrect static camera setup, falling back to transform rendering for static cameras.')
			else:
				if not no_batch:
					if force_static: raise ValueError('Static cameras only work without data batch.')
					else: self.log.warning('Incorrect static camera setup, falling back to transform rendering for static cameras.')
				else: sample_lut=True
		
		apply_step_channel = False
		if use_step_channel is not None and use_step_channel!=[]:
			if np.isscalar(use_step_channel):
				data = data * use_step_channel
			else:
				step_channel = [_%data_shape[-1] for _ in use_step_channel if ((-data_shape[-1]) <= _ and _ < data_shape[-1])]
				step_channel = sorted(step_channel)
				depth_steps = [cam.depth_step for cam in cameras]
				#if: same for every camera, can multiply before sampling (?). should be fine with lerp and grid before is usually smaller
				if np.all([step==depth_steps[0] for step in depth_steps]):
				#	if: every channel is included -> premultiply with scalar
					if step_channel==list(range(data_shape[-1])):
						data = data * depth_steps[0]
				#	else -> premultiply with channel vector
					else:
						data = data * tf.constant([(depth_steps[0] if _ in step_channel else 1) for _ in range(data_shape[-1])], dtype=tf.float32)
				#else -> multiply after sampling
				else:
					apply_step_channel = True
		
		if sample_lut:
			sampled = self._sample_camera_LuT(data, transformations[0], cameras, inverse)
		else:
			sampled = self._sample_camera_transform(data, transformations, cameras, inverse) #NVDHWC
		
		
		# depth-step size correction
		if apply_step_channel:
			shape = shape_list(sampled)
			step = tf.constant([[(depth_step if _ in step_channel else no_step) for _ in range(shape[-1])] for depth_step in depth_steps], dtype=tf.float32) #VC
			step = tf.reshape(step, (1,shape[-5],1,1,1,shape[-1])) #NVDHWC
			shape[-1]=1
			shape[-5]=1
			step = tf.tile(step, shape)
			sampled = sampled * tf.stop_gradient(step)
		
		
		return tf.squeeze(sampled, 0) if no_batch else sampled
	
	def raymarch_camera(self, data, transformations, cameras, use_step_channel=None, **raymarch_kwargs):
		#check data
		data_shape = tf.shape(data).numpy()
		if not len(data_shape)==5: raise ValueError('data must be 5D (NDHWC)')
		if not isinstance(transformations, Iterable):
			transformations =[transformations]
			if data_shape[0]!=1: raise ValueError('transformation and data batch size mismatch.')
			no_batch = True
		else:
			if len(transformations)!=data_shape[0]: raise ValueError('transformation and data batch size mismatch.')
			no_batch = False
		
		# check cameras
		if not isinstance(cameras, Iterable): cameras = [cameras] #compat
		if len(cameras)>1 and not all((cam.transform.grid_size==cameras[0].transform.grid_size for cam in cameras[1:])):
			raise ValueError('all cameras must have the same resolution (DHW). (use Renderer._sort_cameras() for batching.)')
		
		if use_step_channel is not None and use_step_channel!=[]:
			if np.isscalar(use_step_channel):
				data = data * use_step_channel
			else:
				step_channel = [_%data_shape[-1] for _ in use_step_channel if ((-data_shape[-1]) <= _ and _ < data_shape[-1])]
				step_channel = sorted(step_channel)
				depth_steps = [cam.depth_step for cam in cameras]
				#if: same for every camera, can multiply before sampling (?). should be fine with lerp and grid before is usually smaller
				if np.all([step==depth_steps[0] for step in depth_steps]):
				#	if: every channel is included -> premultiply with scalar
					if step_channel==list(range(data_shape[-1])):
						data = data * depth_steps[0]
				#	else -> premultiply with channel vector
					else:
						data = data * tf.constant([(depth_steps[0] if _ in step_channel else 1) for _ in range(data_shape[-1])], dtype=tf.float32)
				#else -> multiply after sampling
				else:
					raise ValueError("All cameras must have the same step size for batched rendering.")
		
		sampled = self._raymarch_camera_transform(data, transformations, cameras, **raymarch_kwargs)
		
		return tf.squeeze(sampled, 0) if no_batch else sampled
	
	# simple 3D lookup sampling (e.g. for warping)
	# lookup coordinates are absolute and not normalized
	def _sample_LuT(self, data, luts, combined_batch=False, relative=False, normalized=False, cell_center_offset=0.0):
		with self.profiler.sample('sample_LuT'):
			luts_shape = GridShape.from_tensor(luts) #shape_list(luts)
			if luts_shape.c==3: #[-1]==3: #pad LoD 0
				with self.profiler.sample('auto pad lod'):
					lut_pad = [[0,0] for _ in range(len(luts_shape))]
					lut_pad[-1][-1] = 1
					luts = tf.pad(luts, lut_pad, "CONSTANT", constant_values=0, name="auto_pad_lut_lod")
			luts_shape = GridShape.from_tensor(luts)
			if luts_shape.c!=4:
				raise ValueError
			with self.profiler.sample('Sampling kernel'):
				resampled = sampling_ops.sample_grid_lut(input=data, lookup=luts, \
					interpolation = self.filter_mode, boundary=self.boundary_mode, mipmapping=self.mip_mode, num_mipmaps=self.num_mips, mip_bias=self.mip_bias, \
					coordinate_mode='LOOKUP', separate_camera_batch= not combined_batch, cell_center_offset=cell_center_offset, relative_coords=relative, normalized_coords=normalized)
		return tf.squeeze(resampled, 1) if combined_batch else resampled
	
	
	def _blend_grid(self, data, blend_mode=None, keep_dims=False):
		'''blends the grid along z / depth according to blend mode
		
		Blend modes:
		MAX, MIN, MEAN: the according reduction operations
		ADDITIVE: sum-reduction
		BEER_LAMBERT: Beer-Lambert without self-attenuation
			N.B. returned density is the normal cumulative sum, while the attenuation uses the exlusive cumulative sum.
		BEER_LAMBERT_SELF: Beer-Lambert with self-attenuation (i.e. non-exlusive cumulative sum for attenuation)
		
		Args:
			data (tf.Tensor): the light/density grid to blend, shape NDHWC
			blend_mode (str): one of MAX, MIN, MEAN, BEER_LAMBERT, BEER_LAMBERT_SELF, ADDITIVE
			keep_dims: keeps the depth dimesion int the output.
				For Min, MAX, MEAN it is 1.
				For BEER_LAMBERT, BEER_LAMBERT_SELF, ADDITIVE it is the original depth D.
		Returns:
			tf.Tensor: blended grid with shape: NDHWC if keep_dims else NHWC
		'''
		with self.profiler.sample('_blend_grid'):
			if blend_mode is None: blend_mode = self.blend_mode
			
			if blend_mode.upper()=='MAX':
				return tf.reduce_max(data, axis=-4, keepdims=keep_dims)
			elif blend_mode.upper()=='MEAN':
				return tf.reduce_mean(data, axis=-4, keepdims=keep_dims)
			elif blend_mode.upper()=='MIN':
				return tf.reduce_min(data, axis=-4, keepdims=keep_dims)
			elif blend_mode.upper()=='BEER_LAMBERT': #BEER_LAMBERT_SELF with cell self-attenuation
				grid_shape = GridShape.from_tensor(data)
				if grid_shape.c==1: #blending for density only
					return tf.math.cumsum(data, axis=-4, exclusive=False) if keep_dims else tf.reduce_sum(data, axis=-4)
				
				light, density = tf.split(data, [grid_shape.c-1,1], axis=-1)
				dens_sum = tf.math.cumsum(density, axis=-4, exclusive=False)
				if keep_dims:
					return tf.concat([tf.math.cumsum(light * tf.math.exp(-dens_sum), axis=-4, exclusive=False), dens_sum], axis=-1)
				else:
					return tf.concat([tf.reduce_sum(light * tf.math.exp(-dens_sum), axis=-4), dens_sum[...,-1,:,:,:]], axis=-1)
					
			elif False: #blend_mode.upper()=='BEER_LAMBERT': # without cell self-attenuation
				grid_shape = GridShape.from_tensor(data)
				if grid_shape.c==1: #blending for density only
					return tf.math.cumsum(data, axis=-4, exclusive=True) if keep_dims else tf.reduce_sum(data, axis=-4)
				
				light, density = tf.split(data, [grid_shape.c-1,1], axis=-1)
				dens_sum = tf.math.cumsum(density, axis=-4, exclusive=True)
				if keep_dims:
					return tf.concat([tf.math.cumsum(light * tf.math.exp(-dens_sum), axis=-4, exclusive=False), dens_sum], axis=-1)
				else:
					return tf.concat([tf.reduce_sum(light * tf.math.exp(-dens_sum), axis=-4), tf.reduce_sum(density, axis=-4, keepdims=keep_dims)], axis=-1)
			
			
			@tf.custom_gradient
			def __blend(x):
				with self.profiler.sample('Blending kernel'):
					y = blending_ops.reduce_grid_blend(x, blend_mode, keep_dims)
				def grad(dy):
					with self.profiler.sample('Blending gradients kernel'):
						dx = blending_ops.reduce_grid_blend_grad(dy, y, x, blend_mode, keep_dims)
					return dx
				return y, grad
			
			return __blend(data)
	
	def _tonemap(self, image, mode='NONE', **kwargs):
		mode = mode.upper()
		if mode=='NONE':
			image_sdr = image
		elif mode=='CLIP_NEGATIVE':
			image_sdr = tf.maximum(image, 0)
		elif mode=='SATURATE':
			image_sdr = tf.clip_by_value(image, 0, 1)
		elif mode=='NORMALIZE':
			min = tf.min(image)
			if min<0: image -= min
			max = tf.max(image)
			if max>0: image_sdr = image/max
		return image_sdr
	
	def _apply_custom_ops(self, tensor, custom_ops, name):
		result = tensor
		if custom_ops is not None and name in custom_ops:
			ops = custom_ops[name]
			if isinstance(ops, Iterable):
				for op in ops:
					result = op(result)
			else:
				result = ops(result)
		return result
		
	#render a batch of cameras with the same size/resolution and scene
	def _render_cameras(self, light_density, grid_transform, cameras, camera_size, custom_ops=None):
		light_density = self._apply_custom_ops(light_density, custom_ops, "GRID")
		if self.can_render_fused and ((custom_ops is None) or ("FRUSTUM" not in custom_ops)):
			with self.profiler.sample('Sampling & Reduction'):
				image_hdr = self.raymarch_camera(light_density, grid_transform, cameras, use_step_channel=[0,1,2,3] if self.blend_mode not in ['MAX','MEAN','MIN'] else None)
		else:
			with self.profiler.sample('Sampling'):
				frustum_grid = self.sample_camera(light_density, grid_transform, cameras, False, use_step_channel=[0,1,2,3] if self.blend_mode not in ['MAX','MEAN','MIN'] else None)
			frustum_grid = self._apply_custom_ops(frustum_grid, custom_ops, "FRUSTUM")
			with self.profiler.sample('Reduction'):
				image_hdr = self._blend_grid(frustum_grid, self.blend_mode)
		image_hdr = self._apply_custom_ops(image_hdr, custom_ops, "IMAGE")
		#self.log.info("image shape: %s", shape_list(image_hdr))
		return image_hdr
	
	def _volume_scatter(self, *, density, light_in, **kwargs):
		return density * (self.scattering_ratio * light_in)
	
	def _build_light_grid(self, density_transform, light_list, monochrome=False):
		with self.profiler.sample('Lighting'):
			self.log.debug('render lights: %d', len(light_list))
			if len(light_list)==0:
				raise ValueError('light list is empty')
			light_shape = density_transform.grid_shape
			light_shape.c = 1 if monochrome else 3
			light_shape.n = 1
			light_data = tf.zeros(light_shape._value, dtype=tf.float32)
			i=0
			for light in light_list:
				if isinstance(light, Light):
					with self.profiler.sample('Light {}: {}'.format(i, type(light).__name__)):
						light_grid = light.grid_lighting(density_transform, self) # scattering_func=self._volume_scatter
						if monochrome and not light.monochrome: #RGB light to greyscale
							light_grid = tf.reduce_sum(light_grid*self.luma, axis=-1, keep_dims=True)
						if not monochrome and light.monochrome: #greyscale light to RGB
							light_grid = tf.broadcast_to(light_grid, light_shape._value)
				elif isinstance(light, (tf.Tensor, np.ndarray)):
					ls = GridShape.from_tensor(light)
					if ls.c==3 and monochrome:
						light = tf.reduce_sum(light*self.luma, axis=-1, keep_dims=True)
					light_grid =  tf.broadcast_to(light, light_shape._value)
				else:
					raise ValueError("Type of light {} '{}' is not supported.".format(i, type(light).__name__))
				light_data += light_grid
				del light_grid
				i+=1
		return light_data
	
	def render_density(self, density_transform, light_list, camera_list, cut_alpha=True, background=None, monochrome=False, split_cameras=False, custom_ops=None, tonemapping="NONE"):
		
		with self.profiler.sample('Render'):
			## apply lighting to grid
			if custom_ops is not None and "DENSITY" in custom_ops:
				t_density = density_transform.data
				density_transform.set_data(self._apply_custom_ops(t_density, custom_ops, "DENSITY"))
			light_data = self._build_light_grid(density_transform, light_list, monochrome)
			self.log.debug('light shape: %s', tf.shape(light_data))
			data = tf.concat([light_data, density_transform.data], axis=-1)
			del light_data
			## resample to frustum grid
			cam_images = [None]*len(camera_list)
			self.log.debug('render cameras: %d', len(camera_list))
			i=0
			with self.profiler.sample('Render Cameras'):
				with self.profiler.sample('Sort'):
					non_static_cams, static_cams = self._sort_cameras(camera_list)
				cameras = non_static_cams + static_cams
				for cam_size, cams in cameras:
					with self.profiler.sample('Size {} x{}'.format(cam_size, len(cams))):
						if split_cameras:
							images = []
							for cam in cams:
								images.append(self._render_cameras(data, density_transform, [cam], cam_size, custom_ops=custom_ops))
							images = tf.concat(images, axis=0)
						else:
							images = self._render_cameras(data, density_transform, cams, cam_size, custom_ops=custom_ops)
							
						t = tf.exp(-images[...,-1:])
						if cut_alpha:
							images = images[...,:-1]
						img_shape = shape_list(images)
						if background is not None:
							cam_batch_bkgs = tf.stack([tf.broadcast_to(background[camera_list.index(cam)], img_shape[1:]) for cam in cams])
							images += cam_batch_bkgs * t
						with self.profiler.sample('Tonemapping (%s)'%tonemapping):
							images = self._tonemap(images, mode=tonemapping)
						#reorder rendered images to match order of input cameras
						images = tf.split(images, len(cams))
						for img, cam in zip(images, cams):
							if cam.scissor_pad is not None:
								img = tf.pad(img, [(0,0)] + list(cam.scissor_pad))
							cam_images[camera_list.index(cam)] = img
						del images
					i+=1
			if custom_ops is not None and "DENSITY" in custom_ops:
				density_transform.set_data(t_density)
			return cam_images
	
	def resample_grid3D_aligned(self, data, target_shape, align_x='BORDER', align_y='BORDER', align_z='BORDER'):
		'''
		align_: alignment for each dimension, string
			BORDER: align outer cell borders
			CENTER: align outer cell centers
			STAGGER_INPUT: align input center to output border
			STAGGER_OUTPUT: align input border to output center
		wraps _sample_transform, which uses the render sampling with orthogonal projection.
		'''
		in_shape = data.get_shape().as_list()
		assert len(in_shape)== 5, "input shape must be NDHWC, is : {}".format(in_shape)
		assert len(target_shape)==3
		in_shape = GridShape.from_tensor(data)
		target_shape = GridShape(target_shape)
		batch = in_shape.n
		
		#_sample_transform is currently experimental and assumes the output grid to be in a centered [-1,1] cube, so scale input accordingly
		# scale with output shape to get the right offset, depending on alignment for that axis
		def get_scale(align, in_size, out_size):
			if align.upper() == 'BORDER':
				scale = 2./in_size
			elif align.upper() == 'CENTER': #scale to align target corner centers, then scale to target border
				scale = 2./out_size * (out_size-1.)/(in_size-1.)
			elif align.upper() == 'STAGGER_INPUT': #align input center to output border
				scale = 2./(in_size-1.)
			elif align.upper() == 'STAGGER_OUTPUT': #align input border to output center
				scale = 2./(in_size+1.)
			else:
				raise ValueError("unknown alignment {}".format(align))
			return scale
		in_scale = [
			get_scale(align_x, in_shape.x, target_shape.x),
			get_scale(align_y, in_shape.y, target_shape.y),
			get_scale(align_z, in_shape.z, target_shape.z),
		]
		
		in_transform = GridTransform(in_shape.zyx.value, scale=in_scale, center=True)
		# only shape important here
		target_transform = GridTransform(target_shape.zyx.value)
		return tf.squeeze(self._sample_transform(data, [in_transform]*batch, [target_transform]),1)
	
	
	def resample_grid3D_offset(self, data, offsets, target_shape):
		if not isinstance(offsets, tf.Tensor):
			offsets = tf.constant(offsets, dtype=tf.float32)
		offsets_shape = shape_list(offsets)
		if len(offsets_shape)!=2 or offsets_shape[1]!=3:
			raise ValueError("Shape of offsets must be (N,3), is %s"%offsets_shape)
		if len(target_shape)!=3:
			raise ValueError("target_shape must be (3,), is %s"%target_shape)
		data_shape = GridShape.from_tensor(data)
		
		offsets = tf.pad(offsets, ((0,0),(0,1))) #pad to (N,4)
		offsets = tf.reshape(offsets, [offsets_shape[0],1,1,1,4])
		offsets = tf.broadcast_to(offsets, [offsets_shape[0]]+target_shape+[4])
		
		return self._sample_LuT(data, offsets, relative=True)
	
	
	def unproject(self, grid_transform, targets, cameras, blend_func=tf.minimum):
		#--- Volume Estimation ---
		if self.allow_fused and False:
			inp = tf.zeros(grid_transform.grid_shape._value, dtype=tf.float32)
			M = [grid_transform.get_transform_matrix().transpose()]
			V, P, F = self._get_camera_params_batch(cameras)
			tar = tf.expand_dims(targets, 0)
			# use the gradient operation to scatter the targets into the volume. contents of input and output are irrelevant when using ADDITIVE blending
			unprojections = raymarching_ops.raymarch_grid_transform_grad(input=inp, output=tar, output_grad=tar, matrix_m=M, matrix_v=V, matrix_p=P, frustum_params=F, \
				output_shape=[cameras[0].transform.grid_size[0]] + list(targets.shape)[1:3],\
				interpolation=self.filter_mode, boundary=self.boundary_mode, blending_mode="ADDITIVE")
			#targets are [0|1], gradients (here the targets) are summed over the cameras. thus, the voxels seen by all cameras should have the value of the number of cameras.
			unprojections = (unprojections - (len(cameras)-0.5)) * 2
		else:
			unprojections = tf.ones(grid_transform.grid_shape._value, dtype=tf.float32)
			for i in range(len(cameras)):
				cam = cameras[i]
				#expand target to frustum volume (tile along z)
				tar = tf.reshape(targets[i], [1,1] + list(targets[i].shape))
				tar = tf.tile(tar, (1,cam.transform.grid_size[0],1,1,1))
				#sample target to shared volume
				unprojection = self.sample_camera(tar, grid_transform, cam, inverse=True)
				unprojections = blend_func(unprojections, unprojection)
			#unprojection = blend_func(unprojections, axis=0)
		return unprojections
	
	def visual_hull(self, grid_transform, targets, cameras, image_blur=0.0, grid_blur=0.0, threshold=0.5, soft_blur=0.0):
		
		target_hulls = tf_data_gaussDown2D(targets, image_blur, stride=1, channel=1, padding='SAME') if image_blur>0 else targets
		target_hulls = tf.cast(tf.greater_equal(target_hulls, threshold), tf.float32)
		
		hull = self.unproject(grid_transform, target_hulls, cameras)
		
		hull = tf_data_gaussDown3D(hull, grid_blur, stride=1, channel=1, padding='SAME') if grid_blur>0 else hull
		hull = tf.cast(tf.greater_equal(hull, threshold), tf.float32)
		
		if soft_blur>0:
			hull = tf_data_gaussDown3D(hull, soft_blur, stride=1, channel=1, padding='SAME')
		
		target_hulls = self.project_hull(hull, grid_transform, cameras)
		
		return hull, target_hulls
	
	def project_hull(self, hull, grid_transform, cameras):
		if self.allow_fused and False:
			target_hulls = self.raymarch_camera(hull, [grid_transform], cameras, blend_mode="ADDITIVE")
			target_hulls = tf.cast(tf.greater_equal(target_hulls, 0.5), tf.float32)
			return target_hulls
		else:
			target_hulls = []
			for camera in cameras:
				target_hull = tf.squeeze(self._sample_camera_transform(hull, [grid_transform], [camera]), axis=0)
				target_hull = self._blend_grid(target_hull, blend_mode='MAX')
				target_hulls.append(target_hull)
			return tf.concat(target_hulls, axis=0)
	
	def write_images(self, image_batches, file_masks, base_path=None, frame_id=None, use_batch_id=False, format='EXR', y_flip=True, gamma=1.0):
		formats = {'EXR':'.exr', 'PNG':'.png'}
		os.makedirs(base_path, exist_ok=True)
		for image_batch, file_mask in zip(image_batches, file_masks):
			i=0
			for image in image_batch:
				image_shape = shape_list(image)
				if y_flip:# y-flip
					image = tf.reverse(image, axis=[-3])
				if image_shape[-1]==2:
					image = tf.pad(image, [[0,0] for _ in range(len(image_shape)-1)] + [[0,1]])
					image_shape = shape_list(image)
				if frame_id is not None and use_batch_id:
					file_name = file_mask.format(i, frame_id)
				elif frame_id is not None and not use_batch_id:
					file_name = file_mask.format(frame_id)
				elif frame_id is None and use_batch_id:
					file_name = file_mask.format(i)
				else:
					file_name = file_mask
				file_name += formats[format]
				path = os.path.join(base_path, file_name) if base_path is not None else file_name
			#projected_grads = np.flip(projected_grads, axis=0)
				if gamma!=1.0:
					image = gammaCorrection(image, gamma)
				if format=='EXR':
					try:
						imageio.imwrite(path, image, 'EXR-FI')
					except KeyboardInterrupt:
						raise
					except:
						self.log.exception("Failed to write exr image with shape %s to '%s':", image.get_shape().as_list(), path)
						return
				elif format=='PNG':
					image = (np.clip(image, 0.0, 1.0)*255.0).astype(np.uint8)
					try:
						imageio.imwrite(path, image)
					except KeyboardInterrupt:
						raise
					except:
						self.log.exception("Failed to write png image with shape %s to '%s':", image.get_shape().as_list(), path)
						return
				else:
					raise ValueError('format not supported')
				i+=1
	