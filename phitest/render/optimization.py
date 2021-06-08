import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from lib.util import HistoryBuffer, NO_OP, NO_CONTEXT, lerp, lerp_fast, copy_nested_structure
from lib.tf_ops import save_discriminator, tf_None_to_const, tf_cosine_similarity, tf_pad_to_shape, shape_list # tf_laplace_filter_3d,
from lib.scalar_schedule import scalar_schedule
from .renderer import RenderingContext
from .vector import GridShape
import logging

LOG = logging.getLogger("Optimization")

# --- LOSSES ---
class LossSchedules:
	def __init__(self, *, \
			density_target=lambda i: 0.0, density_target_raw=lambda i: 0.0, density_target_depth_smoothness=lambda i: 0.0, \
			density_hull=lambda i: 0.0, \
			density_negative=lambda i: 0.0, density_smoothness=lambda i: 0.0, density_smoothness_2=lambda i: 0.0, \
			density_smoothness_temporal=lambda i: 0.0, density_warp=lambda i: 0.0, density_disc=lambda i: 0.0, \
			velocity_warp_dens=lambda i: 0.0, velocity_warp_vel=lambda i: 0.0, velocity_divergence=lambda i: 0.0, \
			velocity_smoothness=lambda i: 0.0, velocity_cossim=lambda i: 0.0, velocity_magnitude=lambda i: 0.0, \
			density_lr=lambda i: 0.0, light_lr=lambda i: 0.0, velocity_lr=lambda i: 0.0, discriminator_lr=lambda i: 0.0, discriminator_regularization=lambda i: 0.0\
		):
		self.density_target = density_target
		self.density_target_raw = density_target_raw
		self.density_target_depth_smoothness = density_target_depth_smoothness
		self.density_hull = density_hull
		self.density_negative = density_negative
		self.density_smoothness = density_smoothness
		self.density_smoothness_2 = density_smoothness_2
		self.density_smoothness_temporal = density_smoothness_temporal
		self.density_warp = density_warp
		self.density_disc = density_disc
		
		self.velocity_warp_dens = velocity_warp_dens
		self.velocity_warp_vel = velocity_warp_vel
		self.velocity_divergence = velocity_divergence
		self.velocity_smoothness = velocity_smoothness
		self.velocity_cossim = velocity_cossim
		self.velocity_magnitude = velocity_magnitude
		
		self.density_lr = density_lr
		self.light_lr = light_lr
		self.velocity_lr = velocity_lr
		self.discriminator_lr = discriminator_lr
		
		self.discriminator_regularization = discriminator_regularization
	
	def set_schedules(self, **kwargs):
		for name, value in kwargs.items():
			if not hasattr(self, name):
				raise AttributeError('Loss schedule {} does not exist.'.format(name))
			setattr(self, name, value)
		return self

def scale_losses(losses, scale):
	if isinstance(losses, (list, tuple)):
		return [_ * scale for _ in losses]
	elif isinstance(losses, tf.Tensor):
		return losses * scale
	else:
		raise TypeError

def reduce_losses(losses):
	if isinstance(losses, (list, tuple)):
		return tf.reduce_sum([tf.reduce_mean(_) for _ in losses])
	elif isinstance(losses, tf.Tensor):
		return tf.reduce_mean(losses)
	else:
		raise TypeError

class OptimizationContext:
	def __init__(self, setup, iteration, loss_schedules, \
			rendering_context, vel_scale=[1,1,1], warp_order=1, dt=1.0, buoyancy=None, \
			dens_warp_clamp="NONE", vel_warp_clamp="NONE", \
			density_optimizer=None, density_lr=1.0, light_optimizer=None, light_lr=1.0, \
			velocity_optimizer=None, velocity_lr=1.0, \
			frame=None, tf_summary=None, summary_interval=1, summary_pre=None, profiler=None, light_var_list=[]):
		self.setup = setup
		self.profiler = profiler if profiler is not None else Profiler(active=False)
		self.iteration = iteration
		self.frame = frame
		self._losses = {}
		self.l1_loss = lambda i, t: tf.abs(i-t)
		self.l2_loss = tf.math.squared_difference #lambda i, t: (i-t)**2
		def huber_loss(i, t, delta=1.0):
			#tf.losses.huber_loss does not support broadcasting...
			abs_error = tf.abs(t-i)
			sqr = tf.minimum(abs_error, delta)
			lin = abs_error - sqr
			return (0.5*(sqr*sqr)) + (lin*delta)
		
		self.base_loss_functions = {
		#	'L0.5': lambda i, t: tf.sqrt(tf.abs(i-t)),
		#	'L1': self.l1_loss,
		#	'L2': self.l2_loss,
		#	'L3': lambda i, t: tf.pow(tf.abs(i-t), 3),
			'RAE': lambda i, t: tf.sqrt(tf.abs(i-t)),
			'MRAE': lambda i, t: tf.reduce_mean(tf.sqrt(tf.abs(i-t))),
			'AE': lambda i, t: tf.abs(i-t),
			'SAE': lambda i, t: tf.reduce_sum(tf.abs(i-t)),
			'MAE': lambda i, t: tf.reduce_mean(tf.abs(i-t)),
			'SE': tf.math.squared_difference,
			'SSE': lambda i, t: tf.reduce_sum(tf.math.squared_difference(i,t)),
			'MSE': lambda i, t: tf.reduce_mean(tf.math.squared_difference(i,t)),
			'RMSE': lambda i, t: tf.sqrt(tf.reduce_mean(tf.math.squared_difference(i,t))),
			'CAE': lambda i, t: tf.pow(tf.abs(i-t), 3),
			
			'HUBER': huber_loss, #lambda i,t: tf.losses.huber_loss(predictions=i, labels=t, reduction=tf.losses.Reduction.NONE),
			#'LBE': tf_log_barrier_ext,
		}
		self.default_loss_function = None #self.l2_loss
		self.loss_functions = {
			"density/target":		self.base_loss_functions["AE"], #self.l1_loss,
			"density/target_raw":	self.base_loss_functions["AE"], #self.l1_loss,
			"density/target_depth_smooth":	self.base_loss_functions["SE"], #self.l2_loss,
			"density/hull":			self.base_loss_functions["SE"], #self.l2_loss,
			"density/negative":		self.base_loss_functions["SE"], #self.l2_loss,
			"density/edge":			self.base_loss_functions["SE"], #self.l2_loss,
			"density/smooth":		self.base_loss_functions["SE"], #self.l2_loss,
			"density/smooth-temp":		self.base_loss_functions["SE"], #self.l2_loss,
			"density/warp":			self.base_loss_functions["AE"], #self.l1_loss,
			
			"velocity/density_warp":	self.base_loss_functions["AE"], #self.l1_loss,
			"velocity/velocity_warp":	self.base_loss_functions["AE"], #self.l1_loss,
			"velocity/divergence":		self.base_loss_functions["SE"], #self.l2_loss,
			"velocity/magnitude":		self.base_loss_functions["SE"], #self.l2_loss,
			"velocity/smooth":			self.base_loss_functions["SE"],
			"velocity/cossim":			self.base_loss_functions["SE"],
		}
		self.loss_schedules = loss_schedules
		self.density_optimizer = density_optimizer
		self.density_lr = density_lr
		self.light_optimizer = light_optimizer
		self.light_lr = light_lr
		self.velocity_optimizer = velocity_optimizer
		self.velocity_lr = velocity_lr
		self._loss_summary = {}
		self._tf_summary = tf_summary
		self._summary_interval = summary_interval
		self.summary_pre = summary_pre
		self._compute_loss_summary = False
		
		self._target_weights = None
		self._target_weights_norm = 1.0
		
		self.render_ctx = rendering_context
		self.render_ops = {}
		self.vel_scale = vel_scale
		self.buoyancy = buoyancy
		self.warp_order = warp_order
		self.dens_warp_clamp = dens_warp_clamp
		self.vel_warp_clamp = vel_warp_clamp
		self.dt = dt
		self.light_var_list = light_var_list
		
		self.warp_dens_grads = False
		self.warp_dens_grads_decay = 0.9
		self.warp_vel_grads = False
		self.warp_vel_grads_decay = 0.9
		self.custom_dens_grads_weight = 1.0
		self.custom_vel_grads_weight = 1.0
		
		self._gradient_tape = None
		
		self.inspect_gradients = False
		self.inspect_gradients_func = NO_OP
		self.inspect_gradients_images_func = NO_OP
		self.inspect_gradients_images = {}
	
	def start_iteration(self, iteration, force=False, compute_loss_summary=False):
		'''Reset losses and set iteration
			will do nothing if iteration is already set
		'''
		self._compute_loss_summary = compute_loss_summary
		self.set_gradient_tape()
		if self.iteration==iteration and not force:
			return
		LOG.debug("Start iteration %d, update optimization context", iteration)
		self.iteration = iteration
		self._loss_summary = {}
		self._losses = {}
		
		self.density_lr.assign(self.loss_schedules.density_lr(self.iteration))
		self.light_lr.assign(self.loss_schedules.light_lr(self.iteration))
		self.velocity_lr.assign(self.loss_schedules.velocity_lr(self.iteration))
		if self.record_summary:
			summary_names = self.make_summary_names('density/learning_rate')
			self._tf_summary.scalar(summary_names[0], self.density_lr.numpy(), step=self.iteration)
			summary_names = self.make_summary_names('velocity/learning_rate')
			self._tf_summary.scalar(summary_names[0], self.velocity_lr.numpy(), step=self.iteration)
	
	@property
	def target_weights(self):
		return self._target_weights
	@target_weights.setter
	def target_weights(self, weights):
		if weights is None:
			self._target_weights = None
			self._target_weights_norm = 1.0
		else:
			self._target_weights = tf.constant(weights, dtype=tf.float32)[:, np.newaxis, np.newaxis, np.newaxis]
			self._target_weights_norm = tf.constant(1./tf.reduce_sum(self._target_weights), dtype=tf.float32)
	@property
	def target_weights_norm(self):
		return self._target_weights_norm
	
	@property
	def tape(self):
		return self._gradient_tape
	
	def set_gradient_tape(self, tape=None):
		self._gradient_tape = tape
	
	def set_loss_func(self, loss_name, loss_function):
		if callable(loss_function):
			self.loss_functions[loss_name] = loss_function
		elif isinstance(loss_function, str):
			loss_function = loss_function.upper()
			if loss_function in self.base_loss_functions:
				self.loss_functions[loss_name] = self.base_loss_functions[loss_function]
			else:
				raise ValueError("Unknown loss function {} for loss {}".format(loss_function, loss_name))
		else:
			raise TypeError("Invalid loss function for loss {}".format(loss_name))
	
	def get_loss_func(self, loss_name):
		return self.loss_functions.get(loss_name, self.default_loss_function)
	
	def get_losses(self):
		loss_list = []
		for loss_tensors in self._losses.values():
			loss_list.extend(loss_tensors)
		return loss_list
	
	def pop_losses(self):
		'''Return current loss value and reset it to 0'''
		loss = self.get_losses()
		self._losses = {}
		return loss
	
	def pop_loss_summary(self):
		loss_summary = self._loss_summary
		self._loss_summary = {}
		return loss_summary
	
	@property
	def record_summary(self):
		return self._tf_summary is not None and ((self.iteration+1)%self._summary_interval)==0
	
	def compute_loss_summary(self):
		return (self.record_summary or self._compute_loss_summary)
	
	@property
	def scale_density_target(self):
		return self.loss_schedules.density_target(self.iteration)
	
	def CV(self, schedule, it=None):
		'''Current Value of a scalar schedule'''
		if callable(schedule):
			return schedule(self.iteration if it is None else it)
		else: #isinstance(schedule, collection.abs.Mapping) and ('type' in schedule):
			return scalar_schedule(schedule, self.iteration if it is None else it)
	
	def LA(self, loss_scale):
		'''Loss Active'''
		if isinstance(loss_scale, bool):
			return loss_scale
		else:
			return not np.isclose(loss_scale, 0, atol=self.setup.training.loss_active_eps)
	
	def add_loss(self, loss_tensors, loss_value=None, loss_value_scaled=None, loss_scale=None, loss_name=None):
		'''add a loss to the accumulator and write summaries
		change to:
			loss_tensor, for optimization
			loss_value, reduced loss_tensor, for value output
			loss_scale, 
		'''
		#self.loss +=loss
		if isinstance(loss_tensors, (list, tuple)):
			self._losses[loss_name] = loss_tensors
		elif isinstance(loss_tensors, tf.Tensor):
			self._losses[loss_name] = [loss_tensors]
		else:
			raise TypeError
		
		if loss_name is not None:
			self._loss_summary[loss_name] = (loss_value_scaled, loss_value, loss_scale)
			if self.record_summary:
				summary_names = self.make_summary_names(loss_name)
				if loss_value_scaled is not None:
					self._tf_summary.scalar(summary_names[0], loss_value_scaled, step=self.iteration)
				if loss_value is not None:
					self._tf_summary.scalar(summary_names[1], loss_value, step=self.iteration)
				if loss_scale is not None:
					self._tf_summary.scalar(summary_names[2], loss_scale, step=self.iteration)
	
	def get_loss(self, loss_name):
		if loss_name in self._losses:
			return self._losses[loss_name]
		else:
			raise KeyError("Loss '%s' not recorded, available losses: %s"%(loss_name, list(self._losses.keys())))
	
	def add_render_op(self, name, func):
		if not name in self.render_ops: self.render_ops[name] = []
		self.render_ops[name].append(func)
	def remove_render_op(self, name, func):
		if name in self.render_ops:
			try:
				i = self.render_ops[name].index(func)
			except ValueError:
				pass
			else:
				del self.render_ops[name][i]
	def remove_render_ops(self, name):
		if name in self.render_ops:
			del self.render_ops[name]
	
	def RO_grid_dens_grad_scale(self, weight=1.0, sharpness=1.0, eps=1e-5):
	#	LOG.info("SCALE GRAD: init dens grid with weight %s", weight)
		@tf.custom_gradient
		def op(x):
			# input: combined light-density grid NDHWC with C=4 (3 light, 1 dens)
			gs = GridShape.from_tensor(x)
			channel = gs.c
			d = x[...,-1:]
		#	LOG.info("SCALE GRAD: dens grid fwd with shape %s, dens_shape %s", gs, GridShape.from_tensor(d))
			y = tf.identity(x)
			def grad(dy):
				# scale density gradient with exisiting density distribution
				#lerp: (1-w)*dy + w*(dy*(dens/mean(dens)))
				d_s = tf.pow(tf.abs(d), sharpness)
				m = tf.maximum(tf.reduce_max(d_s, axis=[-4,-3,-2], keepdims=True), eps)
				c = (1 - weight) + weight*(d_s/m)
				if channel>1:
					c = tf.pad(c, [(0,0),(0,0),(0,0),(0,0),(channel-1,0)], constant_values=1)
				gs_c = GridShape.from_tensor(c)
				dx = dy * c
				gs_dx = GridShape.from_tensor(dx)
		#		LOG.info("SCALE GRAD: dens grid bwd with shape %s, c-shape %s, weight %s", gs_dx, gs_c, weight)
				return dx
			return y, grad
		return op
	def RO_frustum_dens_grad_scale(self, weight=1.0, sharpness=1.0, eps=1e-5):
		#inspired by 'Single-image Tomography: 3D Volumes from 2D Cranial X-Rays' https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13369
		@tf.custom_gradient
		def op(x):
			# input: sampled frustum grid NDHWC with C=4 (3 light, 1 dens)
			gs = GridShape.from_tensor(x)
			channel = gs.c
			d = x[...,-1:]
			y = tf.identity(x)
			def grad(dy):
				# scale density gradient with exisiting density distribution along view ray (z-axis)
				#lerp: (1-w)*dy + w*(dy*(dens/mean(dens)))
				d_s = tf.pow(tf.abs(d), sharpness)
				m = tf.maximum(tf.reduce_max(d_s, axis=-4, keepdims=True), eps)
			#	dx_c = dy * tf.pad((d/m), [(0,0),(0,0),(0,0),(0,0),(3,0)], constant_values=1)
			#	dx = dy + weight*(dx_c - dy) 
				c = (1 - weight) + weight*(d_s/m)
				if channel>1:
					c = tf.pad(c, [(0,0),(0,0),(0,0),(0,0),(channel-1,0)], constant_values=1)
				return dy * c #tf.pad((1 - weight) + weight*(d_s/m), [(0,0),(0,0),(0,0),(0,0),(3,0)], constant_values=1)
			return y, grad
		return op
	
	def set_inspect_gradient(self, active, func=None, img_func=None):
		self.inspect_gradients_images = {}
		if active:
			self.inspect_gradients = True
			self.inspect_gradients_func = func if func is not None else NO_OP
			self.inspect_gradients_images_func = img_func if img_func is not None else NO_OP
		else:
			self.inspect_gradients = False
			self.inspect_gradients_func = NO_OP
			self.inspect_gradients_images_func = NO_OP
	
	def make_summary_names(self, loss_name):
		summary_name = []
		if self.summary_pre is not None:
			summary_name.append(self.summary_pre)
		summary_name.append(loss_name)
		summary_name.append("{type}")
		if self.frame is not None:
			summary_name.append('f{:04d}'.format(self.frame))
		summary_name = "/".join(summary_name)
		return summary_name.format(type="final"), summary_name.format(type="raw"), summary_name.format(type="scale")
	
	def frame_pre(self, name):
		if self.frame is not None:
			return 'f{:04d}_{}'.format(self.frame, name)
		return name
### Density

def loss_dens_target(ctx, state, loss_func=None):
	# Render loss for density against targets without background
	loss_scale = ctx.scale_density_target
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("density/target")
		with ctx.profiler.sample("target loss"):
			if ctx.target_weights is not None:
				tmp_loss = tf.reduce_sum(loss_func(state.images, state.targets)*ctx.target_weights, axis=0) * ctx.target_weights_norm #weighted mean
			else:
				tmp_loss = tf.reduce_mean(loss_func(state.images, state.targets), axis=0) #mean over batch/cameras to be independent of it
		tmp_loss_scaled = tmp_loss * loss_scale
		if ctx.compute_loss_summary():
			ctx.add_loss([tmp_loss_scaled], tf.reduce_mean(tmp_loss), tf.reduce_mean(tmp_loss_scaled), loss_scale, 'density/target')
		else:
			ctx.add_loss([tmp_loss_scaled], None, None, loss_scale, 'density/target')
		return True
	return False

def loss_dens_target_raw(ctx, state, loss_func=None):
	# Render with blended bkg loss for density against raw targets (with background)
	loss_scale = ctx.loss_schedules.density_target_raw(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("density/target_raw")
		with ctx.profiler.sample("raw target loss"):
			if ctx.target_weights is not None:
				tmp_loss = tf.reduce_sum(loss_func(state.images + state.bkgs*state.t, state.targets_raw)*ctx.target_weights, axis=0) * ctx.target_weights_norm
			else:
				tmp_loss = tf.reduce_mean(loss_func(state.images + state.bkgs*state.t, state.targets_raw), axis=0)
		tmp_loss_scaled = tmp_loss * loss_scale
		if ctx.compute_loss_summary():
			ctx.add_loss([tmp_loss_scaled], tf.reduce_mean(tmp_loss), tf.reduce_mean(tmp_loss_scaled), loss_scale, 'density/target_raw')
		else:
			ctx.add_loss([tmp_loss_scaled], None, None, loss_scale, 'density/target_raw')
		return True
	return False

def loss_dens_target_depth_smooth(ctx, state, frustum_density, loss_func=None):
	# Smoothness loss for density using forward differences (gradient computation of the 3D laplace filter convolution is so slow...)
	loss_scale = ctx.loss_schedules.density_target_depth_smoothness(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("density/target_depth_smooth")
		with ctx.profiler.sample('density target depth gradient loss'):
			tmp_loss = [
				loss_func(frustum_density[:,1:,:,:,:], frustum_density[:,:-1,:,:,:]), #z_grad = d[:,1:,:,:,:] - d[:,:-1,:,:,:]
			]
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, loss_name='density/target_depth_smooth')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, loss_name='density/target_depth_smooth')
		return True
	return False

def loss_dens_hull(ctx, state, loss_func=None):
	""" Loss to reduce density outside the hull: density*(1-hull)
		This loss considers the raw density without the hull applied, even if density.restrict_to_hull is True
	"""
	loss_scale = ctx.loss_schedules.density_hull(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("density/hull")
		with ctx.profiler.sample("density hull loss"):
			tmp_loss = loss_func(state.density._d * (1.-state.density.hull), 0)
		tmp_loss_scaled = tmp_loss * loss_scale
		if ctx.compute_loss_summary():
			ctx.add_loss([tmp_loss_scaled], tf.reduce_mean(tmp_loss), tf.reduce_mean(tmp_loss_scaled), loss_scale, 'density/hull')
		else:
			ctx.add_loss([tmp_loss_scaled], None, None, loss_scale, 'density/hull')
		return True
	return False

def loss_dens_negative(ctx, state, loss_func=None):
	# loss for negative density
	loss_scale = ctx.loss_schedules.density_negative(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("density/negative")
		with ctx.profiler.sample("negative density loss"):
			tmp_loss = tf.reduce_mean(loss_func(tf.maximum(-state.density.d, 0), 0), axis=0)
		tmp_loss_scaled = tmp_loss * loss_scale
		if ctx.compute_loss_summary():
			ctx.add_loss([tmp_loss_scaled], tf.reduce_mean(tmp_loss), tf.reduce_mean(tmp_loss_scaled), loss_scale, 'density/negative')
		else:
			ctx.add_loss([tmp_loss_scaled], None, None, loss_scale, 'density/negative')
		return True
	return False

def loss_dens_smooth(ctx, state, loss_func=None):
	# Smoothness (laplace edge filter) loss for density
	loss_scale = ctx.loss_schedules.density_smoothness(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("density/edge")
		with ctx.profiler.sample('density edge loss'):
			d = state.density.d
			tmp_loss = [
				loss_func(d[:,:,:,:-2,:] - 2* d[:,:,:,1:-1,:] + d[:,:,:,2:,:], 0), #(x-1) - 2(x) + (x+1)
				loss_func(d[:,:,:-2,:,:] - 2* d[:,:,1:-1,:,:] + d[:,:,2:,:,:], 0),
				loss_func(d[:,:-2,:,:,:] - 2* d[:,1:-1,:,:,:] + d[:,2:,:,:,:], 0),
			]
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, loss_name='density/edge')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, loss_name='density/edge')
		return True
	return False

def loss_dens_smooth_2(ctx, state, loss_func=None):
	# Smoothness loss for density using forward differences (gradient computation of the 3D laplace filter convolution is so slow...)
	loss_scale = ctx.loss_schedules.density_smoothness_2(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("density/smooth")
		with ctx.profiler.sample('density gradient loss'):
			d = state.density.d
			tmp_loss = [
				loss_func(d[:,:,:,1:,:] - d[:,:,:,:-1,:], 0), #x_grad = d[:,:,:,1:,:] - d[:,:,:,:-1,:]
				loss_func(d[:,:,1:,:,:] - d[:,:,:-1,:,:], 0), #y_grad = d[:,:,1:,:,:] - d[:,:,:-1,:,:]
				loss_func(d[:,1:,:,:,:] - d[:,:-1,:,:,:], 0), #z_grad = d[:,1:,:,:,:] - d[:,:-1,:,:,:]
			]
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, loss_name='density/smooth')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, loss_name='density/smooth')
		return True
	return False

def loss_dens_smooth_temporal(ctx, state, loss_func=None):
	# Temoral smoothness loss for density, for tomofluid tests
	loss_scale = ctx.loss_schedules.density_smoothness_temporal(ctx.iteration)
	if ctx.LA(loss_scale) and (state.prev is not None or state.next is not None):
		if loss_func is None: loss_func = ctx.get_loss_func("density/smooth-temp")
		with ctx.profiler.sample('density temporal gradient loss'):
			d = state.density.d
			tmp_loss = 0
			if state.prev is not None:
				tmp_loss += loss_func(state.prev.density.d, d)
			if state.next is not None:
				tmp_loss += loss_func(d, state.next.density.d)
		tmp_loss_scaled = tmp_loss * loss_scale
		if ctx.compute_loss_summary():
			ctx.add_loss([tmp_loss_scaled], tf.reduce_mean(tmp_loss), tf.reduce_mean(tmp_loss_scaled), loss_scale, 'density/smooth-temp')
		else:
			ctx.add_loss([tmp_loss_scaled], None, None, loss_scale, 'density/smooth-temp')
		return True
	return False

def loss_dens_warp(ctx, state, loss_func=None):
	#warp loss "loss(A(dt, vt), dt+1)" between prev and current and current and next state for density.
	# will scale the velocity to match the density shape/resolution
	loss_scale = ctx.loss_schedules.density_warp(ctx.iteration)
	if ctx.LA(loss_scale) and (state.prev is not None or state.next is not None):
		if loss_func is None: loss_func = ctx.get_loss_func("density/warp")
		tmp_loss = 0
		with ctx.profiler.sample('density warp loss'):
			if state.prev is not None:
				tmp_loss += loss_func(state.prev.density_advected(order=ctx.warp_order, dt=ctx.dt, clamp=ctx.dens_warp_clamp), state.density.d)
			if state.next is not None:
				tmp_loss += loss_func(state.density_advected(order=ctx.warp_order, dt=ctx.dt, clamp=ctx.dens_warp_clamp), state.next.density.d)
		tmp_loss_scaled = tmp_loss * loss_scale
		if ctx.compute_loss_summary():
			ctx.add_loss([tmp_loss_scaled], tf.reduce_mean(tmp_loss), tf.reduce_mean(tmp_loss_scaled), loss_scale, 'density/warp')
		else:
			ctx.add_loss([tmp_loss_scaled], None, None, loss_scale, 'density/warp')
		return True
	return False

#randomize_rot_cams(disc_cameras, [-30,30], [0,360])
def loss_dens_disc(ctx, state, disc, img_list=None):
	# loss from the discriminator for the density
	if ctx.setup.training.discriminator.active and ctx.setup.training.discriminator.start_delay<=ctx.iteration:
		loss_scale = ctx.loss_schedules.density_disc(ctx.iteration-ctx.setup.training.discriminator.start_delay)
		loss_active = ctx.LA(loss_scale)
		#randomize_rot_cams(disc_cameras, [-30,30], [0,360])
		if loss_active or disc.record_history: #only render if needed for history or loss
			LOG.debug('Render discriminator input for density loss')
			disc_in = disc.fake_samples(state, history_samples=False, concat=False, spatial_augment=False, name="dens_disc_samples")
		#		
			if img_list is not None and isinstance(img_list, list):
				img_list.extend(disc_in)
			if loss_active:
				LOG.debug('Run discriminator loss for density')
				with ctx.profiler.sample("dens disc loss"):
					disc_in = tf.concat(disc_in, axis=0)
					if ctx.inspect_gradients:
						ctx.inspect_gradients_images['density/discriminator'] = disc_in
					#disc_in = disc.check_input(disc_in, "dens_disc")
					disc_in = (disc_in,)if (disc.loss_type in ["SGAN"]) else (disc.real_samples(spatial_augment=True, intensity_augment=True),disc_in)
					tmp_loss, disc_scores = disc.loss(disc_in, flip_target=not (disc.loss_type in ["SGAN"]), training=False)
				tmp_loss_scaled = scale_losses(tmp_loss, loss_scale)
				if ctx.compute_loss_summary():
					ctx.add_loss(tmp_loss_scaled, reduce_losses(tmp_loss), reduce_losses(tmp_loss_scaled), loss_scale, 'density/discriminator')
				else:
					ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'density/discriminator')
				return True
		#END disc render or density loss
	return False

def warp_dens_grads(opt_ctx, state, grads, order='FWD'):
	if order.upper()=='FWD': #propagate density gradients to next state, simple forward warp. do not clamp negative gradients
		return state.prev.velocity.warp(grads, order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp="NONE" if opt_ctx.dens_warp_clamp=="NEGATIVE" else opt_ctx.dens_warp_clamp), tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)
	elif order.upper()=='BWD': #propagate density gradients to previous state, backprop through prev->warp
		var_list = state.get_variables()
		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(var_list)
			d_warp = state.density_advected(order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=opt_ctx.dens_warp_clamp)
		return tape.gradient([d_warp], var_list, output_gradients=[grads])
	else:
		raise ValueError

def apply_custom_grads_with_check(grad, custom_grad, custom_grad_scale=1.):
	if grad is None and custom_grad is None:
		LOG.warning("apply_custom_grads: base and custom gradient is None.")
	
	if grad is None:
		#LOG.warning("apply_custom_grads: base gradient is None.")
		grad = 0.
	if custom_grad is None:
		#LOG.warning("apply_custom_grads: custom gradient is None.")
		custom_grad = 0.
	
	return grad + custom_grad * custom_grad_scale

def optStep_density(opt_ctx, state, use_vel=False, disc_ctx=None, disc_samples_list=None, custom_dens_grads=None, apply_dens_grads=True):
	dens_vars = state.density.get_variables()
	with opt_ctx.profiler.sample('optStep_density'):
		with opt_ctx.profiler.sample('loss'), tf.GradientTape(watch_accessed_variables=False, persistent=opt_ctx.inspect_gradients) as dens_tape:
			dens_tape.watch(dens_vars)
			if opt_ctx.light_var_list:
				dens_tape.watch(opt_ctx.light_var_list)
			opt_ctx.set_gradient_tape(dens_tape)
			
			catch_frustum_grid = opt_ctx.LA(opt_ctx.loss_schedules.density_target_depth_smoothness(opt_ctx.iteration))
			fg_container = []
			if catch_frustum_grid:
				# use the custom render op hooks to catch the reference to the frutum grid tensor
				def _catch_FG(fg):
					fg_container.append(fg)
					return fg
				opt_ctx.add_render_op("FRUSTUM", _catch_FG)
			else:
				fg_container.append(None)
			state.render_density(opt_ctx.render_ctx, custom_ops=opt_ctx.render_ops)
			
			if catch_frustum_grid:
				opt_ctx.remove_render_op("FRUSTUM", _catch_FG)
				#LOG.debug("frustum grids: %d, frustum ops: %d", len(fg_container), len(opt_ctx.render_ops["FRUSTUM"]))
			
			LOG.debug("Density losses")
			active_density_loss = False
			# Render loss for density against targets without background
			active_density_loss = loss_dens_target(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_target_raw(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_target_depth_smooth(opt_ctx, state, fg_container[0]) or active_density_loss
			active_density_loss = loss_dens_hull(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_negative(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_smooth(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_smooth_2(opt_ctx, state) or active_density_loss
			active_density_loss = loss_dens_smooth_temporal(opt_ctx, state) or active_density_loss
			
			if use_vel and state.velocity is not None:
				active_density_loss = loss_dens_warp(opt_ctx, state) or active_density_loss
			
			if disc_ctx is not None:
				active_density_loss = loss_dens_disc(opt_ctx, state, disc_ctx, disc_samples_list) or active_density_loss
			
		#END gradient tape
		if active_density_loss:
			if custom_dens_grads is not None:
				cdg_scale = opt_ctx.CV(opt_ctx.custom_dens_grads_weight)
			with opt_ctx.profiler.sample('gradient'):
				if opt_ctx.inspect_gradients:
					for loss_name in opt_ctx._losses:
						dens_grads = dens_tape.gradient(opt_ctx.get_loss(loss_name), dens_vars)
						if dens_grads['density'] is not None:
							opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=dens_grads['density'], name=loss_name)
						del dens_grads
						if loss_name in opt_ctx.inspect_gradients_images:
							img_grads = dens_tape.gradient(opt_ctx.get_loss(loss_name), [opt_ctx.inspect_gradients_images[loss_name]])
							opt_ctx.inspect_gradients_images_func(opt_ctx=opt_ctx, gradients=img_grads[0], name=loss_name)
							del img_grads
							del opt_ctx.inspect_gradients_images[loss_name]
					if custom_dens_grads is not None and opt_ctx.LA(cdg_scale):
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=custom_dens_grads['density']*cdg_scale, name="custom_dens_grad")
						if custom_dens_grads.get('inflow') is not None:
							opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=custom_dens_grads['inflow']*cdg_scale, name="custom_inflow_grad")
					opt_ctx.inspect_gradients_images = {}
				
				LOG.debug('Compute and apply density gradients')
				if opt_ctx.light_var_list:
					dens_vars['lights'] = opt_ctx.light_var_list
				dens_grads = dens_tape.gradient(opt_ctx.get_losses(), dens_vars)
				
				if opt_ctx.inspect_gradients:
					if dens_grads['density'] is not None:
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=dens_grads['density'], name="density/total")
					if dens_grads.get('inflow') is not None:
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=dens_grads['inflow'], name="density/inflow")
					if dens_grads.get('lights') is not None:
						for i, lg in enumerate(dens_grads['lights']):
							opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=lg, name="density/light_{}".format(i))
				
				if opt_ctx.light_var_list:
					opt_ctx.light_optimizer.apply_gradients(zip(dens_grads['lights'], opt_ctx.light_var_list))
					del dens_grads['lights']
					del dens_vars['lights']
				
				curr_dens_grads = copy_nested_structure(dens_grads)
				if custom_dens_grads is not None and opt_ctx.LA(cdg_scale):
					dens_grads = nest.map_structure(lambda d, c: apply_custom_grads_with_check(d, c, cdg_scale), dens_grads, custom_dens_grads)
				if not apply_dens_grads: #still have to update inflow
					del dens_grads['density']
					del dens_vars['density']
					LOG.debug("Gradients for density of frame %d not applied, updated vars: %s, grads: %s", state.frame, dens_vars.keys(), dens_grads.keys())
				dens_grads_vars = [_ for _ in zip(nest.flatten(dens_grads), nest.flatten(dens_vars)) if _[0] is not None]
				if len(dens_grads_vars)>0:
					LOG.debug("Update %d density variables.", len(dens_grads_vars))
					opt_ctx.density_optimizer.apply_gradients(dens_grads_vars)
		else:
			curr_dens_grads = nest.map_structure(lambda v: tf.constant(0, dtype=tf.float32), dens_vars)
		
		opt_ctx.set_gradient_tape()
		del dens_tape
		
		with opt_ctx.profiler.sample('clamp density'): #necessary as negative density really breaks the rendering
			d = state.density.d
			state.density.apply_clamp(opt_ctx.CV(opt_ctx.setup.data.density.min), opt_ctx.CV(opt_ctx.setup.data.density.max))
				
	return active_density_loss, curr_dens_grads

### Velocity

def loss_vel_warp_dens(ctx, state, loss_func=None):
	#warp loss "loss(A(dt, vt), dt+1)" between current and next state for velocity.
	loss_scale = ctx.loss_schedules.velocity_warp_dens(ctx.iteration)
	if ctx.LA(loss_scale) and state.next is not None:
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/density_warp")
		LOG.debug("Run dens-warp loss for velocity, curr to next")
		with ctx.profiler.sample('velocity dens warp loss'):
			curr_dens = state.density.scaled(state.velocity.centered_shape, with_inflow=True)
			next_dens = state.next.density.scaled(state.velocity.centered_shape)
			tmp_loss = loss_func(state.velocity.warp(curr_dens, order=ctx.warp_order, dt=ctx.dt, clamp=ctx.dens_warp_clamp), next_dens)
		tmp_loss_scaled = tmp_loss * loss_scale
		if ctx.compute_loss_summary():
			ctx.add_loss([tmp_loss_scaled], tf.reduce_mean(tmp_loss), tf.reduce_mean(tmp_loss_scaled), loss_scale, 'velocity/density_warp')
		else:
			ctx.add_loss([tmp_loss_scaled], None, None, loss_scale, 'velocity/density_warp')
		return True
	return False

def loss_vel_warp_vel(ctx, state, loss_func=None):
	#warp loss "loss(A(vt, vt), vt+1)" between prev and current and current and next state for velocity.
	loss_scale = ctx.loss_schedules.velocity_warp_vel(ctx.iteration)
	if ctx.LA(loss_scale) and (state.prev is not None or state.next is not None):
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/velocity_warp")
		LOG.debug("Run vel-warp loss for velocity")
		tmp_loss = [0,0,0]
		with ctx.profiler.sample('velocity vel warp loss'):
			if state.prev is not None:
				LOG.debug("Warp loss prev to curr")
				prev_warped = state.prev.velocity_advected(order=ctx.warp_order, dt=ctx.dt, clamp=ctx.vel_warp_clamp)
				
				# buoyancy
				if ctx.setup.training.optimize_buoyancy or tf.reduce_any(tf.not_equal(ctx.buoyancy, 0.0)):
					prev_warped = prev_warped.with_buoyancy(ctx.buoyancy, state.density)
				
				tmp_loss[0] += loss_func(prev_warped.x, state.velocity.x)
				tmp_loss[1] += loss_func(prev_warped.y, state.velocity.y)
				tmp_loss[2] += loss_func(prev_warped.z, state.velocity.z)
			if state.next is not None:
				LOG.debug("Warp loss curr to next")
				curr_warped = state.velocity_advected(order=ctx.warp_order, dt=ctx.dt, clamp=ctx.vel_warp_clamp)
				
				# buoyancy
				if ctx.setup.training.optimize_buoyancy or tf.reduce_any(tf.not_equal(ctx.buoyancy, 0.0)):
					curr_warped = curr_warped.with_buoyancy(ctx.buoyancy, state.next.density)
				
				tmp_loss[0] += loss_func(curr_warped.x, state.next.velocity.x)
				tmp_loss[1] += loss_func(curr_warped.y, state.next.velocity.y)
				tmp_loss[2] += loss_func(curr_warped.z, state.next.velocity.z)
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, 'velocity/velocity_warp')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, 'velocity/velocity_warp')
		return True
	return False

def loss_vel_smooth(ctx, state, loss_func=None):
	'''Smoothness (forward differences) loss for velocity'''
	loss_scale = ctx.loss_schedules.velocity_smoothness(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/smooth")
		with ctx.profiler.sample('velocity gradient loss'):
			vel_components = state.velocity.var_list()
			tmp_loss = []
			for c in vel_components:
				tmp_loss.append(loss_func(c[:,:,:,1:,:] - c[:,:,:,:-1,:], 0)) #x_grad = d[:,:,:,1:,:] - d[:,:,:,:-1,:]
				tmp_loss.append(loss_func(c[:,:,1:,:,:] - c[:,:,:-1,:,:], 0)) #y_grad = d[:,:,1:,:,:] - d[:,:,:-1,:,:]
				tmp_loss.append(loss_func(c[:,1:,:,:,:] - c[:,:-1,:,:,:], 0)) #z_grad = d[:,1:,:,:,:] - d[:,:-1,:,:,:]
			
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, loss_name='velocity/smooth')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, loss_name='velocity/smooth')
		return True
	return False

def loss_vel_cossim(ctx, state, loss_func=None):
	'''Smoothness (forward differences) loss for velocity'''
	loss_scale = ctx.loss_schedules.velocity_cossim(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/cossim")
		with ctx.profiler.sample('velocity cosine loss'):
			v = state.velocity.centered()
			tmp_loss = [
				loss_func(tf_cosine_similarity(v[:,:,:,1:,:], v[:,:,:,:-1,:], axis=-1)*(-0.5)+0.5, 0),
				loss_func(tf_cosine_similarity(v[:,:,1:,:,:], v[:,:,:-1,:,:], axis=-1)*(-0.5)+0.5, 0),
				loss_func(tf_cosine_similarity(v[:,1:,:,:,:], v[:,:-1,:,:,:], axis=-1)*(-0.5)+0.5, 0),
			]
			
		tmp_loss_scaled = [_ * loss_scale for _ in tmp_loss]
		if ctx.compute_loss_summary():
			ctx.add_loss(tmp_loss_scaled, tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss]), tf.reduce_sum([tf.reduce_mean(_) for _ in tmp_loss_scaled]), loss_scale, loss_name='velocity/cossim')
		else:
			ctx.add_loss(tmp_loss_scaled, None, None, loss_scale, loss_name='velocity/cossim')
		return True
	return False

def loss_vel_divergence(ctx, state, loss_func=None):
	'''divergence loss'''
	loss_scale = ctx.loss_schedules.velocity_divergence(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/divergence")
		LOG.debug("Run divergence loss for velocity")
		with ctx.profiler.sample('divergence loss'):
			tmp_loss = loss_func(state.velocity.divergence(ctx.vel_scale), 0)
		tmp_loss_scaled = tmp_loss * loss_scale
		if ctx.compute_loss_summary():
			ctx.add_loss([tmp_loss_scaled], tf.reduce_mean(tmp_loss), tf.reduce_mean(tmp_loss_scaled), loss_scale, 'velocity/divergence')
		else:
			ctx.add_loss([tmp_loss_scaled], None, None, loss_scale, 'velocity/divergence')
		return True
	return False

def loss_vel_magnitude(ctx, state, loss_func=None):
	'''
		loss to minimize velocities
		tf.norm can cause issues (NaN gradients at 0 magnitude): https://github.com/tensorflow/tensorflow/issues/12071
	'''
	loss_scale = ctx.loss_schedules.velocity_magnitude(ctx.iteration)
	if ctx.LA(loss_scale):
		if loss_func is None: loss_func = ctx.get_loss_func("velocity/magnitude")
		LOG.debug("Run vector magnitude loss for velocity")
		with ctx.profiler.sample('magnitude loss'):
			tmp_loss = loss_func(state.velocity.magnitude(ctx.vel_scale), 0)
		tmp_loss_scaled = tmp_loss * loss_scale
		if ctx.compute_loss_summary():
			ctx.add_loss([tmp_loss_scaled], tf.reduce_mean(tmp_loss), tf.reduce_mean(tmp_loss_scaled), loss_scale, 'velocity/magnitude')
		else:
			ctx.add_loss([tmp_loss_scaled], None, None, loss_scale, 'velocity/magnitude')
		return True
	return False

def warp_vel_grads(opt_ctx, state, grads, order='FWD'):
	raise NotImplementedError
	if order.upper()=='FWD': #propagate velocity gradients to next state, simple forward warp
		v = state.veloctiy
		grads = VelocityGrid(v.centered_shape, x=grads[0], y=grads[1], z=grads[2], as_var=False, boundary=v.boundary, \
			warp_renderer=v.warp_renderer, scale_renderer=v.scale_renderer)
		return grads.warped(vel_grid=v, order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=opt_ctx.vel_warp_clamp)
	elif order.upper()=='BWD': #propagate velocity gradients to previous state, backprop through prev->warp
		var_list = state.velocity.var_list()
		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(var_list)
			v_warp = state.velocity.warped(order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=opt_ctx.vel_warp_clamp)
		return tape.gradient(v_warp, var_list, grads)
	else:
		raise ValueError

def optStep_velocity(opt_ctx, state, custom_vel_grads=None, optimize_inflow=False, apply_vel_grads=True):
	vel_vars = state.velocity.get_variables()
	if optimize_inflow:
		dens_vars = state.density.get_variables()
		if 'inflow' in dens_vars: # inflow variable available
			vel_vars['inflow'] = dens_vars['inflow']
	if opt_ctx.setup.training.optimize_buoyancy:
		vel_vars['buoyancy'] = opt_ctx.buoyancy
	with  opt_ctx.profiler.sample('optStep_velocity'):
		with opt_ctx.profiler.sample('loss'), tf.GradientTape(watch_accessed_variables=False, persistent=opt_ctx.inspect_gradients) as vel_tape:
			vel_tape.watch(vel_vars)
		#	velocity_loss = 0
			active_velocity_loss = False
			LOG.debug("velocity losses")
			
			#warp losses
			active_velocity_loss = loss_vel_warp_dens(opt_ctx, state) or active_velocity_loss
			active_velocity_loss = loss_vel_warp_vel(opt_ctx, state) or active_velocity_loss
			
			#direct losses
			active_velocity_loss = loss_vel_smooth(opt_ctx, state) or active_velocity_loss
			active_velocity_loss = loss_vel_cossim(opt_ctx, state) or active_velocity_loss
			active_velocity_loss = loss_vel_divergence(opt_ctx, state) or active_velocity_loss
			active_velocity_loss = loss_vel_magnitude(opt_ctx, state) or active_velocity_loss
			
		#	velocity_loss = opt_ctx.loss #opt_ctx.pop_loss()
		#END gradient tape
		if active_velocity_loss:
			with opt_ctx.profiler.sample('gradient'):
				if custom_vel_grads is not None:
					cvg_scale = opt_ctx.CV(opt_ctx.custom_vel_grads_weight)
				if opt_ctx.inspect_gradients:
					for loss_name in opt_ctx._losses:
						vel_grads = vel_tape.gradient(opt_ctx.get_loss(loss_name), state.velocity.get_variables())
						if vel_grads.get('velocity_x') is not None:
							opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['velocity_x'], name=loss_name+"_x")
						if vel_grads.get('velocity_y') is not None:
							opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['velocity_y'], name=loss_name+"_y")
						if vel_grads.get('velocity_z') is not None:
							opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['velocity_z'], name=loss_name+"_z")
						del vel_grads
					if custom_vel_grads is not None and opt_ctx.LA(cvg_scale):
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=custom_vel_grads['velocity_x']*cvg_scale, name="custom_vel_grad_x")
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=custom_vel_grads['velocity_y']*cvg_scale, name="custom_vel_grad_y")
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=custom_vel_grads['velocity_z']*cvg_scale, name="custom_vel_grad_z")
					opt_ctx.inspect_gradients_images = {}
				
				LOG.debug('Compute and apply velocity gradients')
				vel_grads = vel_tape.gradient(opt_ctx.get_losses(), vel_vars)
				if opt_ctx.inspect_gradients:
					if vel_grads.get('velocity_x') is not None:
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['velocity_x'], name="velocity/total_x")
					if vel_grads.get('velocity_y') is not None:
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['velocity_y'], name="velocity/total_y")
					if vel_grads.get('velocity_z') is not None:
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['velocity_z'], name="velocity/total_z")
					if vel_grads.get('buoyancy') is not None:
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['buoyancy'][0], name="velocity/buoyancy_x")
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['buoyancy'][1], name="velocity/buoyancy_y")
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['buoyancy'][2], name="velocity/buoyancy_z")
					if vel_grads.get('inflow') is not None:
						opt_ctx.inspect_gradients_func(opt_ctx=opt_ctx, gradients=vel_grads['inflow'], name="velocity/inflow")
					
				curr_vel_grads = copy_nested_structure(vel_grads)
				if apply_vel_grads:
					if custom_vel_grads is not None and opt_ctx.LA(cvg_scale):
						for k in vel_grads: #depth 1 sufficient for now...
							if k in custom_vel_grads: 
								LOG.debug("Update velocity gradient '%s' with custom gradient.", k)
								vel_grads[k] += custom_vel_grads[k]*cvg_scale
					if vel_grads.get('inflow') is not None:
						inflow_grads_vars = ((vel_grads['inflow'], vel_vars['inflow']),)
						opt_ctx.density_optimizer.apply_gradients(inflow_grads_vars)
						del vel_grads['inflow']
						del vel_vars['inflow']
					vel_grads_vars = zip(nest.flatten(vel_grads), nest.flatten(vel_vars))
					opt_ctx.velocity_optimizer.apply_gradients(vel_grads_vars)
		else:
			curr_vel_grads = nest.map_structure(lambda v: tf.constant(0, dtype=tf.float32), vel_vars)
		del vel_tape
	return active_velocity_loss, curr_vel_grads

def optStep_state(opt_ctx, state, disc_ctx=None, disc_samples_list=None, custom_dens_grads=None, custom_vel_grads=None, apply_dens_grads=True, apply_vel_grads=True):
	LOG.debug("Optimization step for state: frame %d", state.frame)
	with  opt_ctx.profiler.sample('optStep_state'):
		prev_losses = opt_ctx._losses
		opt_ctx.pop_losses()
		opt_ctx.frame = state.frame
		
		dens_active, dens_grads = optStep_density(opt_ctx, state, use_vel=True, disc_ctx=disc_ctx, disc_samples_list=disc_samples_list, custom_dens_grads=custom_dens_grads, apply_dens_grads=apply_dens_grads)
		
		opt_ctx.pop_losses()
		vel_active, vel_grads = optStep_velocity(opt_ctx, state, custom_vel_grads=custom_vel_grads, apply_vel_grads=apply_vel_grads)
		
		opt_ctx.pop_losses()
		opt_ctx._losses = prev_losses
	
	return dens_grads, vel_grads

def optStep_sequence(opt_ctx, sequence, disc_ctx=None, disc_samples_list=None, order='FWD'):
	LOG.debug("Optimization step for sequence with %d states", len(sequence))
	total_losses = []
	loss_summaries = {}
	if order.upper()=='FWD':
		optim_order = list(range(len(sequence)))
	elif order.upper()=='BWD':
		optim_order = list(reversed(range(len(sequence))))
	elif order.upper()=='RAND':
		optim_order = np.random.permutation(len(sequence)).tolist()
	else:
		raise ValueError
	#for state in sequence:
	first_state = True
	dens_grad_keys = ['density']
	vel_grad_keys = ['velocity_x', 'velocity_y', 'velocity_z']
	dens_grads = {_:tf.constant(0, dtype=tf.float32) for _ in dens_grad_keys}
	#vel_grads = [_:tf.constant(0, dtype=tf.float32) for _ in dens_grad_keys]
	wdg_decay = opt_ctx.CV(opt_ctx.warp_dens_grads_decay)
	wvg_decay = opt_ctx.CV(opt_ctx.warp_vel_grads_decay)
	num_step = 0
	for i in optim_order:
		state = sequence[i]
		
		custom_dens_grads = None
		custom_vel_grads = None
		if opt_ctx.warp_vel_grads(opt_ctx.iteration): # and not first_state and not order=='RAND'
			raise NotImplementedError
			#	warped_vel_grads = warp_vel_grads(opt_ctx, )
		#			vel_grads = warped_vel_grads
		#			custom_vel_grads = vel_grads
		if opt_ctx.warp_dens_grads(opt_ctx.iteration) and not first_state and not order=='RAND':
			with opt_ctx.profiler.sample("Warp dens grads "+order):
				warped_dens_grads = warp_dens_grads(opt_ctx, state=state, grads=dens_grads['density'], order=order)
				dens_grads = {k: warped_dens_grads[k] for k in dens_grad_keys} #cutoff inflow
				custom_dens_grads = {k:warped_dens_grads[k] for k in state.density.get_variables()} #warped_dens_grads[:-3] #provide inflow gradients to current state
				if order=='BWD':
				#	if custom_vel_grads is not None: TODO
				#		vel_grads = [v+w for v,w in zip(vel_grads, warped_dens_grads[1:])]
				#	else:
				#		vel_grads = {k: warped_dens_grads[k] for k in vel_grad_keys} #warped_dens_grads[-3:]
					vel_grads = {k: warped_dens_grads[k] for k in vel_grad_keys} #warped_dens_grads[-3:]
					custom_vel_grads = copy_nested_structure(vel_grads)#[:]
		if custom_dens_grads is not None and wdg_decay==0: #no decay, average
			custom_dens_grads = nest.map_structure(lambda s: s * (1/num_step), custom_dens_grads) #[_ * (1/num_step) for _ in custom_dens_grads]
		if custom_vel_grads is not None and wvg_decay==0: #no decay, average
			custom_vel_grads = nest.map_structure(lambda s: s * (1/num_step), custom_vel_grads) #[_ * (1/num_step) for _ in custom_vel_grads]
		
		with opt_ctx.profiler.sample("Frame"):
			last_dens_grads, last_vel_grads = optStep_state(opt_ctx, state, disc_ctx, disc_samples_list, \
				custom_dens_grads=custom_dens_grads, custom_vel_grads=custom_vel_grads,
				apply_dens_grads= (not opt_ctx.update_first_dens_only(opt_ctx.iteration)) or i==0) #don't apply density gradients if the sequence is warped fwd every iteration, except for the first frame
			loss_summaries[state.frame]=opt_ctx.pop_loss_summary()
		
		if opt_ctx.warp_dens_grads(opt_ctx.iteration):
			#last_dens_grads = last_dens_grads[:1] #exclude inflow if available
			if 'inflow' in last_dens_grads: del last_dens_grads['inflow']
			dens_grads = nest.map_structure(lambda l, d: lerp(l,d,wdg_decay) if wdg_decay>0 else l+d, last_dens_grads, dens_grads)
		#	if first_state:
		#		dens_grads = [l*(1-wdg_decay) if wdg_decay>0 else l for l in last_dens_grads]
		#	else:
		#		dens_grads = [lerp(l,d,wdg_decay) if wdg_decay>0 else l+d for d,l in zip(dens_grads, last_dens_grads)]
		#if opt_ctx.warp_vel_grads(opt_ctx.iteration): #opt_ctx.warp_dens_grads or 
		#	last_vel_grads = last_vel_grads[:3] #exclude exclude buoyancy and inflow if available
		#	if first_state:
		#		vel_grads = [l*(1-wvg_decay) if wvg_decay>0 else l for l in last_vel_grads]
		#	else:
		#		vel_grads = [lerp(l,v,wvg_decay) if wvg_decay>0 else l+v for v,l in zip(vel_grads, last_vel_grads)]
			# no decay version
		#	dens_grads = [l+d for d,l in zip(dens_grads, last_dens_grads)]
		#	vel_grads = [l+v for v,l in zip(vel_grads, last_vel_grads)]
		first_state = False
		num_step+=1
	return loss_summaries

### Discriminator
class DiscriminatorContext:
	CHECK_INPUT_NONE = 0x0
	CHECK_INPUT_CLAMP = 0x1
	CHECK_INPUT_DUMP  = 0x2
	CHECK_INPUT_CHECK = 0x10
	CHECK_INPUT_CHECK_NOTFINITE = 0x20
	CHECK_INPUT_SIZE = 0x40
	CHECK_INPUT_RAISE = 0x100
	CHECK_INPUT_RAISE_NOTFINITE = 0x200
	def __init__(self, ctx, model, rendering_context, real_data, loss_type, optimizer, learning_rate, crop_size=None, scale_range=[1,1], rotation_mode="NONE", check_input=0, check_info_path=None, \
			resource_device=None, scale_samples_to_input_resolution=False, \
			use_temporal_input=False, temporal_input_steps=None):
		assert isinstance(ctx, OptimizationContext)
		assert isinstance(rendering_context, RenderingContext)
		assert isinstance(learning_rate, tf.Variable)
		
		self.model = model
		self.real_data = real_data
		self.opt_ctx = ctx
		self.render_ctx = rendering_context
		self.history = None
		self._train_base = True
		self._train = True
		if ctx.setup.training.discriminator.history.samples>0:
			self.history = HistoryBuffer(ctx.setup.training.discriminator.history.size)
		#self._label_fake = 0.0
		self._label_real = ctx.setup.training.discriminator.target_label
		self._conditional_hull = ctx.setup.training.discriminator.conditional_hull
		self.optimizer = optimizer
		self.lr = learning_rate
		self._last_it = self.opt_ctx.iteration
		self.center_crop = True # center crop on density center of mass + random offset
		self.crop_size = crop_size
		self.scale_range = scale_range
		self.rotation_mode = rotation_mode
		self.dump_path = None
		self._check_input = check_input #bit-mask
		self._check_info_path = check_info_path
		self._input_range = [0.0, 10.0]
		self.resource_device = resource_device
		loss_types = ["SGAN", "RpSGAN", "RpLSGAN", "RaSGAN", "RaLSGAN"]
		if loss_type not in loss_types:
			raise ValueError("Unknown Discriminator loss_type {}. Available losses: {}".format(loss_type, loss_types))
		self.loss_type = loss_type #SGAN, RpSGAN, RaSGAN, RcSGAN
		
		self.scale_to_input_res = scale_samples_to_input_resolution
		
		self._temporal_input = use_temporal_input
		self._temporal_input_steps = temporal_input_steps
	
	@property
	def train(self):
		return self._train and self._train_base
	
	@train.setter
	def train(self, train):
		self._train_base = train
	
	def start_iteration(self, iteration, force=False, compute_loss_summary=False):
		self.opt_ctx.start_iteration(iteration, force, compute_loss_summary)
		if iteration==self._last_it and not force:
			return
		
		curr_lr = self.opt_ctx.loss_schedules.discriminator_lr(self.opt_ctx.iteration - self.opt_ctx.setup.training.discriminator.start_delay)
		self._train = self.opt_ctx.LA(curr_lr)
		self.lr.assign(curr_lr)
		
		if self.opt_ctx.record_summary:
			summary_names = self.opt_ctx.make_summary_names('discriminator/learning_rate')
			self.opt_ctx._tf_summary.scalar(summary_names[0], self.lr.numpy(), step=self.opt_ctx.iteration)
		self._last_it = self.opt_ctx.iteration
	
	@property
	def input_res(self):
		return self.model.input_shape[-3:-1]
	
	@property
	def record_history(self):
		return self.train and self.history is not None
	
	def var_list(self):
		return self.model.trainable_variables
		
	def real_labels(self, logits):
		return tf.ones_like(logits)*self._label_real
	def fake_labels(self, logits):
		return tf.zeros_like(logits)
	
	def _scale_range(self, shape, target_shape, max_scale_range):
		scale = np.amax([t/i for t,i in zip(target_shape, shape)])
		if scale>max_scale_range[1]:
			raise ValueError("Scaling impossible with shape {}, target shape {}, resulting scale {} and scale_range {}".format(shape, target_shape, scale, max_scale_range))
		return [max(scale, max_scale_range[0]), max_scale_range[1]]
	
	def dump_samples(self, sample_batch, is_real):
		if self.dump_path is not None:
			with self.opt_ctx.profiler.sample("dump disc samples"):
				if is_real: name = 'real_s{:06d}_i{:06d}'
				else: name = 'fake_s{:06d}_i{:06d}'
				if self._temporal_input:
					for i, batch in enumerate(tf.split(sample_batch, 3, axis=-1)):
						self.render_ctx.dens_renderer.write_images([batch], [name + '_t%02d'%i], base_path=self.dump_path, use_batch_id=True, frame_id=self.opt_ctx.iteration, format='EXR')
				else:
					self.render_ctx.dens_renderer.write_images([sample_batch], [name], base_path=self.dump_path, use_batch_id=True, frame_id=self.opt_ctx.iteration, format='EXR')
	
	def check_input(self, input, name="input"):
		# debug discriminator failure
		dump_input = False
		nan_input = False
		if (self._check_input & (self.CHECK_INPUT_CHECK_NOTFINITE | self.CHECK_INPUT_CHECK))>0:
			nan_in = tf.reduce_any(tf.is_nan(input), axis=[1,2,3])
			if tf.reduce_any(nan_in).numpy():
				LOG.warning("NaN in samples %s of discriminator input '%s' in iteration %d.", np.where(nan_in.numpy())[0], name, self.opt_ctx.iteration)
				dump_input = True
				nan_input = True
		if (self._check_input & self.CHECK_INPUT_SIZE)>0:
			input_shape = shape_list(input)
			if tf.reduce_any(tf.not_equal(input.get_shape()[-3:], self.model.input_shape[-3:])).numpy():
				LOG.warning("shape %s of input '%s' does not match discriminator input shape %s", shape_list(input), name, self.model.input_shape)
		if (self._check_input & self.CHECK_INPUT_CHECK)>0:
			if tf.reduce_any(tf.less(input, self._input_range[0])).numpy():
				in_min = tf.reduce_min(input).numpy()
				LOG.warning("Minimum value %f of discriminator input '%s' exceeds minimum %f in iteration %d.", in_min, name, self._input_range[0], self.opt_ctx.iteration)
				dump_input = True
			if tf.reduce_any(tf.greater(input, self._input_range[1])).numpy():
				in_max = tf.reduce_max(input).numpy()
				LOG.warning("Maximum value %f of discriminator input '%s' exceeds maximum %f in iteration %d.", in_max, name, self._input_range[1], self.opt_ctx.iteration)
				dump_input = True
		if dump_input and (self._check_input & self.CHECK_INPUT_DUMP)>0 and self._check_info_path is not None:
			name = "{}_{:06d}".format(name, self.opt_ctx.iteration) + "_{:04d}"
			self.render_ctx.dens_renderer.write_images([input], [name], base_path=self._check_info_path, use_batch_id=True, format='EXR')
		if (dump_input and (self._check_input & self.CHECK_INPUT_RAISE)>0) or (nan_input and (self._check_input & self.CHECK_INPUT_RAISE_NOTFINITE)>0):
			raise ValueError("Discriminator input {} error.".format(name))
			
		
		if (self._check_input & self.CHECK_INPUT_CLAMP)>0:
			return tf.minimum(tf.maximum(input, self._input_range[0]), self._input_range[1]) #also makes nan and -inf to min and inf to max (TF 1.12 on GPU)
		else:
			return input
	
	def check_output(self, output, loss, input, name='output'):
		if (self._check_input & (self.CHECK_INPUT_CHECK_NOTFINITE | self.CHECK_INPUT_CHECK))>0:
			dump_input = False
			if not tf.reduce_all(tf.is_finite(output)):
				LOG.warning("Discriminator output '%s' in iteration %d is not finite: %s", name, self.opt_ctx.iteration, output.numpy())
				dump_input = True
			if not tf.reduce_all(tf.is_finite(loss)):
				LOG.warning("Discriminator loss '%s' in iteration %d is not finite: %s", name, self.opt_ctx.iteration, loss.numpy())
				dump_input = True
			if dump_input and (self._check_input & self.CHECK_INPUT_DUMP)>0 and self._check_info_path is not None:
				file_name = "{}_{:06d}".format(name, self.opt_ctx.iteration) + "_{:04d}"
				self.render_ctx.dens_renderer.write_images([input], [file_name], base_path=self._check_info_path, use_batch_id=True, format='EXR')
			if dump_input and (self._check_input & (self.CHECK_INPUT_RAISE_NOTFINITE |self.CHECK_INPUT_RAISE))>0:
				raise ValueError("Discriminator output {} error.".format(name))
	
	def _scale_samples_to_input_res(self, *samples_raw):
		if self.scale_to_input_res:
			#LOG.info("Scaling disc input before augmentation from %s to %s.", [shape_list(_) for _ in samples_raw], self.input_res)
			return [tf.image.resize_bilinear(sample_raw, self.input_res) for sample_raw in samples_raw]
		else:
			return samples_raw
	
	def _pad_samples_to_input_res(self, *samples_raw):
		return [tf_pad_to_shape(sample_raw, [-1]+ list(self.input_res) +[-1], allow_larger_dims=True) for sample_raw in samples_raw]
	
	def image_center_of_mass(self, img):
		sample_mean_y = tf.reduce_mean(sample, axis=[-3,-1]) #NW
	#	LOG.info("mean y shape: %s", sample_mean_y.get_shape().as_list())
		coords_x = tf.reshape(tf.range(0, scale_shape[-1], 1,dtype=tf.float32), (1,scale_shape[-1])) #1W
		center_x = tf.reduce_sum(coords_x*sample_mean_y, axis=-1)/tf.reduce_sum(sample_mean_y, axis=-1) #N
		sample_mean_x = tf.reduce_mean(sample, axis=[-2,-1]) #NH
		coords_y = tf.reshape(tf.range(0, scale_shape[-2], 1,dtype=tf.float32), (1,scale_shape[-2])) #1H
		center_y = tf.reduce_sum(coords_y*sample_mean_x, axis=-1)/tf.reduce_sum(sample_mean_x, axis=-1) #N
		return Float2(center_x, center_y)
	
	
	def _prepare_samples(self, *samples_raw, scale_range=(1.,1.), rotation_mode="90", crop_shape="INPUT"):
		""" Data augmentation for discriminator input.
		
		1. scale image resolution with random scaling factor from scale_range using bilinear interpolation.
		2. appy random rotation
		3. pad the image to be at have least size crop_shape
		4. apply random crop, focusing on the center of mass, if possible
		
		"""
		samples = []
		with self.opt_ctx.profiler.sample('prepare_crop_flip'):
			for sample_raw in samples_raw:
				sample_shape = shape_list(sample_raw)
				#raw shape & target/crop shape -> scale range
				#check allowed scale range
				#now allow any scale range and pad later if necessary
				#scale_range = self._scale_range(sample_shape[-3:-1], crop_shape, scale_range)
				if not (scale_range==None or scale_range==(1.,1.)):
					scale = np.random.uniform(*scale_range)
					if scale!=1.:
						scale_shape = [int(np.ceil(_*scale)) for _ in sample_shape[-3:-1]]
						sample = tf.image.resize_bilinear(sample_raw, scale_shape)
					else:
						sample = sample_raw
				
				#random 90deg rotation and mirroring
				if rotation_mode==90 or rotation_mode=="90":
					r = np.random.randint(2, size=3)
					if r[0]==1:
						sample = tf.transpose(sample, (0,2,1,3)) #swap x and y of NHWC tensor
					flip_axes = []
					if r[1]==1:
						flip_axes.append(-2) #flip x
					if r[2]==1:
						flip_axes.append(-3) #flip y
					if flip_axes:
						sample = tf.reverse(sample, flip_axes)
				elif rotation_mode.upper()=="CONTINUOUS":
					raise NotImplementedError
				elif not (rotation_mode is None or rotation_mode.upper()=="NONE"):
					raise ValueError("Unknown rotation_mode %s"%rotation_mode)
				
				if crop_shape is not None:
					if crop_shape=="INPUT":
						crop_shape = self.input_res
					sample_shape = shape_list(sample)
					
					if np.any(np.less(sample_shape[-3:-1], crop_shape)):
						sample = tf_pad_to_shape(sample, [-1]+ list(crop_shape) +[-1], allow_larger_dims=True) #, mode="REFLECT")
						sample_shape = shape_list(sample)
					
					# don't crop if shape already matches
					if not np.all(np.equal(sample_shape[-3:-1], crop_shape)):
						# -> find a "center of mass" and crop around that, with some random offset
						# what if sample is empty/all 0?:
						crop_eps = 1e-4
						if self.center_crop and tf.reduce_mean(sample).numpy()>crop_eps:
						#	LOG.info("scale shape: %s", scale_shape)
							sample_mean_y = tf.reduce_mean(sample, axis=[-3,-1]) #NW
						#	LOG.info("mean y shape: %s", sample_mean_y.get_shape().as_list())
							coords_x = tf.reshape(tf.range(0, sample_shape[-2], 1,dtype=tf.float32), (1,sample_shape[-2])) #1W
							center_x = tf.reduce_sum(coords_x*sample_mean_y, axis=-1)/tf.reduce_sum(sample_mean_y, axis=-1) #N
							sample_mean_x = tf.reduce_mean(sample, axis=[-2,-1]) #NH
							coords_y = tf.reshape(tf.range(0, sample_shape[-3], 1,dtype=tf.float32), (1,sample_shape[-3])) #1H
							center_y = tf.reduce_sum(coords_y*sample_mean_x, axis=-1)/tf.reduce_sum(sample_mean_x, axis=-1) #N
							
							# get offset s.t. crop is in bounds, centered on center of mass (+ noise)
							crop_shape = tf.constant(crop_shape, dtype=tf.int32) #HW
							offset_bounds = sample.get_shape()[-3:-1] - crop_shape #2
							offset = tf.stack([center_y, center_x], axis=-1) + tf.random.uniform([sample.get_shape()[0], 2], -20,21, dtype=tf.float32) - tf.cast(crop_shape/2, dtype=tf.float32) #N2
							offset = tf.clip_by_value(tf.cast(offset, dtype=tf.int32), [0,0], offset_bounds)
							sample = tf.stack([tf.image.crop_to_bounding_box(s, *o, *crop_shape) for s, o in zip(sample, offset)], axis=0)
						else:
							sample = tf.random_crop(sample, [sample_shape[0]]+list(crop_shape)+[sample_shape[-1]])
				
				samples.append(sample)
		return samples if len(samples)>1 else samples[0]
	
	def augment_intensity(self, samples, scale_range, gamma_range):
		scale_shape = (shape_list(samples)[0],1,1,1)
		scale = tf.random.uniform(scale_shape, *scale_range, dtype=samples.dtype)
		gamma = tf.random.uniform(scale_shape, *gamma_range, dtype=samples.dtype)
		scale = [scale,scale,scale]
		gamma = [gamma,gamma,gamma]
		if self._conditional_hull: #do not scale the intensity of the hull, the disc should be invariant to intensities
			scale.append(tf.ones(scale_shape, dtype=samples.dtype))
			gamma.append(tf.ones(scale_shape, dtype=samples.dtype))
		if self._temporal_input:
			scale *=3
			gamma *=3
		samples = tf.pow(tf.multiply(samples, tf.concat(scale, axis=-1)), tf.concat(gamma, axis=-1))
		return samples
	
	def real_samples(self, spatial_augment=True, intensity_augment=True):
		with self.opt_ctx.profiler.sample('real_samples'):
			samples = self.real_data.get_next()
			samples = self._scale_samples_to_input_res(samples)[0]
			if spatial_augment:
				samples = self._prepare_samples(*tf.split(samples, samples.get_shape()[0], axis=0), \
					crop_shape="INPUT" if self.scale_to_input_res else self.crop_size, scale_range=self.scale_range, rotation_mode=self.rotation_mode)
			else:
				samples = self._pad_samples_to_input_res(*tf.split(samples, samples.get_shape()[0], axis=0))
			if intensity_augment:
				samples = self.augment_intensity(tf.concat(samples, axis=0), self.opt_ctx.setup.data.discriminator.scale_real, self.opt_ctx.setup.data.discriminator.gamma_real)
		return samples
	
	def _render_fake_samples(self, state, name="render_fake_samples"):
		dens_transform = state.get_density_transform()
		#LOG.debug("Render fake samples '%s' with jitter %s", name, [_.jitter for _ in self.render_ctx.cameras])
		imgs_fake = self.render_ctx.dens_renderer.render_density(dens_transform, self.render_ctx.lights, self.render_ctx.cameras, monochrome=self.render_ctx.monochrome, custom_ops=self.opt_ctx.render_ops) #[1DWC]*N
		if self._conditional_hull:
			imgs_hull = self.render_ctx.dens_renderer.project_hull(state.hull, dens_transform, self.render_ctx.cameras) #NDWC
			imgs_hull = tf.split(imgs_hull, len(self.render_ctx.cameras), axis=0) #[1DWC]*N
			imgs_fake = [tf.concat([f,h], axis=-1) for f,h in zip(imgs_fake, imgs_hull)]
		return imgs_fake
		
	def fake_samples(self, state, history_samples=True, concat=True, spatial_augment=True, intensity_augment=False, name="fake_samples"):
		with self.opt_ctx.profiler.sample('fake_samples'):
			#prepare fake samples
			in_fake = []
			if state is not None:
				self.render_ctx.randomize_camera_rotation(z_range=[0,0])
			
			#		
				
				with self.opt_ctx.profiler.sample('render fake'):
					# TODO temporal disc input:
					if self._temporal_input:
						cur_idx = 1
						tmp_fake = [None]*3
						with NO_CONTEXT() if self.opt_ctx.tape is None else self.opt_ctx.tape.stop_recording(): # don't need gradients for cmp images (and probably don't have memory for it...)
							#TODO random step. consider data/reconstuction step (i.e. frame skipping) vs dataset steps?
							#	for testing, use fixed prev/next step
							#TODO border handling. use border image in disc triplet? also randomly for other frames/states?
							#	curr: needs at least 3 frame sequence or will break
							#	or black inputs. TODO needs black prev/next in real data.
							# use fixed random camera transform from current disc input
							if state.prev is None:
								tmp_fake[1] = self._render_fake_samples(state.next, name=name + "_next")
								tmp_fake[2] = self._render_fake_samples(state.next.next, name=name + "_nnext")
								cur_idx = 0
							elif state.next is None:
								tmp_fake[0] = self._render_fake_samples(state.prev.prev, name=name + "_pprev")
								tmp_fake[1] = self._render_fake_samples(state.prev, name=name + "_prev")
								cur_idx = 2
							else:
								tmp_fake[0] = self._render_fake_samples(state.prev, name=name + "_prev")
								tmp_fake[2] = self._render_fake_samples(state.next, name=name + "_next")
								cur_idx = 1
							LOG.debug("Render temporal fake disc input '%s', current idx %d. tape available: %s", name, cur_idx, self.opt_ctx.tape is not None)
					
					imgs_fake = self._render_fake_samples(state, name=name)
					
					if self._temporal_input:
						tmp_fake[cur_idx] = imgs_fake
						imgs_fake = [tf.concat(_, axis=-1) for _ in zip(*tmp_fake)]
					in_fake += imgs_fake
			with NO_CONTEXT() if self.opt_ctx.tape is None else self.opt_ctx.tape.stop_recording():
				if self.record_history:
					if history_samples:
						in_history = self.history.get_samples(self.opt_ctx.setup.training.discriminator.history.samples, replace=False, allow_partial=True)
					with tf.device(self.resource_device): #copy to resource device
						hist_samples = [tf.identity(_) for _ in in_fake]
						self.history.push_samples(hist_samples, self.opt_ctx.setup.training.discriminator.history.keep_chance, 'RAND')
					if history_samples:
						in_fake += in_history
					#if disc_dump_samples and len(disc_in_history)>0: self.render_ctx.dens_renderer.write_images([tf.concat(disc_in_history, axis=0)], ['zz_disc_{1:04d}_fake_history{0}'], base_path=setup.paths.data, use_batch_id=True, frame_id=it, format='PNG')
					if self.opt_ctx.record_summary:
						summary_names = self.opt_ctx.make_summary_names('discriminator/history_size')
						self.opt_ctx._tf_summary.scalar(summary_names[0], len(self.history), step=self.opt_ctx.iteration)
			
			in_fake = self._scale_samples_to_input_res(*in_fake)
			if spatial_augment:
				in_fake = self._prepare_samples(*in_fake, crop_shape="INPUT" if self.scale_to_input_res else self.crop_size, scale_range=self.scale_range, rotation_mode=self.rotation_mode)
			else:
				in_fake = self._pad_samples_to_input_res(*in_fake)
			
			if intensity_augment:
				raise NotImplementedError
		return tf.concat(in_fake, axis=0) if concat else in_fake
	
	def postprocess_loss(self, loss, out, name="loss"):
		self.opt_ctx.add_loss(tf.math.reduce_mean(loss), loss_name='discriminator/'+name)
		if self.opt_ctx.setup.training.discriminator.use_fc:
			scores = tf.math.sigmoid(out)
		else:
			scores = tf.reduce_mean(tf.math.sigmoid(out), axis=[1,2,3])
		return scores
	
	def loss(self, input, flip_target=False, training=True):
		'''
			Relativistic discriminator: https://github.com/AlexiaJM/relativistic-f-divergences
		input: (real, fake), (input) for SGAN
		flip_target: 
		'''
		name = "fake_loss" if flip_target else "real_loss"
		if self.loss_type=="SGAN":
			if flip_target:
				out = self.model(self.check_input(input[0], "fake"), training=training)
				loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=self.fake_labels(out)))
			else:
				out = self.model(self.check_input(input[0], "real"), training=training)
				loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=self.real_labels(out)))
			
			if self.opt_ctx.setup.training.discriminator.use_fc:
				scores = tf.math.sigmoid(out)
			else:
				scores = tf.reduce_mean(tf.math.sigmoid(out), axis=[1,2,3])
		else:
			out_real = self.model(self.check_input(input[0], "real"), training=training)
			out_fake = self.model(self.check_input(input[1], "fake"), training=training)
			if self.loss_type in ["RpSGAN", "RpLSGAN"]:
				#relativistic paired
				#batch and (disc out) resolution of fake and real must match here
				if flip_target:
					out_rel = out_fake-out_real
				else:
					out_rel = out_real-out_fake
				
				if self.loss_type=="RpSGAN":
					loss = 2*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out_rel, labels=self.real_labels(out_rel)))
					if self.opt_ctx.setup.training.discriminator.use_fc:
						scores = tf.math.sigmoid(out)
					else:
						scores = tf.reduce_mean(tf.math.sigmoid(out), axis=[1,2,3])
				elif self.loss_type=="RpLSGAN":
					loss = 2*tf.reduce_mean(tf.math.squared_difference(out_rel, self._label_real))
					if self.opt_ctx.setup.training.discriminator.use_fc:
						scores = out_rel
					else:
						scores = tf.reduce_mean(out_rel, axis=[1,2,3])
				out = out_rel
				
				
			elif self.loss_type in ["RaSGAN", "RaLSGAN"]:
				# relativistic average. patch gan/disc: cmp to average value of every patch
				if flip_target:
					out_rel_real = out_fake-tf.reduce_mean(out_real)#, axis=0)
					out_rel_fake = out_real-tf.reduce_mean(out_fake)#, axis=0)
				else:
					out_rel_real = out_real-tf.reduce_mean(out_fake)#, axis=0)
					out_rel_fake = out_fake-tf.reduce_mean(out_real)#, axis=0)
				
				if self.loss_type=="RaSGAN":
					loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out_rel_real, labels=self.real_labels(out_rel_real))) \
						+ tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out_rel_fake, labels=self.fake_labels(out_rel_fake)))
				elif self.loss_type=="RaLSGAN":
					loss = tf.reduce_mean(tf.math.squared_difference(out_rel_real, self._label_real)) \
						+ tf.reduce_mean(tf.math.squared_difference(out_rel_fake, -self._label_real))
				
				out = (out_rel_real, out_rel_fake)
				scores = tf.zeros([1], dtype=tf.float32)
		
		return loss, scores#, name

def loss_disc_real(disc, in_real, training=True):
	in_real = disc.check_input(in_real, "real")
	out_real = disc.model(in_real, training=training)
	loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_real, labels=disc.real_labels(out_real))
	disc.check_output(out_real, loss_real, in_real, "real")
	loss_real = tf.math.reduce_mean(loss_real)
	disc.opt_ctx.add_loss([loss_real], loss_value_scaled=loss_real, loss_name='discriminator/real_loss')
	if disc.opt_ctx.setup.training.discriminator.use_fc:
		scores_real = tf.math.sigmoid(out_real)
	else:
		scores_real = tf.reduce_mean(tf.math.sigmoid(out_real), axis=[1,2,3])
	return scores_real

def loss_disc_fake(disc, in_fake, training=True):
	in_fake = disc.check_input(in_fake, "fake")
	out_fake = disc.model(in_fake, training=training)
	loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=out_fake, labels=disc.fake_labels(out_fake))
	disc.check_output(out_fake, loss_fake, in_fake, "fake")
	loss_fake = tf.math.reduce_mean(loss_fake)
	disc.opt_ctx.add_loss([loss_fake], loss_value_scaled=loss_fake, loss_name='discriminator/fake_loss')
	if disc.opt_ctx.setup.training.discriminator.use_fc:
		scores_fake = tf.math.sigmoid(out_fake)
	else:
		scores_fake = tf.reduce_mean(tf.math.sigmoid(out_fake), axis=[1,2,3])
	return scores_fake

def loss_disc_weights(disc):
	loss_scale = disc.opt_ctx.loss_schedules.discriminator_regularization(disc.opt_ctx.iteration)
	if disc.opt_ctx.LA(loss_scale):
		with disc.opt_ctx.profiler.sample('discriminator regularization'):
			disc_weights = disc.var_list()
			tmp_loss = tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(var)) for var in disc_weights])
		tmp_loss_scaled = tmp_loss * loss_scale
		disc.opt_ctx.add_loss([tmp_loss_scaled], tmp_loss, tmp_loss_scaled, loss_scale, 'discriminator/regularization')
		return True
	return False

def optStep_discriminator(disc_ctx, state=None, additional_fake_samples=None):
	if disc_ctx.train and disc_ctx.opt_ctx.setup.training.discriminator.start_delay<=disc_ctx.opt_ctx.iteration:
		LOG.debug("Optimization step for discriminator")
		with disc_ctx.opt_ctx.profiler.sample('optStep_discriminator'):
			#prepare real samples
			disc_in_real = disc_ctx.real_samples(spatial_augment=True, intensity_augment=True)
			disc_ctx.dump_samples(disc_in_real, True)
			
			if disc_ctx.loss_type in ["SGAN"]:
				with disc_ctx.opt_ctx.profiler.sample('real'):
					with tf.GradientTape() as disc_tape:
						disc_loss_real, disc_scores_real = disc_ctx.loss((disc_in_real,), flip_target=False, training=True) #disc_scores_real = loss_disc_real(disc_ctx, disc_in_real)
						disc_ctx.opt_ctx.add_loss(disc_loss_real, loss_value_scaled=reduce_losses(disc_loss_real), loss_name='discriminator/loss_real')
						loss_disc_weights(disc_ctx)
						disc_loss_real = disc_ctx.opt_ctx.pop_losses()
					grads = disc_tape.gradient(disc_loss_real, disc_ctx.var_list())
					disc_ctx.optimizer.apply_gradients(zip(grads, disc_ctx.var_list()))
			
			#prepare fake samples
			disc_in_fake = []
			if additional_fake_samples is not None:
				r = np.random.choice(len(additional_fake_samples), len(disc_ctx.render_ctx.cameras), replace=False)
				disc_in_fake.extend([additional_fake_samples[_] for _ in r])
			disc_in_fake.extend(disc_ctx.fake_samples(state, concat=False, spatial_augment=False, name="disc_fake_samples"))
			if disc_ctx.crop_size is not None or disc_ctx.scale_to_input_res:
				disc_in_fake = disc_ctx._prepare_samples(*disc_in_fake, crop_shape="INPUT" if disc_ctx.scale_to_input_res else disc_ctx.crop_size, scale_range=disc_ctx.scale_range, rotation_mode=disc_ctx.rotation_mode)
			
			with disc_ctx.opt_ctx.profiler.sample('fake'):
				disc_in_fake = disc_ctx.augment_intensity(tf.concat(disc_in_fake, axis=0), disc_ctx.opt_ctx.setup.data.discriminator.scale_fake, disc_ctx.opt_ctx.setup.data.discriminator.gamma_fake)
				disc_ctx.dump_samples(disc_in_fake, False)
				if disc_ctx.loss_type in ["SGAN"]:
					with tf.GradientTape() as disc_tape:
						disc_loss_fake, disc_scores_fake = disc_ctx.loss((disc_in_fake,), flip_target=True, training=True) #disc_scores_fake = loss_disc_fake(disc_ctx, disc_in_fake)
						disc_ctx.opt_ctx.add_loss(disc_loss_fake, loss_value_scaled=reduce_losses(disc_loss_fake), loss_name='discriminator/loss_fake')
						loss_disc_weights(disc_ctx)
						disc_loss_fake = disc_ctx.opt_ctx.pop_losses()
					grads = disc_tape.gradient(disc_loss_fake, disc_ctx.var_list())
					disc_ctx.optimizer.apply_gradients(zip(grads, disc_ctx.var_list()))
			
			if not (disc_ctx.loss_type in ["SGAN"]):
				with disc_ctx.opt_ctx.profiler.sample(disc_ctx.loss_type):
					with tf.GradientTape() as disc_tape:
						disc_loss, disc_scores = disc_ctx.loss((disc_in_real, disc_in_fake), False, True) #disc_scores_fake = loss_disc_fake(disc_ctx, disc_in_fake)
						disc_ctx.opt_ctx.add_loss(disc_loss, loss_value_scaled=reduce_losses(disc_loss), loss_name='discriminator/'+disc_ctx.loss_type)
						loss_disc_weights(disc_ctx)
						disc_loss = disc_ctx.opt_ctx.pop_losses()
					grads = disc_tape.gradient(disc_loss, disc_ctx.var_list())
					disc_ctx.optimizer.apply_gradients(zip(grads, disc_ctx.var_list()))
		
		if (disc_ctx.loss_type in ["SGAN"]):
			if disc_ctx.opt_ctx.record_summary:
				summary_names = disc_ctx.opt_ctx.make_summary_names('discriminator/real_score')
				disc_ctx.opt_ctx._tf_summary.scalar(summary_names[0], tf.reduce_mean(disc_scores_real), step=disc_ctx.opt_ctx.iteration)
				summary_names = disc_ctx.opt_ctx.make_summary_names('discriminator/fake_score')
				disc_ctx.opt_ctx._tf_summary.scalar(summary_names[0], tf.reduce_mean(disc_scores_fake), step=disc_ctx.opt_ctx.iteration)
			return disc_loss_real, disc_loss_fake, disc_scores_real, disc_scores_fake
		else:
			if disc_ctx.opt_ctx.record_summary:
				summary_names = disc_ctx.opt_ctx.make_summary_names('discriminator/score')
				disc_ctx.opt_ctx._tf_summary.scalar(summary_names[0], tf.reduce_mean(disc_scores), step=disc_ctx.opt_ctx.iteration)
			return disc_loss[0], disc_scores
	else:
		LOG.debug("Optimization discriminator inactive")
		return 0,0,0,0
