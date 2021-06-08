import copy, os
import tensorflow as tf
import numpy as np
from lib.tf_ops import shape_list, spacial_shape_list, tf_tensor_stats, tf_norm2, tf_angle_between
from lib.util import load_numpy
from .renderer import Renderer
from .transform import GridTransform
from .vector import GridShape, Vector3
import logging

LOG = logging.getLogger("Structs")

# --- DATA Structs ---

def get_coord_field(shape, offset=[0,0,0], lod=0.0, concat=True):
	'''
	shape: z,y,x
	offset: x,y,z
	returns: 1,z,y,x,c with c=x,z,y,lod
	'''
	coord_z, coord_y, coord_x = tf.meshgrid(tf.range(shape[0], dtype=tf.float32), tf.range(shape[1], dtype=tf.float32), tf.range(shape[2], dtype=tf.float32), indexing='ij') #z,y,x
	coord_data = [tf.reshape(coord_x + offset[0], [1]+shape+[1]),
		tf.reshape(coord_y + offset[1], [1]+shape+[1]),
		tf.reshape(coord_z + offset[2], [1]+shape+[1])] #3 x 1DHW1
	if lod is not None:
		lod_data = tf.constant(lod, shape=[1]+shape+[1], dtype=tf.float32) #tf.ones([1]+shape+[1])*lod
		coord_data.append(lod_data)#4 x 1DHW1
	if concat:
		coord_data = tf.concat(coord_data, axis=-1)

	
	return coord_data

class Zeroset:
	def __init__(self, initial_value, shape=None, as_var=True, outer_bounds="OPEN", device=None, var_name="zeroset", trainable=True):
		self.outer_bounds = outer_bounds
		self.is_var = as_var
		self._device = device
		self._name = var_name
		self._is_trainable = trainable
		
		with tf.device(self._device):
			if shape is not None:
				assert isinstance(shape, GridShape)
				initial_value = tf.constant(initial_value, shape=shape.value, dtype=tf.float32)
			if as_var:
				self._levelset = tf.Variable(initial_value=initial_value, name=var_name, trainable=trainable)
			else:
				self._levelset = tf.identity(initial_value)
	
	@property
	def grid_shape(self):
		return GridShape.from_tensor(self._levelset)
	
	def _hull_staggered_lerp_weight(self, a, b):
		a_leq = tf.less_equal(a,0)
		return tf.where( tf.logical_xor(a_leq, tf.less_equal(b,0)), #sign change along iterpolation
				tf.abs( tf.divide( tf.minimum(a,b), tf.subtract(a,b) ) ),
				tf.cast(a_leq, dtype=a.dtype)
			)
	
	def _hull_simple_staggered_component(self, axis):
		assert axis in [1,2,3,-2,-3,-4]
		axis = axis%5
		pad = [(0,0),(0,0),(0,0),(0,0),(0,0)]
		pad[axis]=(1,1)
		shape = self.grid_shape.value
		shape[axis] -= 1
		offset = np.zeros((5,), dtype=np.int32)
		cells_prev = tf.slice(self._levelset, offset, shape) #self._levelset[:,:,:,:-1,:]
		offset[axis] += 1
		cells_next = tf.slice(self._levelset, offset, shape) #self._levelset[:,:,:, 1:,:]
		hull = self._hull_staggered_lerp_weight(cells_prev,cells_next)
		hull = tf.pad(hull, pad, constant_values=1 if self.outer_bounds=="OPEN" else 0)
		return hull
	
	def to_hull_simple_staggered(self):
		return self._hull_simple_staggered_component(-2), self._hull_simple_staggered_component(-3), self._hull_simple_staggered_component(-4)
	
	def to_hull_simple_centered(self):
		raise NotImplementedError()
	
	def to_denstiy_simple_centered(self):
		return tf.where(tf.greater(self._levelset, 0), 250, 0)
	
	def resize(self, shape):
		assert shape_list(shape)==[3]
		new_shape = GridShape(shape)
		if new_shape==self.grid_shape:
			return
		raise NotImplementedError("Zeroset.resize() not implemented.")
	
	def assign(levelset):
		raise NotImplementedError()
		

class DensityGrid:
	def __init__(self, shape, constant=0.1, as_var=True, d=None, scale_renderer=None, hull=None, inflow=None, inflow_offset=None, inflow_mask=None, device=None, var_name="denstiy", trainable=True, restrict_to_hull=True):
		self.shape = shape
		if d is not None:
			d_shape = shape_list(d)
			if not len(d_shape)==5 or not d_shape[-1]==1 or not self.shape==spacial_shape_list(d):
				raise ValueError("Invalid shape of density on assignment: %s"%d_shape)
		self.is_var = as_var
		self._device = device
		self._name = var_name
		self._is_trainable = trainable
		if as_var:
			rand_init = tf.constant_initializer(constant)
			with tf.device(self._device):
				self._d = tf.Variable(initial_value=d if d is not None else rand_init(shape=[1]+self.shape+[1], dtype=tf.float32), name=var_name+'_dens', trainable=True)
			
		else:
			with tf.device(self._device):
				if d is not None:
					self._d = tf.constant(d, dtype=tf.float32)
				else:
					self._d = tf.constant(constant, shape=[1]+self.shape+[1], dtype=tf.float32)
			
		self.scale_renderer = scale_renderer
		with tf.device(self._device):
			self.hull = tf.constant(hull, dtype=tf.float32) if hull is not None else None
		self.restrict_to_hull = restrict_to_hull
		
		if inflow is not None:
			with tf.device(self._device):
				if isinstance(inflow, str) and inflow=='CONST':
					assert isinstance(inflow_mask, (tf.Tensor, np.ndarray))
					inflow = rand_init(shape=shape_list(inflow_mask), dtype=tf.float32)
				if as_var:
					self._inflow = tf.Variable(initial_value=inflow, name=var_name+'_inflow', trainable=True)
				else:
					self._inflow = tf.constant(inflow, dtype=tf.float32)
				self.inflow_mask = tf.constant(inflow_mask, dtype=tf.float32) if inflow_mask is not None else None
			inflow_shape = spacial_shape_list(self._inflow) #.get_shape().as_list()[-4:-1]
			self._inflow_padding = [[0,0]]+[[inflow_offset[_],self.shape[_]-inflow_offset[_]-inflow_shape[_]] for _ in range(3)]+[[0,0]]
			self.inflow_offset = inflow_offset
		else:
			self._inflow = None
	
	@property
	def trainable(self):
		return self._is_trainable and self.is_var
	
	@property
	def d(self):
		if self.restrict_to_hull:
			return self.with_hull()
		else:
			return tf.identity(self._d)
	
	def with_hull(self):
		if self.hull is not None:
			return self._d * self.hull # hull is a (smooth) binary mask
		else:
			return tf.identity(self._d)
	
	@property
	def inflow(self):
		if self._inflow is None:
			return tf.zeros_like(self._d, dtype=tf.float32)
		elif self.inflow_mask is not None: #hasattr(self, 'inflow_mask') and 
			return tf.pad(self._inflow*self.inflow_mask, self._inflow_padding)
		else:
			return tf.pad(self._inflow, self._inflow_padding)
	
	def with_inflow(self):
		density = self.d
		if self._inflow is not None:
			density = tf.maximum(density+self.inflow, 0)
		return density
	
	@classmethod
	def from_file(cls, path, as_var=True, scale_renderer=None, hull=None, inflow=None, inflow_offset=None, inflow_mask=None, device=None, var_name="denstiy", trainable=True, restrict_to_hull=True):
		try:
			with np.load(path) as np_data:
				d = np_data['arr_0']
				shape =spacial_shape_list(d)
				if 'hull' in np_data and hull is None:
					hull = np_data['hull']
				if 'inflow' in np_data and inflow is None:
					inflow=np_data['inflow']
					if 'inflow_mask' in np_data and inflow_mask is None:
						inflow_mask=np_data['inflow_mask']
					if 'inflow_offset' in np_data and inflow_offset is None:
						inflow_offset=np_data['inflow_offset'].tolist()
				grid = cls(shape, d=d, as_var=as_var, scale_renderer=scale_renderer, hull=hull, inflow=inflow, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
					device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull)
		except:
			LOG.warning("Failed to load density from '%s':", path, exc_info=True)
			return None
		else:
			return grid
		
	@classmethod
	def from_scalarFlow_file(cls, path, as_var=True, shape=None, scale_renderer=None, hull=None, inflow=None, inflow_offset=None, inflow_mask=None, device=None, var_name="sF_denstiy", trainable=True, restrict_to_hull=True):
		# if shape is set the loaded grid will be reshaped if necessary
		density = load_numpy(path).astype(np.float32)[::-1]
		density = density.reshape([1] + list(density.shape)) # 
		density = tf.constant(density, dtype=tf.float32)
		d_shape = spacial_shape_list(density)
		if shape is not None and shape!=d_shape:
			if scale_renderer is None:
				raise ValueError("No renderer provided to scale density.")
			LOG.debug("scaling scalarFlow density from %s to %s", d_shape, shape)
			density = scale_renderer.resample_grid3D_aligned(density, shape)
			d_shape = shape
		else:
			# cut of SF inflow region and set as inflow. or is it already cut off in SF dataset? it is, but not in the synth dataset or my own sF runs.
			# lower 15 cells...
			inflow, density= tf.split(density, [15, d_shape[1]-15], axis=-3)
			inflow_mask = tf.ones_like(inflow, dtype=tf.float32)
			inflow_offset = [0,0,0]
			density = tf.concat([tf.zeros_like(inflow, dtype=tf.float32), density], axis=-3)
		return cls(d_shape, d=density, as_var=as_var, scale_renderer=scale_renderer, hull=hull, inflow=inflow, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
			device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull)
	
	def copy(self, as_var=None, device=None, var_name=None, trainable=None, restrict_to_hull=None):
		if as_var is None:
			as_var = self.is_var
		if as_var and var_name is None:
			var_name = self._name + '_cpy'
		if trainable is None:
			trainable = self._is_trainable
		if restrict_to_hull is None:
			restrict_to_hull = self.restrict_to_hull
		if self._inflow is not None:
			grid = DensityGrid(self.shape, d=tf.identity(self._d), as_var=as_var, scale_renderer=self.scale_renderer, hull=self.hull, \
				inflow=tf.identity(self._inflow), inflow_offset=self.inflow_offset, inflow_mask=self.inflow_mask, \
				device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull)
		else:
			grid = DensityGrid(self.shape, d=tf.identity(self._d), as_var=as_var, scale_renderer=self.scale_renderer, hull=self.hull, \
				device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull)
		return grid
	
	def scaled(self, new_shape, with_inflow=False):
		if not (isinstance(new_shape, list) and len(new_shape)==3):
			raise ValueError("Invalid shape")
		density = self.d if not with_inflow else self.with_inflow()
		if new_shape!=self.shape:
			LOG.debug("Scaling density from %s to %s", self.shape, new_shape)
			with self.scale_renderer.profiler.sample("scale density"):
				d_scaled = self.scale_renderer.resample_grid3D_aligned(density, new_shape)
		else:
			LOG.debug("No need to scale density to same shape %s", self.shape)
			d_scaled = tf.identity(density)
		return d_scaled
	
	def copy_scaled(self, new_shape, as_var=None, device=None, var_name=None, trainable=None, restrict_to_hull=None):
		'''Does not copy inflow and hull, TODO'''
		if as_var is None:
			as_var = self.is_var
		if as_var and var_name is None:
			var_name = self._name + '_scaled'
		if trainable is None:
			trainable = self._is_trainable
		if restrict_to_hull is None:
			restrict_to_hull = self.restrict_to_hull
		d_scaled = self.scaled(new_shape)
		grid = DensityGrid(new_shape, d=d_scaled, as_var=as_var, scale_renderer=self.scale_renderer, device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull)
		return grid
	
	def warped(self, vel_grid, order=1, dt=1.0, clamp="NONE"):
		if not (isinstance(vel_grid, VelocityGrid)):
			raise ValueError("Invalid velocity grid")
		return vel_grid.warp(self.with_inflow(), order=order, dt=dt, clamp=clamp)
	
	def copy_warped(self, vel_grid, as_var=None, order=1, dt=1.0, device=None, var_name=None, clamp="NONE", trainable=None, restrict_to_hull=None):
		'''Does not copy inflow and hull, TODO'''
		if as_var is None:
			as_var = self.is_var
		if as_var and var_name is None:
			var_name = self._name + '_warped'
		if trainable is None:
			trainable = self._is_trainable
		if restrict_to_hull is None:
			restrict_to_hull = self.restrict_to_hull
		d_warped = self.warped(vel_grid, order=order, dt=dt, clamp=clamp)
		grid = DensityGrid(self.shape, d=d_warped, as_var=as_var, scale_renderer=self.scale_renderer, device=device, var_name=var_name, trainable=trainable, restrict_to_hull=restrict_to_hull)
		return grid
	
	def scale(self, scale):
		self.assign(self._d*scale)
	
	def apply_clamp(self, vmin, vmax):
		vmin = tf.maximum(vmin, 0)
		d = tf.clip_by_value(self._d, vmin, vmax)
		inflow = None
		if self._inflow is not None:
			# use already clamped density for consistency
			denstiy_shape = shape_list(d)
			density_cropped = d[self._inflow_padding[0][0] : denstiy_shape[0]-self._inflow_padding[0][1],
				self._inflow_padding[1][0] :  denstiy_shape[1]-self._inflow_padding[1][1],
				self._inflow_padding[2][0] :  denstiy_shape[2]-self._inflow_padding[2][1],
				self._inflow_padding[3][0] :  denstiy_shape[3]-self._inflow_padding[3][1],
				self._inflow_padding[4][0] :  denstiy_shape[4]-self._inflow_padding[4][1]]
			inflow = tf.clip_by_value(self._inflow, vmin - density_cropped, vmax - density_cropped)
		self.assign(d, inflow)
	
	def assign(self, d, inflow=None):
		shape = shape_list(d)
		if not len(shape)==5 or not shape[-1]==1 or not shape[-4:-1]==self.shape:
			raise ValueError("Invalid or incompatible shape of density on assignment: is {}, required: NDHW1 with DHW={}".format(shape, self.shape))
		if self.is_var:
			self._d.assign(d)
			if self._inflow is not None and inflow is not None:
				self._inflow.assign(inflow)
		else:
			with tf.device(self._device):
				self._d = tf.identity(d)
				if self._inflow is not None and inflow is not None:
					self._inflow = tf.identity(inflow)
	
	def var_list(self):
		if self.is_var:
			if self._inflow is not None:
				return [self._d, self._inflow]
			return [self._d]
		else:
			raise TypeError("This DensityGrid is not a variable.")
	
	def get_variables(self):
		if self.is_var:
			var_dict = {'density': self._d}
			if self._inflow is not None:
				var_dict['inflow'] = self._inflow
			return var_dict
		else:
			raise TypeError("This DensityGrid is not a variable.")
		
	
	def save(self, path):
		density = self._d
		if isinstance(density, (tf.Tensor, tf.Variable)):
			density = density.numpy()
		save = {}
		if self.hull is not None:
			hull = self.hull
			if isinstance(hull, (tf.Tensor, tf.Variable)):
				hull = hull.numpy()
			save['hull']=hull
		if self._inflow is not None:
			inflow = self._inflow
			if isinstance(inflow, (tf.Tensor, tf.Variable)):
				inflow = inflow.numpy()
			save['inflow']=inflow
			if self.inflow_mask is not None:
				inflow_mask = self.inflow_mask
				if isinstance(inflow_mask, (tf.Tensor, tf.Variable)):
					inflow_mask = inflow_mask.numpy()
				save['inflow_mask']=inflow_mask
			save['inflow_offset']=np.asarray(self.inflow_offset)
		np.savez_compressed(path, density, **save)
	
	def mean(self):
		return tf.reduce_mean(self.d)
	
	def stats(self, mask=None, state=None, **warp_kwargs):
		'''
			mask: optional binary float mask, stats only consider cells>0.5
		'''
		d = self.d
		if mask is not None:
			mask =  mask if mask.dtype==tf.bool else tf.greater(mask, 0.5)
			d = tf.boolean_mask(d, mask)
		
		stats = {
			'density': tf_tensor_stats(d, as_dict=True),
			'shape':self.shape,
		}
		if state is not None and state.prev is not None and state.prev.density is not None and state.prev.velocity is not None:
			warp_SE = tf.squared_difference(state.prev.density_advected(**warp_kwargs), self.d)
			if mask is not None:
				warp_SE = tf.boolean_mask(warp_SE, mask)
			stats["warp_SE"] = tf_tensor_stats(warp_SE, as_dict=True)
		else:
			stats["warp_SE"] = tf_tensor_stats(tf.zeros([1,1,1,1,1], dtype=tf.float32), as_dict=True)
		return stats

class VelocityGrid:
	@staticmethod
	def component_shapes(centered_shape):
		x_shape = copy.copy(centered_shape)
		x_shape[2] +=1
		y_shape = copy.copy(centered_shape)
		y_shape[1] +=1
		z_shape = copy.copy(centered_shape)
		z_shape[0] +=1
		return x_shape, y_shape, z_shape
		
	def __init__(self, centered_shape, std=0.1, as_var=True, x=None, y=None, z=None, boundary=None, scale_renderer=None, warp_renderer=None, *, coords=None, lod=None, device=None, var_name="velocity", trainable=True):
		self.centered_shape = centered_shape.tolist() if isinstance(centered_shape, np.ndarray) else centered_shape
		self.x_shape, self.y_shape, self.z_shape = VelocityGrid.component_shapes(self.centered_shape)
		self.set_boundary(boundary)
		self.is_var = as_var
		self._device = device
		self._name = var_name
		self._is_trainable = trainable
		if as_var:
			if x is not None:
				x_shape = shape_list(x)
				if not len(x_shape)==5 or not x_shape[-1]==1 or not x_shape[-4:-1]==self.x_shape:
					raise ValueError("Invalid shape of velocity x component on assignment")
			if y is not None:
				y_shape = shape_list(y)
				if not len(y_shape)==5 or not y_shape[-1]==1 or not y_shape[-4:-1]==self.y_shape:
					raise ValueError("Invalid shape of velocity y component on assignment")
			if z is not None:
				z_shape = shape_list(z)
				if not len(z_shape)==5 or not z_shape[-1]==1 or not z_shape[-4:-1]==self.z_shape:
					raise ValueError("Invalid shape of velocity z component on assignment")
			# in a box
			#rand_init = tf.random_normal_initializer(0.0, std)
			std = tf.abs(std)
			rand_init = tf.random_uniform_initializer(-std, std)
			# maybe even uniformly in space and in a sphere?: http://6degreesoffreedom.co/circle-random-sampling/
			with tf.device(self._device):
				self._x = tf.Variable(initial_value=x if x is not None else rand_init(shape=[1]+self.x_shape+[1], dtype=tf.float32), name=var_name + '_x', trainable=True)
				self._y = tf.Variable(initial_value=y if y is not None else rand_init(shape=[1]+self.y_shape+[1], dtype=tf.float32), name=var_name + '_y', trainable=True)
				self._z = tf.Variable(initial_value=z if z is not None else rand_init(shape=[1]+self.z_shape+[1], dtype=tf.float32), name=var_name + '_z', trainable=True)
		else:
			if x is None:
				x = tf.constant(tf.random.uniform([1]+self.x_shape+[1], -std, std, dtype=tf.float32))
			if y is None:
				y = tf.constant(tf.random.uniform([1]+self.y_shape+[1], -std, std, dtype=tf.float32))
			if z is None:
				z = tf.constant(tf.random.uniform([1]+self.z_shape+[1], -std, std, dtype=tf.float32))
			self.assign(x,y,z)
		
		if lod is None:
			lod = tf.zeros([1]+self.centered_shape+[1])
		with tf.device(self._device):
			self.lod_pad = tf.identity(lod)
		
		self.scale_renderer = scale_renderer
		if self.scale_renderer is not None:
			if (self.outer_bounds=='CLOSED' and self.scale_renderer.boundary_mode!='BORDER') \
				or (self.outer_bounds=='OPEN' and self.scale_renderer.boundary_mode!='CLAMP'):
				LOG.warning("Velocity outer boundary %s does not match scale renderer boundary mode %s", self.outer_bounds, self.scale_renderer.boundary_mode)
		self.warp_renderer = warp_renderer
		if self.warp_renderer is not None:
			if (self.outer_bounds=='CLOSED' and self.warp_renderer.boundary_mode!='BORDER') \
				or (self.outer_bounds=='OPEN' and self.warp_renderer.boundary_mode!='CLAMP'):
				LOG.warning("Velocity outer boundary %s does not match scale renderer boundary mode %s", self.outer_bounds, self.warp_renderer.boundary_mode)
	
	def set_boundary(self, boundary):
		assert (boundary is None) or isinstance(boundary, Zeroset)
		self.boundary = boundary
		self.outer_bounds = self.boundary.outer_bounds if self.boundary is not None else "OPEN"
	
	@property
	def trainable(self):
		return self._is_trainable and self.is_var
	
	@property
	def x(self):
		v = self._x
		if self.boundary is not None:
			v*= self.boundary._hull_simple_staggered_component(-2)
		return v
	@property
	def y(self):
		v = self._y
		if self.boundary is not None:
			v*= self.boundary._hull_simple_staggered_component(-3)
		return v
	@property
	def z(self):
		v = self._z
		if self.boundary is not None:
			v*= self.boundary._hull_simple_staggered_component(-4)
		return v
	
	@classmethod
	def from_centered(cls, centered_grid, as_var=True, boundary=None, scale_renderer=None, warp_renderer=None, device=None, var_name="velocity", trainable=True):
		centered_shape = shape_list(centered_grid)
		assert len(centered_shape)==5
		assert centered_shape[-1]==3
		assert centered_shape[0]==1
		centered_shape = centered_shape[-4:-1]
		vel_grid = cls(centered_shape, as_var=as_var, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name, trainable=trainable)
		x,y,z = vel_grid._centered_to_staggered(centered_grid)
		vel_grid.assign(x,y,z)
		return vel_grid
		
	@classmethod
	def from_file(cls, path, as_var=True, boundary=None, scale_renderer=None, warp_renderer=None, device=None, var_name="velocity", trainable=True):
		try:
			with np.load(path) as vel:
				if 'centered_shape' not in vel:#legacy
					shape = shape_list(vel["vel_x"])
					LOG.debug("%s", shape)
					shape[-2] -=1
					shape = shape[1:-1]
				else:
					shape = vel['centered_shape'].tolist()
				vel_grid = cls(shape, x=vel["vel_x"].astype(np.float32), y=vel["vel_y"].astype(np.float32), z=vel["vel_z"].astype(np.float32), \
					as_var=as_var, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name, trainable=trainable)
		except:
			LOG.warning("Failed to load velocity from '%s':", path, exc_info=True)
			return None
		else:
			return vel_grid
	
	@classmethod
	def from_scalarFlow_file(cls, path, as_var=True, shape=None, boundary=None, scale_renderer=None, warp_renderer=None, device=None, var_name="sF_velocity", trainable=True):
		# sF velocities are stored as combined staggered grid with upper cells missing, DHWC with C=3
		velocity = load_numpy(path).astype(np.float32)[::-1]
		v_shape = GridShape.from_tensor(velocity)
		velocity = v_shape.normalize_tensor_shape(velocity) #.reshape([1] + list(velocity.shape)) # NDHWC
		velocity = tf.constant(velocity, dtype=tf.float32)
		v_shape = v_shape.zyx.value
		v_x, v_y, v_z = tf.split(velocity, 3, axis=-1)
		p0 = (0,0)
		# extend missing upper cell
		v_x = tf.pad(v_x, [p0,p0,p0,(0,1),p0], "SYMMETRIC")
		v_y = tf.pad(v_y, [p0,p0,(0,1),p0,p0], "SYMMETRIC")
		v_z = tf.pad(-v_z, [p0,(1,0),p0,p0,p0], "SYMMETRIC") #z value/direction reversed, pad lower value as axis is reversed (?)
		#v_shape = spacial_shape_list(velocity)
		if shape is not None and v_shape!=shape:
			assert len(shape)==3
			if scale_renderer is None:
				raise ValueError("No renderer provided to scale velocity.")
		#	shape = GridShape(shape).zyx
		#	vel_scale = shape/v_shape #[o/i for i,o in zip(v_shape, shape)] #z,y,x
			LOG.debug("scaling scalarFlow velocity from %s to %s with magnitude scale %s", v_shape, shape)
			v_tmp = cls(v_shape, x=v_x, y=v_y, z=v_z, as_var=False, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name="sF_tmp", trainable=False)
			v_x, v_y, v_z = v_tmp.scaled(shape, scale_magnitude=True)
			# can only scale 1 and 4 channel grids
		#	v_x = scale_renderer.resample_grid3D_aligned(v_x, shape.value)*vel_scale.x#[2]
		#	v_y = scale_renderer.resample_grid3D_aligned(v_y, shape.value)*vel_scale.y#[1]
		#	v_z = scale_renderer.resample_grid3D_aligned(v_z, shape.value)*vel_scale.z#[0]
		#	velocity = tf.concat([v_x, v_y, v_z], axis=-1)
			v_shape = shape
		
		#return cls.from_centered(velocity,as_var=as_var, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name)
		return cls(v_shape, x=v_x, y=v_y, z=v_z,as_var=as_var, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name, trainable=trainable)
	
	def copy(self, as_var=None, device=None, var_name=None, trainable=None):
		if as_var is None:
			as_var = self.is_var
		if as_var and var_name is None:
			var_name = self._name + '_cpy'
		if trainable is None:
			trainable = self._is_trainable
		grid = VelocityGrid(self.centered_shape, x=tf.identity(self._x), y=tf.identity(self._y), z=tf.identity(self._z), as_var=as_var, \
			boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=device, var_name=var_name, trainable=trainable)
		return grid
		
	
	def scaled(self, centered_shape, scale_magnitude=True):
		if not (isinstance(centered_shape, list) and len(centered_shape)==3):
			raise ValueError("Invalid shape")
		#resample velocity
		if centered_shape!=self.centered_shape:
			with self.scale_renderer.profiler.sample("scale velocity"):
				x_shape, y_shape, z_shape = VelocityGrid.component_shapes(centered_shape)
				LOG.debug("Scaling velocity from %s to %s", self.centered_shape, centered_shape)
				x_scaled = self.scale_renderer.resample_grid3D_aligned(self.x, x_shape, align_x='center')
				y_scaled = self.scale_renderer.resample_grid3D_aligned(self.y, y_shape, align_y='center')
				z_scaled = self.scale_renderer.resample_grid3D_aligned(self.z, z_shape, align_z='center')
				if scale_magnitude:
					vel_scale = [o/i for i,o in zip(self.centered_shape, centered_shape)] #z,y,x
					LOG.debug("Scaling velocity magnitude with %s", vel_scale)
					x_scaled *= vel_scale[2]
					y_scaled *= vel_scale[1]
					z_scaled *= vel_scale[0]
		else:
			LOG.debug("No need to scale velocity to same shape %s", self.centered_shape)
			x_scaled = tf.identity(self.x)
			y_scaled = tf.identity(self.y)
			z_scaled = tf.identity(self.z)
		return x_scaled, y_scaled, z_scaled
	
	def copy_scaled(self, centered_shape, scale_magnitude=True, as_var=None, device=None, var_name=None, trainable=None):
		if as_var is None:
			as_var = self.is_var
		if as_var and var_name is None:
			var_name = self._name + '_scaled'
		if trainable is None:
			trainable = self._is_trainable
		x_scaled, y_scaled, z_scaled = self.scaled(centered_shape, scale_magnitude)
		grid = VelocityGrid(centered_shape, x=x_scaled, y=y_scaled, z=z_scaled, as_var=as_var, \
			boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=device, var_name=var_name, trainable=trainable)
		return grid
	
	def _lut_warp_vel(self, shape, dt=1.0):
		# use to get lookup positions to warp velocity components
		vel = self._sampled_to_shape(shape) #3 x 1DHW1
		vel_lut = [- vel[i]*dt for i in range(len(vel))] #3 x 1DHW1
		vel_lut = tf.concat(vel_lut, axis = -1) #1DHW3
		return vel_lut
	
	def _warp_vel_component(self, data, lut, order=1, dt=1.0, clamp="NONE"):
		if order<1 or order>2:
			raise ValueError("Unsupported warp order '{}'".format(order))
		warped = self.warp_renderer._sample_LuT(data, lut, True, relative=True)
		clamp = clamp.upper()
		if order==2: #MacCormack
			warped_back = self.warp_renderer._sample_LuT(warped, -lut, True, relative=True)
			corrected = warped + 0.5*(data-warped_back)
			if clamp=="MC" or clamp=="MC_SMOOTH":
				#raise NotImplementedError("MacCormack clamping has not been implemented.")
				fm = self.warp_renderer.filter_mode
				self.warp_renderer.filter_mode = "MIN"
				data_min = self.warp_renderer._sample_LuT(data, lut, True, relative=True)
				self.warp_renderer.filter_mode = "MAX"
				data_max = self.warp_renderer._sample_LuT(data, lut, True, relative=True)
				self.warp_renderer.filter_mode = fm
				if clamp=='MC':
					#LOG.warning("Experimental clamp for MacCormack velocity advection.")
					raise NotImplementedError("MIM and MAX warp sampling have wrong gradients.")
					corrected = tf.clip_by_value(corrected, data_min, data_max)
				if clamp=='MC_SMOOTH':
					#LOG.warning("Experimental 'revert' clamp for MacCormack velocity advection.")
					clamp_OOB = tf.logical_or(tf.less(corrected, data_min), tf.greater(corrected, data_max))
					corrected = tf.where(clamp_OOB, warped, corrected)
			warped = corrected
		return warped
	
	def warped(self, vel_grid=None, order=1, dt=1.0, clamp="NONE"):
		if vel_grid is None:
			#vel_grid = self
			pass
		elif not isinstance(vel_grid, VelocityGrid):
			raise TypeError("Invalid VelocityGrid")
		with self.warp_renderer.profiler.sample("warp velocity"):
			LOG.debug("Warping velocity grid")
			#TODO will cause errors if grid shapes do not match, resample if necessary?
			if vel_grid is None:
				lut_x = tf.concat([-vel*dt for vel in self._sampled_to_component_shape('X', concat=False)], axis=-1)
			else:
				lut_x = vel_grid._lut_warp_vel(self.x_shape, dt)
			x_warped = self._warp_vel_component(self.x, lut_x, order=order, dt=dt, clamp=clamp)
			del lut_x
			
			if vel_grid is None:
				lut_y = tf.concat([-vel*dt for vel in self._sampled_to_component_shape('Y', concat=False)], axis=-1)
			else:
				lut_y = vel_grid._lut_warp_vel(self.y_shape, dt)
			y_warped = self._warp_vel_component(self.y, lut_y, order=order, dt=dt, clamp=clamp)
			del lut_y
			
			
			if vel_grid is None:
				lut_z = tf.concat([-vel*dt for vel in self._sampled_to_component_shape('Z', concat=False)], axis=-1)
			else:
				lut_z = vel_grid._lut_warp_vel(self.z_shape, dt)
			z_warped = self._warp_vel_component(self.z, lut_z, order=order, dt=dt, clamp=clamp)
			del lut_z
		return x_warped, y_warped, z_warped
	
	def copy_warped(self, vel_grid=None, as_var=None, order=1, dt=1.0, device=None, var_name=None, clamp="NONE", trainable=None):
		if as_var is None:
			as_var = self.is_var
		if as_var and var_name is None:
			var_name = self._name + '_warped'
		if trainable is None:
			trainable = self._is_trainable
		x_warped, y_warped, z_warped = self.warped(vel_grid, order, dt, clamp=clamp)
		grid = VelocityGrid(self.centered_shape, x=x_warped, y=y_warped, z=z_warped, as_var=as_var, \
			boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=device, var_name=var_name, trainable=trainable)
		return grid
	
	def divergence_free(self, residual=1e-5):
		raise NotImplementedError
	
	def var_list(self):
		if self.is_var:
			return [self._x, self._y, self._z]
		else:
			raise TypeError("This VelocityGrid is not a variable.")
	
	def get_variables(self):
		if self.is_var:
			return {'velocity_x': self._x, 'velocity_y': self._y, 'velocity_z': self._z}
		else:
			raise TypeError("This VelocityGrid is not a variable.")
	
	def save(self, path):
		np.savez_compressed(path, centered_shape=self.centered_shape, vel_x=self.x.numpy(), vel_y=self.y.numpy(), vel_z=self.z.numpy())
	
	
	def assign(self, x,y,z):
		x_shape = shape_list(x)
		if not len(x_shape)==5 or not x_shape[-1]==1 or not x_shape[-4:-1]==self.x_shape:
			raise ValueError("Invalid or incompatible shape of velocity x component on assignment: is {}, required: NDHW1 with DHW={}".format(x_shape, self.x_shape))
		y_shape = shape_list(y)
		if not len(y_shape)==5 or not y_shape[-1]==1 or not y_shape[-4:-1]==self.y_shape:
			raise ValueError("Invalid or incompatible shape of velocity y component on assignment: is {}, required: NDHW1 with DHW={}".format(y_shape, self.y_shape))
		z_shape = shape_list(z)
		if not len(z_shape)==5 or not z_shape[-1]==1 or not z_shape[-4:-1]==self.z_shape:
			raise ValueError("Invalid or incompatible shape of velocity z component on assignment: is {}, required: NDHW1 with DHW={}".format(z_shape, self.z_shape))
		if self.is_var:
			self._x.assign(x)
			self._y.assign(y)
			self._z.assign(z)
		else:
			with tf.device(self._device):
				self._x = tf.identity(x)
				self._y = tf.identity(y)
				self._z = tf.identity(z)
	
	def assign_add(self, x,y,z):
		x_shape = shape_list(x)
		if not len(x_shape)==5 or not x_shape[-1]==1 or not x_shape[-4:-1]==self.x_shape:
			raise ValueError("Invalid or incompatible shape of velocity x component on assignment: is {}, required: NDHW1 with DHW={}".format(x_shape, self.x_shape))
		y_shape = shape_list(y)
		if not len(y_shape)==5 or not y_shape[-1]==1 or not y_shape[-4:-1]==self.y_shape:
			raise ValueError("Invalid or incompatible shape of velocity y component on assignment: is {}, required: NDHW1 with DHW={}".format(y_shape, self.y_shape))
		z_shape = shape_list(z)
		if not len(z_shape)==5 or not z_shape[-1]==1 or not z_shape[-4:-1]==self.z_shape:
			raise ValueError("Invalid or incompatible shape of velocity z component on assignment: is {}, required: NDHW1 with DHW={}".format(z_shape, self.z_shape))
		if self.is_var:
			self._x.assign_add(x)
			self._y.assign_add(y)
			self._z.assign_add(z)
		else:
			with tf.device(self._device):
				self._x = tf.identity(self._x+x)
				self._y = tf.identity(self._y+y)
				self._z = tf.identity(self._z+z)
	
	def assign_sub(self, x,y,z):
		x_shape = shape_list(x)
		if not len(x_shape)==5 or not x_shape[-1]==1 or not x_shape[-4:-1]==self.x_shape:
			raise ValueError("Invalid or incompatible shape of velocity x component on assignment: is {}, required: NDHW1 with DHW={}".format(x_shape, self.x_shape))
		y_shape = shape_list(y)
		if not len(y_shape)==5 or not y_shape[-1]==1 or not y_shape[-4:-1]==self.y_shape:
			raise ValueError("Invalid or incompatible shape of velocity y component on assignment: is {}, required: NDHW1 with DHW={}".format(y_shape, self.y_shape))
		z_shape = shape_list(z)
		if not len(z_shape)==5 or not z_shape[-1]==1 or not z_shape[-4:-1]==self.z_shape:
			raise ValueError("Invalid or incompatible shape of velocity z component on assignment: is {}, required: NDHW1 with DHW={}".format(z_shape, self.z_shape))
		if self.is_var:
			self._x.assign_sub(x)
			self._y.assign_sub(y)
			self._z.assign_sub(z)
		else:
			with tf.device(self._device):
				self._x = tf.identity(self._x-x)
				self._y = tf.identity(self._y-y)
				self._z = tf.identity(self._z-z)
	
	def scale_magnitude(self, scale):
		if np.isscalar(scale):
			scale = [scale]*3
		assert len(scale)==3
		self.assign(self.x*scale[0],self.y*scale[1], self.z*scale[2])
	
	def _centered_to_staggered(self, centered):
		centered_shape = shape_list(centered)
		assert len(centered_shape)==5
		assert centered_shape[-1]==3
		assert centered_shape[0]==1
		assert self.centered_shape==centered_shape[-4:-1]
		with self.scale_renderer.profiler.sample("centered velocity to staggered"):
			x,y,z= tf.split(centered, 3, axis=-1)
			centered_x_transform = GridTransform(self.centered_shape, scale=[2./_ for _ in self.x_shape[::-1]], center=True)
			centered_y_transform = GridTransform(self.centered_shape, scale=[2./_ for _ in self.y_shape[::-1]], center=True)
			centered_z_transform = GridTransform(self.centered_shape, scale=[2./_ for _ in self.z_shape[::-1]], center=True)
			# only shape important here
			staggered_x_transform = GridTransform(self.x_shape)#,translation=[0.5,0,0])
			staggered_y_transform = GridTransform(self.y_shape)#,translation=[0,0.5,0])
			staggered_z_transform = GridTransform(self.z_shape)#,translation=[0,0,0.5])
			x = tf.squeeze(self.scale_renderer._sample_transform(x, [centered_x_transform], [staggered_x_transform]),1)
			y = tf.squeeze(self.scale_renderer._sample_transform(y, [centered_y_transform], [staggered_y_transform]),1)
			z = tf.squeeze(self.scale_renderer._sample_transform(z, [centered_z_transform], [staggered_z_transform]),1)
		return x,y,z
	
	def _staggeredTensor_to_components(self, tensor, reverse=False):
		tensor_shape = GridShape.from_tensor(tensor)
	#	assert len(tensor_shape)==5
		assert tensor_shape.c==3
		assert tensor_shape.n==1
		assert np.asarray(self.tensor_shape)+np.asarray([1,1,1])== tensor_shape.xyz.as_shape() #tensor_shape[-4:-1]
		tensor = tensor_shape.normalize_tensor_shape(tensor)
		components = tf.split(tensor, 3, axis=-1)
		if reverse:
			components = components[::-1]
		x = components[0][:,:-1,:-1,:]
		y = components[0][:,:-1,:,:-1]
		z = components[0][:,:,:-1,:-1]
		return x,y,z
		
	def as_staggeredTensor(self, reverse=False):
		z = (0,0)
		p = (0,1)
		components = [
			tf.pad(self.x, [z,p,p,z,z]),
			tf.pad(self.y, [z,p,z,p,z]),
			tf.pad(self.z, [z,z,p,p,z]),
		]
		if reverse:
			components = components[::-1]
		return tf.concat(components, axis=-1)
	
	def _sampled_to_shape(self, shape):
		with self.scale_renderer.profiler.sample("velocity to shape"):
			# uniform scaling, centered grids
			#_sample_transform assumes the output grid to be in a centered [-1,1] cube, so scale input accordingly
			# scale with output shape to get the right 0.5 offset
			scale = [2./_ for _ in shape[::-1]]
			staggered_x_transform = GridTransform(self.x_shape, scale=scale, center=True)
			staggered_y_transform = GridTransform(self.y_shape, scale=scale, center=True)
			staggered_z_transform = GridTransform(self.z_shape, scale=scale, center=True)
			# only shape important here
			sample_transform = GridTransform(shape)
			#check if shape matches component shape to avoid sampling (e.g. for self warping)
			vel_sampled = [
				tf.squeeze(self.scale_renderer._sample_transform(self.x, [staggered_x_transform], [sample_transform]),1) \
					if not shape==self.x_shape else tf.identity(self.x), #1DHW1
				tf.squeeze(self.scale_renderer._sample_transform(self.y, [staggered_y_transform], [sample_transform]),1) \
					if not shape==self.y_shape else tf.identity(self.y),
				tf.squeeze(self.scale_renderer._sample_transform(self.z, [staggered_z_transform], [sample_transform]),1) \
					if not shape==self.z_shape else tf.identity(self.z),
			]
		return vel_sampled
	
	def centered(self, pad_lod=False, concat=True):#, shape=None):
		shape = self.centered_shape
		with self.warp_renderer.profiler.sample("velocity to centered"):
			#vel_centered = self._sampled_to_shape(shape)#3 x 1DHW1
			h = tf.constant(0.5, dtype=tf.float32)
			vel_centered = [
				(self.x[:,:,:,1:] + self.x[:,:,:,:-1])*h,
				(self.y[:,:,1:] + self.y[:,:,:-1])*h,
				(self.z[:,1:] + self.z[:,:-1])*h,
			]
			if pad_lod:
				vel_centered.append(self.lod_pad)#4 x 1DHW1
			if concat:
				vel_centered = tf.concat(vel_centered, axis=-1) #1DHW[3|4]
		return vel_centered
	
	def _sampled_to_component_shape(self, component, pad_lod=False, concat=True):
		# grids have the same spacing/resolution, so global/constant offset
		component = component.upper()
		offset_coord_from = 0.5
		offset_coord_to = -0.5
		with self.warp_renderer.profiler.sample("velocity to component shape"):
			vel_sampled = []
			# sample x
			vel_sampled.append(tf.identity(self.x) if component=='X' else \
				tf.squeeze(self.warp_renderer.resample_grid3D_offset(self.x, \
					offsets = [[offset_coord_from,offset_coord_to,0.0] if component=='Y' else [offset_coord_from,0.0,offset_coord_to],], \
					target_shape = self.y_shape if component=='Y' else self.z_shape), 1))
			# sample y
			vel_sampled.append(tf.identity(self.y) if component=='Y' else \
				tf.squeeze(self.warp_renderer.resample_grid3D_offset(self.y, \
					offsets = [[offset_coord_to,offset_coord_from,0.0] if component=='X' else [0.0,offset_coord_from,offset_coord_to],], \
					target_shape = self.x_shape if component=='X' else self.z_shape), 1))
			# sample z
			vel_sampled.append(tf.identity(self.z) if component=='Z' else \
				tf.squeeze(self.warp_renderer.resample_grid3D_offset(self.z, \
					offsets = [[offset_coord_to,0.0,offset_coord_from] if component=='X' else [0.0,offset_coord_to,offset_coord_from],], \
					target_shape = self.x_shape if component=='X' else self.y_shape), 1))
			
			if pad_lod:
				vel_sampled.append(self.lod_pad)#4 x 1DHW1
			if concat:
				vel_sampled = tf.concat(vel_sampled, axis=-1) #1DHW[3|4]
		return vel_sampled
	
	def centered_lut_grid(self, dt=1.0):
		vel_centered = self.centered()
		#vel_lut = tf.concat([self.coords - vel_centered * dt, self.lod_pad], axis = -1)
		vel_lut = vel_centered * (- dt)
		return vel_lut
	
	def warp(self, data, order=1, dt=1.0, clamp="NONE"):
		with self.warp_renderer.profiler.sample("warp scalar"):
			v = self.centered_lut_grid(dt)
			data_shape = spacial_shape_list(data)
			if data_shape!=self.centered_shape:
				raise ValueError("Shape mismatch")
			LOG.debug("Warping density grid")
			data_warped = self.warp_renderer._sample_LuT(data, v, True, relative=True)
			
			clamp = clamp.upper()
			if order==2: #MacCormack
				data_warped_back = self.warp_renderer._sample_LuT(data_warped, -v, True, relative=True)
				data_corr = data_warped + 0.5*(data-data_warped_back)
				if clamp=='MC' or clamp=='MC_SMOOTH': #smooth clamp
					fm = self.warp_renderer.filter_mode
					self.warp_renderer.filter_mode = "MIN"
					data_min = self.warp_renderer._sample_LuT(data, v, True, relative=True)
					self.warp_renderer.filter_mode = "MAX"
					data_max = self.warp_renderer._sample_LuT(data, v, True, relative=True)
					self.warp_renderer.filter_mode = fm
					if clamp=='MC':
						#LOG.warning("Experimental clamp for MacCormack density advection.")
						raise NotImplementedError("MIM and MAX warp sampling have wrong gradients.")
						data_corr = tf.clip_by_value(data_corr, data_min, data_max)
					if clamp=='MC_SMOOTH':
						#LOG.warning("Experimental 'revert' clamp for MacCormack density advection.")
						clamp_OOB = tf.logical_or(tf.less(data_corr, data_min), tf.greater(data_corr, data_max))
						data_corr = tf.where(clamp_OOB, data_warped, data_corr)
				data_warped = data_corr
			elif order>2:
				raise ValueError("Unsupported warp order '{}'".format(order))
			
			if clamp=='NEGATIVE':
				data_warped = tf.maximum(data_warped, 0)
			
			return data_warped
	
	def with_buoyancy(self, value, scale_grid):
		# value: [x,y,z]
		# scale_grid: density 1DHW1
		if isinstance(scale_grid, DensityGrid):
			scale_grid = scale_grid.with_inflow() #.d
		assert len(shape_list(value))==1
		if not isinstance(value, (tf.Tensor, tf.Variable)):
			value = tf.constant(value, dtype=tf.float32)
		value = tf.reshape(value, [1,1,1,1,shape_list(value)[0]])
		buoyancy = value*scale_grid # 1DHW3
		return self + buoyancy
	
	"""
	def apply_buoyancy(self, value, scale_grid):
		# value: [x,y,z]
		# scale_grid: density 1DHW1
		assert len(shape_list(value))==1
		value = tf.reshape(tf.constant(value, dtype=tf.float32), [1,1,1,1,shape_list(value)[0]])
		buoyancy = value*scale_grid # 1DHW3
		self += buoyancy
	"""
	#centered
	def divergence(self, world_scale=[1,1,1]):
		#out - in per cell, per axis
		x_div = self.x[:,:,:,1:,:] - self.x[:,:,:,:-1,:]
		y_div = self.y[:,:,1:,:,:] - self.y[:,:,:-1,:,:]
		z_div = self.z[:,1:,:,:,:] - self.z[:,:-1,:,:,:]
		# sum to get total divergence per cell
		div = x_div*world_scale[0]+y_div*world_scale[1]+z_div*world_scale[2]
		return div
	#centered
	def magnitude(self, world_scale=[1,1,1]):
		with self.warp_renderer.profiler.sample("magnitude"):
			v = self.centered(pad_lod=False)*tf.constant(world_scale, dtype=tf.float32)
			return tf_norm2(v, axis=-1, keepdims=True)
	
	def stats(self, world_scale=[1,1,1], mask=None, state=None, **warp_kwargs):
		'''
			mask: optional binary float mask, stats only consider cells>0.5
		'''
		x = self.x
		if mask is not None:
			mask_x = tf.greater(self.scale_renderer.resample_grid3D_aligned(mask, self.x_shape, align_x='stagger_output'), 0.5)
			x = tf.boolean_mask(x, mask_x)
		y = self.y
		if mask is not None:
			mask_y = tf.greater(self.scale_renderer.resample_grid3D_aligned(mask, self.y_shape, align_y='stagger_output'), 0.5)
			y = tf.boolean_mask(y, mask_y)
		z = self.z
		if mask is not None:
			mask_z = tf.greater(self.scale_renderer.resample_grid3D_aligned(mask, self.z_shape, align_z='stagger_output'), 0.5)
			z = tf.boolean_mask(z, mask_z)
		if mask is not None and mask.dtype!=tf.bool:
			mask = tf.greater(mask, 0.5)
		
		divergence = self.divergence(world_scale)
		if mask is not None: divergence = tf.boolean_mask(divergence, mask)
		magnitude = self.magnitude(world_scale)
		if mask is not None: magnitude = tf.boolean_mask(magnitude, mask)
		
		stats = {
			'divergence': tf_tensor_stats(divergence, as_dict=True),
			'magnitude': tf_tensor_stats(magnitude, as_dict=True),
			'velocity_x': tf_tensor_stats(x, as_dict=True),
			'velocity_y': tf_tensor_stats(y, as_dict=True),
			'velocity_z': tf_tensor_stats(z, as_dict=True),
			'shape':self.centered_shape, 'bounds':self.outer_bounds,
		}
		
		if state is not None and state.prev is not None and state.prev.velocity is not None:
			prev_warped = state.prev.velocity_advected(**warp_kwargs)
			
			def vel_warp_SE_stats(prev, curr, mask):
				warp_SE = tf.squared_difference(prev, curr)
				if mask is not None:
					warp_SE = tf.boolean_mask(warp_SE, mask)
				return tf_tensor_stats(warp_SE, as_dict=True)
			stats["warp_x_SE"] = vel_warp_SE_stats(prev_warped.x, self.x, mask_x if mask is not None else None)
			stats["warp_y_SE"] = vel_warp_SE_stats(prev_warped.y, self.y, mask_y if mask is not None else None)
			stats["warp_z_SE"] = vel_warp_SE_stats(prev_warped.z, self.z, mask_z if mask is not None else None)
			
			warp_vdiff_mag = (prev_warped-self).magnitude()
			if mask is not None:
				warp_vdiff_mag = tf.boolean_mask(warp_vdiff_mag, mask)
			stats["warp_vdiff_mag"] = tf_tensor_stats(warp_vdiff_mag, as_dict=True)
			del warp_vdiff_mag
			
			vel_CangleRad_mask = tf.greater(state.prev.velocity.magnitude() * self.magnitude(), 1e-8)
			if mask is not None:
				vel_CangleRad_mask = tf.logical_and(mask, vel_CangleRad_mask)
			warp_CangleRad = tf_angle_between(state.prev.velocity.centered(), self.centered(), axis=-1, keepdims=True)
			stats["warp_angleCM_rad"] = tf_tensor_stats(tf.boolean_mask(warp_CangleRad, vel_CangleRad_mask), as_dict=True)
			del warp_CangleRad
			
		else:
			stats["warp_x_SE"] = tf_tensor_stats(tf.zeros([1,1,1,1,1], dtype=tf.float32), as_dict=True)
			stats["warp_y_SE"] = tf_tensor_stats(tf.zeros([1,1,1,1,1], dtype=tf.float32), as_dict=True)
			stats["warp_z_SE"] = tf_tensor_stats(tf.zeros([1,1,1,1,1], dtype=tf.float32), as_dict=True)
			stats["warp_vdiff_mag"] = tf_tensor_stats(tf.zeros([1,1,1,1,1], dtype=tf.float32), as_dict=True)
			stats["warp_angleCM_rad"] = tf_tensor_stats(tf.zeros([1,1,1,1,1], dtype=tf.float32), as_dict=True)
		
		return stats
	
	def __add__(self, other):
		if isinstance(other, VelocityGrid):
			if self.centered_shape!=other.centered_shape:
				raise ValueError("VelocityGrids of shape %s and %s are not compatible"%(self.centered_shape, other.centered_shape))
			return VelocityGrid(self.centered_shape, x=self.x+other.x, y=self.y+other.y, z=self.z+other.z, as_var=False, \
				boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=None)
		if isinstance(other, (np.ndarray, tf.Tensor, tf.Variable)):
			other_shape = shape_list(other)
			if self.centered_shape!=spacial_shape_list(other) or other_shape[0]!=1 or other_shape[-1]!=3:
				raise ValueError("VelocityGrid of shape %s is not compatible with tensor of shape %s are not compatible"%(self.centered_shape, spacial_shape_list(other)))
			x,y,z = self._centered_to_staggered(other)
			return VelocityGrid(self.centered_shape, x=self.x+x, y=self.y+y, z=self.z+z, as_var=False, \
				boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=None)
		else:
			return NotImplemented
	
	def __iadd__(self, other):
		if isinstance(other, VelocityGrid):
			if self.centered_shape!=other.centered_shape:
				raise ValueError("VelocityGrids of shape %s and %s are not compatible"%(self.centered_shape, other.centered_shape))
			self.assign_add(other.x, other.y, other.z)
			return self
		if isinstance(other, (np.ndarray, tf.Tensor, tf.Variable)):
			other_shape = shape_list(other)
			if self.centered_shape!=spacial_shape_list(other) or other_shape[0]!=1 or other_shape[-1]!=3:
				raise ValueError("VelocityGrid of shape %s is not compatible with tensor of shape %s are not compatible"%(self.centered_shape, spacial_shape_list(other)))
			x,y,z = self._centered_to_staggered(other)
			self.assign_add(x, y, z)
			return self
		else:
			return NotImplemented
	
	def __sub__(self, other):
		if isinstance(other, VelocityGrid):
			if self.centered_shape!=other.centered_shape:
				raise ValueError("VelocityGrids of shape %s and %s are not compatible"%(self.centered_shape, other.centered_shape))
			return VelocityGrid(self.centered_shape, x=self.x-other.x, y=self.y-other.y, z=self.z-other.z, as_var=False, \
				boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=None)
		if isinstance(other, (np.ndarray, tf.Tensor, tf.Variable)):
			other_shape = shape_list(other)
			if self.centered_shape!=spacial_shape_list(other) or other_shape[0]!=1 or other_shape[-1]!=3:
				raise ValueError("VelocityGrid of shape %s is not compatible with tensor of shape %s are not compatible"%(self.centered_shape, spacial_shape_list(other)))
			x,y,z = self._centered_to_staggered(other)
			return VelocityGrid(self.centered_shape, x=self.x-x, y=self.y-y, z=self.z-z, as_var=False, \
				boundary=self.boundary, scale_renderer=self.scale_renderer, warp_renderer=self.warp_renderer, device=None)
		else:
			return NotImplemented
	
	def __isub__(self, other):
		if isinstance(other, VelocityGrid):
			if self.centered_shape!=other.centered_shape:
				raise ValueError("VelocityGrids of shape %s and %s are not compatible"%(self.centered_shape, other.centered_shape))
			self.assign_sub(other.x, other.y, other.z)
			return self
		if isinstance(other, (np.ndarray, tf.Tensor, tf.Variable)):
			other_shape = shape_list(other)
			if self.centered_shape!=spacial_shape_list(other) or other_shape[0]!=1 or other_shape[-1]!=3:
				raise ValueError("VelocityGrid of shape %s is not compatible with tensor of shape %s are not compatible"%(self.centered_shape, spacial_shape_list(other)))
			x,y,z = self._centered_to_staggered(other)
			self.assign_sub(x, y, z)
			return self
		else:
			return NotImplemented

class State:
	def __init__(self, density, velocity, frame, prev=None, next=None, transform=None, targets=None, targets_raw=None, bkgs=None):
		self._density = None
		if density is not None:
			assert isinstance(density, DensityGrid)
			self._density = density
		self._velocity = None
		if velocity is not None:
			assert isinstance(velocity, VelocityGrid)
			self._velocity = velocity
		
		self.frame = frame
		self.prev = prev
		self.next = next
		
		self.transform = transform
		self.targets = targets
		self.targets_raw = targets_raw
		self.bkgs = bkgs
		self.target_cameras = None
		self.images = None
		self.t = None
	
	class StateIterator:
		def __init__(self, state):
			self.curr_state = state
		def __next__(self):
			if self.curr_state is not None:
				state = self.curr_state
				self.curr_state = state.next
				return state
			raise StopIteration
	def __iter__(self):
		return self.StateIterator(self)
	
	@property
	def density(self):
		if self._density is not None:
			return self._density
		else:
			raise AttributeError("State for frame {} does not contain density".format(self.frame))
	@property
	def velocity(self):
		if self._velocity is not None:
			return self._velocity
		else:
			raise AttributeError("State for frame {} does not contain velocity".format(self.frame))
	
	@classmethod
	def from_file(cls, path, frame, transform=None, as_var=True, boundary=None, scale_renderer=None, warp_renderer=None, device=None, density_filename="density.npz", velocity_filename="velocity.npz"):
		density = DensityGrid.from_file(os.path.join(path, density_filename), as_var=as_var, scale_renderer=scale_renderer, device=device)
		velocity = VelocityGrid.from_file(os.path.join(path, velocity_filename), as_var=as_var, \
			boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device)
		state = cls(density, velocity, frame, transform=transform)
		return state
	
	@classmethod
	def from_scalarFlow_file(cls, density_path, velocity_path, frame, transform=None, as_var=True, boundary=None, scale_renderer=None, warp_renderer=None, device=None):
		density = DensityGrid.from_scalarFlow_file(density_path, as_var=as_var, scale_renderer=scale_renderer, device=device)
		velocity = VelocityGrid.from_scalarFlow_file(velocity_path, as_var=as_var, \
			boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device)
		state = cls(density, velocity, frame, transform=transform)
		return state
	
	def copy(self, as_var=None, device=None):
		s = State(self.density.copy(as_var=as_var, device=device), self.velocity.copy(as_var=as_var, device=device), self.frame)
		m = copy.copy(self.__dict__)
		del m["_velocity"]
		del m["_density"]
		del m["prev"]
		del m["next"]
		for k,v in m.items():
			setattr(s,k,v)
		return s
	
	def copy_warped(self, order=1, dt=1.0, frame=None, as_var=None, targets=None, targets_raw=None, bkgs=None, device=None, clamp="NONE"):
		d = self.density.copy_warped(order=order, dt=dt, as_var=as_var, device=device, clamp=clamp)
		v = self.velocity.copy_warped(order=order, dt=dt, as_var=as_var, device=device, clamp=clamp)
		return State(d, v, frame, transform=self.transform, targets=targets, targets_raw=targets_raw, bkgs=bkgs)
	
	def get_density_transform(self):
		if isinstance(self.transform, GridTransform):
			return self.transform.copy_new_data(self.density.d)
		else:
			raise TypeError("state.transform is not a GridTransform")
	
	def get_velocity_transform(self):
		if isinstance(self.transform, GridTransform):
			return self.transform.copy_new_data(self.velocity.lod_pad)
		else:
			raise TypeError("state.transform is not a GridTransform")
	
	def render_density(self, render_ctx, custom_ops=None):
		imgs = tf.concat(render_ctx.dens_renderer.render_density(self.get_density_transform(), light_list=render_ctx.lights, camera_list=self.target_cameras, cut_alpha=False, monochrome=render_ctx.monochrome, custom_ops=custom_ops), axis=0) #, background=bkg
		imgs, d = tf.split(imgs, [3,1], axis=-1)
		t = tf.exp(-d)
		self.images = imgs
		self.t = t
	
	def density_advected(self, dt=1.0, order=1, clamp="NONE"):
		return self.density.warped(self.velocity, order=order, dt=dt, clamp=clamp)#self.velocity.warp(self.density, scale_renderer)
	def velocity_advected(self, dt=1.0, order=1, clamp="NONE"):
		return self.velocity.copy_warped(order=order, dt=dt, as_var=False, clamp=clamp)
	
	def rescale_density(self, shape, device=None):
		self._density = self.density.copy_scaled(shape, device=device)
	def rescale_velocity(self, shape, scale_magnitude=True, device=None):
		self._velocity = self.velocity.copy_scaled(shape, scale_magnitude=scale_magnitude, device=device)
	def rescale(self, dens_shape, vel_shape, device=None):
		rescale_density(self, dens_shape, device=device)
		rescale_velocity(self, vel_shape, device=device)
	
	def var_list(self):
		var_list = []
		if self._density is not None:
			var_list += self.density.var_list()
		if self._velocity is not None:
			var_list += self.velocity.var_list()
		return var_list
	
	def get_variables(self):
		var_dict = {}
		if self._density is not None:
			var_dict.update(self.density.get_variables())
		if self._velocity is not None:
			var_dict.update(self.velocity.get_variables())
		return var_dict
	
	def stats(self, vel_scale=[1,1,1], mask=None, render_ctx=None, **warp_kwargs):
		target_stats = None
		if render_ctx is not None and getattr(self, "target_cameras", None) is not None:
			target_stats = {}
			self.render_density(render_ctx)
			if getattr(self, "targets_raw") is not None and getattr(self, "bkgs") is not None:
				target_stats["SE_raw"] = tf_tensor_stats(tf.math.squared_difference(self.images + self.bkgs*self.t, self.targets_raw), as_dict=True)
			if getattr(self, "targets") is not None:
				target_stats["SE"] = tf_tensor_stats(tf.math.squared_difference(self.images, self.targets), as_dict=True)
		return self.density.stats(mask=mask, state=self, **warp_kwargs), self.velocity.stats(vel_scale, mask=mask, state=self, **warp_kwargs), target_stats
	
	def save(self, path, suffix=None):
		self.density.save(os.path.join(path, 'density.npz' if suffix is None else 'density_'+suffix+'.npz'))
		self.velocity.save(os.path.join(path, 'velocity.npz' if suffix is None else 'velocity_'+suffix+'.npz'))

class Sequence:
	def __init__(self, states):
		self.sequence = [state for state in states]
	
	class SequenceIterator:
		def __init__(self, sequence):
			self.seq = sequence
			self.idx = 0
		def __next__(self):
			if self.idx<len(self.seq):
				idx = self.idx
				self.idx +=1
				return self.seq[idx]
			raise StopIteration
	def __iter__(self):
		return self.SequenceIterator(self)
	
	def __getitem__(self, idx):
		return self.sequence[idx]
	
	def __len__(self):
		return len(self.sequence)
	
	@classmethod
	def from_file(cls, load_path, frames, transform=None, as_var=True, base_path=None, boundary=None, scale_renderer=None, warp_renderer=None, device=None, density_filename="density.npz", velocity_filename="velocity.npz", frame_callback=lambda idx, frame: None):
		sequence = []
		prev = None
		for idx, frame in enumerate(frames):
			frame_callback(idx, frame)
			sub_dir = 'frame_{:06d}'.format(frame)
			data_path = os.path.join(load_path, sub_dir)
			state = State.from_file(data_path, frame, transform=transform, as_var=as_var, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, \
				device=device, density_filename=density_filename, velocity_filename=velocity_filename)
			if base_path is not None:
				state.data_path = os.path.join(base_path, sub_dir)
				os.makedirs(state.data_path, exist_ok=True)
			state.prev = prev
			prev = state
			sequence.append(state)
		for i in range(len(sequence)-1):
			sequence[i].next = sequence[i+1]
		return cls(sequence)
	
	@classmethod
	def from_scalarFlow_file(cls, density_path_mask, velocity_path_mask, frames, transform=None, as_var=True, base_path=None, boundary=None, scale_renderer=None, warp_renderer=None, device=None, vel_frame_offset=1, frame_callback=lambda idx, frame: None):
		sequence = []
		prev = None
		for idx, frame in enumerate(frames):
			frame_callback(idx, frame)
			sub_dir = 'frame_{:06d}'.format(frame)
			density_path = density_path_mask.format(frame=frame)
			velocity_path = velocity_path_mask.format(frame=frame+vel_frame_offset)
			state = State.from_scalarFlow_file(density_path, velocity_path, frame=frame, transform=transform, as_var=as_var, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device)
			if base_path is not None:
				state.data_path = os.path.join(base_path, sub_dir)
				os.makedirs(state.data_path, exist_ok=True)
			state.prev = prev
			prev = state
			sequence.append(state)
		for i in range(len(sequence)-1):
			sequence[i].next = sequence[i+1]
		return cls(sequence)
	
	def copy(self, as_var=None, device=None):
		s = [_.copy(as_var=as_var, device=device) for _ in self]
		for i in range(len(s)):
			if i>0:
				s[i].prev = s[i-1]
			if i<(len(s)-1):
				s[i].next = s[i+1]
		return Sequence(s)
	
	def insert_state(self, state, idx):
		self.sequence.insert(state, idx)
	
	def append_state(self, state):
		self.sequence.append(state)
	
	def start_iteration(self, iteration):
		for state in self:
			ctx.start_iteration(iteration)
	
	def stats(self, vel_scale=[1,1,1], mask=None, **warp_kwargs):
		return [_.stats(vel_scale, mask=mask, state=_, **warp_kwargs) for _ in self]
	
	def save(self, path=None, suffix=None):
		for state in self:
			if path is None and hasattr(state, 'data_path'):
				state.save(state.data_path, suffix)
			else:
				state.save(os.path.join(path, 'frame_{:06d}'.format(state.frame)), suffix)
	
	def densities_advect_fwd(self, dt=1.0, order=1, clamp='NONE'):
		if clamp is None or clamp.upper()not in ['LOCAL', 'GLOBAL']:
			for i in range(1, len(self)):
				self[i].density.assign(self[i-1].density_advected(order=order, dt=dt, clamp=clamp))
		elif clamp.upper()=='LOCAL': #clamp after each step, before the next warp
			for i in range(1, len(self)):
				self[i].density.assign(tf.maximum(self[i-1].density_advected(order=order, dt=dt), 0))
		elif clamp.upper()=='GLOBAL': #clamp after all warping
			for i in range(1, len(self)):
				self[i].density.assign(self[i-1].density_advected(order=order, dt=dt))
			for i in range(1, len(self)):
				self[i].density.assign(tf.maximum(self[i].density._d, 0))
	def velocities_advect_fwd(self, dt=1.0, order=1, clamp='NONE'):
		for i in range(1, len(self)):
			self[i].velocity.assign(*self[i-1].velocity.warped(order=order, dt=dt, clamp=clamp))