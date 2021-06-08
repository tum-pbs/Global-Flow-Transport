import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .serialization import to_dict, from_dict

from .vector import *

class MatrixTransform(object):
	def __init__(self, transform_matrix=None, parent=None, grid_size=None, static=False):
		if transform_matrix is not None:
			self._matrix = transform_matrix
		else:
			self._matrix = self.identity_matrix()
		if parent is not None and not isinstance(parent, MatrixTransform):
			raise TypeError("parent must be a Transform object or None.")
		self.parent = parent
		self.grid_size=grid_size
	@classmethod
	def from_dict(cls, d):
		p = d.pop("parent")
		p = from_dict(p)
		return cls(parent=p, **d)
	
	@classmethod
	def from_lookat(cls, eye, lookat, parent=None):
		pass
	
	@classmethod
	def from_fwd_up_right_pos(cls, fwd, up, right, pos, parent=None):
		mat = np.asarray(
		[[right[0],up[0],fwd[0],pos[0]],
		 [right[1],up[1],fwd[1],pos[1]],
		 [right[2],up[2],fwd[2],pos[2]],
		 [0,0,0,1]],
		dtype=np.float32)
		return cls(mat, parent)
	
	@classmethod
	def from_transform(cls, transform, parent=None):
		raise NotImplementedError()
		
	@staticmethod
	def translation_matrix(translation):
		return np.asarray(
		[[1,0,0,translation[0]],
		 [0,1,0,translation[1]],
		 [0,0,1,translation[2]],
		 [0,0,0,1]],
		dtype=np.float32)
	@staticmethod
	def rotation_matrix(rotation):
		rot = Rotation.from_euler('xyz', rotation, degrees=True).as_dcm()
		rot = np.pad(rot, (0,1), mode='constant')
		rot[-1,-1]=1
		return rot
	@staticmethod
	def scale_matrix(scale):
		return np.asarray(
		[[scale[0],0,0,0],
		 [0,scale[1],0,0],
		 [0,0,scale[2],0],
		 [0,0,0,1]],
		dtype=np.float32)
	@staticmethod
	def identity_matrix():
		return np.asarray(
		[[1,0,0,0],
		 [0,1,0,0],
		 [0,0,1,0],
		 [0,0,0,1]],
		dtype=np.float32)
	
	def set_parent(self, parent_transform):
		if parent_transform is not None and not isinstance(parent_transform, MatrixTransform):
			raise TypeError("parent must be a Transform object or None.")
		self.parent = parent_transform
	
	def copy_no_data(self):
		return copy.deepcopy(self)
	
	def get_local_transform(self):
		return self._matrix
		
	def position_global(self):
		return self.get_transform_matrix()@np.asarray([0,0,0,1])
	def forward_global(self):
		v = self.get_transform_matrix()@np.asarray([0,0,1,0])
		return v/np.linalg.norm(v)
	def up_global(self):
		v = self.get_transform_matrix()@np.asarray([0,1,0,0])
		return v/np.linalg.norm(v)
	def right_global(self):
		v = self.get_transform_matrix()@np.asarray([1,0,0,0])
		return v/np.linalg.norm(v)
	
	def transform(self, vector):
		if isinstance(vector, (Vector2,Vector3)):
			raise TypeError("use Vector4 for transformations")
		elif isinstance(vector, Vector4):
			return Vector4(self.get_transform_matrix() @ vector.value)
		else:
			return self.get_transform_matrix() @ np.asarray(vector)
	
	def transform_AABB(self, corner_min=[0,0,0], corner_max=[1,1,1], expand_corners=False):
		corners = []
		cmin = Vector3(corner_min)
		cmax = Vector3(corner_max)
		corners.append(self.transform([cmin.x, cmin.y, cmin.z, 1]))
		if expand_corners:
			corners.append(self.transform([cmax.x,cmin.y,cmin.z, 1]))
			corners.append(self.transform([cmin.x,cmax.y,cmin.z, 1]))
			corners.append(self.transform([cmin.x,cmin.y,cmax.z, 1]))
			corners.append(self.transform([cmax.x,cmax.y,cmin.z, 1]))
			corners.append(self.transform([cmax.x,cmin.y,cmax.z, 1]))
			corners.append(self.transform([cmin.x,cmax.y,cmax.z, 1]))
		corners.append(self.transform([cmax.x,cmax.y,cmax.z, 1]))
		return corners
		
		
	def get_transform_matrix(self):
		if self.parent is not None:
			return self.parent.get_transform_matrix() @ self.get_local_transform()
		else:
			return self.get_local_transform()
	def get_inverse_transform(self):
		return np.linalg.inv(self.get_transform_matrix())
	def inverse(self):
		return MatrixTransform(self.get_inverse_transform()) #includes parent transform!
	def is_static(self):
		if not self.static:
			return False
		elif self.parent is not None:
			return self.parent.is_static()
		else:
			return True
	#operators
	def __eq__(self, other):
		return self.get_transform_matrix() == other.get_transform_matrix()
	
	def to_dict(self):
		return {
				"transform_matrix":np.asarray(self._matrix).tolist(),
				"parent":to_dict(self.parent),
				"grid_size":self.grid_size,
			}

class Transform(MatrixTransform):
	def __init__(self, translation=[0,0,0], rotation_deg=[0,0,0], scale=[1,1,1], parent=None, static=False):
		self.translation = translation
		self.rotation_deg = rotation_deg
		self.scale = scale
		self.parent = parent
	@classmethod
	def from_dict(cls, d):
		p = d.pop("parent")
		p = from_dict(p)
		return cls(parent=p, **d)
	
	def set_translation(self, translation):
		pass
	def set_rotation_angle(self, angle_deg):
		pass
	def set_scale(self, scale):
		pass
	def set_rotation_quaternion(self, rotation):
		pass
	
	def translate_local(self, translation):
		pass
	def rotate_around_local(self, axis, angle_deg):
		pass
	def scale_local(self, scale):
		pass
	
	def get_local_transform(self):
		M_scale = Transform.scale_matrix(self.scale)
		M_rot = Transform.rotation_matrix(self.rotation_deg)
		M_trans = Transform.translation_matrix(self.translation)
		return M_trans@(M_rot@M_scale)
	#operators
	def __eq__(self, other):
		return self.get_transform_matrix() == other.get_transform_matrix()
	def __str__(self):
		return '{}: t={}, r={}, s={}; p=({})'.format(type(self).__name__, self.translation, self.rotation_deg, self.scale ,self.parent)
	
	def to_dict(self):
		return {
				"translation":list(self.translation),
				"rotation_deg":list(self.rotation_deg),
				"scale":list(self.scale),
				"parent":to_dict(self.parent),
			}
	

class GridTransform(Transform):
	# center: offset grid s.t. its center is at (0,0,0) is OS
	# normalize: normalize size to (1,1,1) with 1/grid-size
	def __init__(self, grid_size, translation=[0,0,0], rotation_deg=[0,0,0], scale=[1,1,1], center=False, normalize='NONE', parent=None, static=False):
		super().__init__(translation, rotation_deg, scale, parent)
		self.grid_size=grid_size
		self.center = center
		self.normalize = normalize
		self.data=None
	@classmethod
	def from_dict(cls, d):
		p = d.pop("parent")
		p = from_dict(p)
		return cls(parent=p, **d)
	
	@classmethod
	def from_transform(cls, transform, grid_size, center=False, normalize='NONE'):
		return cls(grid_size, transform.translation, transform.rotation_deg, transform.scale, center, normalize, transform.parent)
	@classmethod
	def from_grid(cls, grid, translation=[0,0,0], rotation_deg=[0,0,0], scale=[1,1,1], center=False, normalize='NONE', parent=None):
		pass
	@classmethod
	def from_grid_transform(cls, grid, transform, center=False, normalize='NONE'):
		pass
	
	def set_data(self, data, format='NDHWC'): #TODO rename: set_grid
		data_shape = data.get_shape().as_list()
		self.grid_size = [data_shape[format.index(_)] for _ in 'DHW']
		self.data = data
	def get_grid(self):
		return data
	
	def get_grid_size(self): #TODO rename: get_grid_shape
		return np.asarray(self.grid_size)
	
	def get_channel(self):
		return self.data.get_shape().as_list()[-1]
	
	@property
	def grid_shape(self):
		if self.data is not None:
			return GridShape.from_tensor(self.data)
		else:
			return GridShape(self.grid_size)
	
	def copy_no_data(self):
		gt = copy.copy(self)
		gt.data = None
		return copy.deepcopy(gt)
	
	def copy_new_data(self, data):
		gt = self.copy_no_data()
		gt.set_data(data)
		return gt
	
	def copy_same_data(self):
		return copy_new_data(self.data)
	
	def get_local_transform(self):
		size = np.flip(self.get_grid_size()) # shape is zyx, but coordinates are xyz
		M_center = Transform.translation_matrix(-size/2.0)
		if self.normalize=='ALL':
			M_norm_scale = Transform.scale_matrix(1.0/size)
		if self.normalize=='MIN':
			M_norm_scale = Transform.scale_matrix(np.asarray([1.0/np.min(size)]*3, dtype=np.float32))
		if self.normalize=='MAX':
			M_norm_scale = Transform.scale_matrix(np.asarray([1.0/np.max(size)]*3, dtype=np.float32))
		M_scale = Transform.scale_matrix(self.scale)
		M_rot = Transform.rotation_matrix(self.rotation_deg)
		M_trans = Transform.translation_matrix(self.translation)
		M = M_scale@M_center if self.center else M_scale
		M = M_norm_scale@M if self.normalize!='NONE' else M
		return M_trans@(M_rot@M)
	
	def grid_corners_world(self, all_corners=False):
		gs = self.grid_shape
		return self.transform_AABB(corner_max=[gs.x,gs.y,gs.z], expand_corners=all_corners)
	
	def grid_size_world(self):
		gs = self.grid_shape
		dir_x = self.transform(Float4(gs.x,0,0,0)).xyz
		dir_y = self.transform(Float4(0,gs.y,0,0)).xyz
		dir_z = self.transform(Float4(0,0,gs.z,0)).xyz
		return Float3(dir_x.magnitude, dir_y.magnitude, dir_z.magnitude)
	
	def cell_size_world(self):
		dir_x = self.transform(Float4(1,0,0,0)).xyz
		dir_y = self.transform(Float4(0,1,0,0)).xyz
		dir_z = self.transform(Float4(0,0,1,0)).xyz
		return Float3(dir_x.magnitude, dir_y.magnitude, dir_z.magnitude)
	
	def __eq__(self, other):
		if np.any(np.not_equal(self.get_transform_matrix(), other.get_transform_matrix())): return False
		if np.any(np.not_equal(self.get_grid_size(), other.get_grid_size())): return False
		return True
	
	def __str__(self):
		return '{}: {}, t={}, r={}, s={}; p=({})'.format(type(self).__name__, self.grid_size, self.translation, self.rotation_deg, self.scale ,self.parent)
	
	def to_dict(self):
		return {
				"translation":list(self.translation),
				"rotation_deg":list(self.rotation_deg),
				"scale":list(self.scale),
				"center":bool(self.center),
				"normalize":str(self.normalize),
				"grid_size":list(self.grid_size),
				"parent":to_dict(self.parent),
			}