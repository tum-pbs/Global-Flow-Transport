import tensorflow as tf
import numpy as np
import logging, numbers

LOG = logging.getLogger("Vector")


class Matrix(object):
	def __init__(self, *args, **kwargs):
		raise NotImplementedError

#V_ATTR_MAP = {'x':0,'y':1,'z':2,'w':3,'r':0,'g':1,'b':2,'a':3}
class Vector(Matrix):
	def __init__(self, *args, **kwargs):
		raise NotImplementedError
		
	@classmethod
	def from_dict(cls, d):
		return cls(d["value"])
	
	def to_dict(self):
		return {
				"value":self.value,
			}
	
	@property
	def value(self):
		raise NotImplementedError
	
	@property
	def as_shape(self):
		#raise NotImplementedError
		return self.value[::-1].astype(np.int32)
	
	@property
	def magnitude(self):
		return np.linalg.norm(self._value)
	
	@property
	def normalized(self):
		mag = self.magnitude
		if mag==0:
			return self
		return self/mag
	
	@property
	def sq_magnitude(self):
		return np.dot(self._value, self._value)
	
	@property
	def prod(self):
		return np.prod(self._value)
	@property
	def sum(self):
		return np.sum(self._value)
	
	def __len__(self):
		return len(self._value)
	
	def __getitem__(self, index):
		return self._value[index]
	
	def __getattr__(self, attr):
		'''
		support for generic swizzling
		'''
		if len(attr)>4: raise AttributeError("At most 4 elements of ({}) can be selected from {}, selection: '{}'".format(", ".join(self._attr_map.keys()), self.__class__.__name__, attr))
		r = []
		for a in attr.lower():
			if a not in self._attr_map: raise AttributeError("Element '{}' from choice '{}' not available for {}, available: ({})".format(a, attr, self.__class__.__name__, ", ".join(self._attr_map.keys())))
			i = self._attr_map[a]
			if i>=len(self): raise AttributeError(attr)
			r.append(i)
		if len(r)==1: return self[r[0]]
		elif len(r)==2: return Vector2(self[r])
		elif len(r)==3: return Vector3(self[r])
		elif len(r)==4: return Vector4(self[r])
	
	def __setattr__(self, name, value):
		if name.lower() in self._attr_map:
			if not np.isscalar(value):
				raise ValueError("Only scalar values can be assigned to vector elements.")
			self._value[self._attr_map[name.lower()]] = value
		else:
			super().__setattr__(name, value)
	
	def __array__(self, dtype=None):
		if dtype:
			return np.array(self._value, dtype=dtype)
		else:
			return self.value
	
	def __add__(self, other):
		cls = self.__class__
		try:
			r = np.add(self, other)
		except:
			return NotImplemented
		return cls(r)
	
	def __radd__(self, other):
		cls = self.__class__
		try:
			r = np.add(other, self)
		except:
			return NotImplemented
		return cls(r)
	
	def __iadd__(self, other):
		return self.__add__(other)
	
	def __sub__(self, other):
		cls = self.__class__
		try:
			r = np.subtract(self, other)
		except:
			return NotImplemented
		return cls(r)
	
	def __rsub__(self, other):
		cls = self.__class__
		try:
			r = np.subtract(other, self)
		except:
			return NotImplemented
		return cls(r)
	
	def __isub__(self, other):
		return self.__sub__(other)
	
	def __mul__(self, other):
		cls = self.__class__
		try:
			r = np.multiply(self, other)
		except:
			return NotImplemented
		return cls(r)
	
	def __rmul__(self, other):
		cls = self.__class__
		try:
			r = np.multiply(other, self)
		except:
			return NotImplemented
		return cls(r)
	
	def __imul__(self, other):
		return self.__mul__(other)
	
	def __truediv__(self, other):
		cls = self.__class__
		try:
			r = np.true_divide(self, other)
		except:
			return NotImplemented
		return cls(r)
	
	def __rtruediv__(self, other):
		cls = self.__class__
		try:
			r = np.true_divide(other, self)
		except:
			return NotImplemented
		return cls(r)
	
	def __itruediv__(self, other):
		return self.__truediv__(other)
	
	def __floordiv__(self, other):
		cls = self.__class__
		try:
			r = np.floor_divide(self, other)
		except:
			return NotImplemented
		return cls(r)
	
	def __rfloordiv__(self, other):
		cls = self.__class__
		try:
			r = np.floor_divide(other, self)
		except:
			return NotImplemented
		return cls(r)
	
	def __ifloordiv__(self, other):
		return self.__floordiv__(other)
	
	def __mod__(self, other):
		cls = self.__class__
		try:
			r = np.mod(self, other)
		except:
			return NotImplemented
		return cls(r)
	
	def __rmod__(self, other):
		cls = self.__class__
		try:
			r = np.mod(other, self)
		except:
			return NotImplemented
		return cls(r)
	
	def __imod__(self, other):
		return self.__mod__(other)
	
	def __matmul__(self, other):
		cls = self.__class__
		try:
			r = np.dot(self, other)
		except:
			return NotImplemented
		return r
	
	def __rmatmul__(self, other):
		cls = self.__class__
		try:
			r = np.dot(other, self)
		except:
			return NotImplemented
		return r
	
	def __imatmul__(self, other):
		return self.__matmul__(other)
	
	def __eq__(self, other):
		cls = self.__class__
		if isinstance(other, cls):
			if len(self)!=len(other): return False
			if (self._value!=other._value).any(): return False
			return True
		elif isinstance(other, (list, tuple, np.ndarray)):
			if len(self)!=len(other): return False
			if (self._value!=other).any(): return False
			return True
		else:
			return NotImplemented
		
	
	def __ne__(self, other):
		return not self==other
	
	def __repr__(self):
		return "{}: {}".format(self.__class__.__name__, self._value)
	
	def __deepcopy__(self, *args):
		return type(self)(self._value)
	
	def __copy__(self):
		return self.__deepcopy__()
	
	def copy(self):
		return self.__copy__()
	

class GridShape(Vector):
	_attr_map = {'n':0,'x':3,'y':2,'z':1,'c':4}
	def __init__(self, v):
		'''
		v: 3, 4 or 4 element vector: DHW, DHWC, NDHWC. z,y,x order, will be copied
		'''
		if isinstance(v, (list, tuple, np.ndarray)):
			v = np.array(v, copy=True)
			assert v.shape==(3,) or v.shape==(4,) or v.shape==(5,)
		elif isinstance(v, GridShape):
			v = v.value
		elif isinstance(v, Vector3):
			v = v.zyx.value
		elif isinstance(v, Vector4):
			v = v.zyxw.value
		else:
			raise ValueError("Unsupported input type")
		
		if len(v)<4:
			v = np.append(v, [1])
		if len(v)<5:
			v = np.append([1], v)
		
		self._value = v.astype(np.int32)
	
	@classmethod
	def from_tensor(cls, tensor):
		if isinstance(tensor, (tf.Tensor, tf.Variable)):
			shape = tensor.get_shape().as_list()
		elif isinstance(tensor, np.ndarray):
			shape = list(tensor.shape)
		elif isinstance(tensor, (list, tuple)):
			shape = list(np.asarray(tensor).shape)
		return cls(shape)
	
	def normalize_tensor_shape(self, tensor):
		if isinstance(tensor, (tf.Tensor, tf.Variable)):
			return tf.reshape(tensor, self._value)
		elif isinstance(tensor, np.ndarray):
			return np.reshape(tensor, self._value)
		else:
			raise ValueError("Unsupported tensor type")
	
	@property
	def as_shape(self):
		return self.value
	
	@property
	def value(self):
		return np.array(self._value, copy=True)
	@property
	def spatial_vector(self):
		return Int3(self.x,self.y,self.z)
	
	def padding_to(self, shape, offset=[0,0,0,0,0]):
		if len(shape)!=5 or len(offset)!=5:
			raise ValueError
		padding = []
		for i in range(5):
			padding.append([offset[i], shape[i] - self[i] - offset[i]])
		return padding
	
class Vector2(Vector):
	_attr_map = {'x':0,'y':1,'r':0,'g':1}
	def __init__(self, *args, dtype=None):
		'''
		v: 3 element vector with x,y,z order, will be copied
		'''
		num_args = len(args)
		if num_args==0:
			self._value = np.array([0,0], dtype=dtype)
		elif num_args==1:
			v = args[0]
			if np.isscalar(v):
				v=[v,v]
			if isinstance(v, (list, tuple, np.ndarray)):
				self._value = np.array(v, dtype=dtype)
				assert self._value.shape==(2,)
			elif isinstance(v, Vector2):
				self._value = v.__array__(dtype=dtype)
			elif isinstance(v, Vector3):
				self._value = v.__array__(dtype=dtype)[:-1]
			elif isinstance(v, Vector4):
				self._value = v.__array__(dtype=dtype)[:-2]
			else:
				raise ValueError("Unsupported input type")
		elif num_args==2:
			x,y = args
			assert np.isscalar(x)
			assert np.isscalar(y)
			self._value = np.array([x,y], dtype=dtype)
		else:
			raise ValueError("Unsupported parameters for Vector2: %s"%args)
		
	
	def __len__(self):
		return 2
	@property
	def value(self):
		return np.array(self._value, copy=True)
	
	@classmethod
	def from_elements(cls,*, x, y):
		return cls([x,y])
	
	@classmethod
	def from_shape(cls, shape):
		return cls(shape[::-1])

class Float2(Vector2):
	def __init__(self, *args):
		super().__init__(*args, dtype=np.float32)

class Int2(Vector2):
	def __init__(self, *args):
		super().__init__(*args, dtype=np.int32)

class Vector3(Vector):
	_attr_map = {'x':0,'y':1,'z':2,'r':0,'g':1,'b':2}
	def __init__(self, *args, dtype=None):
		'''
		v: 3 element vector with x,y,z order, will be copied
		'''
		num_args = len(args)
		if num_args==0:
			self._value = np.array([0,0,0], dtype=dtype)
		elif num_args==1:
			v = args[0]
			if np.isscalar(v):
				v=[v,v,v]
			
			if isinstance(v, (list, tuple, np.ndarray)):
				self._value = np.array(v, dtype=dtype)
				assert self._value.shape==(3,), "Input shape %s must be (3,)"%(self._value.shape,)
			elif isinstance(v, Vector3):
				self._value = v.__array__(dtype=dtype)
			elif isinstance(v, Vector4):
				self._value = v.__array__(dtype=dtype)[:-1]
			else:
				raise ValueError("Unsupported input types: %s"%([type(_).__name__ for _ in args],))
		elif num_args==2:
			v1, v2 = args
			if isinstance(v1, Vector2) and isinstance(v2, numbers.Number):
				self._value = np.array([v1.x,v1.y,v2], dtype=dtype)
			elif isinstance(v1, numbers.Number) and isinstance(v2, Vector2):
				self._value = np.array([v1,v2.x,v2.y], dtype=dtype)
			else:
				raise ValueError("Unsupported input types: %s"%([type(_).__name__ for _ in args],))
		elif num_args==3:
			x,y,z = args
			assert np.isscalar(x)
			assert np.isscalar(y)
			assert np.isscalar(z)
			self._value = np.array([x,y,z], dtype=dtype)
		else:
			raise ValueError("Unsupported parameters for Vector3: %s"%(args,))
		
	
	def __len__(self):
		return 3
	@property
	def value(self):
		return np.array(self._value, copy=True)
	
	@classmethod
	def from_elements(cls,*, x, y, z):
		return cls([x,y,z])
	
	@classmethod
	def from_shape(cls, shape):
		return cls(shape[::-1])

class Float3(Vector3):
	def __init__(self, *args):
		super().__init__(*args, dtype=np.float32)

class Int3(Vector3):
	def __init__(self, *args):
		super().__init__(*args, dtype=np.int32)

class Vector4(Vector):
	_attr_map = {'x':0,'y':1,'z':2,'w':3,'r':0,'g':1,'b':2,'a':3}
	def __init__(self, *args, dtype=None):
		'''
		v: 4 element vector with x,y,z,w order, will be copied
		'''
		num_args = len(args)
		if num_args==1:
			v = args[0]
			if np.isscalar(v):
				v=[v,v,v,v]
			if isinstance(v, (list, tuple, np.ndarray)):
				self._value = np.array(v, dtype=dtype)
				assert self._value.shape==(4,)
			elif isinstance(v, Vector3):
				self._value = np.append(v.value, [1])
			elif isinstance(v, Vector4):
				self._value = v.value
			else:
				raise ValueError("Unsupported input types: %s"%([type(_).__name__ for _ in args],))
		elif num_args==2:
			v1, v2 = args
			if isinstance(v1, Vector2) and isinstance(v2, Vector2):
				self._value = np.array([v1.x,v1.y,v2.x,v2.y], dtype=dtype)
			elif isinstance(v1, Vector3) and isinstance(v2, numbers.Number):
				self._value = np.array([v1.x,v1.y,v1.z,v2], dtype=dtype)
			elif isinstance(v1, numbers.Number) and isinstance(v2, Vector3):
				self._value = np.array([v1,v2.x,v2.y,v2.z], dtype=dtype)
			else:
				raise ValueError("Unsupported input types: %s"%([type(_).__name__ for _ in args],))
		elif num_args==3:
			v1, v2, v3 = args
			if isinstance(v1, Vector2) and isinstance(v2, numbers.Number) and isinstance(v3, numbers.Number):
				self._value = np.array([v1.x,v1.y,v2,v3], dtype=dtype)
			elif isinstance(v1, numbers.Number) and isinstance(v2, Vector2) and isinstance(v3, numbers.Number):
				self._value = np.array([v1,v2.x,v2.y,v3], dtype=dtype)
			elif isinstance(v1, numbers.Number) and isinstance(v2, numbers.Number) and isinstance(v3, Vector2) :
				self._value = np.array([v1,v2,v3.x,v3.y], dtype=dtype)
			else:
				raise ValueError("Unsupported input types: %s"%([type(_).__name__ for _ in args],))
		elif num_args==4:
			x,y,z,w = args
			assert np.isscalar(x)
			assert np.isscalar(y)
			assert np.isscalar(z)
			assert np.isscalar(w)
			self._value = np.array([x,y,z,w], dtype=dtype)
		else:
			raise ValueError("Unsupported parameters for Vector4: %s"%(args,))
		
	
	def __len__(self):
		return 4
	@property
	def value(self):
		return np.array(self._value, copy=True)
	
	@classmethod
	def from_elements(cls,*, x, y, z, w):
		return cls([x,y,z,w])
	
	@classmethod
	def from_shape(cls, shape):
		return cls(shape[::-1])

class Float4(Vector4):
	def __init__(self, *args):
		super().__init__(*args, dtype=np.float32)

class Int4(Vector4):
	def __init__(self, *args):
		super().__init__(*args, dtype=np.int32)