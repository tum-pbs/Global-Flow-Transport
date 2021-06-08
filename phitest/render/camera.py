import math
import numpy as np
from .serialization import to_dict, from_dict
from .transform import MatrixTransform, GridTransform
from .vector import Float2, Float3, Float4
import logging

LOG = logging.getLogger("Camera")

class LuTCache(object):
	def __init__(self, object_grid_transform, camera):
		self.obj = object_grid_transform
		self.cam = camera
		self.LuT = None
		self.inverseLuT = None
		
	
	def check_lut(self, object_grid_transform, camera):
		return (object_grid_transform==self.obj) and (camera==self.cam)
	
	def get_size():
		pass

class Camera(object):
	# fov: horizontal degree
	# aspect: width/height
	# jitter: scalar or array-like with shape (3,)
	# static: precompute LuT and LoD w.r.t this grid transform
	def __init__(self, grid_transform, nearFar=[1,10], topRightBottomLeft=[1,1,-1,-1], fov=None, aspect=1, perspective=True, static=None, jitter=None):
		self.transform=grid_transform
		self.jitter = jitter
		self.trbl = topRightBottomLeft
		self.clip = nearFar
		self.perspective = perspective
		if fov is not None:
			self.set_fov(fov, aspect)
		self.static = static
		self.LuT = None
		self.inverseLuT = None
		self.scissor_pad = None
	
	@classmethod
	def from_dict(cls, d):
		t = d.pop("grid_transform")
		t = from_dict(t)
		return cls(grid_transform=t,**d)
	
	def view_transform_inverse(self, with_jitter=True):
		if self.jitter is not None and with_jitter:
			jitter_position = np.random.uniform(-self.jitter,self.jitter, [3]).astype(np.float32)
			#LOG.debug("Camera jitter position: %s", jitter_position)
			view_transform = GridTransform(self.transform.grid_size, translation=jitter_position, parent=self.transform)
		else:
			view_transform = self.transform
		return view_transform
	def view_matrix_inverse(self, with_jitter=True):
		return self.view_transform_inverse(with_jitter).get_transform_matrix()
	def view_matrix(self, with_jitter=True):
		return self.view_transform_inverse(with_jitter).get_inverse_transform()
	
	def frustum(self):
		return np.asarray([self.clip[0], self.clip[1], self.trbl[3], self.trbl[1], self.trbl[0], self.trbl[2]], dtype=np.float32)
	#aspect: width/height if not vertical, else height/width
	def set_fov(self, fov, aspect=1, vertical=False):
		half_h = np.tan(fov/2. * np.pi/180.)*self.clip[0]
		half_v = half_h/aspect
		if vertical:
			self.trbl = [half_h, half_v, -half_h, -half_v]
		else:
			self.trbl = [half_v, half_h, -half_v, -half_h]
	def projection_matrix(self):
		t, r, b, l = self.trbl
		n, f = self.clip
		
		#OpenGL projection matrices
		#http://www.songho.ca/opengl/gl_projectionmatrix.html
		if not self.perspective:
			return np.asarray(
			[[2/(r-l),0,0,0],
			 [0,2/(t-b),0,0],
			 [0,0,-2/(f-n),0],
			 [-(r+l)/(r-l),-(t+b)/(t-b),-(f+n)/(f-n),1]],
			dtype=np.float32).transpose()
		# https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix
		#https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
		else:
			return np.asarray(
			[[2*n/(r-l), 0, 0, 0],
			 [0, 2*n/(t-b), 0, 0],
			 [(r+l)/(r-l), (t+b)/(t-b), -(f+n)/(f-n), -1],
			 [0, 0, -2*f*n/(f-n), 0]],
			dtype=np.float32).transpose()
	
	@property
	def depth_step(self):
		'''step size in view space'''
		return (self.far - self.near)/self.transform.grid_size[0]
	@property
	def pix_height_near(self):
		'''step size in view space'''
		return (self.top - self.bottom)/self.transform.grid_size[1]
	@property
	def pix_width_near(self):
		'''step size in view space'''
		return (self.right - self.left)/self.transform.grid_size[2]
	
	@property
	def position_global(self):
		return self.transform.position_global()
	@property
	def forward_global(self):
		return self.transform.forward_global()
	@property
	def up_global(self):
		return self.transform.up_global()
	@property
	def right_global(self):
		return self.transform.right_global()
	@property
	def near(self):
		return self.clip[0]
	@property
	def far(self):
		return self.clip[1]
	@property
	def top(self):
		return self.trbl[0]
	@property
	def right(self):
		return self.trbl[1]
	@property
	def bottom(self):
		return self.trbl[2]
	@property
	def left(self):
		return self.trbl[3]
	@property
	def view_height(self):
		return self.top-self.bottom
	@property
	def view_width(self):
		return self.right-self.left
	
	@property
	def aspect(self):
		return self.view_height/self.view_width #h/v
	
	def _near_plane_intersection(self, pos_view, near=None):
		''' https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
			near plane: n = (0,0,-1), p0=(0,0,-near)
			line: l0= (0,0,0), l= pos_view
		'''
		if near==None:
			near = self.near
		pos_view = Float3(np.asarray(pos_view, dtype=np.float32)[:3]).normalized
		n = Float3(0,0,-1)
		den = np.dot(pos_view, n)
		if den==0:
			return None
		#num = dot([0,0,-near],n) = near
		#d= num/den
		return pos_view*(near/den)
	
	def _project_on_near(self, pos_view, near=None):
		if near==None:
			near = self.near
		pos_view = np.asarray(pos_view, dtype=np.float32)[:3]
		#intercept theorem
		scale = near/(-pos_view[2])
		return pos_view*scale
	
	def project_world_to_screenUV(self, pos_world):
		if len(pos_world)==3:
			pos_world = Float4(Float3(pos_world), 1)
		mat_view = MatrixTransform(self.view_matrix(with_jitter=False))
		pos_view = mat_view.transform(pos_world)
		pos_near = Float3(self._project_on_near(pos_view))
		pos_uv = (pos_near.xy - Float2(self.left, self.bottom)) / Float2(self.view_width, self.view_height)
		return pos_uv
	
	def project_world_to_screenPIX(self, pos_world):
		pos_uv = self.project_world_to_screenUV(pos_world)
		return pos_uv * Float2(self.transform.grid_size[2], self.transform.grid_size[1])
	
	def screenUV_to_worldRay(self, pos_uv):
		'''
		return a ray, starting at pos on near plane
		'''
		mat_view_inv =  self.view_transform_inverse(with_jitter=False)
		pos_near = (pos_uv * Float2(self.view_width, self.view_height)) + Float2(self.left, self.bottom)
		pos_near = Float4(pos_near, -self.near, 1) #in view space (VS)
		pos_world = mat_view_inv.transform(pos_near).xyz
		dir_world = (pos_world - Float4(self.position_global).xyz).normalized
		return pos_world, dir_world
	
	def screenPIX_to_worldRay(self, pos_pix):
		'''
		return a ray, starting at pos on near plane
		'''
		pos_uv = pos_pix / Float2(self.transform.grid_size[2], self.transform.grid_size[1])
		return self.screenUV_to_worldRay(pos_uv)
	
	def copy_clipped_to_world_coords_old(self, coordinate_list, preserve_aspect=True, pad=0.0, clip_to_original=True):
		'''
			copy camera with trbl set to the minimum hull of the original trbl and the projected coordinates (like scissor rect)
			does not change resolution -> higher spacial resolution
			coordinate_list: list of 3D world-space coorinates as 4D verctors (with w=1)
		'''
		# get x,y in view space in near plane
		view = MatrixTransform(self.view_matrix())
		pos_view = []
		pos_view_z = []
		for p in coordinate_list:
			p_view = view.transform(p)
			pos_view.append(p_view)
			pos_view_z.append(p_view[2])
		pos_view_z = np.asarray(pos_view_z)
		
		near = self.near #np.amin(-pos_view_z)-pad
		far  = self.far #np.amax(-pos_view_z)+pad
		if clip_to_original:
			near = np.clip(near, self.near, self.far)
			far  = np.clip(far , self.near, self.far)
		
		pos_near = []
		for p_view in pos_view:
			p_near = self._near_plane_intersection(p_view, near)
			if p_near is not None:
				pos_near.append(p_near)
		if len(pos_near)==0:
			raise ValueError("")
		x,y,_ = np.split(np.asarray(pos_near), 3, axis=-1)
		
		# project original trbl to new near for comparison
		pr, pt, _ = self._near_plane_intersection([self.right,self.top,-self.near], near)
		pl, pb, _ = self._near_plane_intersection([self.left,self.bottom,-self.near], near)
		
		# min/max considering (projected) original trbl
		t = np.amax(y)+pad
		r = np.amax(x)+pad
		b = np.amin(y)-pad
		l = np.amin(x)-pad
		if clip_to_original:
			t = np.clip(t, pb, pt)
			r = np.clip(r, pl, pr)
			b = np.clip(b, pb, pt)
			l = np.clip(l, pl, pr)
		
		# perserve aspect ratio
		if preserve_aspect:
			aspect = self.aspect
			h = t-b
			v = r-l
			a_v = h/aspect
			if a_v<v:
				a_h = v*aspect
				d_h_half = (a_h-h)/2.0
				t += d_h_half
				b -= d_h_half
			else:
				d_v_half = (a_v-v)/2.0
				r += d_v_half
				l -= d_v_half
		
		H = pt-pb #self.view_height
		W = pr-pl #self.view_width
	#	# project new trbl to original near for comparison
	#	pl, pb, _ = self._near_plane_intersection([l,b,near])
	#	pr, pt, _ = self._near_plane_intersection([r,t,near])
		scissor_rect = (((l-pl)/W,(b-pb)/H),((r-pl)/W,(t-pb)/H))
		
		new_cam = Camera(self.transform.copy_no_data(), nearFar=[near, far], topRightBottomLeft=[t,r,b,l], perspective=self.perspective, static=self.static, jitter=self.jitter)
		
		#print("Changed cam from {},{} ({})\nto {},{}\nsr: {}".format(self.clip, [pt,pr,pb,pl], self.trbl, new_cam.clip, new_cam.trbl, scissor_rect))
		return new_cam, scissor_rect
	
	def copy_clipped_to_world_coords(self, coordinate_list, preserve_aspect=True, pad=0.0, clamp_to_original=True, preserve_clip=False):
		'''
			copy camera with trbl and clip set to the minimum hull of the original and the projected coordinates (like scissor rect)
			does not change resolution -> higher spacial resolution
			coordinate_list: list of 3D world-space coorinates as 4D verctors (with w=1)
			preserve_clip: don't move near/far clip, preserves sampling step size
		'''
		
		# get x,y in view space in near plane
		view = MatrixTransform(self.view_matrix())
		pos_view = []
		pos_view_z = []
		for p in coordinate_list:
			p_view = view.transform(p)
			pos_view.append(p_view)
			pos_view_z.append(p_view[2])
		pos_view_z = np.asarray(pos_view_z)
		
		if not preserve_clip:
			near = np.amin(-pos_view_z)-pad
			far  = np.amax(-pos_view_z)+pad
			if clamp_to_original:
				near = np.clip(near, self.near, self.far)
				far  = np.clip(far , self.near, self.far)
		else:
			near = self.near
			far  = self.far
		
		#project frustum corners to new near
		n_scale = near / self.near
		pt = self.top * n_scale
		pr = self.right * n_scale
		pb = self.bottom * n_scale
		pl = self.left * n_scale
		
		pos_near = []
		for p_view in pos_view:
			pos_near.append(self._project_on_near(p_view, near))
		x,y,_ = np.split(np.asarray(pos_near), 3, axis=-1)
		
		# min/max considering (projected) original trbl
		t = np.amax(y)+pad
		r = np.amax(x)+pad
		b = np.amin(y)-pad
		l = np.amin(x)-pad
		if clamp_to_original:
			t = np.clip(t, pb, pt)
			r = np.clip(r, pl, pr)
			b = np.clip(b, pb, pt)
			l = np.clip(l, pl, pr)
		
		# perserve aspect ratio
		if preserve_aspect:
			aspect = self.aspect
			h = t-b
			v = r-l
			a_v = h/aspect
			if a_v<v:
				a_h = v*aspect
				d_h_half = (a_h-h)/2.0
				t += d_h_half
				b -= d_h_half
			else:
				d_v_half = (a_v-v)/2.0
				r += d_v_half
				l -= d_v_half
		
		H = pt-pb #self.view_height
		W = pr-pl #self.view_width
	#	# project new trbl to original near for comparison
	#	pl, pb, _ = self._near_plane_intersection([l,b,near])
	#	pr, pt, _ = self._near_plane_intersection([r,t,near])
		scissor_rect = (((l-pl)/W,(b-pb)/H),((r-pl)/W,(t-pb)/H))
		
		new_cam = Camera(self.transform.copy_no_data(), nearFar=[near, far], topRightBottomLeft=[t,r,b,l], perspective=self.perspective, static=self.static, jitter=self.jitter)
		
		#print("Changed cam from {},{} ({})\nto {},{}\nsr: {}".format(self.clip, [pt,pr,pb,pl], self.trbl, new_cam.clip, new_cam.trbl, scissor_rect))
		return new_cam, scissor_rect
	
	def copy_with_frustum_crop(self, coordinate_list, pad=0): #, preserve_aspect=True
		'''
			copy camera with trbl and clip set to the minimum hull of the original trbl and the projected coordinates,
			also reduces resolution to match
			coordinate_list: list of 3D world-space coorinates as 4D verctors (with w=1)
			pad: in grid cells
		'''
		def roundUp(x, base=1.0, offset=0.0):
			return math.ceil((x-offset)/base)*base + offset
		def roundDown(x, base=1.0, offset=0.0):
			return math.floor((x-offset)/base)*base + offset
		
		# get coordinates in view space
		view = MatrixTransform(self.view_matrix())
		pos_view = []
		pos_view_z = []
		for p in coordinate_list:
			p_view = view.transform(p)
			pos_view.append(p_view)
			pos_view_z.append(p_view[2])
		pos_view_z = np.asarray(pos_view_z)
		
		#clamp near and for to new view coords depth
		n = np.amin(-pos_view_z)
		f = np.amax(-pos_view_z)
		
		#match/round to original sampling coords
		pix_depth = self.depth_step
		n = roundDown(n, pix_depth, self.near)
		f = roundUp(f, pix_depth, self.far)
		
		#pad clamp to original clip
		n = np.clip(n - pix_depth*pad, self.near, self.far)
		f = np.clip(f + pix_depth*pad, self.near, self.far)
		grid_z = int(round((f-n)/pix_depth))
		offset_z = int(round((n-self.near)/pix_depth))
		
		#project frustum corners to new near
		n_scale = n / self.near
		pt = self.top * n_scale
		pr = self.right * n_scale
		pb = self.bottom * n_scale
		pl = self.left * n_scale
		
		pos_near = []
		for p_view in pos_view:
			pos_near.append(self._project_on_near(p_view, n))
		x,y,_ = np.split(np.asarray(pos_near), 3, axis=-1)
		
		t = np.amax(y)
		r = np.amax(x)
		b = np.amin(y)
		l = np.amin(x)
		
		#match/round to original sampling coords
		pix_height = self.pix_height_near * n_scale
		t = roundUp(t, pix_height, pb)
		b = roundDown(b, pix_height, pb)
		pix_width = self.pix_width_near * n_scale
		r = roundUp(r, pix_width, pl)
		l = roundDown(l, pix_width, pl)
		
		#pad and clamp
		t = np.clip(t + pix_height*pad, pb, pt)
		b = np.clip(b - pix_height*pad, pb, pt)
		grid_y = int(round((t-b)/pix_depth))
		offset_y = int(round((b-pb)/pix_depth))
		r = np.clip(r + pix_width*pad, pl, pr)
		l = np.clip(l - pix_width*pad, pl, pr)
		grid_x = int(round((r-l)/pix_depth))
		offset_x = int(round((l-pl)/pix_depth))
		
		scissor_size = (grid_y, grid_x)
		scissor_offset = (offset_y, offset_x)
		img_pad = [(offset_y, self.transform.grid_size[1] - grid_y - offset_y), (offset_x, self.transform.grid_size[2] - grid_x - offset_x), (0,0)]
		
		transform = self.transform.copy_no_data()
		transform.grid_size = [grid_z,grid_y, grid_x]
		new_cam = Camera(transform, nearFar=[n, f], topRightBottomLeft=[t,r,b,l], perspective=self.perspective, static=self.static, jitter=self.jitter)
		new_cam.scissor_pad = img_pad
		
		return new_cam
	
	def is_static(self):
		return self.transform.is_static()
	
	def set_LuT(self, lut, object_grid_transform):
		pass
	
	def has_LuT(self, object_grid_transform):
		pass
	
	def get_LuT(self, object_grid_transform):
		pass
	
	def to_dict(self):
		return {
				"grid_transform":to_dict(self.transform),
				"topRightBottomLeft":list(self.trbl),
				"nearFar":list(self.clip),
				"perspective":bool(self.perspective),
				"jitter":self.jitter,
			}