import os, sys, shutil, socket, faulthandler, signal, math, copy, random
import datetime, time
import logging, argparse
import json
import munch
import imageio


parser = argparse.ArgumentParser(description='Reconstruct volumetric smoke densities from 2D views.')
parser.add_argument('-s', '--setup', dest='setup_file', default=None, help='setup from JSON file to use')
parser.add_argument('-d', '--deviceID', dest="cudaID", default='0', help='id of cuda device to use')
parser.add_argument('-r', '--noRender', dest='render', action='store_false', help='turn off final rendering.')
parser.add_argument('-f', '--fit', dest='fit', action='store_true', help='run density volume optimization.')
parser.add_argument('-c', '--noConsole', dest='console', action='store_false', help='turn off console output')
parser.add_argument('--debug', dest='debug', action='store_true', help='enable debug output.')
args = parser.parse_args()

cudaID = args.cudaID

os.environ["CUDA_VISIBLE_DEVICES"]=cudaID
import numpy as np
import tensorflow as tf

from phitest.render import *
import phitest.render.render_helper as render_helper
from phitest.render.profiling import Profiler
from phitest.render.serialization import to_dict, from_dict

from lib.logger import StreamCapture
from lib.progress_bar import ProgressBar

tf.enable_eager_execution()

from lib.util import *
from lib.scalar_schedule import *
from lib.tf_ops import *
from lib.data import *
from lib.tf_colormap import *



def get_clip_nearFar(position, focus, depth):
	cam_dh = depth*0.5 #depth half
	dist = np.linalg.norm(focus-position)
	return [dist-cam_dh,dist+cam_dh]

def build_camera_from_sFcallibration(position, forward, up, right, resolution, fov_horizontal, fov_vertical, focus, focus_depth_clip=1.0, **kwargs):
	flip_z = lambda v: np.asarray(v)*np.asarray([1,1,-1])
	invert_v = lambda v: np.asarray(v)*(-1)
	pos = flip_z(position)
	fwd = invert_v(flip_z(forward))
	up = flip_z(up)
	right = flip_z(right)
	cam_focus = flip_z(focus)
	aspect = fov_horizontal/fov_vertical #resolution[2]/resolution[1] #W/H
	cam_dh = focus_depth_clip*0.5 #depth half
	
	dist = np.linalg.norm(cam_focus-pos)
	cam = Camera(MatrixTransform.from_fwd_up_right_pos(fwd, up, right, pos), nearFar=[dist-cam_dh,dist+cam_dh], fov=fov_horizontal, aspect=aspect, static=None)
	cam.transform.grid_size = copy.copy(resolution)
	
	return cam

def build_scalarFlow_cameras(setup, ids=[2,1,0,4,3], focus_depth_clip=1.0, interpolation_weights=[]):
	
	scalarFlow_cameras = []
	cam_resolution_scale = 1./setup.training.train_res_down #0.125#0.3
	train_cam_resolution = copy.copy(setup.rendering.main_camera.base_resolution)
	train_cam_resolution[1] = int(train_cam_resolution[1]*cam_resolution_scale)
	train_cam_resolution[2] = int(train_cam_resolution[2]*cam_resolution_scale)
	log.info('scalarFlow train camera resolution: %s', str(train_cam_resolution))
#	cam_dh = focus_depth_clip*0.5 #depth half
	
	for cam_id in ids:
		cam_calib = setup.calibration[str(cam_id)]
		if cam_calib.fov_horizontal is None:
			cam_calib.fov_horizontal = setup.calibration.fov_horizontal_average
		if cam_calib.fov_vertical is None:
			cam_calib.fov_vertical = setup.calibration.fov_vertical_average
	
	for i in range(len(ids)):
		cam_calib = setup.calibration[str(ids[i])]
		cam = build_camera_from_sFcallibration(**cam_calib, **setup.calibration, resolution=train_cam_resolution, focus_depth_clip=focus_depth_clip)
		scalarFlow_cameras.append(cam)
		
		if interpolation_weights and i<(len(ids)-1):
			for w in interpolation_weights:
				cam_calib = interpolate_camera_callibration(setup.calibration[str(ids[i])], setup.calibration[str(ids[i+1])], w, setup.calibration)
				cam = build_camera_from_sFcallibration(**cam_calib, **setup.calibration, resolution=train_cam_resolution, focus_depth_clip=focus_depth_clip)
				scalarFlow_cameras.append(cam)
		
	return scalarFlow_cameras

#requires cv2
#from lib.view_interpolation_test import get_dense_optical_flow, lerp_image, lerp_image_2, lerp_vector, slerp_vector

def interpolate_camera_callibration(cal1, cal2, interpolation_weight, calib_base):
	calib = munch.Munch()
	t = interpolation_weight
	calib["forward"] = slerp_vector(cal1["forward"], cal2["forward"], t, normalized=True)
	calib["up"] = slerp_vector(cal1["up"], cal2["up"], t, normalized=True)
	calib["right"] = slerp_vector(cal1["right"], cal2["right"], t, normalized=True)
	if True: #focus_slerp is not None:
		p1 = np.subtract(cal1["position"], calib_base["focus"])
		p2 = np.subtract(cal2["position"], calib_base["focus"])
		calib["position"] = np.add(slerp_vector(p1, p2, t, normalized=False), calib_base["focus"])
	else:
		calib["position"] = lerp_vector(cal1["position"], cal2["position"], t)
	calib["fov_horizontal"] = lerp(cal1["fov_horizontal"], cal2["fov_horizontal"], t)
	calib["fov_vertical"] = lerp(cal1["fov_vertical"], cal2["fov_vertical"], t)
	
	return calib

def interpolate_image(target1, target2, interpolation_weights, use_backwards_flow=True):
	single = False
	if np.isscalar(interpolation_weights):
		single = True
		interpolation_weights = [interpolation_weights]
	
	is_tf = False
	if isinstance(target1, tf.Tensor):
		target1 = target1.numpy()
		is_tf = True
	if isinstance(target2, tf.Tensor):
		target2 = target2.numpy()
		is_tf = True
	
	flow = get_dense_optical_flow(target1, target2)
	if use_backwards_flow:
		flow_back = get_dense_optical_flow(target2, target1)
	targets = [lerp_image_2(target1, target2, w, flow, flow_back) if use_backwards_flow else lerp_image(target1, target2, w, flow) for w in interpolation_weights]
	
	if is_tf:
		targets = [tf.constant(_) if len(_.shape)==3 else tf.constant(_[...,np.newaxis]) for _ in targets]
	
	if single:
		return targets[0]
	else:
		return targets

def interpolate_images(images, interpolation_weights, use_backwards_flow=True):
	ret = []
	for img1, img2 in zip(images[:-1], images[1:]):
		ret.append(img1)
		ret.extend(interpolate_image(img1, img2, interpolation_weights, use_backwards_flow))
	ret.append(images[-1])
	return ret

def setup_target_cameras(base_cameras, frustum_resolution, crop_coordinates=None, crop_pad=0, normalize_resolution=False, jitter=False):
	cams = copy.deepcopy(base_cameras)
	for cam in cams:
		cam.transform.grid_size = frustum_resolution
	if crop_coordinates is not None:
		cams = [cam.copy_with_frustum_crop(crop_coordinates, crop_pad) for cam in cams]
		#normalize cams to same grid size to allow sampling batching
		if normalize_resolution:
			resolutions = [cam.transform.grid_size for cam in cams]
			resolution_hull = np.amax(resolutions, axis=0)
			for cam in cams:
				pass
				#cam.pad_frustum_to(resolution_hull) TODO
	if jitter:
		#log.info("Setup target cameras with jitter")
		raise NotImplementedError("TODO: fix too large uv jitter.")
		for cam in cams:
			cam.jitter = cam.depth_step
	return cams


def preestimate_volume(grid_transform, targets, cameras):
#--- Volume Estimation ---
	unprojections = []
	for i in range(len(cameras)):
		cam = cameras[i]
		#expand target to frustum volume (tile along z)
		tar = tf.reshape(targets[i], [1,1] + list(targets[i].shape))
		tar = tf.tile(tar, (1,cam.transform.grid_size[0],1,1,1))
		#sample target to shared volume
		unprojections.append(renderer.sample_camera(tar, grid_transform, cam, inverse=True))
	unprojection = tf.reduce_min(unprojections, axis=0)
	return unprojection
#	tf_print_stats(unprojection, 'volume estimate', log=log)
#	grid_transform.set_data(unprojection*setup.data.density_scale)
#	np.savez_compressed(os.path.join(setup.paths.data, 'density_estimate.npz'), unprojection.numpy())

def generate_volume(grid_transform, targets, cameras, gen_model, setup, render_cameras=None, cut_alpha=True, random_rotation_pivot=None):
	# set a random rotation of the (shared) volume to make the generator rotationally invariant
	if random_rotation_pivot is not None:
		random_rotation_pivot.rotation_deg = np.random.uniform(0,360, 3).tolist()
	# get initial estimate from unprojected targets
	with profiler.sample('pre-estimate volume'):
		volume_estimate = preestimate_volume(sim_transform, targets, cameras)
	# let the generator refine the volume
	with profiler.sample('generate volume'):
		volume = volume_estimate
		for rec in range(setup.training.generator.recursion):
			volume = tf.clip_by_value(gen_model(volume)*setup.training.generator.out_scale, setup.training.generator.out_min, setup.training.generator.out_max)
	sim_transform.set_data(volume)
	# render images from refined volume for loss
	if render_cameras is not None:
		imgs = renderer.render_density(sim_transform, lights, render_cameras, cut_alpha=cut_alpha)
		return volume, imgs
	return volume

def hull_AABB_OS(hull, hull_threshold = 0.1):
	'''min and max coord in object-space for each axis'''
	assert len(hull.get_shape().as_list())==3, "hull must be 3D DHW"
	def min_max_coords(flat_hull):
		coords = tf.cast(tf.where(tf.greater_equal(flat_hull, hull_threshold)), tf.float32)
		min_coord = tf.minimum(tf.reduce_min(coords), tf.cast(tf.shape(flat_hull)[0], tf.float32))
		max_coord = tf.maximum(tf.reduce_max(coords), 0.)
		return min_coord, max_coord #tf.reduce_min(coords), tf.reduce_max(coords)
	x_min, x_max = min_max_coords(tf.reduce_max(hull, axis=(-2,-3))) #W
	y_min, y_max = min_max_coords(tf.reduce_max(hull, axis=(-1,-3))) #H
	z_min, z_max = min_max_coords(tf.reduce_max(hull, axis=(-1,-2))) #D
	return ([x_min, y_min, z_min],[x_max, y_max, z_max])

def create_inflow(hull, hull_height, height):
	hull_threshold = 0.1
	assert len(hull.get_shape().as_list())==3, "hull must be 3D DHW"
	if tf.reduce_max(hull)<1.0:
		log.warning("Empty hull -> no inflow.")#raise ValueError('Empty hull')
		return None, None, None
	#find lowest part of hull, https://stackoverflow.com/questions/42184663/how-to-find-an-index-of-the-first-matching-element-in-tensorflow
	y_hull = tf.reduce_max(hull, axis=(-1,-3)) #H
	y_idx_min = tf.reduce_min(tf.where(tf.greater_equal(y_hull, hull_threshold)))
	y_idx_max = y_idx_min + hull_height
	
	#take max xz extend of hull from hull_min to hull_min+hull_height
	hull_slice = hull[:,y_idx_min:y_idx_max,:]
	flat_hull_slice = tf.reduce_max(hull_slice, axis=(-2), keepdims=True)
	x_hull_slice_idx = tf.where(tf.greater_equal(tf.reduce_max(flat_hull_slice, axis=(-2,-3)), hull_threshold))
	x_hull_slice_idx_min = tf.reduce_min(x_hull_slice_idx)
	x_hull_slice_idx_max = tf.reduce_max(x_hull_slice_idx) +1
	z_hull_slice_idx = tf.where(tf.greater_equal(tf.reduce_max(flat_hull_slice, axis=(-2,-1)), hull_threshold))
	z_hull_slice_idx_min = tf.reduce_min(z_hull_slice_idx)
	z_hull_slice_idx_max = tf.reduce_max(z_hull_slice_idx) +1
	
	flat_hull_slice = flat_hull_slice[z_hull_slice_idx_min:z_hull_slice_idx_max, :, x_hull_slice_idx_min:x_hull_slice_idx_max]
	
	if height=='MAX': #extend inflow all the way to the lower end of the grid
		height = y_idx_max.numpy().tolist()
	else:
		height = max(y_idx_max.numpy().tolist(), height)
	inflow_mask = tf.tile(flat_hull_slice, (1,height,1))
	inflow_shape = [(z_hull_slice_idx_max-z_hull_slice_idx_min).numpy().tolist(), height, (x_hull_slice_idx_max-x_hull_slice_idx_min).numpy().tolist()]
	inflow_offset = [z_hull_slice_idx_min.numpy().tolist(), (y_idx_max-height).numpy().tolist(), x_hull_slice_idx_min.numpy().tolist()]
	
	#return size and (corner)position
	return inflow_mask, inflow_shape, inflow_offset

# --- RENDERING ---

def render_cameras(grid_transform, cameras, lights, renderer, img_path, name_pre='img', bkg=None, \
		format='EXR', img_transfer=None, cut_alpha=True, img_normalize=False):
	
	imgs = renderer.render_density(grid_transform, lights, cameras, background=bkg, cut_alpha=cut_alpha)#, background=bkg
	imgs = tf.concat(imgs, axis=0)
	if not cut_alpha:
		imgs, imgs_d = tf.split(imgs, [3,1], axis=-1)
		imgs = tf.concat([imgs, tf.exp(-imgs_d)], axis=-1)
	if img_normalize:
		imgs /= tf.reduce_max(imgs, axis=(-3,-2,-1))
	if img_transfer:
		imgs = tf_element_transfer_func(imgs, img_transfer)
	with renderer.profiler.sample("save image"):
		renderer.write_images([imgs], [name_pre+'_cam_{:04d}'], base_path=img_path, use_batch_id=True, format=format)

def render_cycle(grid_transform, cameras, lights, renderer, img_path, name_pre='img', steps=12, steps_per_cycle=12, bkg=None, \
		format='EXR', img_transfer=None, img_stats=True, rotate_cameras=False, cut_alpha=True, img_normalize=False):
	
	r_step = 360.0/steps_per_cycle
	
	if renderer.can_render_fused and rotate_cameras:
		cams = []
		for camera in cameras:
			for i in range(steps):
				cam = copy.deepcopy(camera)
				cam.transform.parent.rotation_deg[1] -= r_step*i
				cams.append(cam)
		if bkg is not None:
			bkg = [_ for _ in bkg for i in range(steps)]
		render_cameras(grid_transform, cams, lights, renderer, img_path, name_pre, bkg, format, img_transfer, cut_alpha, img_normalize)
		return
	
	if rotate_cameras:
		cameras = copy.deepcopy(cameras)
	else:
		rot = grid_transform.rotation_deg
		grid_transform.rotation_deg = [0,0,0]
	#if setup.rendering.background.type=='COLOR': bkg_render=[bkg_render]*len(cameras)
	with renderer.profiler.sample("render cycle "+name_pre):
		for i in range(steps):
			#log.debug('render frame %d', i)
			if not rotate_cameras:
				grid_transform.rotation_deg = [0,i*r_step,0]
			with renderer.profiler.sample("render step"):
				imgs = renderer.render_density(grid_transform, lights, cameras, background=bkg, cut_alpha=cut_alpha)#, background=bkg
				imgs = tf.concat(imgs, axis=0)
				if not cut_alpha:
					imgs, imgs_d = tf.split(imgs, [3,1], axis=-1)
					imgs = tf.concat([imgs, tf.exp(-imgs_d)], axis=-1)
				if img_normalize:
					imgs /= tf.reduce_max(imgs, axis=(-3,-2,-1))
				if img_transfer:
					if isinstance(img_transfer, tuple):
						imgs = tf_cmap_nearest(imgs, *img_transfer)
						#log.debug("Remapped image shape: %s", imgs.get_shape().as_list())
					else:
						imgs = tf_element_transfer_func(imgs, img_transfer)
			if args.console and img_stats:
				print_stats(imgs, 'frame '+str(i))
			with renderer.profiler.sample("save image"):
				renderer.write_images([imgs], [name_pre+'_cam{}_{:04d}'], base_path=img_path, use_batch_id=True, frame_id=i, format=format)
			if rotate_cameras:
				for camera in cameras:
					camera.transform.parent.rotation_deg[1] -= r_step #counter-rotate cam to match same object-view as object rotation
	if not rotate_cameras: grid_transform.rotation_deg = rot

def render_gradients(gradients, grid_transform, cameras, renderer, path, image_mask, steps=12, steps_per_cycle=12, format='EXR', img_stats=True, name="gradients", log=None):
	tf_print_stats(gradients, "gradients " + name, log=log)
#	print("gradient shape:", tf.shape(gradients))
	os.makedirs(path, exist_ok=True)
	grad_shape = GridShape.from_tensor(gradients)
	if grad_shape.c==1: #density gradients
		grad_light = tf.concat([tf.maximum(gradients,0), tf.zeros_like(gradients), tf.maximum(-gradients, 0)], axis=-1)
	elif grad_shape.c==3: #velocity gradients
		grad_light = tf.abs(gradients)
	grid_transform = grid_transform.copy_new_data(tf.zeros_like(gradients))
	grid_transform.rotation_deg = [0,0,0]
	r_step = 360.0/steps_per_cycle
	with renderer.profiler.sample("render gradients cycle"):
		for i in range(steps):
			#log.debug('render frame %d', i)
			grid_transform.rotation_deg = [0,i*r_step,0]
			with renderer.profiler.sample("render step"):
				imgs = renderer.render_density(grid_transform, [grad_light], cameras, cut_alpha=True)#, background=bkg
				imgs = tf.concat(imgs, axis=0)
		#	if args.console and img_stats:
		#		print_stats(imgs, 'gradients frame '+str(i))
		#	print("gradient images shape:", tf.shape(imgs))
		#	print("gradient images type:", imgs.dtype)
		#	with tf.device("/gpu:0"):
		#		imgs = tf.identity(imgs)
			imgs /=tf.reduce_max(imgs)
			with renderer.profiler.sample("save image"):
				renderer.write_images([imgs], [image_mask], base_path=path, use_batch_id=True, frame_id=i, format=format)

def write_image_gradients(gradient_images, renderer, path, image_mask, image_neg_mask, format='EXR', img_stats=True):
	os.makedirs(path, exist_ok=True)
	if args.console and img_stats:
		print_stats(gradient_images, 'gradients frame '+str(i))
	imgs = gradient_images / tf.reduce_max(tf.abs(gradient_images))
	imgs_neg = tf.maximum(-imgs, 0)
	imgs = tf.maximum(imgs, 0)
	with renderer.profiler.sample("save image"):
		renderer.write_images([imgs, imgs_neg], [image_mask, image_neg_mask], base_path=path, use_batch_id=True, format=format)

'''
def advect_step(density, velocity):
	density = velocity.warp(density)
	velocity = velocity.copy_warped()
	return density, velocity
'''
def world_scale(shape, size=None, width=None, as_np=True):
	'''
	shape and size are z,y,x
	width corresponds to x and keeps aspect/cubic cells
	'''
	assert len(shape)==3
	if size is not None and width is not None:
		raise ValueError("Specify only one of size or width.")
	if size is not None:
		assert len(size)==3
		scale = np.asarray(size, dtype=np.float32)/np.asarray(shape, dtype=np.float32)
	elif width is not None:
		scale = np.asarray([width/shape[-1]]*3, dtype=np.float32)
	else:
		raise ValueError("Specify one of size or width.")
	if as_np:
		return scale
	else:
		return scale.tolist()


if __name__=='__main__':
	from common_setups import RECONSTRUCT_SEQUENCE_SETUP
	#setup = default_setup
	if args.setup_file is not None:
		try:
			with open(args.setup_file, 'r') as setup_json:
				setup = json.load(setup_json)
		except:
			raise #log.exception('Could not read setup from file: %s', args.setup_file)
			#exit(1)
	else:
		raise ValueError("No setup provided")
	setup = update_dict_recursive(RECONSTRUCT_SEQUENCE_SETUP, setup, deepcopy=True, new_key='DISCARD_WARN')
	
	with open(setup["rendering"]["target_cameras"]["calibration_file"], 'r') as calibration_file:
		cam_setup = json.load(calibration_file)
	setup['calibration']=cam_setup
	def flip_z(v):
		return v*np.asarray([1,1,-1])
	
	setup = munch.munchify(setup)
	cam_setup = setup.calibration
	
	hostname = socket.gethostname()
	now = datetime.datetime.now()
	now_str = now.strftime("%y%m%d-%H%M%S")
	try:
		paths = setup.paths
	except AttributeError:
		setup.paths = munch.Munch()
	prefix = 'seq'
	try:
		base_path = setup.paths.base
	except AttributeError:
		setup.paths.base = "./"
		base_path = setup.paths.base
	try:
		run_path = setup.paths.run
	except AttributeError:
		if args.fit:
			setup.paths.run = 'recon_{}_{}_{}'.format(prefix, now_str, setup.title)
		else:
			setup.paths.run = 'render_{}_{}_{}'.format(prefix, now_str, setup.title)
	if hasattr(setup.paths, 'group'):
		setup.paths.path = os.path.join(setup.paths.base, setup.paths.group, setup.paths.run)
	else:
		setup.paths.path = os.path.join(setup.paths.base, setup.paths.run)
	#run_path = setup.paths.run
	
	
	if os.path.isdir(setup.paths.path):
		setup.paths.path, _ = makeNextGenericPath(setup.paths.path)
	else:
		os.makedirs(setup.paths.path)
	
	setup.paths.log = os.path.join(setup.paths.path, 'log')
	os.makedirs(setup.paths.log)
	setup.paths.config = os.path.join(setup.paths.path, 'config')
	os.makedirs(setup.paths.config)
	setup.paths.data = setup.paths.path #os.path.join(setup.paths.run, 'images')
	if setup.validation.warp_test:
		setup.paths.warp_test = os.path.join(setup.paths.path, 'warp_test')
		os.makedirs(setup.paths.warp_test)
	#os.makedirs(setup.paths.data)
	
	sys.stderr = StreamCapture(os.path.join(setup.paths.log, 'stderr.log'), sys.stderr)
	
	#setup logging
	log_format = '[%(asctime)s][%(name)s:%(levelname)s] %(message)s'
	log_formatter = logging.Formatter(log_format)
	root_logger = logging.getLogger()
	root_logger.setLevel(logging.INFO)
	logfile = logging.FileHandler(os.path.join(setup.paths.log, 'logfile.log'))
	logfile.setLevel(logging.INFO)
	logfile.setFormatter(log_formatter)
	root_logger.addHandler(logfile)
	errlog = logging.FileHandler(os.path.join(setup.paths.log, 'error.log'))
	errlog.setLevel(logging.WARNING)
	errlog.setFormatter(log_formatter)
	root_logger.addHandler(errlog)
	if args.debug:
		debuglog = logging.FileHandler(os.path.join(setup.paths.log, 'debug.log'))
		debuglog.setLevel(logging.DEBUG)
		debuglog.setFormatter(log_formatter)
		root_logger.addHandler(debuglog)
	if args.console:
		console = logging.StreamHandler(sys.stdout)
		console.setLevel(logging.INFO)
		console_format = logging.Formatter('[%(name)s:%(levelname)s] %(message)s')
		console.setFormatter(console_format)
		root_logger.addHandler(console)
	log = logging.getLogger('train')
	log.setLevel(logging.DEBUG)
	
	if args.debug:
		root_logger.setLevel(logging.DEBUG)
		log.info("Debug output active")
	
	
	
	with open(os.path.join(setup.paths.config, 'setup.json'), 'w') as config:
		json.dump(setup, config, sort_keys=True, indent=2)
	
	sources = [sys.argv[0], "common_setups.py"]
	sources.extend(os.path.join("lib", _) for _ in os.listdir("lib") if _.endswith(".py"))
	sources.extend(os.path.join("phitest/render", _) for _ in os.listdir("phitest/render") if _.endswith(".py"))
	archive_files(os.path.join(setup.paths.config,'sources.zip'), *sources)
	
	log.info('--- Running test: %s ---', setup.title)
	log.info('Test description: %s', setup.desc)
	log.info('Test directory: %s', setup.paths.path)
	log.info('Python: %s', sys.version)
	log.info('TensorFlow version: %s', tf.__version__)
	log.info('host: %s, device: %s, pid: %d', hostname, cudaID, os.getpid())
	
	if setup.data.rand_seed_global is not None:
		os.environ['PYTHONHASHSEED']=str(setup.data.rand_seed_global)
		random.seed(setup.data.rand_seed_global)
		np.random.seed(setup.data.rand_seed_global)
		tf.set_random_seed(setup.data.rand_seed_global)
	log.info("global random seed: %s", setup.data.rand_seed_global)
	
	profiler = Profiler()
	renderer = Renderer(profiler,
		filter_mode=setup.rendering.filter_mode,
		mipmapping=setup.rendering.mip.mode,
		num_mips=setup.rendering.mip.level,
		mip_bias=setup.rendering.mip.bias,
		blend_mode=setup.rendering.blend_mode,
		sample_gradients=setup.rendering.sample_gradients,
		fast_gradient_mip_bias_add=0.0,
		luma = setup.rendering.luma,
		fused=setup.rendering.allow_fused_rendering)
	vel_renderer = Renderer(profiler,
		filter_mode=setup.rendering.filter_mode,
		mipmapping=setup.rendering.mip.mode,
		num_mips=setup.rendering.mip.level,
		mip_bias=setup.rendering.mip.bias,
		blend_mode='ADDITIVE',
		sample_gradients=setup.rendering.sample_gradients,
		luma = setup.rendering.luma,
		fused=setup.rendering.allow_fused_rendering)
	scale_renderer = Renderer(profiler,
		filter_mode='LINEAR',
		boundary_mode=setup.data.velocity.boundary.upper(),
		mipmapping=setup.rendering.mip.mode,
		num_mips=setup.rendering.mip.level,
		mip_bias=setup.rendering.mip.bias,
		blend_mode='ADDITIVE',
		sample_gradients=setup.rendering.sample_gradients)
	warp_renderer = Renderer(profiler,
		filter_mode='LINEAR',
		boundary_mode=setup.data.velocity.boundary.upper(),
		mipmapping='NONE',
		blend_mode='ADDITIVE',
		sample_gradients=False)
	
	synth_target_renderer = Renderer(profiler,
		filter_mode=setup.rendering.synthetic_target.filter_mode,
		mipmapping=setup.rendering.mip.mode,
		num_mips=setup.rendering.mip.level,
		mip_bias=setup.rendering.mip.bias,
		blend_mode=setup.rendering.synthetic_target.blend_mode,
		sample_gradients=setup.rendering.sample_gradients,
		fast_gradient_mip_bias_add=0.0,
		luma = setup.rendering.luma,
		fused=setup.rendering.allow_fused_rendering)
		
	upscale_renderer = Renderer(profiler,
		filter_mode='LINEAR',
		boundary_mode=setup.data.velocity.boundary.upper(),
		mipmapping='NONE',
		blend_mode='ADDITIVE',
		sample_gradients=setup.rendering.sample_gradients)
	max_renderer = Renderer(profiler,
		filter_mode=setup.rendering.filter_mode,
		mipmapping=setup.rendering.mip.mode,
		num_mips=setup.rendering.mip.level,
		mip_bias=setup.rendering.mip.bias,
		blend_mode='MAX',
		sample_gradients=setup.rendering.sample_gradients,
		fast_gradient_mip_bias_add=0.0,
		fused=setup.rendering.allow_fused_rendering)
	
	grad_renderer = vel_renderer

	
	
	pFmt = PartialFormatter()
	run_index = RunIndex(setup.data.run_dirs, ['recon_seq',])#prefix_voldifrender, recursive=True)

	def load_velocity(mask, fmt=None, boundary=None, scale_renderer=None, warp_renderer=None, device=None, var_name="velocity"):
	#	id_end = mask.find(']')
		sf = RunIndex.parse_scalarFlow(mask)
		load_mask = run_index[mask]
		if load_mask is not None:
			load_mask = pFmt.format(load_mask, **fmt) if fmt is not None else load_mask
			log.info("load velocity grid from run %s", load_mask)
			vel_grid = VelocityGrid.from_file(load_mask, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name)
		elif sf is not None: #mask.startswith('[SF:') and id_end!=-1: # explicit scalarFlow
			if sf["frame"] is None:
				raise ValueError("Missing frame from scalarFlow specifier.")
			#sim_offset, frame_offset = mask[4:id_end].split(':')
			fmt['sim'] += sf["sim"] #int(sim_offset)
			fmt['frame'] += sf["frame"] #int(frame_offset)
			#sub_path = mask[id_end+1:]
			run_path = os.path.normpath(os.path.join(setup.data.velocity.scalarFlow_reconstruction, sf["relpath"])).format(**fmt)
			log.info("load velocity grid from ScalarFlow %s", run_path)
			vel_grid = VelocityGrid.from_scalarFlow_file(run_path, boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name)
		else: # not in runs, assume centered vel
			load_mask = mask.format(**fmt) if fmt is not None else mask
			log.info("load centered velocity grid from file %s", load_mask)
			with np.load(load_mask) as np_data:
				vel_centered = reshape_array_format(np_data['data'], 'DHWC')
			vel_grid = VelocityGrid.from_centered(tf.constant(vel_centered, dtype=tf.float32), boundary=boundary, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=device, var_name=var_name)
		return vel_grid
	
	
	
	color_channel = 1 if setup.rendering.monochrome else 3
	train_res_down=setup.training.train_res_down
	
	# using scaled up cube here as we have random volume rotation as data augmentation for the generator
	density_size = [setup.data.grid_size]*3
	density_size[1] = int(math.ceil(density_size[1] * cam_setup.scale_y))
	volume_size = [cam_setup.marker_width, cam_setup.marker_width * cam_setup.scale_y, cam_setup.marker_width]#cam_setup.volume_size']
	sim_center = flip_z(cam_setup.volume_offset + np.asarray(volume_size)/2.0) #for easy rotation around the volume center
	sim_transform = GridTransform(density_size, translation=sim_center, scale=[cam_setup.marker_width]*3, center=True, normalize='MIN')
	sF_transform = GridTransform([100,178,100], translation=flip_z(cam_setup.volume_offset + np.asarray([0,0,cam_setup.marker_width])), scale=[cam_setup.marker_width]*3, normalize='MIN')
	density_size = Int3(density_size[::-1])
	
	cam_resolution = copy.copy(setup.rendering.main_camera.base_resolution) #[256,int(1920*cam_resolution_scale),int(1080*cam_resolution_scale)] #zyx
	aspect = cam_resolution[2]/cam_resolution[1]
	cam_resolution[1] *= setup.rendering.main_camera.resolution_scale
	cam_resolution[2] *= setup.rendering.main_camera.resolution_scale
	#cam_yh, cam_xh = 1920.0/2000.0 *cam_xy_scale, 1080.0/2000.0 *cam_xy_scale#height, width half
	cam_dist = setup.rendering.main_camera.distance
	main_camera = Camera(GridTransform(cam_resolution, translation=[0,0,cam_dist], parent=Transform(translation=[0.33913451, 0.38741691, -0.25786148], rotation_deg=[0,0,0.])), nearFar=[setup.rendering.main_camera.near,setup.rendering.main_camera.far], #topRightBottomLeft=[cam_yh,cam_xh,-cam_yh,-cam_xh])
	fov = setup.rendering.main_camera.fov, aspect=aspect)
	cameras = [
		main_camera,
	]
	if not args.fit:
	#	num_views = 8
	#	for i in range(1,num_views):
	#		tmp_cam = copy.deepcopy(main_camera)
	#		tmp_cam.transform.parent.rotation_deg = [0,(360./num_views)*i,0]
	#		cameras.append(tmp_cam)
	#	del tmp_cam
		tmp_cam = copy.deepcopy(main_camera)
		tmp_cam.transform.parent.rotation_deg = [0,90,0]
		cameras.append(tmp_cam)
	#	tmp_cam = copy.deepcopy(main_camera)
	#	tmp_cam.transform.parent.rotation_deg = [0,180,0]
	#	cameras.append(tmp_cam)
		tmp_cam = copy.deepcopy(main_camera)
		tmp_cam.transform.parent.rotation_deg = [0,225,0]
		cameras.append(tmp_cam)
		del tmp_cam
	else:
		tmp_cam = copy.deepcopy(main_camera)
		tmp_cam.transform.parent.rotation_deg = [0,90,0]
		cameras.append(tmp_cam)
		del tmp_cam
	
	for _cam in cameras:
		renderer.check_LoD(sim_transform, _cam, check_inverse=True, name="main camera")
	
	scalarFlow_cam_ids = setup.rendering.target_cameras.camera_ids #[2,1,0,4,3] #[0,2,3] #
	if setup.data.density.target_cam_ids =="ALL":
		setup.data.density.target_cam_ids = list(range(len(scalarFlow_cam_ids)))
	view_interpolation_weights = [(_+1)/(setup.training.density.view_interpolation.steps+1) for _ in range(setup.training.density.view_interpolation.steps)]
	scalarFlow_cameras = build_scalarFlow_cameras(setup, scalarFlow_cam_ids, interpolation_weights=view_interpolation_weights)
	scalarFlow_cameras_base = [scalarFlow_cameras[_*(setup.training.density.view_interpolation.steps+1)] for _ in range(len(scalarFlow_cam_ids))]
	
	#train_down_pool = (1, setup.training.train_res_down, setup.training.train_res_down, 1)
	scalarFlow_cam_focus = flip_z(setup.calibration.focus)#[0.33913451, 0.38741691 -0.05, -0.25786148] #cam_setup.focus']#
	cam_resolution_scale = 1./setup.training.train_res_down #0.125#0.3
	train_cam_resolution = copy.copy(setup.rendering.main_camera.base_resolution)
	train_cam_resolution[1] = int(train_cam_resolution[1]*cam_resolution_scale)
	train_cam_resolution[2] = int(train_cam_resolution[2]*cam_resolution_scale)
	log.info('scalarFlow train camera resolution: %s', str(train_cam_resolution))
	for sF_cam in scalarFlow_cameras:
		renderer.check_LoD(sim_transform, sF_cam, check_inverse=True, name="scalarFlow camera")
	cam_dh = 0.5 #depth half
	if setup.training.discriminator.active:
		disc_cam_resolution_scale = 1./setup.training.discriminator.cam_res_down #0.125#0.3
		disc_cam_resolution = copy.copy(setup.rendering.main_camera.base_resolution)
		disc_cam_resolution[1] = int(disc_cam_resolution[1]*disc_cam_resolution_scale)
		disc_cam_resolution[2] = int(disc_cam_resolution[2]*disc_cam_resolution_scale)
		disc_cam_dist = 1.3
		disc_camera = Camera(GridTransform(disc_cam_resolution, translation=[0,0,disc_cam_dist], parent=Transform(translation=scalarFlow_cam_focus, rotation_deg=[0,0,0])), nearFar=[disc_cam_dist-cam_dh,disc_cam_dist+cam_dh], fov=cam_setup.fov_horizontal_average, aspect=aspect)
		disc_cameras = [copy.deepcopy(disc_camera) for _ in range(setup.training.discriminator.num_fake)]
		if setup.training.discriminator.fake_camera_jitter:
			raise NotImplementedError("TODO: fix too large uv jitter.")
			for cam in disc_cameras:
				cam.jitter = cam.depth_step
		log.info('discriminator camera resolution: %s, jitter: %s', str(disc_cam_resolution), setup.training.discriminator.fake_camera_jitter)
		renderer.check_LoD(sim_transform, disc_camera, check_inverse=True, name="discriminator camera")
	log.debug('Main camera transform: %s', main_camera.transform)
	
	lights = []
		#PointLight(Transform(translation=[-0.5,-0.5,-0.5], parent=Transform()), color=[0.5,0.9,1.0], intensity=2.4, range_scale=7.4),
		#PointLight(Transform(translation=[0,0,2], parent=Transform(translation=[0,0.2,0], rotation_deg=[0,45,0])), color=[0.75,0.95,1.0], intensity=4, range_scale=1),
		#SpotLight(Transform(translation=[0,0,2], parent=Transform(translation=[0,0.2,0], rotation_deg=[0,45,0])), color=[0.7,0.95,1.0], intensity=4, cast_shadows=True, shadow_clip=[1.25, 2.75], range_scale=1, angle_deg=35., shadow_resolution=[256]*3, cone_mask=False),
		#SpotLight(Transform(translation=[0,0,2], parent=Transform(translation=[0,0,0], rotation_deg=[90,0,0])), color=[0.7,0.95,1.0], intensity=6, cast_shadows=True, shadow_clip=[1.25, 2.75], range_scale=1, angle_deg=25., shadow_resolution=[256]*3, cone_mask=False),
		#PointLight(Transform(translation=[0.2,0,0], parent=Transform()), color=[0.4,0.0,1.0], intensity=2.0, range_scale=12),
		#PointLight(Transform(translation=[-0.2,0,0], parent=Transform()), color=[0.2,1.0,0.0], intensity=1.4, range_scale=12),
		#PointLight(Transform(translation=[0,0,2], parent=Transform(translation=scalarFlow_cam_focus, rotation_deg=[-40,0,0])), intensity=setup.rendering.lighting.initial_intensity, range_scale=0.5),
	
	if setup.rendering.lighting.initial_intensity>=0:
		lights.append(
			SpotLight(Transform(translation=[0,0,2], parent=Transform(translation=scalarFlow_cam_focus, rotation_deg=[-40,0,0])), intensity=setup.rendering.lighting.initial_intensity, \
			cast_shadows=True, shadow_clip=[1.35, 2.65], range_scale=0.825, angle_deg=25., shadow_resolution=setup.rendering.lighting.shadow_resolution, cone_mask=False, \
			static=sim_transform if setup.rendering.allow_static_cameras else None)
		)
	
	if setup.rendering.lighting.ambient_intensity>=0:
		lights.append(Light(intensity=setup.rendering.lighting.ambient_intensity)) #some simple constant/ambient light as scattering approximation
	
	for light in lights:
		if isinstance(light, SpotLight) and light.cast_shadows:
			renderer.check_LoD(sim_transform, light.shadow_cam, check_inverse=True, name="shadow camera")
	
	
	shadow_lights = [
		SpotLight(Transform(translation=[0,0,2], parent=Transform(translation=scalarFlow_cam_focus, rotation_deg=[-40,0,0])), intensity=3.4, \
		cast_shadows=True, shadow_clip=[1.35, 2.65], range_scale=0.825, angle_deg=25., shadow_resolution=setup.rendering.lighting.shadow_resolution, cone_mask=False, \
		static=sim_transform if setup.rendering.allow_static_cameras else None),
		Light(intensity=0.08),
	]
	
	synth_target_lights = []
	if setup.rendering.synthetic_target.initial_intensity>=0:
		synth_target_lights.append(
			SpotLight(Transform(translation=[0,0,2], parent=Transform(translation=scalarFlow_cam_focus, rotation_deg=[-40,0,0])), intensity=setup.rendering.synthetic_target.initial_intensity, \
			cast_shadows=True, shadow_clip=[1.35, 2.65], range_scale=0.825, angle_deg=25., shadow_resolution=setup.rendering.lighting.shadow_resolution, cone_mask=False, \
			static=sim_transform if setup.rendering.allow_static_cameras else None)
		)
	
	if setup.rendering.synthetic_target.ambient_intensity>=0:
		synth_target_lights.append(Light(intensity=setup.rendering.synthetic_target.ambient_intensity))
	#synth_target_renderer = renderer
	
	
	if not args.fit:
		# scene serialization
		scene = {
			"cameras":cameras,
			"sFcameras":scalarFlow_cameras,
			"lighting":lights,
			"objects":[sim_transform],
		}
		scene_file = os.path.join(setup.paths.config, "scene.json")
		#log.debug("Serializing scene to %s ...", scene_file)
		with open(scene_file, "w") as file:
			try:
				json.dump(scene, file, default=to_dict, sort_keys=True)#, indent=2)
			except:
				log.exception("Scene serialization failed.")
	
	main_render_ctx = RenderingContext([main_camera], lights, renderer, vel_renderer, setup.rendering.monochrome)
	
	def render_sequence(sequence, vel_pad, cycle=True, cycle_steps=12, sF_cam=False, render_density=True, render_shadow=True, render_velocity=True):
		log.debug("Render images for sequence")
		clip_cams = True
		with profiler.sample('render sequence'):
			if cycle:
				cycle_cams = [main_camera]
			
			if render_shadow:
				shadow_cams = [copy.deepcopy(main_camera) for _ in range(1)]
				shadow_cams[0].transform.parent.rotation_deg[1] -= 60
				#shadow_cams[1].transform.parent.rotation_deg[1] += 42
				shadow_cams_cycle = [main_camera]
				shadow_dens_scale = 4.
			
			if clip_cams:
				AABB_corners_WS = []
				AABB_corners_WS_cycle = []
				GRID_corners_WS_cycle = []
				for state in sequence:
					dens_transform = state.get_density_transform()
					dens_hull = state.density.hull #state.hull if hasattr(state, "hull") else 
					if dens_hull is None:
						continue
					corners_OS = hull_AABB_OS(tf.squeeze(dens_hull, (0,-1)))
					AABB_corners_WS += dens_transform.transform_AABB(*corners_OS, True)
					
					dens_shape = dens_transform.grid_shape
					grid_OS = (np.asarray([0,0,0], dtype=np.float32), np.asarray(dens_shape.xyz, dtype=np.float32))
					cycle_transform = dens_transform.copy_no_data()
					AABB_corners_WS_cycle.extend(cycle_transform.transform_AABB(*corners_OS, True))
					GRID_corners_WS_cycle.extend(cycle_transform.transform_AABB(*grid_OS, True))
					for i in range(1, cycle_steps):
						cycle_transform.rotation_deg[1] += i * 360/cycle_steps
						AABB_corners_WS_cycle.extend(cycle_transform.transform_AABB(*corners_OS, True))
						GRID_corners_WS_cycle.extend(cycle_transform.transform_AABB(*grid_OS, True))
					
					del dens_hull
				if AABB_corners_WS:
					seq_cams = [cam.copy_clipped_to_world_coords(AABB_corners_WS)[0] for cam in cameras]
				else:
					seq_cams = cameras
				
				if cycle and AABB_corners_WS_cycle:
					cycle_cams = [cam.copy_clipped_to_world_coords(AABB_corners_WS_cycle)[0] for cam in cycle_cams]
				
				if render_shadow and GRID_corners_WS_cycle:
					shadow_cams = [cam.copy_clipped_to_world_coords(GRID_corners_WS_cycle)[0] for cam in shadow_cams]
					if cycle:
						shadow_cams_cycle = [cam.copy_clipped_to_world_coords(GRID_corners_WS_cycle)[0] for cam in shadow_cams_cycle]
				
				split_cams = True
			else:
				seq_cams = cameras
				split_cams = False
			
			i=0
			if args.console:
				substeps = 0
				if render_density: 
					substeps += 3 if render_cycle else 1
					#if render_shadow: substeps += 1
					if sF_cam: substeps += 1
				if render_velocity: substeps += 3 if render_cycle else 1
				#2 + (4 if cycle else 0) + (1 if sF_cam else 0)
				cycle_pbar = ProgressBar(len(sequence)*substeps, name="Render Sequence: ")
				substep = 0
				def update_pbar(frame, desc):
					nonlocal substep
					cycle_pbar.update(i*substeps + substep, desc="Frame {:03d} ({:03d}/{:03d}): {:30}".format(frame, i+1, len(sequence), desc))
					substep +=1
			
			for state in sequence:
				if render_density:
					log.debug("Render density frame %d (%d)", state.frame, i)
					if args.console: update_pbar(state.frame, "Density, main cameras") #progress_bar(i*7,len(sequence)*7, "Frame {:03d} ({:03d}/{:03d}): {:30}".format(state.frame, i+1,len(sequence), "Density, main cameras"), length=30)
					bkg_render = None
					bkg_render_alpha = None
					if setup.rendering.background.type=='CAM':
						bkg_render = state.bkgs
					elif setup.rendering.background.type=='COLOR':
						bkg_render = [tf.constant(setup.rendering.background.color, dtype=tf.float32)]*len(seq_cams)
						bkg_render_alpha = [tf.constant(list(setup.rendering.background.color) + [0.], dtype=tf.float32)]*len(seq_cams)
					dens_transform = state.get_density_transform()
					#sim_transform.set_data(state.density)
					val_imgs = renderer.render_density(dens_transform, lights, seq_cams, background=bkg_render, split_cameras=split_cams)
					renderer.write_images([tf.concat(val_imgs, axis=0)], ['seq_img_cam{}_{:04d}'], base_path=setup.paths.data, use_batch_id=True, frame_id=i, format='PNG')
					if render_shadow:
						shadow_dens = dens_transform.copy_new_data(render_helper.with_border_planes(dens_transform.data *shadow_dens_scale, planes=["Z-","Y-"], density=100., width=3, offset=2))
						shadow_imgs = renderer.render_density(shadow_dens, shadow_lights, shadow_cams, background=bkg_render, split_cameras=split_cams)
						renderer.write_images([tf.concat(shadow_imgs, axis=0)], ['seq_sdw_cam{}_{:04d}'], base_path=setup.paths.data, use_batch_id=True, frame_id=i, format='PNG')
					if cycle or sF_cam:
						tmp_transform = state.get_density_transform()
						tmp_transform.set_data(tf.zeros_like(state.density.d))
						# synth:
						#dens_grads = ("viridis", 0., 1.1) #np.percentile(state.density.d.numpy(), 95))
						#dens_grads = ("viridis", 0., 5.5) #np.percentile(state.density.d.numpy(), 95))
						# sF real:
						dens_grads = ("viridis", 0., 2.5) #np.percentile(state.density.d.numpy(), 95))
					if cycle:
						if args.console: update_pbar(state.frame, "Density, cycle") #progress_bar(i*7+1,len(sequence)*7, "Frame {:03d} ({:03d}/{:03d}): {:30}".format(state.frame, i+1,len(sequence), "Density, cycle"), length=30)
						render_cycle(dens_transform, cycle_cams, lights, renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='img', bkg=bkg_render_alpha, img_stats=False, rotate_cameras=True, cut_alpha=False, format='PNG')
						if render_shadow:
							if True:
								del shadow_dens
								shadow_dens = dens_transform.copy_new_data(dens_transform.data *shadow_dens_scale)
							render_cycle(shadow_dens, shadow_cams_cycle, shadow_lights, renderer, \
								state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='img_sdw', bkg=bkg_render, img_stats=False, rotate_cameras=True, format="PNG")
							del shadow_dens
					#	dens_mean = tf.reduce_mean(state.density.d)
						#dens_max = tf.reduce_max(state.density.d)
					#	dens_grads = [(0.0, tf.constant([0,0,0], dtype=tf.float32)),
					#		(dens_mean*2, tf.constant([0.0,0.0,1.0], dtype=tf.float32)),
					#		(dens_max, tf.constant([1.0,1.0,0.0], dtype=tf.float32)),
					#		(10.0, tf.constant([1.0,0.0,0.0], dtype=tf.float32))]
						#dens_norm = state.density.d/dens_max
						#dens_gamma = tf.pow(dens_norm, 1/2.2)
						#dens_grads_max = ("inferno", 0., 1.)
						#dens = tf_element_transfer_func(state.density.d, dens_grads)
					#	render_cycle(tmp_transform, [main_camera], [tf.concat([state.density.d]*3, axis=-1)], max_renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='dens_max', img_transfer=dens_grads_max, img_stats=False, format="PNG")
						#dens_ecm = (tf_cmap_nearest(tf.pow(state.density.d/dens_max, 1/2.2), *dens_grads_max)-0.016)
						#render_cycle(dens_transform, [main_camera], [dens_ecm*17], renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='dens_ecma', img_stats=False, format="PNG")
						#render_cycle(tmp_transform, [main_camera], [dens_ecm*17], vel_renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='dens_ecm', img_stats=False, format="PNG")
						render_cycle(tmp_transform, [main_camera], [tf.concat([state.density.d]*3, axis=-1)], vel_renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='dens', img_transfer=dens_grads, img_stats=False, format="PNG")
						if args.console: update_pbar(state.frame, "Density inflow, cycle") #progress_bar(i*7+2,len(sequence)*7, "Frame {:03d} ({:03d}/{:03d}): {:30}".format(state.frame, i+1,len(sequence), "Density inflow, cycle"), length=30)
					#	render_cycle(tmp_transform, [main_camera], [tf.concat([state.density.inflow]*3, axis=-1)], max_renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='inflow', img_transfer=dens_grads, img_stats=False, format="PNG")
					if sF_cam and getattr(state, "target_cameras", None) is not None:
						if args.console: update_pbar(state.frame, "Density, target cameras") #progress_bar(i*7+3,len(sequence)*7, "Frame {:03d} ({:03d}/{:03d}): {:30}".format(state.frame, i+1,len(sequence), "Density, sF cameras"), length=30)
						imgs = tf.concat(renderer.render_density(dens_transform, lights, state.target_cameras, cut_alpha=False), axis=0)#, background=bkg
						imgs, imgs_d = tf.split(imgs, [3,1], axis=-1)
						renderer.write_images([imgs, tf.exp(-imgs_d)], ['train_cam{}', 'train_trans_cam{}'], base_path=state.data_path, use_batch_id=True, format='PNG')
						render_cycle(tmp_transform, state.target_cameras, [tf.concat([state.density.d]*3, axis=-1)], vel_renderer, state.data_path, steps=1, steps_per_cycle=1, name_pre="train_dens", \
							img_transfer=dens_grads, img_stats=False, format="PNG")
						if getattr(state, 'targets_raw', None) is not None and (len(state.target_cameras)==state.targets_raw.get_shape().as_list()[0]):
							#imgs = tf.concat(renderer.render_density(dens_transform, lights, scalarFlow_cameras, cut_alpha=True), axis=0)#, background=[state.bkgs]
							#imgs, d = tf.split(imgs, [3,1], axis=-1)
							imgs = tf.stack(imgs) #tf.stack([imgs[_] for _ in setup.data.density.target_cam_ids])
							imgs_d = tf.stack(imgs_d)
							imgs_bkg = imgs + state.bkgs*tf.exp(-imgs_d)#t = tf.exp(-d)
							renderer.write_images([imgs_bkg], ['train_bkg_cam{}'], base_path=state.data_path, use_batch_id=True, format='PNG')
							renderer.write_images([tf.abs(imgs_bkg - state.targets_raw)], ['train_err_cam{}'], base_path=state.data_path, use_batch_id=True, format='EXR')
							renderer.write_images([tf.abs(imgs - state.targets_raw)], ['train_err_bkg_cam{}'], base_path=state.data_path, use_batch_id=True, format='PNG')
				
				if render_velocity:
					vel_transform = state.get_velocity_transform()
					vel_scale = vel_transform.cell_size_world().value #world_scale(state.velocity.centered_shape, width=1.)
					log.debug("Render velocity frame %d (%d) with cell size %s", state.frame, i, vel_scale)
					if args.console: update_pbar(state.frame, "Velocity, main cameras") #progress_bar(i*7+4,len(sequence)*7, "Frame {:03d} ({:03d}/{:03d}): {:30}".format(state.frame, i+1,len(sequence), "Velocity, main cameras"), length=30)
					#sim_transform.set_data(vel_pad)
					vel_centered = state.velocity.centered()*vel_scale/float(setup.data.step)*setup.rendering.velocity_scale
					#val_imgs = vel_renderer.render_density(vel_transform, [tf.maximum(vel_centered, 0)], cameras)
					val_imgs = vel_renderer.render_density(vel_transform, [tf.abs(vel_centered)], cameras, split_cameras=split_cams)
					vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['seq_velA_cam{}_{:04d}'], base_path=setup.paths.data, use_batch_id=True, frame_id=i, format='PNG')
					#val_imgs = vel_renderer.render_density(vel_transform, [tf.maximum(-vel_centered, 0)], cameras)
					#vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['seq_velN_cam{}_{:04d}'], base_path=setup.paths.data, use_batch_id=True, frame_id=i, format='EXR')
					if cycle:
						if args.console: update_pbar(state.frame, "Velocity, cycle") #progress_bar(i*7+5,len(sequence)*7, "Frame {:03d} ({:03d}/{:03d}): {:30}".format(state.frame, i+1,len(sequence), "Velocity, cycle"), length=30)
						#render_cycle(vel_transform, [main_camera], [tf.maximum(vel_centered, 0)], vel_renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='velP', img_stats=False)
						render_cycle(vel_transform, [main_camera], [tf.abs(vel_centered)], vel_renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='velA', img_stats=False, format='PNG')
						#render_cycle(vel_transform, [main_camera], [tf.maximum(-vel_centered, 0)], vel_renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='velN')
						
					#	vel_mag = state.velocity.magnitude()
					#	max_mag = tf.reduce_max(vel_mag)
					#	vel_mag_grads = [(0.0, tf.constant([0,0,0], dtype=tf.float32)),
					#		(1.0, tf.constant([1.0,1.0,1.0], dtype=tf.float32)),
					#		(1.0, tf.constant([0.5,0.5,1.0], dtype=tf.float32)),
					#		(max_mag, tf.constant([1.0,0.0,0.0], dtype=tf.float32))]
						#vel_mag = tf_element_transfer_func(vel_mag, vel_mag_grads)
						if args.console: update_pbar(state.frame, "Velocity magnitude, cycle") #progress_bar(i*7+6,len(sequence)*7, "Frame {:03d} ({:03d}/{:03d}): {:30}".format(state.frame, i+1,len(sequence), "Velocity magnitude, cycle"), length=30)
					#	render_cycle(vel_transform, [main_camera], [tf.concat([vel_mag]*3, axis=-1)], max_renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='velM', img_transfer=vel_mag_grads, img_stats=False, format="PNG")
					#	del vel_mag
						
						vel_div = state.velocity.divergence()
						vel_div = tf.concat((tf.maximum(vel_div, 0), tf.abs(vel_div), tf.maximum(-vel_div, 0)), axis=-1)
						render_cycle(vel_transform, [main_camera], [vel_div], vel_renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='velDiv', img_stats=False, img_normalize=True, format="PNG")
						del vel_div
					if False:
						vel_shape = GridShape.from_tensor(vel_centered)
						vel_slices_z = render_helper.image_volume_slices(vel_centered, axis=-4, abs_value=False)
						vel_renderer.write_images([tf.stack(vel_slices_z, axis=0)], ['vel_slice_{:04d}'], base_path=os.path.join(state.data_path, "vel_xy"), use_batch_id=True, format='EXR')
						vel_slices_x = render_helper.image_volume_slices(vel_centered, axis=-2, abs_value=False)
						vel_slices_x = (tf.transpose(_, (1,0,2)) for _ in vel_slices_x)
						vel_slices_x = list(tf.concat((tf.abs(_), tf.maximum(_,0), tf.maximum(-_,0)), axis=-2) for _ in vel_slices_x)
						vel_renderer.write_images([tf.stack(vel_slices_x, axis=0)], ['vel_slice_{:04d}'], base_path=os.path.join(state.data_path, "vel_zy"), use_batch_id=True, format='EXR')
				
				i +=1
				substep = 0
			if args.console:
				cycle_pbar.update(cycle_pbar._max_steps, "Done")
				cycle_pbar.close()
				#progress_bar(i*7,len(sequence)*7, "Frame {:03d} ({:03d}/{:03d}): {:30}".format(state.frame, i,len(sequence), "Done"), length=30)
	
	stop_training = False
	def handle_train_interrupt(sig, frame):
		global stop_training
		if stop_training:
			log.info('Training still stopping...')
		else:
			log.warning('Training interrupted, stopping...')
		stop_training = True
	
	resource_device = setup.training.resource_device #'/cpu:0'
	compute_device = '/gpu:0'
	log.debug("resource device (volumes and images): %s", resource_device)
	log.debug("compute device: %s", compute_device)
	
	def wrap_resize_images(images, size):
		return tf_image_resize_mip(images, size, mip_bias=0.5, method=tf.image.ResizeMethod.BILINEAR)
		#return tf.image.resize_bilinear(images, size)
	
	# rebuild the interpolation of the active cameras here to match the target interpolation
	target_cameras = build_scalarFlow_cameras(setup, [scalarFlow_cam_ids[_] for _ in setup.data.density.target_cam_ids], interpolation_weights=view_interpolation_weights) #[scalarFlow_cameras[_] for _ in setup.data.density.target_cam_ids]
	#log.debug("%d target cameras, %d cam calib ids.", len(target_cameras), len(scalarFlow_cam_ids))
	target_cameras_base = [target_cameras[_] for _ in range(0, len(target_cameras), setup.training.density.view_interpolation.steps +1)]
	
	frames = list(range(setup.data.start, setup.data.stop, setup.data.step))
	if args.fit:
		log.info("Reconstructing sequence for frames %s", frames)
		setup.paths.data = setup.paths.path
		os.makedirs(setup.paths.data, exist_ok=True)
		try:
			faultlog = open(os.path.join(setup.paths.log, 'fault.log'), 'a')
			faulthandler.enable(file=faultlog)
			summary = tf.contrib.summary
			summary_writer = summary.create_file_writer(setup.paths.log)
			
			
			if True:
				plot_schedule(setup.training.density.learning_rate, setup.training.iterations, os.path.join(setup.paths.config, 'dens_lr.png'), 'Density LR')
				plot_schedule(setup.training.velocity.learning_rate, setup.training.iterations, os.path.join(setup.paths.config, 'vel_lr.png'), 'Velocity LR')
				
				plot_schedule(setup.training.density.warp_loss, setup.training.iterations, os.path.join(setup.paths.config, 'dens_warp_loss.png'), 'Density Warp Loss Scale')
				plot_schedule(setup.training.velocity.density_warp_loss, setup.training.iterations, os.path.join(setup.paths.config, 'vel_dens_warp_loss.png'), 'Velocity Density Warp Loss Scale')
				plot_schedule(setup.training.velocity.velocity_warp_loss, setup.training.iterations, os.path.join(setup.paths.config, 'vel_vel_warp_loss.png'), 'Velocity Velocity Warp Loss Scale')
				plot_schedule(setup.training.velocity.divergence_loss, setup.training.iterations, os.path.join(setup.paths.config, 'vel_div_loss.png'), 'Velocity Divergence Loss Scale')
				
				labels = ['Dens warp', 'Vel dens-warp', 'Vel vel-warp']#, 'Vel div']
				schedules = [setup.training.density.warp_loss, setup.training.velocity.density_warp_loss, setup.training.velocity.velocity_warp_loss]#, setup.training.velocity.divergence_loss]
				plot_schedules(schedules, setup.training.iterations, os.path.join(setup.paths.config, 'warp_loss_cmp.png'), labels=labels, title='Warp Loss Comparison')
				
				if setup.training.discriminator.active:
					plot_schedule(setup.training.discriminator.learning_rate, setup.training.iterations, os.path.join(setup.paths.config, 'disc_lr.png'), 'Discriminator LR')
					#plot_schedule(setup.training.discriminator.loss_scale, setup.training.iterations, os.path.join(setup.paths.config, 'disc_loss_scale.png'), 'Discriminator Loss Scale')
			
			frustum_half = 0.75
			dist = 4.
			log.debug("Setup validation")
			val_cameras = [
				Camera(GridTransform(cam_resolution, translation=[0,0,0.8], parent=Transform(rotation_deg=[-30,0,0], parent=Transform(translation=scalarFlow_cam_focus, rotation_deg=[0,-85,0]))), nearFar=[0.3,1.3], fov=40, aspect=aspect, static=sim_transform if setup.rendering.allow_static_cameras else None),
			]
			
			
			if len(frames)<1:
				log.error("Not enough frames for sequence reconstruction: %s", frames)
				sys.exit(1)
			if len(frames)==1:
				log.warning("Single frame reconstruction can not provide meaningfull velocity.")
			
					
			log.info("--- Pre-setup ---")
			
			if setup.data.hull=='ALL':
				#setup.data.hull=[]
				hull_cameras = scalarFlow_cameras
			elif setup.data.hull=='TARGETS' or setup.data.hull=='TARGETS_ROT' or setup.data.hull=='ROT':
				#setup.data.hull=setup.data.density.target_cam_ids
				hull_cameras = target_cameras
			elif setup.data.hull=='ORIGINAL_ALL':
				hull_cameras = scalarFlow_cameras_base
			elif setup.data.hull=='ORIGINAL_TARGETS' or setup.data.hull=='ORIGINAL_TARGETS_ROT':
				#setup.data.hull=[_*(setup.training.density.view_interpolation.steps +1) for _ in setup.data.density.target_cam_ids] #only from not-interpolated views
				hull_cameras = target_cameras_base
			elif isinstance(setup.data.hull, (list, tuple)):
				hull_cameras = [scalarFlow_cameras[_] for _ in setup.data.hull]
					
			# TomoFluid: interpolated targets have less weight, based on the angle to a 'real' target
			def get_target_weights(cameras, real_cameras, focus, mode="COS", **kwargs):
				focus = Float3(focus)
				# get angles from cameras to next base/real camera
				angles = []
				for camera in cameras:
					dir_from_focus = (Float3(camera.transform.transform(Float4(0,0,0,1)))-focus).normalized
					angle = np.pi
					for cam in real_cameras:
						dff = (Float3(cam.transform.transform(Float4(0,0,0,1)))-focus).normalized
						angle = np.minimum(np.arccos(np.dot(dir_from_focus, dff)), angle)
					angles.append(angle)
				
				if mode=="COS":
					weights = [np.cos(_*2)*0.5+0.5 for _ in angles]
			#	elif mode=="POW":
			#		min_weight = kwargs.get("min_weight", 1e-3)
			#		base = min_weight ** (-2./np.pi)
			#		weights = [base**(-_) for _ in angles]
				elif mode=="EXP":
					min_weight = kwargs.get("min_weight", 1e-4)
					scale = np.log(min_weight) * (-2./np.pi)
					weights = [np.exp(-_*scale) for _ in angles]
				else:
					raise ValueError("Unknown target weight mode %s"%mode)
				return weights
			view_interpolation_target_weights = get_target_weights(target_cameras, target_cameras_base, cam_setup.focus, mode="EXP") if setup.training.density.view_interpolation.steps>0 else None
			
			def frame_loadTargets(frame, sim_transform):
				'''
				load or render targets
				preprocessing: background subtraction, masking, view interpolation
				choose targets used for visual hull
				'''
				# setup targets and hulls at base_res
				if setup.data.density.target_type!='RAW': #SYNTHETIC or PRERPROC
					if setup.data.density.target_type=='SYNTHETIC' and setup.rendering.background.type=='COLOR':
						bkgs = np.ones([len(scalarFlow_cameras_base), train_cam_resolution[1], train_cam_resolution[2], 1], dtype=np.float32) * np.asarray(setup.rendering.background.color, dtype=np.float32)
					else:
						# load backgrounds from raw targets
						_, bkgs = load_targets(setup.data.density.target, simulations=[setup.data.simulation], frames=[frame], bkg_subtract=False, flip_y=setup.data.density.target_flip_y)
					#	targets_raw = [targets_raw[_] for _ in scalarFlow_cam_ids]
						bkgs = [bkgs[_] for _ in scalarFlow_cam_ids]
					#	targets_raw = tf.nn.avg_pool(targets_raw, (1,train_res_down, train_res_down, 1), (1,train_res_down, train_res_down, 1), 'VALID')
						bkgs = tf.nn.avg_pool(bkgs, (1,train_res_down, train_res_down, 1), (1,train_res_down, train_res_down, 1), 'VALID')
				#		renderer.write_images([targets_raw], ['target_raw_cam{}'], base_path=image_path, use_batch_id=True)
					
					offset=setup.data.scalarFlow_frame_offset #-11
					if setup.data.density.target_type=='SYNTHETIC':
						log.info('Rendering scalar Flow reconstruction (sim %d, frame %d) as optimization target.', setup.data.simulation, frame)
						#load sF density, scale to current grid size, normalize to current mean (i.e. same total density), report MSE
						tmp_sF_dens = DensityGrid.from_scalarFlow_file(setup.data.density.scalarFlow_reconstruction.format(sim=setup.data.simulation, frame=frame+offset), as_var=False, scale_renderer=scale_renderer)
						tmp_sF_dens.scale(setup.data.density.synthetic_target_density_scale)
					#	tmp_transform = copy.deepcopy(sim_transform)
					#	tmp_transform.set_data(tmp_sF_dens.d)
						tmp_transform = sim_transform.copy_new_data(tmp_sF_dens.d)
						targets = tf.concat(synth_target_renderer.render_density(tmp_transform, synth_target_lights, scalarFlow_cameras_base, cut_alpha=False, monochrome=setup.rendering.monochrome), axis=0) #, background=bkg
						del tmp_sF_dens
						del tmp_transform
						targets, target_densities = tf.split(targets, [color_channel,1], axis=-1)
						t = tf.exp(-target_densities)
						
						targets_monochrome = tf.reduce_mean(targets*synth_target_renderer.luma, axis=-1, keepdims=True)
						targets_raw = targets + t*bkgs
						del t
					elif setup.data.density.target_type=='PREPROC':
						targets, _ = load_targets(setup.data.density.target_preproc, simulations=[setup.data.simulation], frames=[frame+offset], bkg_subtract=False, flip_y=setup.data.density.target_flip_y)
						targets = [targets[_] for _ in scalarFlow_cam_ids]
						targets = tf.nn.avg_pool(targets, (1,train_res_down, train_res_down, 1), (1,train_res_down, train_res_down, 1), 'VALID')
						targets_monochrome = targets
						targets_raw = targets + bkgs
					else:
						raise ValueError("Unknown target_type '%s'"%setup.data.density.target_type)
					
					targets_for_hull = targets_monochrome
				else:
					targets_raw, bkgs = load_targets(setup.data.density.target, simulations=[setup.data.simulation], frames=[frame], bkg_subtract=False, flip_y=setup.data.density.target_flip_y)
					targets_raw = [targets_raw[_] for _ in scalarFlow_cam_ids]
					bkgs = [bkgs[_] for _ in scalarFlow_cam_ids]
					targets_raw = tf.nn.avg_pool(targets_raw, (1,train_res_down, train_res_down, 1), (1,train_res_down, train_res_down, 1), 'VALID')
					bkgs = tf.nn.avg_pool(bkgs, (1,train_res_down, train_res_down, 1), (1,train_res_down, train_res_down, 1), 'VALID')
					targets_bsub = tf.maximum(targets_raw-bkgs, 0)
					targets_for_hull = targets_bsub
					#use mask to remove noise from target/background mismatch
					_, tight_image_hulls = renderer.visual_hull(sim_transform, targets_bsub, scalarFlow_cameras_base, 0.0, \
						0.0, setup.data.density.hull_threshold, 0.0)
					targets_for_hull = targets_bsub
					targets_raw = targets_raw*tight_image_hulls + bkgs*(1-tight_image_hulls)
					targets = targets_bsub*tight_image_hulls
					targets = targets*setup.data.density.target_scale
				
				aux = munch.Munch()
				
				#target selection and interpolation
				targets_raw = [targets_raw[_] for _ in setup.data.density.target_cam_ids]
				targets = [targets[_] for _ in setup.data.density.target_cam_ids]
				bkgs = [bkgs[_] for _ in setup.data.density.target_cam_ids]
				
				if view_interpolation_weights:
					targets_raw = interpolate_images(targets_raw, view_interpolation_weights)
					targets = interpolate_images(targets, view_interpolation_weights)
					bkgs = interpolate_images(bkgs, view_interpolation_weights)
				
				
				if not len(target_cameras) == len(targets):
					raise RuntimeError("Number of targets does not match number of cameras.")
				targets_raw = tf.stack(targets_raw)
				targets = tf.stack(targets)
				bkgs = tf.stack(bkgs)
				
				# hull setup
				if setup.data.hull=='ALL':
					interpolate_hulls = True
					targets_for_hull = targets_for_hull
					#hull_cameras = scalarFlow_cameras
				elif setup.data.hull=='TARGETS' or setup.data.hull=='TARGETS_ROT' or setup.data.hull=='ROT':
					interpolate_hulls = True
					targets_for_hull = [targets_for_hull[_] for _ in setup.data.density.target_cam_ids]
					#hull_cameras = target_cameras
				elif setup.data.hull=='ORIGINAL_ALL':
					interpolate_hulls = False
					targets_for_hull = targets_for_hull
					#hull_cameras = scalarFlow_cameras_base
				elif setup.data.hull=='ORIGINAL_TARGETS' or setup.data.hull=='ORIGINAL_TARGETS_ROT':
					interpolate_hulls = False
					targets_for_hull = [targets_for_hull[_] for _ in setup.data.density.target_cam_ids]
					#hull_cameras = [target_cameras[_*(setup.training.density.view_interpolation.steps +1)] for _ in setup.data.density.target_cam_ids]
				elif isinstance(setup.data.hull, (list, tuple)):
					interpolate_hulls = True
					targets_for_hull = [targets_for_hull[_] for _ in setup.data.hull]
					#hull_cameras = [scalarFlow_cameras[_] for _ in setup.data.hull]
				
				if view_interpolation_weights and interpolate_hulls:
					targets_for_hull = interpolate_images(targets_for_hull, view_interpolation_weights)
				
				if not len(hull_cameras) == len(targets_for_hull):
					raise RuntimeError("Number of hull targets does not match number of hull cameras.")
				targets_for_hull = tf.stack(targets_for_hull)
				
				with tf.device(resource_device):
					aux.targets_raw = tf.identity(targets_raw)
					aux.targets = tf.identity(targets)
					aux.bkgs = tf.identity(bkgs)
					aux.targets_for_hull = tf.identity(targets_for_hull)
				
				return aux
			
			aux_sequence = {}
		#	if setup.data.clip_grid:
			log.info("Load targets")
			for frame in frames:
				aux_sequence[frame] = frame_loadTargets(frame, sim_transform)
			
			
			### setup hull construction ###
			log.info("setup hull construction")
			
			#num_hull_cam_rot = 0
			if setup.data.hull.endswith('ROT'):
				rotations = [45,90,270,315]
				log.info("Construct hull from %s degree rotated copies with mirroring of %d targets.", rotations, len(target_cameras))
				
				def get_target_rotation_pivot(aux_sequence, frames, sim_transform):
					n_views = len(hull_cameras)
					# combine targets of all frames per view
					combined_targets = [tf.reduce_sum([aux_sequence[f].targets_for_hull[v] for f in frames], axis=[0,-1]) for v in range(n_views)] # NVHWC -> VHW
					
					def get_CoM(img):
						# img: HW
						height, width = shape_list(img)
						
						sample_mean_y = tf.reduce_mean(img, axis=-2) #W
						coords_x = tf.range(0, width, 1,dtype=tf.float32) #W
						center_x = tf.reduce_sum(coords_x*sample_mean_y, axis=-1)/tf.reduce_sum(sample_mean_y, axis=-1) #1
						
						sample_mean_x = tf.reduce_mean(img, axis=-1) #H
						coords_y = tf.range(0, height, 1,dtype=tf.float32) #H
						center_y = tf.reduce_sum(coords_y*sample_mean_x, axis=-1)/tf.reduce_sum(sample_mean_x, axis=-1) #1
						return Float2(center_x.numpy().tolist(), center_y.numpy().tolist())
					
					def cam_coords_to_ray(pix, cam):
						pos_world, dir_world = cam.screenPIX_to_worldRay(pix)
						#log.debug("re-projected ray: %s, %s", cam.project_world_to_screenPIX(pos_world), cam.project_world_to_screenPIX(pos_world + dir_world))
						return pos_world, dir_world
					
					def ray_plane_intersection(r_pos, r_dir, p_pos, p_n):
						# https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
						d = np.dot(p_pos - r_pos, p_n) / np.dot(r_dir, p_n)
						pos = (r_pos + r_dir * d)
						#log.debug("ray-plane %s %s %s %s: %s, %f", r_pos, r_dir, p_pos, p_n, pos, d)
						return pos
						
					
					pix_pivots = [get_CoM(_) for _ in combined_targets]
					#log.debug("Screen CoM coords: %s", pix_pivots)
					
					if n_views==1:
						# intersect CoM ray with central depth
						cam = hull_cameras[0]
						pivot_ray = cam_coords_to_ray(pix_pivots[0], cam)
						volume_center = Float3(sim_transform.translation)
						depth_plane_normal = Float3(np.cross(Float3(0,1,0), Float4(cam.right_global).xyz)).normalized
						#ray_plane_intersection(pivot_ray[0], pivot_ray[1], volume_center, depth_plane_normal)
						#depth_plane_normal = Float4(cam.forward_global).xyz.normalized
						#log.debug("depth_plane_normal: %s, cam fwd: %s, right: %s", depth_plane_normal, cam.forward_global, cam.right_global)
						pivot = ray_plane_intersection(pivot_ray[0], pivot_ray[1], volume_center, depth_plane_normal)
					else:
						# find point closes to all CoM rays
						raise NotImplementedError("Pivot calculation for rotational hull with more than 1 base view has not been implemented.")
					
					return pivot.value
				
				rotation_pivot = get_target_rotation_pivot(aux_sequence, frames, sim_transform)
				log.info("pivot for ROT hull at %s. volume center: %s, SF-focus: %s", rotation_pivot, sim_center, scalarFlow_cam_focus)
				#rotation_pivot = scalarFlow_cam_focus
				
				def build_visual_hull(transform, targets, cameras, image_blur=0.0, grid_blur=0.0, threshold=0.5, soft_blur=0.0):
					num_views = len(cameras)
					tmp_hull_cameras = list(cameras)
					tmp_hull_targets = list(targets)
					for r in rotations:
						for cam, target in zip(cameras, targets):
							# copy camera, rotated around y-axis of common focus
							cam = copy.deepcopy(cam)
							p = Transform(translation=-rotation_pivot, parent=Transform(translation=rotation_pivot, rotation_deg=[0,r,0]))
							cam.transform.set_parent(p)
							# y-mirror target at projected focus
							target_width = shape_list(target)[-2]
							focus_pix = cam.project_world_to_screenPIX(rotation_pivot)
							if focus_pix.x<0 or focus_pix.x>target_width:
								continue
							#log.debug("Focus mirror coords: %s", focus_pix)
							shift_x = int(2 * (focus_pix.x - target_width*0.5))
							#print("Rotation {}: cam_focus {}, target_shape {} (width {}), focus_pix {}, shift {}".format(r, rotation_pivot, shape_list(target), target_width, focus_pix, shift_x))
							target_mirrored = tf.reverse(target, axis=[-2])
							target_mirrored = tf_shift(target_mirrored, shift=shift_x, axis=-2)
							tmp_hull_cameras.append(cam)
							tmp_hull_targets.append(tf.maximum(target, target_mirrored))
					
					
					hull, image_hulls = renderer.visual_hull(transform, tmp_hull_targets, tmp_hull_cameras, \
						image_blur=image_blur, grid_blur=grid_blur, threshold=threshold, soft_blur=soft_blur)
					return hull, image_hulls #[:num_views:len(rotations)]
			else:
		#		hull_cameras = [scalarFlow_cameras[_] for _ in setup.data.hull] if setup.data.hull else scalarFlow_cameras
				build_visual_hull = renderer.visual_hull
			
			def frame_hullSetupPre(aux, frame, sim_transform):
				
				#log.info("hull targets shape: %s, cameras: %d", aux.targets_for_hull.get_shape().as_list(), len(hull_cameras))
				hull, image_hulls = build_visual_hull(sim_transform, aux.targets_for_hull, hull_cameras, setup.data.density.hull_image_blur_std, \
					setup.data.density.hull_volume_blur_std, setup.data.density.hull_threshold, setup.data.density.hull_smooth_blur_std)
				_, tight_image_hulls = build_visual_hull(sim_transform, aux.targets_for_hull, hull_cameras, 0.0, \
					0.0, setup.data.density.hull_threshold, 0.0)
				
				with tf.device(resource_device):
					aux.image_hulls = tf.identity(image_hulls)
					aux.image_hulls_tight = tf.identity(tight_image_hulls)
				
				#log.info("hull shape: %s", hull.get_shape().as_list())
				# setup bounding-box
				hull_bounds = hull_AABB_OS(tf.squeeze(hull, (0,-1)))
				#log.info("hull bounds of frame %d: %s", frame, hull_bounds)
				# if there is no density min bounds will be 'inf' and max bound '-inf', TODO handle
				BB_start = Int3(hull_bounds[0])
				BB_end = Int3(hull_bounds[1])
				BB_size = BB_end - BB_start
				# report grid occupancy
				volume_size = density_size.prod
				#	visual hull
				visHull_size = tf.reduce_sum(hull).numpy()
				#	bounding box
				#BB_size = BB_size.x*BB_size.y*BB_size.z
				log.info("F%04d Visual hull: %.03f%% (%d cells); AABB: %.03f%% (%d cells), %s at %s",frame, 100*visHull_size/volume_size, int(visHull_size), 100*(BB_size.x*BB_size.y*BB_size.z)/volume_size, \
					(BB_size.x*BB_size.y*BB_size.z), BB_size._value[::-1], BB_start._value[::-1])
				aux.hull_bound = [BB_start, BB_end]
				
				return aux
			
			
			
			
			log.info("construct visual hulls")
			
			max_hull_start = density_size.copy()
			max_hull_end = Int3(0)
			for frame in frames:
				frame_hullSetupPre(aux_sequence[frame], frame, sim_transform)
				max_hull_start = Int3(np.minimum(aux_sequence[frame].hull_bound[0], max_hull_start))
				max_hull_end = Int3(np.maximum(aux_sequence[frame].hull_bound[1], max_hull_end))
			
			
			
			if setup.data.clip_grid:
				min_mult = 1 #2**setup.rendering.mip.level
				max_hull_pad = Int3(setup.data.clip_grid_pad)
				max_hull_start = max_hull_start-max_hull_pad #Int3(np.maximum(max_hull_start-max_hull_pad, 0))
				max_hull_end = max_hull_end+max_hull_pad #Int3(np.minimum(max_hull_end+max_hull_pad, density_size))
				max_hull_size = max_hull_end - max_hull_start
				new_hull_size = ((max_hull_size + (min_mult-1))//min_mult)*min_mult
				diff_hull_size = new_hull_size - max_hull_size
				max_hull_start -= diff_hull_size//2
				max_hull_end = max_hull_start + new_hull_size
				max_hull_size = new_hull_size
				log.info("Clip grid to maximum hull BB: %.03f%% (%d cells, pad %s), %s at %s", 100*(max_hull_size.prod)/density_size.prod, \
					(max_hull_size.prod), max_hull_pad.as_shape, max_hull_size.as_shape, max_hull_start.as_shape)
				
				
				max_hull_scale = (Float4(sim_transform.transform(Float4(max_hull_end, 1))) - Float4(sim_transform.transform(Float4(max_hull_start, 1)))).xyz
				max_hull_center_WS = Float4(sim_transform.transform(Float4(Float3(max_hull_start) + Float3(max_hull_size)/2.0, 1))).xyz
				
				if setup.data.crop_grid:
					log.info("Crop grid size to preserve spacial resolution")
					density_size = max_hull_size
				log.info("Original grid transform: %s", sim_transform)
				sim_transform = GridTransform(density_size.as_shape, translation=max_hull_center_WS.value, scale=max_hull_scale.value, center=True, normalize='ALL')
				log.info("New grid transform: %s", sim_transform)
			
			#prep for loading cropped grid sequence: shape and transform
			if run_index.can_get_run_id(setup.data.density.initial_value):
				try:
					load_entry = run_index.get_run_entry(setup.data.density.initial_value)
				#	t_dict = load_entry.scalars["sim_transform"]
					t = from_dict(load_entry.scalars["sim_transform"])
					# Manual Override
					#t.grid_size= [74, 134, 63]#[87, 140, 67]
				except:
					log.warning("Failed to load grid transform, using default.", exc_info=True)
				else:
					log.info("Loaded transform '%s' from run %s", t, load_entry.runid)
					density_size = t.grid_shape.spatial_vector
					sim_transform = t
				
			curr_cam_res = current_grow_shape(train_cam_resolution, 0, setup.training.density.grow.factor, setup.training.density.grow.intervals)
			base_shape = density_size.as_shape #copy.copy(density_size)
		#	print(density_size, base_shape)
			main_opt_start_dens_shape = current_grow_shape(base_shape, 0, setup.training.density.grow.factor, setup.training.density.grow.intervals)
			main_opt_start_vel_shape = current_grow_shape(base_shape, 0, setup.training.velocity.grow.factor, setup.training.velocity.grow.intervals)
			pre_opt_start_dens_shape = copy.deepcopy(main_opt_start_dens_shape)
			pre_opt_first_start_vel_shape = current_grow_shape(main_opt_start_vel_shape, 0, setup.training.velocity.pre_opt.first.grow.factor, setup.training.velocity.pre_opt.first.grow.intervals)
			pre_opt_start_vel_shape = current_grow_shape(main_opt_start_vel_shape, 0, setup.training.velocity.pre_opt.grow.factor, setup.training.velocity.pre_opt.grow.intervals)
			curr_dens_shape = pre_opt_start_dens_shape if setup.training.density.pre_optimization else main_opt_start_dens_shape
			curr_vel_shape = pre_opt_start_vel_shape if setup.training.velocity.pre_optimization else main_opt_start_vel_shape
			#vel_scale = world_scale(curr_vel_shape, width=1.)
			z = tf.zeros([1] + curr_vel_shape + [1])
			sim_transform.grid_size = curr_dens_shape
			log.info("Inital setup for sequence reconstruction:\n\tbase shape %s,\n\tinitial density shape %s,\n\tinitial render shape %s,\n\tpre-opt velocity shape %s,\n\tinitial velocity shape %s", \
				base_shape, main_opt_start_dens_shape, curr_cam_res, pre_opt_start_vel_shape, main_opt_start_vel_shape)
			
			vel_bounds = None if setup.data.velocity.boundary.upper()=='CLAMP' else \
				Zeroset(-1, shape=density_size, outer_bounds="OPEN" if setup.data.velocity.boundary.upper()=='CLAMP' else 'CLOSED', as_var=False, device=resource_device)
			
			def frame_hullSetup(aux, frame):
				#re-generate hulls for potentially cropped base resolution
				transform = sim_transform.copy_no_data()
				transform.grid_size = base_shape
				hull, _ = build_visual_hull(transform, aux.targets_for_hull, hull_cameras, setup.data.density.hull_image_blur_std, \
					setup.data.density.hull_volume_blur_std, setup.data.density.hull_threshold, setup.data.density.hull_smooth_blur_std)
				tight_hull, _ = build_visual_hull(transform, aux.targets_for_hull, hull_cameras, 0.0, \
					0.0, setup.data.density.hull_threshold, 0.0)
				
				aux.hull = hull
				aux.hull_tight = tight_hull
				
				# setup bounding-box
				hull_bounds = hull_AABB_OS(tf.squeeze(hull, (0,-1)))
				BB_start = Vector3(hull_bounds[0])
				BB_end = Vector3(hull_bounds[1])
				BB_size = BB_end - BB_start
				# report grid occupancy
				volume_size = base_shape[2]*base_shape[1]*base_shape[0]
				#	visual hull
				visHull_size = tf.reduce_sum(hull).numpy()
				#	bounding box
				#BB_size = BB_size.x*BB_size.y*BB_size.z
				log.info("F%04d Visual hull: %.03f%% (%d cells); AABB: %.03f%% (%d cells), %s at %s",frame, 100*visHull_size/volume_size, int(visHull_size), 100*(BB_size.x*BB_size.y*BB_size.z)/volume_size, \
					(BB_size.x*BB_size.y*BB_size.z), BB_size._value[::-1], BB_start._value[::-1])
			for frame in frames:
				frame_hullSetup(aux_sequence[frame], frame)
			del frame_hullSetup
			
			log.info("--- Sequence setup ---")
			def frame_velSetup(aux_sequence, frame, first_frame, vel_init=None):
				#setup velocity
				#with tf.device(resource_device):
				vel_var_name = "velocity_f{:06d}".format(frame)
				if first_frame and setup.training.velocity.pre_optimization and setup.training.velocity.pre_opt.first.grow.intervals:
					vel_var_name = "velocity_f{:06d}_g000".format(frame)
				elif setup.training.velocity.grow.intervals:
					vel_var_name = "velocity_f{:06d}_g000".format(frame)
				
				vel_shape = (pre_opt_first_start_vel_shape if first_frame else pre_opt_start_vel_shape) if setup.training.velocity.pre_optimization else main_opt_start_vel_shape
				
				if not setup.data.velocity.initial_value.upper().startswith('RAND'):
					if first_frame or not setup.training.velocity.pre_optimization:
						velocity = load_velocity(setup.data.velocity.initial_value, {'sim':setup.data.simulation,'frame':frame}, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, var_name=vel_var_name)
						rel_vel_scale = float(setup.data.step/setup.data.velocity.load_step)
						if rel_vel_scale!=1.:
							log.debug("scale loaded velocity with %f", rel_vel_scale)
							velocity.scale_magnitude(rel_vel_scale)
						if velocity.centered_shape != vel_shape:
							log.error("Shape %s of loaded velocity does not match required shape %s.", velocity.centered_shape, \
								(pre_opt_start_vel_shape if setup.training.velocity.pre_optimization else main_opt_start_vel_shape))
							sys.exit(1)
					else: #not first and pre-opt. will be overwritten, put dummy data as file might not exist
						velocity = VelocityGrid(main_opt_start_vel_shape, setup.data.velocity.init_std * setup.data.step, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, var_name='vel_dummy_f{:06d}'.format(frame))
				else:
					velocity = VelocityGrid(vel_shape, setup.data.velocity.init_std * setup.data.step, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, var_name=vel_var_name)
					if vel_init is not None:
						velocity.assign(vel_init.x, vel_init.y, vel_init.z)
				
				if setup.data.velocity.init_mask != 'NONE':
					if setup.data.velocity.init_mask == 'HULL':
						vel_mask = aux_sequence[frame].hull
					if setup.data.velocity.init_mask == 'HULL_NEXT':
						frame_next = frame+setup.data.step
						if frame_next in aux_sequence:
							vel_mask = aux_sequence[frame_next].hull
						else:
							vel_mask = aux_sequence[frame].hull
					elif setup.data.velocity.init_mask == 'HULL_TIGHT':
						vel_mask = aux_sequence[frame].hull_tight
					elif setup.data.velocity.init_mask == 'HULL_TIGHT_NEXT':
						frame_next = frame+setup.data.step
						if frame_next in aux_sequence:
							vel_mask = aux_sequence[frame_next].hull_tight
						else:
							vel_mask = aux_sequence[frame].hull_tight
					else:
						raise ValueError("Unknown velocity mask %s"%setup.data.velocity.init_mask)
					hull_x = scale_renderer.resample_grid3D_aligned(vel_mask, velocity.x_shape, align_x='STAGGER_OUTPUT')
					hull_y = scale_renderer.resample_grid3D_aligned(vel_mask, velocity.y_shape, align_y='STAGGER_OUTPUT')
					hull_z = scale_renderer.resample_grid3D_aligned(vel_mask, velocity.z_shape, align_z='STAGGER_OUTPUT')
					velocity.assign(x=velocity.x*hull_x, y=velocity.y*hull_y, z=velocity.z*hull_z)
				return velocity
			
			def frame_densSetup(aux_sequence, frame, first_frame):
				inflow_init = None
				inflow_mask = None
				inflow_offset = None
				if setup.data.density.inflow.active:
					base_inflow_mask, base_inflow_shape, base_inflow_offset = create_inflow(tf.squeeze(aux_sequence[frame].hull, (0,-1)), setup.data.density.inflow.hull_height, setup.data.density.inflow.height)
					if base_inflow_mask is not None: #setup.data.density.inflow.active:
						base_inflow_mask = tf.reshape(base_inflow_mask, [1]+base_inflow_shape+[1])
						log.info("Base Inflow: %s at %s", base_inflow_shape, base_inflow_offset)
						inflow_init = 'CONST'
						inflow_offset = current_grow_shape(base_inflow_offset, 0, setup.training.density.grow.factor, setup.training.density.grow.intervals)
						inflow_shape = current_grow_shape(base_inflow_shape, 0, setup.training.density.grow.factor, setup.training.density.grow.intervals, cast_fn=lambda x: max(round(x),1))
						inflow_mask = scale_renderer.resample_grid3D_aligned(base_inflow_mask, inflow_shape)
					else:
						log.error("Failed to build inflow.")
				
				#setup density with start scale
				dens_hull = scale_renderer.resample_grid3D_aligned(aux_sequence[frame].hull, curr_dens_shape) # if setup.training.density.use_hull else None
				#with tf.device(resource_device):
				dens_var_name = "density_f{:06d}".format(frame)
				if setup.data.density.initial_value.upper()=="CONST":
					density = DensityGrid(curr_dens_shape, setup.data.density.scale, scale_renderer=scale_renderer, hull=dens_hull, inflow=inflow_init, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
						device=resource_device, var_name=dens_var_name, restrict_to_hull=setup.training.density.use_hull)
					log.debug("Initialized density with constant value")
				elif setup.data.density.initial_value.upper()=="ESTIMATE":
					density_estimate = renderer.unproject(sim_transform, aux_sequence[frame].targets, target_cameras)#*setup.data.density.scale
					density = DensityGrid(curr_dens_shape, d=density_estimate, scale_renderer=scale_renderer, hull=dens_hull, inflow=inflow_init, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
						device=resource_device, var_name=dens_var_name, restrict_to_hull=setup.training.density.use_hull)
					log.debug("Initialized density with estimate from targets")
				elif setup.data.density.initial_value.upper()=="HULL":
					density = DensityGrid(curr_dens_shape, d=dens_hull*setup.data.density.scale, scale_renderer=scale_renderer, hull=dens_hull, inflow=inflow_init, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
						device=resource_device, var_name=dens_var_name, restrict_to_hull=setup.training.density.use_hull)
					log.debug("Initialized density with visual hull from targets")
				elif setup.data.density.initial_value.upper()=="HULL_TIGHT":
					h = scale_renderer.resample_grid3D_aligned(aux_sequence[frame].hull_tight, curr_dens_shape)
					density = DensityGrid(curr_dens_shape, d=h*setup.data.density.scale, scale_renderer=scale_renderer, hull=dens_hull, inflow=inflow_init, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
						device=resource_device, var_name=dens_var_name, restrict_to_hull=setup.training.density.use_hull)
					del h
					log.debug("Initialized density with visual hull from targets")
				else: #load
					if first_frame or not setup.training.density.pre_optimization:
						try:
							path = run_index[setup.data.density.initial_value]
							if path is None:
								path = setup.data.density.initial_value
							path = path.format(sim=setup.data.simulation, frame=frame)
							density = DensityGrid.from_file(path, scale_renderer=scale_renderer, device=resource_device, var_name=dens_var_name, restrict_to_hull=setup.training.density.use_hull) #, hull=dens_hull, inflow=inflow_mask, inflow_offset=inflow_offset
							if density.hull is None: #no hull loaded, but a hull should be used, so use the newly build one
								density.hull = dens_hull
						except:
							log.exception("Falied to load density for frame %d from '%s'", frame, path)
							sys.exit(1)
						else:
							log.debug("Initialized density for frame %d with value loaded from %s", frame, setup.data.density.initial_value)
						if density.shape != curr_dens_shape:
							log.error("Shape %s of density loaded from '%s' does not match required shape %s.", velocity.centered_shape, path, curr_dens_shape)
							sys.exit(1)
						#density.scale(setup.data.density.scale)
					else: #not first and pre-opt. will be overwritten, put dummy data as file might not exist
						density = DensityGrid(curr_dens_shape, setup.data.density.scale, scale_renderer=scale_renderer, hull=dens_hull, inflow=inflow_init, inflow_offset=inflow_offset, inflow_mask=inflow_mask, \
							device=resource_device, var_name=dens_var_name, restrict_to_hull=setup.training.density.use_hull)
				
				with tf.device(resource_device):
					density.base_hull = tf.identity(aux_sequence[frame].hull) #if setup.training.density.use_hull else None
					if inflow_mask is not None: #setup.data.density.inflow.active:
						density.base_inflow_mask = tf.identity(base_inflow_mask)
						density.base_inflow_shape = base_inflow_shape
						density.base_inflow_offset = base_inflow_offset
				return density
			
			def frame_setup(aux_sequence, frame, first_frame, prev=None, vel_init=None):
				log.info("--- Setup frame %d ---", frame)
				velocity = frame_velSetup(aux_sequence, frame, first_frame, vel_init)
				density = frame_densSetup(aux_sequence, frame, first_frame)
				
				state = State(density, velocity, frame=frame, prev=prev, transform=sim_transform.copy_no_data())
				state.data_path = os.path.join(setup.paths.data, 'frame_{:06d}'.format(frame))
				os.makedirs(state.data_path, exist_ok=True)
				#setup targets
				
				with tf.device(resource_device):
					state.base_targets_raw = tf.identity(aux_sequence[frame].targets_raw)
					state.base_targets = tf.identity(aux_sequence[frame].targets)
					state.base_bkgs = tf.identity(aux_sequence[frame].bkgs)
				targets_raw = wrap_resize_images(state.base_targets_raw, curr_cam_res[1:])
				targets = wrap_resize_images(state.base_targets, curr_cam_res[1:])
				bkgs = wrap_resize_images(state.base_bkgs, curr_cam_res[1:])
				with tf.device(resource_device):
					state.targets_raw = tf.identity(targets_raw)
					state.targets = tf.identity(targets)
					state.bkgs = tf.identity(bkgs)
					state.hull = tf.identity(aux_sequence[frame].hull_tight)
				return state
			
			def sequence_setup(aux_sequence):
				sequence = []
				prev = None
				vel_init = None
				first_frame = True
				for frame in frames:
					state = frame_setup(aux_sequence, frame, first_frame, prev, vel_init)
					if setup.rendering.target_cameras.crop_frustum:
						raise NotImplementedError("Camera frustum cropping is currently not working.")
					AABB_corners_WS = sim_transform.transform_AABB(*hull_AABB_OS(tf.squeeze(state.density.hull, (0,-1))), True) if setup.rendering.target_cameras.crop_frustum else None
					state.target_cameras = setup_target_cameras(target_cameras, curr_cam_res, AABB_corners_WS, setup.rendering.target_cameras.crop_frustum_pad, jitter=setup.training.density.camera_jitter)
					
					
					render_hull = aux_sequence[frame].hull*4
					if hasattr(state.density, "base_inflow_mask"): #setup.data.density.inflow.active:
						render_hull = render_hull + tf.pad(state.density.base_inflow_mask * 0.1, \
							[[0,0]]+[[state.density.base_inflow_offset[_],base_shape[_]-state.density.base_inflow_offset[_]-state.density.base_inflow_shape[_]] for _ in range(3)]+[[0,0]])
					hull_transform = sim_transform.copy_new_data(render_hull)
					render_cycle(hull_transform, [main_camera], lights, renderer, state.data_path, name_pre='hull_inflow', format='PNG')
					
					log.debug('Write target images')
					renderer.write_images([state.base_targets_raw], ['target_raw_base_cam{}'], base_path=state.data_path, use_batch_id=True)
					renderer.write_images([state.base_targets], ['target_base_cam{}'], base_path=state.data_path, use_batch_id=True)
					renderer.write_images([state.base_bkgs], ['bkg_base_cam{}'], base_path=state.data_path, use_batch_id=True)
					renderer.write_images([state.targets_raw], ['target_raw_{}-{}_cam{}'.format(*curr_cam_res[1:], '{}')], base_path=state.data_path, use_batch_id=True, format='PNG')
					renderer.write_images([state.targets], ['target_{}-{}_cam{}'.format(*curr_cam_res[1:], '{}')], base_path=state.data_path, use_batch_id=True, format='PNG')
					renderer.write_images([state.bkgs], ['bkg_{}-{}_cam{}'.format(*curr_cam_res[1:], '{}')], base_path=state.data_path, use_batch_id=True, format='PNG')
					renderer.write_images([aux_sequence[frame].image_hulls], ['hull_base_cam{}'], base_path=state.data_path, use_batch_id=True, format='PNG')
					renderer.write_images([aux_sequence[frame].image_hulls_tight], ['hull_tight_base_cam{}'], base_path=state.data_path, use_batch_id=True, format='PNG')
					np.savez_compressed(os.path.join(state.data_path, 'volume_hull.npz'), aux_sequence[frame].hull)
					np.savez_compressed(os.path.join(state.data_path, 'volume_hull_tight.npz'), aux_sequence[frame].hull_tight)
					
					
					# random velocity initialization, but same for each frame
					if setup.data.velocity.initial_value.upper()=='RAND_CONST':
						vel_init = state.velocity
					prev = state
					sequence.append(state)
					first_frame = False
				return sequence
			
			sequence = sequence_setup(aux_sequence)
			del sequence_setup
			del frame_setup
			del frame_densSetup
			del frame_velSetup
			del aux_sequence
			
			for i in range(len(sequence)-1):
				sequence[i].next = sequence[i+1]
			sequence = Sequence(sequence)
			
			with tf.device(resource_device):
				if setup.training.optimize_buoyancy:
					buoyancy = tf.Variable(initial_value=setup.data.initial_buoyancy, dtype=tf.float32, name='buoyancy', trainable=True)
				else:
					buoyancy = tf.constant(setup.data.initial_buoyancy, dtype=tf.float32)
			
			if setup.training.velocity.pre_opt.first.iterations>0 and setup.training.velocity.pre_optimization: #and setup.training.velocity.pre_optimization ?
				curr_vel_shape = pre_opt_start_vel_shape
			else:
				curr_vel_shape = main_opt_start_vel_shape
				
			s = "Sequence setup:"
			i=0
			for state in sequence:
				p = -1 if state.prev is None else state.prev.frame
				n = -1 if state.next is None else state.next.frame
				s += "\n{:4d}: frame {:06d}: p={:06d},  n={:06d}".format(i, state.frame, p, n)
				i +=1
			log.info(s)
			
			log.debug('Initialize variables (density %f and light intensity %f)', setup.data.density.scale, lights[0].i)
			light_var_list = []
			if setup.training.light.optimize:
				var_intensity = tf.get_variable(name='intensity', initializer=lights[0].i, constraint=lambda var: tf.clip_by_value(var, setup.training.light.min, setup.training.light.max), dtype=tf.float32, trainable=True)#tf.Variable(initial_value=lights[0].i, dtype=tf.float32, name='intensity', trainable=True)
				lights[0].i=var_intensity
				var_ambient_intensity = tf.get_variable(name='ambient_intensity', initializer=lights[1].i, constraint=lambda var: tf.clip_by_value(var, setup.training.light.min, setup.training.light.max), dtype=tf.float32, trainable=True)#tf.Variable(initial_value=lights[1].i, dtype=tf.float32, name='ambient_intensity', trainable=True)
				lights[1].i=var_ambient_intensity
				light_var_list = [var_intensity, var_ambient_intensity]
			
			train_disc = False
			disc_dump_samples = setup.debug.disc_dump_samples
			if setup.training.discriminator.active:
				log.debug('Setup discriminator')
				train_disc = setup.training.discriminator.train or setup.training.discriminator.pre_opt.train or setup.training.discriminator.pre_opt.first.train
				disc_real_data = None
				disc_input_steps = None
				if setup.training.discriminator.temporal_input.active:
					disc_input_steps = list(range(*setup.training.discriminator.temporal_input.step_range))
					if 0 in disc_input_steps:
						disc_input_steps.remove(0)
				if train_disc or setup.training.discriminator.loss_type not in ["SGAN"]:
					log.debug('Setup discriminator training data (tf.data.Dataset)')
					with tf.device(resource_device):
						disc_dataset = get_scalarflow_dataset(setup.data.discriminator.simulations, setup.data.discriminator.frames, \
							path_mask=setup.data.discriminator.target, cam_ids=scalarFlow_cam_ids, down_scale=setup.data.discriminator.real_res_down, \
							threshold=setup.data.density.target_threshold, raw=False, preproc=True, bkg=False, hull=setup.training.discriminator.conditional_hull, \
							path_preproc=setup.data.discriminator.target_preproc if setup.data.discriminator.target_type else None, \
							temporal_input_steps=disc_input_steps \
							).batch(setup.training.discriminator.num_real).prefetch(12)
					
					# roll datatypes into channels and expand monochrome to RGB
					if setup.training.discriminator.temporal_input.active:
						if setup.training.discriminator.conditional_hull:
							disc_dataset = disc_dataset.map(lambda p,ph,y,h,n,nh: tf.concat([p]*color_channel+[ph] + [y]*color_channel+[h] + [n]*color_channel+[nh], axis=-1))
						else:
							disc_dataset = disc_dataset.map(lambda p,y,n: tf.concat([p]*color_channel + [y]*color_channel + [n]*color_channel, axis=-1))
					else:
						if setup.training.discriminator.conditional_hull:
							disc_dataset = disc_dataset.map(lambda y,h: tf.concat([y]*color_channel+[h], axis=-1))
						else:
							disc_dataset = disc_dataset.map(lambda y: tf.concat([y]*color_channel, axis=-1))
					#.map(lambda x,y,z: tf.concat([y]*3, axis=-1)*setup.data.density.target_scale)
					
					disc_real_res = tf.Variable(initial_value=disc_cam_resolution[1:], name='disc_real_res', dtype=tf.int32, trainable=False)
					if setup.data.discriminator.scale_real_to_cam:
						disc_dataset = disc_dataset.map(lambda i: tf.py_func(lambda j: wrap_resize_images(j, disc_real_res.numpy()), [i], tf.float32)) #tf.image.resize_bilinear(s, disc_real_res))
					
					disc_real_data = disc_dataset.make_one_shot_iterator()
					#train_disc = setup.training.discriminator.train
				disc_in_channel = color_channel
				if setup.training.discriminator.conditional_hull:
					disc_in_channel += 1
				if setup.training.discriminator.temporal_input.active:
					disc_in_channel *= 3
				disc_in_shape=list(setup.data.discriminator.crop_size) + [disc_in_channel] #disc_targets_shape[-3:]
				if train_disc and setup.training.discriminator.history.samples>0:
					log.debug("Initializing fake samples history buffer for discriminator experience replay.")
					if setup.training.discriminator.history.load is not None:
						history_path = run_index[setup.training.discriminator.history.load]
						if history_path is None:
							history_path = setup.training.discriminator.history.load
						raise NotImplementedError()
					#	disc_history = HistoryBuffer.deserialize(history_path)
					#	log.debug("Loaded history with %d/%d elements.", disc_history.elements, disc_history.size)
					#else:
					#	disc_history = HistoryBuffer(setup.training.discriminator.history.size)
					#	log.debug("initialized history")
				log.debug('Setup discriminator model')
				if setup.training.discriminator.model is not None:
					model_path = run_index[setup.training.discriminator.model]
					if model_path is None:
						model_path = setup.training.discriminator.model
					disc_model=tf.keras.models.load_model(model_path, custom_objects={'MirrorPadND':MirrorPadND})
					log.info('Restored model from %s', model_path)
				else:
					disc_model=discriminator(disc_in_shape, layers=setup.training.discriminator.layers, strides=setup.training.discriminator.stride, kernel_size=setup.training.discriminator.kernel_size, final_fc=setup.training.discriminator.use_fc, activation=setup.training.discriminator.activation, alpha=setup.training.discriminator.activation_alpha, noise_std=setup.training.discriminator.noise_std, padding=setup.training.discriminator.padding)
					log.debug('Built discriminator keras model')
				#disc_model(disc_targets, training=False)
				disc_weights = disc_model.get_weights()
				disc_model.summary(print_fn= lambda s: log.info(s))
				
				if np.any(np.less(disc_in_shape[:-1], disc_cam_resolution[1:])) and setup.training.discriminator.use_fc and (not setup.data.discriminator.scale_input_to_crop):
					log.warning("Base fake sample camera resolution exeeds rigid discriminator input resolution. Use the patch discriminator or enable input scaling.")
				curr_disc_cam_res = current_grow_shape(disc_cam_resolution, 0, setup.training.density.grow.factor, setup.training.density.grow.intervals)
				for camera in disc_cameras:
					camera.transform.grid_size = curr_disc_cam_res
					if setup.training.discriminator.fake_camera_jitter:
						camera.jitter = camera.depth_step
				disc_real_res.assign(curr_disc_cam_res[1:])
			#END if setup.training.discriminator.active
			
			
			def scale_density(sequence, iteration, factor, intervals, base_shape, actions=[]):
				global curr_dens_shape, curr_cam_res
				dens_shape = current_grow_shape(base_shape, iteration, factor, intervals)
				#vel_shape = current_grow_shape(base_shape, it, setup.training.velocity.grow.factor, setup.training.velocity.grow.intervals)
				if dens_shape!=curr_dens_shape:
					curr_cam_res = current_grow_shape(train_cam_resolution, iteration, factor, intervals)
					#curr_cam_res[0] = train_cam_resolution[0] #keep depth samples the same as there is no correction for changing sample resolution yet
					log.info("Rescaling density in iteration %d from %s to %s, cameras to %s", iteration, curr_dens_shape, dens_shape, curr_cam_res)
					log.debug("Saving sequence")
					with profiler.sample("Save sequence"):
						try:
							for state in sequence:
								state.density.save(os.path.join(state.data_path, \
									"density_{}-{}-{}_{}.npz".format(*state.density.shape, iteration)))
						except:
							log.warning("Failed to save density before scaling from %s to %s in iteration %d:", \
								curr_dens_shape, dens_shape, iteration, exc_info=True)
					log.debug("Rescaling density sequence")
					with profiler.sample('Rescale densities'):
						for state in sequence:
							#state.rescale_density(dens_shape, device=resource_device)
							d = state.density.scaled(dens_shape)
							#scale hull and inflow to new shape based on base values
							hull = state.density.scale_renderer.resample_grid3D_aligned(state.density.base_hull, dens_shape)if state.density.hull is not None else None
							if state.density._inflow is not None:
								if_off = current_grow_shape(state.density.base_inflow_offset, iteration, factor, intervals)
								if_shape = current_grow_shape(state.density.base_inflow_shape, iteration, factor, intervals, cast_fn=lambda x: max(round(x),1)) #cast_fn=math.ceil
								if_scaled = upscale_renderer.resample_grid3D_aligned(state.density._inflow, if_shape)
								if_mask = None if state.density.inflow_mask is None else state.density.scale_renderer.resample_grid3D_aligned(state.density.base_inflow_mask, if_shape)
								log.info("Frame %04d: inflow to %s, offset to %s", state.frame, if_shape, if_off)
								density = DensityGrid(shape=dens_shape, d=d, as_var=state.density.is_var, hull=hull, inflow=if_scaled, inflow_offset=if_off, inflow_mask=if_mask, \
									scale_renderer=state.density.scale_renderer, device=state.density._device, var_name=state.density._name+"_scaled", restrict_to_hull=state.density.restrict_to_hull)
							else:
								density = DensityGrid(shape=dens_shape, d=d, as_var=state.density.is_var, hull=hull, \
									scale_renderer=state.density.scale_renderer, device=state.density._device, var_name=state.density._name+"_scaled", restrict_to_hull=state.density.restrict_to_hull)
							if hull is not None:
								density.base_hull = state._density.base_hull
							if density._inflow is not None:
								density.base_inflow_mask = state._density.base_inflow_mask
								density.base_inflow_shape = state._density.base_inflow_shape
								density.base_inflow_offset = state._density.base_inflow_offset
							state._density = density
							
							AABB_corners_WS = dens_transform.transform_AABB(*hull_AABB_OS(tf.squeeze(state.density.hull, (0,-1))), True) if setup.rendering.target_cameras.crop_frustum else None
							state.target_cameras = setup_target_cameras(target_cameras, curr_cam_res, AABB_corners_WS,setup.rendering.target_cameras.crop_frustum_pad, jitter=setup.training.density.camera_jitter)
						curr_dens_shape = dens_shape
						#target and cams scale
					log.debug("Rescaling cameras")
					if setup.training.discriminator.active:
						global curr_disc_cam_res
						curr_disc_cam_res = current_grow_shape(disc_cam_resolution, iteration, factor, intervals)
						log.info("Scaling discriminator camera resolution to %s", curr_disc_cam_res)
						for camera in disc_cameras:
							camera.transform.grid_size = curr_disc_cam_res
							if setup.training.discriminator.fake_camera_jitter:
								camera.jitter = camera.depth_step
						disc_real_res.assign(curr_disc_cam_res[1:])
					if train_disc and setup.training.discriminator.history.samples>0:
						if setup.training.discriminator.history.reset_on_density_grow:
							log.info("Reset disc history after rescale")
							disc_ctx.history.reset()
					# targets scaled from base
					log.debug("Rescaling targets from base")
					with profiler.sample('Rescale targets'):
						for state in sequence:
							with tf.device(resource_device):
								state.targets_raw = wrap_resize_images(state.base_targets_raw, curr_cam_res[1:])
								state.targets = wrap_resize_images(state.base_targets, curr_cam_res[1:])
								state.bkgs = wrap_resize_images(state.base_bkgs, curr_cam_res[1:])
							log.debug("Writing scaled targets for frame %d", state.frame)
							renderer.write_images([state.targets_raw], ['target_raw_{}-{}_cam{}'.format(*curr_cam_res[1:], '{}')], \
								base_path=state.data_path, use_batch_id=True, format='PNG')
							renderer.write_images([state.targets], ['target_{}-{}_cam{}'.format(*curr_cam_res[1:], '{}')], \
								base_path=state.data_path, use_batch_id=True, format='PNG')
							renderer.write_images([state.bkgs], ['bkg_{}-{}_cam{}'.format(*curr_cam_res[1:], '{}')], \
								base_path=state.data_path, use_batch_id=True, format='PNG')
					return True
				else:
					return False
				#END if rescale density
			
			def scale_velocity(sequence, iteration, factor, scale_magnitude, intervals, base_shape):
				global curr_vel_shape, z
				vel_shape = current_grow_shape(base_shape, iteration, factor, intervals)
				scale_sequence = []
				for state in sequence:
					if state.velocity.centered_shape!=vel_shape:
						scale_sequence.append(state)
				if scale_sequence:
					log.info("Rescaling velocity of frames %s from %s to %s in iteration %d, magnitude: %s", [_.frame for _ in scale_sequence], \
						[_.velocity.centered_shape for _ in scale_sequence], vel_shape, iteration, scale_magnitude)
					log.debug("Saving sequence")
					with profiler.sample("Save sequence"):
							try:
								for state in scale_sequence:
									state.velocity.save(os.path.join(state.data_path, \
										"velocity_{}-{}-{}_{}.npz".format(*state.velocity.centered_shape, iteration)))
							except:
								log.warning("Failed to save velocity before scaling from %s to %s in iteration %d:", \
									state.velocity.centered_shape, vel_shape, iteration, exc_info=True)
					log.debug("Rescaling velocity sequence")
					with profiler.sample('Rescale velocities'):
						z = tf.zeros([1] + vel_shape + [1])
						for state in sequence:
							state.rescale_velocity(vel_shape, scale_magnitude=scale_magnitude, device=resource_device)
					curr_vel_shape = vel_shape
					return True
				else:
					return False
				#END if rescale velocity
			
			
			def render_sequence_val(sequence, vel_pad, it):
				log.debug("Render validation images for sequence, iteration %d", it)
				with profiler.sample('render validation'):
					for state in sequence:
						log.debug("Render density validation frame %d", state.frame)
						#sim_transform.set_data(state.density)
						dens_transform = state.get_density_transform()
						val_imgs = renderer.render_density(dens_transform, lights, val_cameras)
						renderer.write_images([tf.concat(val_imgs, axis=0)], ['val_img_cam{}_{:04d}'], base_path=state.data_path, use_batch_id=True, frame_id=it, format='PNG')
						
						vel_transform = state.get_velocity_transform()
						vel_scale = vel_transform.cell_size_world().value
						log.debug("Render velocity validation frame %d with scale %s", state.frame, vel_scale)
						#sim_transform.set_data(vel_pad)
						vel_centered = state.velocity.centered()*vel_scale/float(setup.data.step)*setup.rendering.velocity_scale #
						val_imgs = vel_renderer.render_density(vel_transform, [tf.abs(vel_centered)], val_cameras)
						vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['val_velA_cam{}_{:04d}'], base_path=state.data_path, use_batch_id=True, frame_id=it, format='PNG')
					#	val_imgs = vel_renderer.render_density(vel_transform, [tf.maximum(vel_centered, 0)], val_cameras)
					#	vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['val_velP_cam{}_{:04d}'], base_path=state.data_path, use_batch_id=True, frame_id=it, format='PNG')
					#	val_imgs = vel_renderer.render_density(vel_transform, [tf.maximum(-vel_centered, 0)], val_cameras)
					#	vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['val_velN_cam{}_{:04d}'], base_path=state.data_path, use_batch_id=True, frame_id=it, format='PNG')
			
			#log.debug("render initial state")
			
			
			# print growing stats
			def log_growth(tar_shape, intervals, factor, max_iter, name):
				s = "Growing {}: {:d} steps with factor {:f}".format(name, len(intervals)+1, factor)
				abs_intervals = abs_grow_intervals(intervals, max_iter)
				if abs_intervals[-1][0]>abs_intervals[-1][1]:
					log.warning("Insufficient iterations for all grow intervals")
				for interval in abs_intervals:
					shape = current_grow_shape(tar_shape, interval[0], factor, intervals)
					s += "\n\t[{:d},{:d}] {}".format(interval[0], interval[1]-1, shape)
				log.info(s)
			if setup.training.velocity.pre_opt.first.iterations>0:
				if setup.training.velocity.pre_opt.first.grow.intervals:
					log_growth(main_opt_start_vel_shape, setup.training.velocity.pre_opt.first.grow.intervals, setup.training.velocity.pre_opt.first.grow.factor, setup.training.velocity.pre_opt.first.iterations, 'pre-opt velocity')
			if setup.training.density.grow.intervals:
				log_growth(base_shape, setup.training.density.grow.intervals, setup.training.density.grow.factor, setup.training.iterations, 'density')
			if setup.training.velocity.grow.intervals:
				log_growth(base_shape, setup.training.velocity.grow.intervals, setup.training.velocity.grow.factor, setup.training.iterations, 'velocity')
			
			loss_schedules = LossSchedules( \
				density_target =		make_schedule(setup.training.density.preprocessed_target_loss), 
				density_target_raw =	make_schedule(setup.training.density.raw_target_loss), 
				density_target_depth_smoothness = make_schedule(setup.training.density.target_depth_smoothness_loss), 
				density_negative =		make_schedule(setup.training.density.negative), 
				density_hull =			make_schedule(setup.training.density.hull), 
				density_smoothness =	make_schedule(setup.training.density.smoothness_loss), 
				density_smoothness_2 =	make_schedule(setup.training.density.smoothness_loss_2), 
				density_smoothness_temporal = make_schedule(setup.training.density.temporal_smoothness_loss), 
				density_warp =			make_schedule(setup.training.density.warp_loss), 
				density_disc =			make_schedule(setup.training.density.discriminator_loss), 
				
				velocity_warp_dens =	make_schedule(setup.training.velocity.density_warp_loss), 
				velocity_warp_vel =		make_schedule(setup.training.velocity.velocity_warp_loss), 
				velocity_divergence =	make_schedule(setup.training.velocity.divergence_loss), 
				velocity_smoothness =	make_schedule(setup.training.velocity.smoothness_loss), 
				velocity_cossim =		make_schedule(setup.training.velocity.cossim_loss), 
				velocity_magnitude =	make_schedule(setup.training.velocity.magnitude_loss), 
				
				density_lr =			make_schedule(setup.training.density.learning_rate), 
				light_lr =				make_schedule(setup.training.light.learning_rate), 
				velocity_lr =			make_schedule(setup.training.velocity.learning_rate), 
				discriminator_lr =		make_schedule(setup.training.discriminator.learning_rate), 
				discriminator_regularization = make_schedule(setup.training.discriminator.regularization) )
			
			
			light_lr = tf.Variable(initial_value=scalar_schedule(setup.training.light.learning_rate, 0), dtype=tf.float32, name='light_lr', trainable=False)
			light_optimizer = tf.train.AdamOptimizer(light_lr, beta1=setup.training.light.optim_beta)
			dens_lr = tf.Variable(initial_value=scalar_schedule(setup.training.density.learning_rate, 0), dtype=tf.float32, name='density_lr', trainable=False)
			dens_optimizer = tf.train.AdamOptimizer(dens_lr, beta1=setup.training.density.optim_beta)
			vel_lr = tf.Variable(initial_value=scalar_schedule(setup.training.velocity.learning_rate, 0), dtype=tf.float32, name='velocity_lr', trainable=False)
			vel_optimizer = tf.train.AdamOptimizer(vel_lr, beta1=setup.training.velocity.optim_beta)
			disc_lr = tf.Variable(initial_value=scalar_schedule(setup.training.discriminator.learning_rate, 0), dtype=tf.float32, name='discriminator_lr', trainable=False)
			disc_optimizer = tf.train.AdamOptimizer(disc_lr, beta1=setup.training.discriminator.optim_beta)
			
			opt_ckpt = tf.train.Checkpoint(dens_optimizer=dens_optimizer, vel_optimizer=vel_optimizer, disc_optimizer=disc_optimizer)
			
			main_ctx = OptimizationContext(setup=setup, iteration=0, loss_schedules=loss_schedules, \
				rendering_context=main_render_ctx, vel_scale=[1,1,1], warp_order=setup.training.velocity.warp_order, dt=1.0, buoyancy=buoyancy, \
				dens_warp_clamp=setup.training.density.warp_clamp, vel_warp_clamp=setup.training.velocity.warp_clamp, \
				density_optimizer=dens_optimizer, density_lr=dens_lr, light_optimizer=light_optimizer, light_lr=light_lr, \
				velocity_optimizer=vel_optimizer, velocity_lr=vel_lr, \
				frame=None, tf_summary=summary, summary_interval=10, summary_pre=None, profiler=profiler,
				light_var_list=light_var_list)
			
			#loss functions for losses
			'''
			"density/target":		self.l1_loss,
			"density/target_raw":	self.l1_loss,
			"density/target_depth_smooth":	self.l2_loss,
			"density/negative":		self.l2_loss,
			"density/edge":			self.l2_loss,
			"density/smooth":		self.l2_loss,
			"density/warp":			self.l1_loss,
			
			"velocity/density_warp":	self.l1_loss,
			"velocity/velocity_warp":	self.l1_loss,
			"velocity/divergence":		self.l2_loss,
			"velocity/magnitude":		self.l2_loss,
			'''
			main_ctx.set_loss_func("density/target_raw", setup.training.density.error_functions.raw_target_loss)
			main_ctx.set_loss_func("density/target", setup.training.density.error_functions.preprocessed_target_loss)
			main_ctx.set_loss_func("density/target_depth_smooth", setup.training.density.error_functions.target_depth_smoothness_loss)
			main_ctx.set_loss_func("density/hull", setup.training.density.error_functions.hull)
			main_ctx.set_loss_func("density/negative", setup.training.density.error_functions.negative)
			main_ctx.set_loss_func("density/edge", setup.training.density.error_functions.smoothness_loss)
			main_ctx.set_loss_func("density/smooth", setup.training.density.error_functions.smoothness_loss_2)
			main_ctx.set_loss_func("density/smooth-temp", setup.training.density.error_functions.temporal_smoothness_loss)
			main_ctx.set_loss_func("density/warp", setup.training.density.error_functions.warp_loss)
			
			main_ctx.set_loss_func("velocity/density_warp", setup.training.velocity.error_functions.density_warp_loss)
			main_ctx.set_loss_func("velocity/velocity_warp", setup.training.velocity.error_functions.velocity_warp_loss)
			main_ctx.set_loss_func("velocity/divergence", setup.training.velocity.error_functions.divergence_loss)
			main_ctx.set_loss_func("velocity/magnitude", setup.training.velocity.error_functions.magnitude_loss)
			
			#gradient warping:
			main_ctx.update_first_dens_only =	make_schedule(setup.training.density.warp_gradients.update_first_only)
			main_ctx.warp_dens_grads =			make_schedule(setup.training.density.warp_gradients.active)
			main_ctx.warp_dens_grads_decay =	make_schedule(setup.training.density.warp_gradients.decay)
			main_ctx.warp_vel_grads =			make_schedule(setup.training.velocity.warp_gradients.active)
			main_ctx.warp_vel_grads_decay =		make_schedule(setup.training.velocity.warp_gradients.decay)
			main_ctx.custom_dens_grads_weight =	make_schedule(setup.training.density.warp_gradients.weight)
			main_ctx.custom_vel_grads_weight =	make_schedule(setup.training.velocity.warp_gradients.weight)
			
			main_ctx.target_weights = view_interpolation_target_weights
			log.info("Target weights: %s", view_interpolation_target_weights)
			
			sF_render_ctx = copy.copy(main_render_ctx)
			sF_render_ctx.cameras = None #scalarFlow_cameras
			opt_ctx = copy.copy(main_ctx)
			opt_ctx.render_ctx = sF_render_ctx
			
			if setup.training.density.scale_render_grads_sharpness>0.0:
				log.info("Scaling density render gradients with exisiting density distribution.")
				opt_ctx.add_render_op('DENSITY', opt_ctx.RO_grid_dens_grad_scale(weight=1, sharpness=setup.training.density.scale_render_grads_sharpness, eps=1e-5))
			
			if setup.training.discriminator.active:
				disc_render_ctx = copy.copy(main_render_ctx)
				disc_render_ctx.cameras = disc_cameras
				#log.info("Disc cam jitter: %s", [_.jitter for _ in disc_render_ctx.cameras])
				#log.warning("Full discriminator in/out debugging enabled!")
				disc_debug_path = os.path.join(setup.paths.data, 'disc_debug')
				os.makedirs(disc_debug_path)
				disc_ctx = DiscriminatorContext(ctx=opt_ctx, model=disc_model, rendering_context=disc_render_ctx, real_data=disc_real_data, \
					loss_type=setup.training.discriminator.loss_type, optimizer=disc_optimizer, learning_rate=disc_lr, \
					crop_size=setup.data.discriminator.crop_size, scale_range=setup.data.discriminator.scale_range, rotation_mode=setup.data.discriminator.rotation_mode, \
					check_input=DiscriminatorContext.CHECK_INPUT_RAISE_NOTFINITE | DiscriminatorContext.CHECK_INPUT_CHECK_NOTFINITE | DiscriminatorContext.CHECK_INPUT_CLAMP | \
					(DiscriminatorContext.CHECK_INPUT_SIZE if setup.training.discriminator.use_fc else 0x0), \
					check_info_path=disc_debug_path, resource_device=resource_device, \
					scale_samples_to_input_resolution=setup.data.discriminator.scale_input_to_crop, \
					use_temporal_input=setup.training.discriminator.temporal_input.active, temporal_input_steps=disc_input_steps)
				#disc_ctx.train = train_disc
				if train_disc and disc_dump_samples:
					disc_ctx.dump_path = os.path.join(setup.paths.data, 'disc_samples')
					log.warning("Dumping ALL discriminator samples to %s.", disc_ctx.dump_path)
					os.makedirs(disc_ctx.dump_path)
				log.info("Discriminator input shape: %s, res: %s", disc_ctx.model.input_shape, disc_ctx.input_res)
			else:
				disc_ctx = DiscriminatorContext(opt_ctx, None, main_render_ctx, None, "SGAN", None, disc_lr)
				disc_ctx.train = False
			
			dump_samples = False
			val_out_step = setup.validation.output_interval
			out_step = setup.training.summary_interval
			loss_summary = []
			
			class StopTraining(Exception):
				pass
			
			def check_loss_summary(loss_summaries, total_losses, it, gradients=None, grad_max=None):
				# check losses and gradients for NaN/Inf
				for f, f_item in loss_summaries.items():
					for k, k_item in f_item.items():
						if not np.all(np.isfinite(k_item)):
							raise ValueError("Loss summary {} of frame {} is not finite.".format(k,f))
				if total_losses and not np.all(np.isfinite(total_losses)):
					raise ValueError("Combined losses are not finite.".format(k,f))
				for f, f_item in gradients.items():
					for k, k_item in f_item.items():
						if not np.all(np.isfinite(k_item)):
							raise ValueError("Gradient summary {} of frame {} is not finite.".format(k,f))
						if grad_max is not None:
							if "density/light" in k or "velocity/buoyancy" in k or "-v" in k:
								continue # gradients of gloabl scalars are higher
							if np.any(np.greater(k_item, grad_max)):
								raise ValueError("Gradient summary {} of frame {} is greater than {}.".format(k,f, grad_max))
			
			def print_loss_summary(loss_summaries, total_losses, start_time, last_time, it, iterations, gradients=None):
				'''
				numpy scalars:
					loss_summaries: {<frame>: {<name/key>: (scaled, raw, scale), ...}, ...}
					gradients: {<frame>: {<name/key>: [grad, ...], ...}, ...}
				'''
				
				log.info('GPU mem: current: %d MiB, max: %d MiB, limit: %d MiB', \
					tf.contrib.memory_stats.BytesInUse().numpy().tolist()/(1024*1024), \
					tf.contrib.memory_stats.MaxBytesInUse().numpy().tolist()/(1024*1024), \
					tf.contrib.memory_stats.BytesLimit().numpy().tolist()/(1024*1024))
				s = ["--- Loss Summary ---\n"]
				now = time.time()
				avg = (now-start_time)/max(1,it)
				avg_last = (now-last_time)/out_step
				s.append('Timing: elapsed {}, avg/step: total {}, last {}, remaining: total {}, last {}\n'.format(format_time(now-start_time), format_time(avg), format_time(avg_last), format_time(avg*(iterations-it)), format_time(avg_last*(iterations-it))))
				s.append('{:26}(x{:>11}): {:>11}({:>11})| ...\n'.format('Last losses', 'Scale', 'Scaled', 'Raw'))
				loss_names = sorted({n for f in loss_summaries for n in loss_summaries[f]})
				loss_frames = sorted((f for f in loss_summaries))
				loss_scales = [loss_summaries[loss_frames[0]][k][-1] for k in loss_names]
				for key, scale in zip(loss_names, loss_scales):
					s.append('{:<26}(x{: 10.04e}):'.format(key, scale))
				#	loss_values = active_losses[key]['frames']
					for f in loss_frames:
						if key in loss_summaries[f]:
							s.append(' {: 10.04e}({: 10.04e})'.format(loss_summaries[f][key][-3], loss_summaries[f][key][-2]))
						else:
							s.append(' {:>11}({:>11})'.format('N/A', 'N/A'))
						s.append('|')
					s.append('\n')
				if gradients is not None:
					s.append('{:26}: {:>11}| ...\n'.format("Per-loss volume gradients", "mean-abs"))
					grad_names = sorted({n for f in gradients for n in gradients[f]})
					grad_frames = sorted((f for f in gradients))
					for key in grad_names:
						s.append('{:<26}:'.format(key))
						#grad_values = active_losses[key]['frames_grad']
						for f in grad_frames:
							if key in gradients[f]:
								s.append(','.join([' {: 10.04e}']*len(gradients[f][key])).format(*gradients[f][key]))
							else:
								s.append(' {:>11}'.format('N/A'))
							s.append('|')
						s.append('\n')
				s.append('total scaled loss (dens, vel):')
				for total_loss in total_losses:
					s.append(' ({: 10.04e},{: 10.04e}),'.format(*total_loss))
				s.append('\n')
				s.append('lr: dens {: 10.04e}, vel {: 10.04e}'.format(dens_lr.numpy(), vel_lr.numpy()))
				log.info(''.join(s))
			
			def print_disc_summary(disc_ctx, disc_loss):#_real, disc_scores_real, disc_loss_fake, disc_scores_fake):
				loss_summaries = [disc_ctx.opt_ctx.pop_loss_summary()]
				active_losses = {}
				f = 0
				for summ in loss_summaries:
					for k, e in summ.items():
						if k not in active_losses:
							active_losses[k] = {'frames':{}, 'scale':e[-1] if e[-1] is not None else 1.0}
						active_losses[k]['frames'][f] = (e[-3], e[-2] if e[-2] is not None else e[-3])
					f +=1
				s = ["--- Disc Summary ---\n"]
				s.append('{:26}(x{:>9}): {:>11}({:>11}), ...\n'.format('Last losses', 'Scale', 'Scaled', 'Raw'))
				for key in sorted(active_losses.keys()):
					s.append('{:<26}(x{: 9.06f}):'.format(key, active_losses[key]['scale']))
					loss_values = active_losses[key]['frames']
					for i in range(f):
						if i in loss_values:
							s.append(' {: 10.04e}({: 10.04e}),'.format(*loss_values[i]))
						else:
							s.append(' {:>11}({:>11}),'.format('N/A', 'N/A'))
					s.append('\n')
				if len(disc_loss)==4:
					s.append('Total loss (scores): real {:.06f} ({}), fake {:.06f} ({})\n'.format(disc_loss[0], disc_loss[2], disc_loss[1], disc_loss[3]))
				elif len(disc_loss)==2:
					s.append('Total loss (scores): {:.06f} ({})\n'.format(*disc_loss))
				s.append('lr: {:.08f}'.format(disc_ctx.lr.numpy()))
				if setup.training.discriminator.history.samples>0:
					s.append(', history size: {}'.format(len(disc_ctx.history)))
				log.info(''.join(s))
			
			def optimize_density(opt_ctx, state, iterations, use_vel=False, disc_ctx=None, grow_factor=1.0, grow_intervals=[], start_iteration=0):
				log.debug('Start density optimization')
				with profiler.sample('Density Optimization', verbose = False):
					last_time = time.time()
					start_time = last_time
					opt_ctx.frame = state.frame
					for it in range(start_iteration, iterations):
						log.debug('Start iteration %d', it)
						summary_iteration = (it+1)%out_step==0 or (it+1)==iterations or stop_training
						opt_ctx.start_iteration(it, compute_loss_summary=summary_iteration)
						scale_density([state], it, grow_factor, grow_intervals, base_shape=main_opt_start_dens_shape)
						if it==0:
							log.info("Render initial state for density of frame %d", state.frame)
							dens_transform = state.get_density_transform()
							val_imgs = renderer.render_density(dens_transform, opt_ctx.render_ctx.lights, val_cameras)
							opt_ctx.render_ctx.dens_renderer.write_images([tf.concat(val_imgs, axis=0)], ['P_init_img_cam{}'], base_path=state.data_path, use_batch_id=True, format='PNG')
							inspect_gradients_list = {state.frame:{}}
							if setup.training.density.pre_opt.inspect_gradients==1:
								#ig_func = lambda opt_ctx, gradients, name: inspect_gradients_list[name] = tf.reduce_mean(tf.abs(gradients)).numpy()
								def ig_func(opt_ctx, gradients, name): inspect_gradients_list[opt_ctx.frame][name] = [tf.reduce_mean(tf.abs(gradients)).numpy()]
								iig_func = None
							if setup.training.density.pre_opt.inspect_gradients==2:
								AABB_corners_WS = dens_transform.transform_AABB(*hull_AABB_OS(tf.squeeze(state.density.hull, (0,-1))), True)
								grad_cams = [main_camera.copy_clipped_to_world_coords(AABB_corners_WS)[0]]
								ig_func = lambda opt_ctx, gradients, name: render_gradients(gradients, dens_transform, grad_cams, grad_renderer, \
									path=os.path.join(setup.paths.data, "gradients", "d-pO_f{:04d}_it{:08d}".format(opt_ctx.frame, opt_ctx.iteration)), \
									image_mask=name.replace("/", ".") + "_cam{:02}_{:04d}", name=name, log=log)
								iig_func = lambda opt_ctx, gradients, name: write_image_gradients(gradients, max_renderer, \
									path=os.path.join(setup.paths.data, "gradients", "d-pO_f{:04d}_it{:08d}".format(opt_ctx.frame, opt_ctx.iteration)), \
									image_mask=name.replace("/", ".") + "_img_cam{:04}", image_neg_mask=name.replace("/", ".") + "_img-neg_cam{:04}")
						#exclude first iteration from measurement as it includes some tf setup time
						elif it==1:
							start_time = time.time()
						
						with profiler.sample('Density Optim Step', verbose = False):
							if summary_iteration and setup.training.density.pre_opt.inspect_gradients:
								for g in inspect_gradients_list: inspect_gradients_list[g].clear()
								opt_ctx.set_inspect_gradient(True, ig_func, iig_func)
							optStep_density(opt_ctx, state, use_vel, disc_ctx)
							opt_ctx.set_inspect_gradient(False)
							opt_ctx.pop_losses()
							loss_summaries = {state.frame:opt_ctx.pop_loss_summary()}
							
							# DISCRIMINATOR
							if disc_ctx is not None:
								disc_ctx.start_iteration(it, compute_loss_summary=summary_iteration)
								#disc_ctx.opt_ctx.frame = None
								for disc_step in range(setup.training.discriminator.steps):
									disc_loss = optStep_discriminator(disc_ctx, state=state) #_real, disc_loss_fake, disc_scores_real, disc_scores_fake
							#END disc training
							
						if args.console and not args.debug:
							progress = it%out_step+1
							progress_bar(progress,out_step, "{:04d}/{:04d}".format(progress, out_step), length=50)
						if summary_iteration:
							log.info('--- Step {:04d}/{:04d} ---'.format(it, iterations-1))
							print_loss_summary(loss_summaries, [], start_time, last_time, it, iterations, inspect_gradients_list if setup.training.density.pre_opt.inspect_gradients==1 else None)
							d_max, d_min, d_mean = tf_print_stats(state.density.d, 'Density', log=log)
							if setup.training.light.optimize:
								log.info("Light intensities: %s", [_.numpy() for _ in light_var_list])
							if disc_ctx is not None and disc_ctx.train and setup.training.discriminator.start_delay<=it:
								print_disc_summary(disc_ctx, disc_loss)#_real, disc_scores_real, disc_loss_fake, disc_scores_fake)
							last_time = time.time()
							summary_writer.flush()
							check_loss_summary(loss_summaries, [], it, inspect_gradients_list if setup.training.density.pre_opt.inspect_gradients==1 else None)
						if val_cameras is not None and (it+1)%val_out_step==0:
							log.info("Render validation view %d for density of frame %d in iteration %d", int((it+1)//val_out_step), state.frame, it)
							try:
								dens_transform = state.get_density_transform()
								val_imgs = renderer.render_density(dens_transform, opt_ctx.render_ctx.lights, val_cameras)
								opt_ctx.render_ctx.dens_renderer.write_images([tf.concat(val_imgs, axis=0)], ['P_val_img_cam{}_{:04d}'], base_path=state.data_path, use_batch_id=True, frame_id=int((it+1)//val_out_step), format='PNG')
							except:
								log.exception('Exception when rendering validation view %d for density of frame %d in iteration %d:', int((it+1)//val_out_step), state.frame, it)
						if stop_training:
							log.warning('Density training of frame %d stopped after %d iterations', state.frame, it+1)
							raise StopTraining
							break
			
			def optimize_velocity(opt_ctx, state, iterations, grow_factor=1.0, grow_intervals=[], grow_scale_magnitude=True, start_iteration=0):
				log.debug('Start velocity optimization')
				with profiler.sample('Velocity Optimization', verbose = False):
					last_time = time.time()
					start_time = last_time
					opt_ctx.frame = state.frame
					for it in range(start_iteration, iterations):
						log.debug('Start iteration %d', it)
						summary_iteration = (it+1)%out_step==0 or (it+1)==iterations or stop_training
						opt_ctx.start_iteration(it, compute_loss_summary=summary_iteration)
						scale_velocity([state], it, grow_factor, grow_scale_magnitude, grow_intervals, base_shape=main_opt_start_vel_shape)
					#	vel_scale = world_scale(curr_vel_shape, width=1.)
					#	opt_ctx.vel_scale = vel_scale
						if it==0:
							vel_transform = state.get_velocity_transform()
							vel_scale = vel_transform.cell_size_world().value
							vel_max, vel_min, vel_mean = tf_print_stats(state.velocity.magnitude(), 'Vel init mag abs', log=log)
							log.info("Render initial state for velocity of frame %d with scale %s", state.frame, vel_scale)
							vel_centered = state.velocity.centered()*vel_scale/float(setup.data.step)*setup.rendering.velocity_scale #
							val_imgs = vel_renderer.render_density(vel_transform, [tf.maximum(vel_centered, 0)], val_cameras)
							vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['P_init_velP_cam{}'], base_path=state.data_path, use_batch_id=True, format='PNG')
							val_imgs = vel_renderer.render_density(vel_transform, [tf.maximum(-vel_centered, 0)], val_cameras)
							vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['P_init_velN_cam{}'], base_path=state.data_path, use_batch_id=True, format='PNG')
							inspect_gradients_list = {state.frame:{}}
							if setup.training.density.pre_opt.inspect_gradients==1:
								#ig_func = lambda opt_ctx, gradients, name: inspect_gradients_list[name] = tf.reduce_mean(tf.abs(gradients)).numpy()
								def ig_func(opt_ctx, gradients, name):
									if name.endswith('_x') or name.endswith('_y') or name.endswith('_z'):
										#velocity gradients are given individually per component with name=loss_name + _(x|y|z)
										c = ['x','y','z'].index(name[-1:])
										name = name[:-2]
										if name not in inspect_gradients_list[opt_ctx.frame]: inspect_gradients_list[opt_ctx.frame][name] = np.asarray([0,0,0], dtype=np.float32)
										inspect_gradients_list[opt_ctx.frame][name][c] = tf.reduce_mean(tf.abs(gradients)).numpy()
									else:
										inspect_gradients_list[opt_ctx.frame][name] = [tf.reduce_mean(tf.abs(gradients)).numpy()]
								iig_func = None
							if setup.training.density.pre_opt.inspect_gradients==2:
								AABB_corners_WS = dens_transform.transform_AABB(*hull_AABB_OS(tf.squeeze(state.density.hull, (0,-1))), True)
								grad_cams = [main_camera.copy_clipped_to_world_coords(AABB_corners_WS)[0]]
								ig_func = lambda opt_ctx, gradients, name: render_gradients(gradients, dens_transform, grad_cams, grad_renderer, \
									path=os.path.join(setup.paths.data, "gradients", "d-pO_f{:04d}_it{:08d}".format(opt_ctx.frame, opt_ctx.iteration)), \
									image_mask=name.replace("/", ".") + "_cam{:02}_{:04d}", name=name, log=log)
								iig_func = lambda opt_ctx, gradients, name: write_image_gradients(gradients, max_renderer, \
									path=os.path.join(setup.paths.data, "gradients", "d-pO_f{:04d}_it{:08d}".format(opt_ctx.frame, opt_ctx.iteration)), \
									image_mask=name.replace("/", ".") + "_img_cam{:04}", image_neg_mask=name.replace("/", ".") + "_img-neg_cam{:04}")
						#exclude first iteration from measurement as it includes some tf setup time
						elif it==1:
							start_time = time.time()
							
						if summary_iteration and setup.training.density.pre_opt.inspect_gradients:
							for g in inspect_gradients_list: inspect_gradients_list[g].clear()
							opt_ctx.set_inspect_gradient(True, ig_func, iig_func)
						optStep_velocity(opt_ctx, state, optimize_inflow=True)
						opt_ctx.set_inspect_gradient(False)
						opt_ctx.pop_losses()
						loss_summaries = {state.frame:opt_ctx.pop_loss_summary()}
							
						if args.console and not args.debug:
							progress = it%out_step+1
							progress_bar(progress,out_step, "{:04d}/{:04d}".format(progress, out_step), length=50)
						if summary_iteration:
							log.info('--- Step {:04d}/{:04d} ---'.format(it, iterations-1))
							print_loss_summary(loss_summaries, [], start_time, last_time, it, iterations, inspect_gradients_list if setup.training.density.pre_opt.inspect_gradients==1 else None)
							log.info("buoyancy: %s", opt_ctx.buoyancy.numpy())
							vel_max, vel_min, vel_mean = tf_print_stats(state.velocity.magnitude(), 'Vel mag abs', log=log)
							last_time = time.time()
							summary_writer.flush()
							check_loss_summary(loss_summaries, [], it, inspect_gradients_list if setup.training.density.pre_opt.inspect_gradients==1 else None)
						if val_cameras is not None and (it+1)%val_out_step==0:
							vel_transform = state.get_velocity_transform()
							vel_scale = vel_transform.cell_size_world().value
							log.info("Render validation view %d for velocity of frame %d with scale %s in iteration %d", int((it+1)//val_out_step), state.frame, vel_scale, it)
							try:
								vel_centered = state.velocity.centered()*vel_scale/float(setup.data.step)*setup.rendering.velocity_scale #
								val_imgs = vel_renderer.render_density(vel_transform, [tf.abs(vel_centered)], val_cameras)
								vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['P_val_velA_cam{}_{:04d}'], base_path=state.data_path, use_batch_id=True, frame_id=int((it+1)//val_out_step), format='PNG')
							#	val_imgs = vel_renderer.render_density(vel_transform, [tf.maximum(vel_centered, 0)], val_cameras)
							#	vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['P_val_velP_cam{}_{:04d}'], base_path=state.data_path, use_batch_id=True, frame_id=int((it+1)//val_out_step), format='PNG')
							#	val_imgs = vel_renderer.render_density(vel_transform, [tf.maximum(-vel_centered, 0)], val_cameras)
							#	vel_renderer.write_images([tf.concat(val_imgs, axis=0)], ['P_val_velN_cam{}_{:04d}'], base_path=state.data_path, use_batch_id=True, frame_id=int((it+1)//val_out_step), format='PNG')
							except:
								log.exception('Exception when rendering validation view %d for velocity of frame %d in iteration %d:', int((it+1)//val_out_step), state.frame, it)
						if stop_training:
							log.warning('Velocity training of frame %d stopped after %d iterations', state.frame, it+1)
							raise StopTraining
							break
			
			# scene serialization
			scene = {
				"cameras":cameras,
				"sFcameras":scalarFlow_cameras,
				"lighting":lights,
				"objects":[sim_transform],
				"pre_opt_vel_shape": main_opt_start_vel_shape,
				"pre_opt_dens_shape": main_opt_start_dens_shape,
				"vel_shape": base_shape,
				"vel_shape": base_shape,
			}
			scene_file = os.path.join(setup.paths.config, "scene.json")
			#log.debug("Serializing scene to %s ...", scene_file)
			with open(scene_file, "w") as file:
				try:
					json.dump(scene, file, default=tf_to_dict, sort_keys=True)#, indent=2)
				except:
					log.exception("Scene serialization failed.")
		
		except KeyboardInterrupt:
			log.warning("Interrupt during setup.")
			sys.exit(0)
		except:
			log.exception('Exception during setup:')
			sys.exit(1)
		
# --- Optimization ---
		'''
			loss_schedules = LossSchedules( \
				density_target =		make_schedule(setup.training.density.preprocessed_target_loss), 
				density_target_raw =	make_schedule(setup.training.density.raw_target_loss), 
				density_smoothness =	make_schedule(setup.training.density.smoothness_loss), 
				density_smoothness_2 =	make_schedule(setup.training.density.smoothness_loss_2), 
				density_warp =			make_schedule(setup.training.density.warp_loss), 
				density_disc =			make_schedule(setup.training.discriminator.loss_scale), 
				
				velocity_warp_dens =	make_schedule(setup.training.velocity.density_warp_loss), 
				velocity_warp_vel =		make_schedule(setup.training.velocity.velocity_warp_loss), 
				velocity_divergence =	make_schedule(setup.training.velocity.divergence_loss), 
				velocity_smoothness =	make_schedule(setup.training.velocity.smoothness_loss), 
				
				density_lr =			make_schedule(setup.training.density.learning_rate), 
				velocity_lr =			make_schedule(setup.training.velocity.learning_rate), 
				discriminator_lr =		make_schedule(setup.training.discriminator.learning_rate) )
		'''
		signal.signal(signal.SIGINT, handle_train_interrupt)
		optim_start = time.time()
		try:
			with summary_writer.as_default(), summary.always_record_summaries():
				if setup.training.density.pre_optimization or setup.training.velocity.pre_optimization:
					pre_optim_start = time.time()
					opt_ctx.summary_pre = "Pre-Optim"
					with profiler.sample("Pre-Optimization"):
						log.info("Run pre-optimization")
						if setup.training.density.pre_optimization:
							#Optimize first frame density
							log.info("--- First frame Density optimization start (%d iterations) ---", setup.training.density.pre_opt.first.iterations)
							loss_schedules.set_schedules(
								density_target =		make_schedule(setup.training.density.pre_opt.first.preprocessed_target_loss), 
								density_target_raw =	make_schedule(setup.training.density.pre_opt.first.raw_target_loss), 
								density_target_depth_smoothness = make_schedule(setup.training.density.pre_opt.first.target_depth_smoothness_loss), 
								density_hull =			make_schedule(setup.training.density.pre_opt.first.hull), 
								density_negative =		make_schedule(setup.training.density.pre_opt.first.negative), 
								density_smoothness =	make_schedule(setup.training.density.pre_opt.first.smoothness_loss), 
								density_smoothness_2 =	make_schedule(setup.training.density.pre_opt.first.smoothness_loss_2), 
								density_smoothness_temporal = make_schedule(setup.training.density.pre_opt.first.temporal_smoothness_loss), 
								density_warp =			make_schedule(setup.training.density.pre_opt.first.warp_loss), 
								density_disc =			make_schedule(setup.training.density.pre_opt.first.discriminator_loss), 
								density_lr =			make_schedule(setup.training.density.pre_opt.first.learning_rate), 
								discriminator_lr =		make_schedule(setup.training.discriminator.pre_opt.first.learning_rate), 
								discriminator_regularization = make_schedule(setup.training.discriminator.pre_opt.first.regularization))
							disc_ctx.train = setup.training.discriminator.pre_opt.first.train and setup.training.discriminator.active
							tmp_state = sequence[0]
							optimize_density(opt_ctx, tmp_state, disc_ctx=disc_ctx, iterations=setup.training.density.pre_opt.first.iterations)
							#copy first frame density to second frame
							if len(sequence)>1: #and setup.training.density.pre_opt.seq_init.upper()!="BASE":
								if (not setup.data.velocity.initial_value.upper().startswith('RAND')) and setup.training.density.pre_opt.iterations==0 and setup.training.density.pre_opt.seq_init.upper()!="WARP":
									# forward warp sequence init with loaded velocity
									sequence[1].density.assign(sequence[0].density.warped(sequence[0].velocity, order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=opt_ctx.dens_warp_clamp))
								elif setup.training.velocity.pre_opt.seq_init.upper()!="BASE": # COPY or WARP if no vel available
									sequence[1].density.assign(sequence[0].density.d)
								
								#Optimize second frame density
								log.info("--- Second frame Density optimization start (%d iterations) ---", setup.training.density.pre_opt.iterations)
								loss_schedules.set_schedules(
									density_target =		make_schedule(setup.training.density.pre_opt.preprocessed_target_loss), 
									density_target_raw =	make_schedule(setup.training.density.pre_opt.raw_target_loss), 
									density_target_depth_smoothness = make_schedule(setup.training.density.pre_opt.target_depth_smoothness_loss), 
									density_hull =			make_schedule(setup.training.density.pre_opt.hull), 
									density_negative =		make_schedule(setup.training.density.pre_opt.negative), 
									density_smoothness =	make_schedule(setup.training.density.pre_opt.smoothness_loss), 
									density_smoothness_2 =	make_schedule(setup.training.density.pre_opt.smoothness_loss_2), 
									density_smoothness_temporal = make_schedule(setup.training.density.pre_opt.temporal_smoothness_loss), 
									density_warp =			make_schedule(setup.training.density.pre_opt.warp_loss), 
									density_disc =			make_schedule(setup.training.density.pre_opt.discriminator_loss), 
									density_lr =			make_schedule(setup.training.density.pre_opt.learning_rate), 
									discriminator_lr =		make_schedule(setup.training.discriminator.pre_opt.learning_rate), 
									discriminator_regularization = make_schedule(setup.training.discriminator.pre_opt.regularization) )
								disc_ctx.train = setup.training.discriminator.pre_opt.train and setup.training.discriminator.active
								optimize_density(opt_ctx, sequence[1], disc_ctx=disc_ctx, iterations=setup.training.density.pre_opt.iterations)
						#END density pre-opt (frame 0,1 )
						
						if setup.training.velocity.pre_optimization and len(sequence)>1:
							#Optimize first frame velocity (using first and second frame density), grow to full size
							log.info("--- First frame Velocity optimization start (%d iterations) ---", setup.training.velocity.pre_opt.first.iterations)
							loss_schedules.set_schedules(
								velocity_warp_dens =	make_schedule(setup.training.velocity.pre_opt.first.density_warp_loss), 
								velocity_warp_vel =		make_schedule(setup.training.velocity.pre_opt.first.velocity_warp_loss), 
								velocity_divergence =	make_schedule(setup.training.velocity.pre_opt.first.divergence_loss), 
								velocity_smoothness =	make_schedule(setup.training.velocity.pre_opt.first.smoothness_loss), 
								velocity_cossim =		make_schedule(setup.training.velocity.pre_opt.first.cossim_loss), 
								velocity_magnitude =	make_schedule(setup.training.velocity.pre_opt.first.magnitude_loss), 
								velocity_lr =			make_schedule(setup.training.velocity.pre_opt.first.learning_rate), )
							optimize_velocity(opt_ctx, sequence[0], \
								iterations=setup.training.velocity.pre_opt.first.iterations, \
								grow_factor=setup.training.velocity.pre_opt.first.grow.factor, \
								grow_intervals=setup.training.velocity.pre_opt.first.grow.intervals, \
								grow_scale_magnitude=setup.training.velocity.pre_opt.first.grow.scale_magnitude)
							#for frame 1 to N-1
							loss_schedules.set_schedules(
								velocity_warp_dens =	make_schedule(setup.training.velocity.pre_opt.density_warp_loss), 
								velocity_warp_vel =		make_schedule(setup.training.velocity.pre_opt.velocity_warp_loss), 
								velocity_divergence =	make_schedule(setup.training.velocity.pre_opt.divergence_loss), 
								velocity_smoothness =	make_schedule(setup.training.velocity.pre_opt.smoothness_loss), 
								velocity_cossim =		make_schedule(setup.training.velocity.pre_opt.cossim_loss), 
								velocity_magnitude =	make_schedule(setup.training.velocity.pre_opt.magnitude_loss), 
								velocity_lr =			make_schedule(setup.training.velocity.pre_opt.learning_rate), )
						#END velocity pre-opt (frame 0)
						
						for n in range(1, len(sequence)-1):
							#advect forward
							if setup.training.velocity.pre_optimization:
								if setup.training.velocity.pre_opt.seq_init.upper()=="WARP":
									#set velocity n to warped n-1
									sequence[n].velocity.assign(*sequence[n-1].velocity.warped(order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=opt_ctx.vel_warp_clamp))
								if setup.training.velocity.pre_opt.seq_init.upper()=="COPY":
									sequence[n].velocity.assign(x=sequence[n-1].velocity.x, y=sequence[n-1].velocity.y, z=sequence[n-1].velocity.z)
							if setup.training.density.pre_optimization:
								if setup.training.density.pre_opt.seq_init.upper()=="WARP":
									#set density n+1 to warped density n (using velocity n)
									sequence[n+1].density.assign(sequence[n].density.warped(sequence[n].velocity, order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=opt_ctx.dens_warp_clamp))
								if setup.training.velocity.pre_opt.seq_init.upper()=="COPY":
									sequence[n+1].density.assign(sequence[n].density.d)
							
							#correct advection results
							if setup.training.density.pre_optimization:
								#Optimize density n+1
								log.info("--- Frame %d Density optimization start (%d iterations) ---", n+1, setup.training.density.pre_opt.iterations)
								optimize_density(opt_ctx, sequence[n+1], disc_ctx=disc_ctx, iterations=setup.training.density.pre_opt.iterations)
							if setup.training.velocity.pre_optimization:
								#Optimize velocity n
								log.info("--- Frame %d Velocity optimization start (%d iterations) ---", n, setup.training.velocity.pre_opt.iterations)
								optimize_velocity(opt_ctx, sequence[n], iterations=setup.training.velocity.pre_opt.iterations, \
									grow_factor=setup.training.velocity.pre_opt.grow.factor, \
									grow_intervals=setup.training.velocity.pre_opt.grow.intervals, \
									grow_scale_magnitude=setup.training.velocity.pre_opt.grow.scale_magnitude)
						
						if setup.training.velocity.pre_optimization and len(sequence)>1:
							if setup.training.velocity.pre_opt.seq_init.upper()=="WARP":
								#set velocity N to warped N-1
								sequence[-1].velocity.assign(*sequence[-2].velocity.warped(order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=opt_ctx.vel_warp_clamp))
							elif setup.training.velocity.pre_opt.seq_init.upper()=="COPY":
								sequence[-1].velocity.assign(x=sequence[-2].velocity.x, y=sequence[-2].velocity.y, z=sequence[-2].velocity.z)
							elif setup.training.velocity.pre_opt.grow.intervals and not setup.training.iterations>0:
								# initialized at lower resolution but not scaled in main
								scale_velocity([sequence[-1]], setup.training.velocity.pre_opt.iterations, setup.training.velocity.pre_opt.grow.factor, \
									setup.training.velocity.pre_opt.grow.scale_magnitude, setup.training.velocity.pre_opt.grow.intervals, base_shape=main_opt_start_vel_shape)
					#profiler
					log.info('Pre-optimization finished after %s', format_time(time.time() - pre_optim_start))
					sequence.save(suffix='pre-opt')
					if setup.training.discriminator.active:
						save_discriminator(disc_ctx.model, 'disc_pre-opt', setup.paths.data)
						if setup.training.discriminator.history.save:
							disc_ctx.history.serialize(setup.paths.data, 'pre-opt')
					
					with open(os.path.join(setup.paths.log, 'profiling.txt'), 'w') as f:
						profiler.stats(f)
				#END pre-optimization
				'''
				if False: #subdivision>0:
					new_sequence = []
					time_scale = 1.0/subdivision
					last_state = None
					for state in sequence[:-1]:
						state.velocity.scale_magnitude(time_scale)
						state.prev = last_state
						new_sequence.append(state)
						last_state = state
						for i in range(subdivision):
							new_state = state.copy_warped()
							new_state.prev = last_state
							last_state.next = new_state
							new_sequence.append(new_state)
							last_state = new_state
					sequence[-1].velocity.scale_magnitude(time_scale)
					sequence[-1].prev = last_state
					last_state.next = sequence[-1]
					new_sequence.append(sequence[-1])
				#END subdivide sequence
				'''
				opt_ctx.summary_pre = "Main-Optim"
				#run full-sequence optimization
				log.info('--- Sequence optimization (order: %s) start (%d iterations) ---', setup.training.frame_order, setup.training.iterations)
				loss_schedules.set_schedules( \
					density_target =		make_schedule(setup.training.density.preprocessed_target_loss), 
					density_target_raw =	make_schedule(setup.training.density.raw_target_loss), 
					density_target_depth_smoothness = make_schedule(setup.training.density.target_depth_smoothness_loss), 
					density_hull =			make_schedule(setup.training.density.hull), 
					density_negative =		make_schedule(setup.training.density.negative), 
					density_smoothness =	make_schedule(setup.training.density.smoothness_loss), 
					density_smoothness_2 =	make_schedule(setup.training.density.smoothness_loss_2), 
					density_smoothness_temporal = make_schedule(setup.training.density.temporal_smoothness_loss), 
					density_warp =			make_schedule(setup.training.density.warp_loss), 
					density_disc =			make_schedule(setup.training.density.discriminator_loss), 
					
					velocity_warp_dens =	make_schedule(setup.training.velocity.density_warp_loss), 
					velocity_warp_vel =		make_schedule(setup.training.velocity.velocity_warp_loss), 
					velocity_divergence =	make_schedule(setup.training.velocity.divergence_loss), 
					velocity_smoothness =	make_schedule(setup.training.velocity.smoothness_loss), 
					velocity_cossim =		make_schedule(setup.training.velocity.cossim_loss), 
					velocity_magnitude =	make_schedule(setup.training.velocity.magnitude_loss), 
					
					density_lr =			make_schedule(setup.training.density.learning_rate), 
					velocity_lr =			make_schedule(setup.training.velocity.learning_rate), 
					discriminator_lr =		make_schedule(setup.training.discriminator.learning_rate), 
					discriminator_regularization = make_schedule(setup.training.discriminator.regularization) )
				
				velocity_noise_schedule = make_schedule(setup.training.velocity.noise_std)
				def seq_vel_add_noise(opt_ctx, seq):
					vel_noise_std = velocity_noise_schedule(opt_ctx.iteration)
					if opt_ctx.LA(vel_noise_std):
						log.debug("Add noise to sequence velocity: std: %f, it: %d.", vel_noise_std, opt_ctx.iteration) #TODO debug
						for state in seq:
							v = state.velocity
							v.assign_add( \
								x = tf.random.normal([1]+v.x_shape+[1], stddev=vel_noise_std, dtype=tf.float32), \
								y = tf.random.normal([1]+v.y_shape+[1], stddev=vel_noise_std, dtype=tf.float32), \
								z = tf.random.normal([1]+v.z_shape+[1], stddev=vel_noise_std, dtype=tf.float32))
				with profiler.sample('Main Optimization'), tf.device(compute_device):
					# special interrupt handling only during training. to avoid saving corrupt states.
					disc_ctx.train = setup.training.discriminator.train and setup.training.discriminator.active
					inspect_gradients_list = {state.frame:{} for state in sequence}
					if setup.training.density.pre_opt.inspect_gradients==1:
						def ig_func(opt_ctx, gradients, name):
							if name.endswith('_x') or name.endswith('_y') or name.endswith('_z'):
								#velocity gradients are given individually per component with name=loss_name + _(x|y|z)
								c = ['x','y','z'].index(name[-1:])
								name = name[:-2]
								if name not in inspect_gradients_list[opt_ctx.frame]: inspect_gradients_list[opt_ctx.frame][name] = np.asarray([0,0,0], dtype=np.float32)
								inspect_gradients_list[opt_ctx.frame][name][c] = tf.reduce_mean(tf.abs(gradients)).numpy()
							else:
								inspect_gradients_list[opt_ctx.frame][name] = [tf.reduce_mean(tf.abs(gradients)).numpy()]
						iig_func = None
					if setup.training.density.pre_opt.inspect_gradients==2:# or True:
						AABB_corners_WS = []
						for state in sequence:
							dens_transform = state.get_density_transform()
							AABB_corners_WS += dens_transform.transform_AABB(*hull_AABB_OS(tf.squeeze(state.density.hull, (0,-1))), True)
						grad_cams = [main_camera.copy_clipped_to_world_coords(AABB_corners_WS)[0]]
						ig_func = lambda opt_ctx, gradients, name: render_gradients(gradients, dens_transform, grad_cams, grad_renderer, \
							path=os.path.join(setup.paths.data, "gradients", "d_f{:04d}_it{:08d}".format(opt_ctx.frame, opt_ctx.iteration)), \
							image_mask=name.replace("/", ".") + "_cam{:02}_{:04d}", name=name, log=log)
						iig_func = lambda opt_ctx, gradients, name: write_image_gradients(gradients, max_renderer, \
							path=os.path.join(setup.paths.data, "gradients", "d_f{:04d}_it{:08d}".format(opt_ctx.frame, opt_ctx.iteration)), \
							image_mask=name.replace("/", ".") + "_img_cam{:04}", image_neg_mask=name.replace("/", ".") + "_img-neg_cam{:04}")
					last_time = time.time()
					start_time = last_time
					
					warp_sequence = False
					fwd_warp_dens_clamp = setup.training.density.warp_clamp#'NONE'
					warp_sequence_schedule = make_schedule(setup.training.density.main_warp_fwd)
					
					for it in range(setup.training.iterations): #more like epochs...
						log.debug('Start iteration %d', it)
						#exclude first iteration from measurement as it includes some tf setup time
						if it==1: start_time = time.time()
						with profiler.sample('Optim Step', verbose = False):
						#	if d_scaled and "WARP" in setup.training.density.grow.pre_grow_actions:
						#		sequence.densities_advect_fwd(order=opt_ctx.warp_order, dt=opt_ctx.dt)
							d_scaled = scale_density(sequence, it, setup.training.density.grow.factor, setup.training.density.grow.intervals, base_shape=base_shape)
							v_scaled = scale_velocity(sequence, it, setup.training.velocity.grow.factor, setup.training.velocity.grow.scale_magnitude, setup.training.velocity.grow.intervals, base_shape=base_shape)
							if d_scaled and "WARP" in setup.training.density.grow.post_grow_actions: #
								log.info("Density set to forward warped after scaling.") #
								sequence.densities_advect_fwd(order=opt_ctx.warp_order, dt=opt_ctx.dt)
							#if setup.training.density.main_warp_fwd and it==0:
							if warp_sequence_schedule(it)!=warp_sequence:
								warp_sequence = warp_sequence_schedule(it)
								if warp_sequence:
									log.info("Density set to forward warped with clamp '%s' in iteration %d.", fwd_warp_dens_clamp, it) #
									sequence.densities_advect_fwd(order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=fwd_warp_dens_clamp)
								else:
									log.info("Density set to per-frame in iteration %d.", it) #
							
							summary_iteration = (it+1)%out_step==0 or (it+1)==setup.training.iterations or stop_training
							opt_ctx.start_iteration(it, compute_loss_summary=summary_iteration)
							
							seq_vel_add_noise(opt_ctx, sequence)
							
							if summary_iteration and setup.training.density.pre_opt.inspect_gradients:# or it==99 or it==200 or it==500:
								for g in inspect_gradients_list: inspect_gradients_list[g].clear()
								opt_ctx.set_inspect_gradient(True, ig_func, iig_func)
							disc_imgs = []
							if setup.training.frame_order=='FWD-BWD':
								loss_summaries = optStep_sequence(opt_ctx, sequence, disc_ctx, disc_samples_list=disc_imgs, order='FWD' if (it%2)==0 else 'BWD')
							elif setup.training.frame_order=='BWD-FWD':
								loss_summaries = optStep_sequence(opt_ctx, sequence, disc_ctx, disc_samples_list=disc_imgs, order='BWD' if (it%2)==0 else 'FWD')
							else:
								loss_summaries = optStep_sequence(opt_ctx, sequence, disc_ctx, disc_samples_list=disc_imgs, order=setup.training.frame_order)
							#print(summary_iteration, opt_ctx.compute_loss_summary(), opt_ctx._compute_loss_summary)
							opt_ctx.set_inspect_gradient(False)
							
							# DISCRIMINATOR
							disc_ctx.start_iteration(it, compute_loss_summary=summary_iteration)
							disc_ctx.opt_ctx.frame = None
							for disc_step in range(setup.training.discriminator.steps):
								disc_loss = optStep_discriminator(disc_ctx, state=None, additional_fake_samples=disc_imgs) #_real, disc_loss_fake, disc_scores_real, disc_scores_fake
							#END disc training
							del disc_imgs
							
							if warp_sequence:
								sequence.densities_advect_fwd(order=opt_ctx.warp_order, dt=opt_ctx.dt, clamp=fwd_warp_dens_clamp)
							
						if args.console and not args.debug:
							progress = it%out_step+1
							progress_bar(progress,out_step, "{:04d}/{:04d}".format(progress, out_step), length=50)
						if summary_iteration:
							log.info('--- Step {:04d}/{:04d} ---'.format(it, setup.training.iterations-1))
							print_loss_summary(loss_summaries, [], start_time, last_time, it, setup.training.iterations, \
								inspect_gradients_list if setup.training.density.pre_opt.inspect_gradients==1 else None)
							log.info("buoyancy: %s", opt_ctx.buoyancy.numpy())
							#d_max, d_min, d_mean = tf_print_stats(state.density.d, 'Density', log=log)
							if setup.training.light.optimize:
								log.info("Light intensities: %s", [_.numpy() for _ in light_var_list])
							if disc_ctx is not None and disc_ctx.train and setup.training.discriminator.start_delay<=it:
								print_disc_summary(disc_ctx, disc_loss)#_real, disc_scores_real, disc_loss_fake, disc_scores_fake)
							last_time = time.time()
							summary_writer.flush()
							check_loss_summary(loss_summaries, [], it, inspect_gradients_list if setup.training.density.pre_opt.inspect_gradients==1 else None, grad_max=5e-2)
						if val_cameras is not None and (it+1)%val_out_step==0:
							log.info("Render validation views %d in iteration %d", int((it+1)//val_out_step), it)
							try:
								render_sequence_val(sequence, z, int((it+1)//val_out_step))
							except:
								log.exception('Exception when rendering validation views %d for sequence in iteration %d:', int((it+1)//val_out_step), it)
						if stop_training:
							log.warning('Training stopped after %d iterations, saving state...', it+1)
							raise StopTraining
							break
						#iteration profiler
					#END for it in iterations (training loop)
				#optimization profiler
			#tf summary
			log.debug('Save sequence')
			sequence.save()
			if setup.training.discriminator.active:
				save_discriminator(disc_ctx.model, 'disc', setup.paths.data)
				if setup.training.discriminator.history.save:
					disc_ctx.history.serialize(setup.paths.data)
			
			# reset signal handling
			
		except StopTraining:
			log.warning('Optimization stopped after %s, saving state...', format_time(time.time() - optim_start))
			log.debug('Save sequence')
			sequence.save(suffix="part")
			if setup.training.discriminator.active:
				save_discriminator(disc_ctx.model, 'disc_part', setup.paths.data)
				if setup.training.discriminator.history.save:
					disc_ctx.history.serialize(setup.paths.data, 'part')
			
		# something unexpected happended. save state if possible and exit.
		except:
			log.exception('Exception during training. Attempting to save state...')
			try:
				summary_writer.close()
			except:
				log.error('Could not close summary writer', exc_info=True)
			if 'sequence' in locals():
				try:
					sequence.save(suffix="exc")
				except:
					log.error('Could not save sequence', exc_info=True)
			
			if 'disc_model' in locals():
				try:
					save_discriminator(disc_ctx.model, 'disc_exc', setup.paths.data)
					if setup.training.discriminator.history.save:
						disc_ctx.history.serialize(setup.paths.data, 'exc')
				except:
					log.exception('Could not save discriminator')
			try:
				with open(os.path.join(setup.paths.log, 'profiling.txt'), 'w') as f:
					profiler.stats(f)
			except:
				log.exception('Could not save profiling')
			faulthandler.disable()
			faultlog.close()
			sys.exit(1)
		else:
			log.info('Optimization finished after %s', format_time(time.time() - optim_start))
		finally:
			signal.signal(signal.SIGINT, signal.SIG_DFL)
		
		with open(os.path.join(setup.paths.log, 'profiling.txt'), 'w') as f:
			profiler.stats(f)
		faulthandler.disable()
		faultlog.close()
		
		scalar_results = munch.Munch()
		scalar_results.buoyancy = buoyancy.numpy().tolist() if setup.training.optimize_buoyancy else None
		scalar_results.light_intensity = [_.numpy().tolist() for _ in light_var_list] if setup.training.light.optimize else None
		final_transform = sim_transform.copy_no_data()
		final_transform.grid_size = sequence[0].density.shape
		scalar_results.sim_transform = final_transform
		
		with open(os.path.join(setup.paths.data, "scalar_results.json"), "w") as f:
			try:
				json.dump(scalar_results, f, default=tf_to_dict, sort_keys=True, indent=2)
			except:
				log.exception("Failed to write scalar_results:")
			
		
	if not args.fit: #args.render and 
		log.debug('Load data')
		if setup.data.load_sequence is None:
			raise ValueError("No sequence specified (setup.data.load_sequence)")
	#	load_path = run_index[setup.data.load_sequence]
		sf = RunIndex.parse_scalarFlow(setup.data.load_sequence)
		if sf is not None:
			log.info("Load scalarFlow sequence for evaluation, sim offset %d, frame offset %d", sf["sim"], sf["frame"])
			frames = list(range(setup.data.start+sf["frame"], setup.data.stop+sf["frame"], setup.data.step))
			vel_bounds = None if setup.data.velocity.boundary.upper()=='CLAMP' else Zeroset(-1, shape=GridShape(), outer_bounds="CLOSED", as_var=False, device=resource_device)
			#sim_transform = sF_transform
			with profiler.sample("load sF sequence"):
				if args.console:
					load_bar = ProgressBar(len(frames), name="Load Sequence: ")
					def update_pbar(step, frame):
						load_bar.update(step, desc="Frame {:03d} ({:03d}/{:03d})".format(frame, step+1, len(frames)))
				else: update_pbar = lambda i, f: None
				sequence = Sequence.from_scalarFlow_file(pFmt.format(setup.data.density.scalarFlow_reconstruction, sim=setup.data.simulation+sf["sim"]), \
					pFmt.format(setup.data.velocity.scalarFlow_reconstruction, sim=setup.data.simulation+sf["sim"]), \
					frames, transform=sim_transform, #sF_transform,
					as_var=False, base_path=setup.paths.data, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, frame_callback=update_pbar)
				if args.console: load_bar.finish(desc="Done")
			frames = list(range(setup.data.start, setup.data.stop, setup.data.step))
			for s,f in zip(sequence, frames):
				s.target_cameras = setup_target_cameras(target_cameras, train_cam_resolution, None, setup.rendering.target_cameras.crop_frustum_pad)
				s.velocity.scale_magnitude(setup.data.step)
				s.density.scale(setup.data.density.scale)
				s.frame = f
		else:
			load_entry = run_index.get_run_entry(setup.data.load_sequence)
			log.info("Load sequence from '%s' for evaluation", load_entry.path)
			try:
				load_setup = munch.munchify(load_entry.setup)
				vel_bounds = None if setup.data.velocity.boundary.upper()=='CLAMP' else Zeroset(-1, shape=GridShape(), outer_bounds="CLOSED", as_var=False, device=resource_device)
				try:
					vel_bounds = None if load_setup.data.velocity.boundary.upper()=='CLAMP' else Zeroset(-1, shape=GridShape(), outer_bounds="CLOSED", as_var=False, device=resource_device)
				except:
					log.info("Using default boundaries: %s", vel_bounds)
			except:
				log.exception("failed to load config from %s:", load_entry.path)
				sys.exit(1)
			frames = list(range(setup.data.start, setup.data.stop, setup.data.step))
			orig_frames = list(range(load_setup.data.start, load_setup.data.stop, load_setup.data.step))
			
			if setup.data.step!=load_setup.data.step:
				log.info("Loaded frame step does not match data, scaling velocity with %f", setup.data.step/load_setup.data.step)
			
			try:
				load_scalars = load_entry.scalars
				t = from_dict(load_scalars["sim_transform"])
			except:
				log.exception("Failed to load transformation, using default.")
			else:
				sim_transform = t
			
			for f in frames:
				if f not in orig_frames:
					raise ValueError("Frame %d not in available frames"%f)
			with profiler.sample("load sequence"):
				if args.console:
					load_bar = ProgressBar(len(frames), name="Load Sequence: ")
					def update_pbar(step, frame):
						load_bar.update(step, desc="Frame {:03d} ({:03d}/{:03d})".format(frame, step+1, len(frames)))
				else: update_pbar = lambda i, f: None
				if not setup.data.load_sequence_pre_opt:
					sequence = Sequence.from_file(load_entry.path, frames, transform=sim_transform, as_var=False, base_path=setup.paths.data, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, frame_callback=update_pbar) # load_sequence(load_path, frames, as_var=False, base_path=setup.paths.data)
				else:
					log.info("Load pre-optimization result")
					sequence = Sequence.from_file(load_entry.path, frames, transform=sim_transform, as_var=False, base_path=setup.paths.data, boundary=vel_bounds, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device, density_filename="density_pre-opt.npz", velocity_filename="velocity_pre-opt.npz", frame_callback=update_pbar)
				if args.console: load_bar.finish(desc="Done")
				for s in sequence:
					s.velocity.scale_magnitude(setup.data.step/load_setup.data.step)
					s.density.scale(setup.data.density.scale)
			
			def load_targets_from_run(frame, cam_ids=[0,1,2,3,4]):
				targets = munch.Munch()
				targets.targets_raw = []
				targets.targets = []
				targets.bkgs = []
				#target.image_hulls = []
				frame_relpath = os.path.relpath(load_entry.frame_path(frame), load_entry.path)
				def read_image_from_entry(name, flip_y=True):
					with load_entry._open_file(os.path.join(frame_relpath, name), mode="rb") as f:
						tmp_img = imageio.imread(f, format='EXR-FI')
						if len(tmp_img.shape)==2:
							tmp_img = np.reshape(tmp_img, list(tmp_img.shape) + [1])
						if flip_y:
							tmp_img = np.flip(tmp_img, axis=0)
					return tmp_img
				
				try:
					for cam_id in cam_ids:
						targets.targets_raw.append(read_image_from_entry("target_raw_base_cam{}.exr".format(cam_id)))
						targets.targets.append(read_image_from_entry("target_base_cam{}.exr".format(cam_id)))
						targets.bkgs.append(read_image_from_entry("bkg_base_cam{}.exr".format(cam_id)))
				except IOError as e:
					log.error("Failed to load targets for cam %d of frame %d: %s", cam_id, frame, e)
				return targets
			
			log.debug('Load run targets')
			#sequence_targets = {}
			setup_targets = (args.render and setup.validation.render_target) or setup.validation.stats
			for state in sequence:
				frame = state.frame
				
				state.target_cameras = setup_target_cameras(target_cameras, train_cam_resolution, None, setup.rendering.target_cameras.crop_frustum_pad)
				
				if setup_targets:
					state_targets = load_targets_from_run(frame, range(len(state.target_cameras)))#frame_preSetup(frame, sim_transform)
				
				with tf.device(resource_device):
					if setup_targets:
						state.targets_raw = tf.identity(state_targets.targets_raw)
						state.targets = tf.identity(state_targets.targets)
						state.bkgs = tf.identity(state_targets.bkgs)
					try:
						with load_entry._open_file("frame_{:06d}/volume_hull_tight.npz".format(frame), mode="rb") as hull_file:
							state.hull = tf.constant(load_numpy(hull_file), dtype=tf.float32)
					except OSError as e:
						log.warning("Failed to load tight hull for frame %d: %s", frame, e)
		
		vel_shape = sequence[0].velocity.centered_shape
		z = tf.zeros([1] + vel_shape + [1])
		
	#	vel_scale = world_scale(vel_shape, width=1.)
	
	if True: #args.render or args.fit:
		
		def print_stats_dict(stats, name, print_fn):
			s = '{}:\n'.format(name)
			for name in sorted(stats.keys()):
				value = stats[name]
				if isinstance(value, tf.Tensor):
					value = value.numpy().tolist()
				if not isinstance(value, float):
					s += '{:<16}: {}\n'.format(name, value)
				else:
					s += '{:<16}: {: 13.06e}\n'.format(name, value)
			print_fn(s)
			
		def render_sequence_cmp(*sequences, cameras, path, name_pre='seq', image_format='PNG', render_velocity=True, background="COLOR", crop_cameras=True):
			#sequences: iterables of states to render or lists with images for every camera.
			assert len(sequences)>1, "need at least 2 sequences to compare"
			length = len(sequences[0])
			for sequence in sequences:
				assert len(sequence)==length, "All sequences must have equal length"
			log.debug("Render comparison of %d sequences:", len(sequences))
			# render image cmp
			AABB_corners_WS = []
			for states in zip(*sequences):
				for state in states:
					if isinstance(state, State):
						dens_hull = state.density.hull #state.hull if hasattr(state, "hull") else 
						if dens_hull is None:
							continue
						dens_transform = state.get_density_transform()
						AABB_corners_WS += dens_transform.transform_AABB(*hull_AABB_OS(tf.squeeze(dens_hull, (0,-1))), True)
			if AABB_corners_WS and crop_cameras:
				seq_cams = [cam.copy_clipped_to_world_coords(AABB_corners_WS)[0] for cam in cameras]
			else:
				seq_cams = cameras
			split_cams = True
			i=0
			for states in zip(*sequences):
				log.debug("Render sequence cmp frame %d", i)
				# density: [orig, dens_warp, veldens_warp]
				if args.console: progress_bar(i*2,len(sequences[0])*2, "Step {:03d}/{:03d}: {:30}".format(i+1,len(sequence), "Sequence cmp Density"), length=30)
				bkg_render = None
				if background=='COLOR':
					bkg_render = [tf.constant(setup.rendering.background.color, dtype=tf.float32)]*len(seq_cams)
				if isinstance(background, list):
					bkg_render = background[i]
				if isinstance(background, (np.ndarray, tf.Tensor)):
					bkg_render = background
				#sim_transform.set_data(state.density)
				dens_imgs = []
				for state in states:
					if isinstance(state, State):
						dens_imgs.append(tf.concat(renderer.render_density(state.get_density_transform(), lights, seq_cams, background=bkg_render, split_cameras=split_cams), axis=0))
					elif isinstance(state, (list, tuple)):
						state = tf.concat(state, axis=0)
					if isinstance(state, (np.ndarray, tf.Tensor)):
						state_shape = shape_list(state)
						if len(state_shape)!=4 or state_shape[0]!=len(seq_cams):
							raise ValueError
						if state_shape[-1]==1:
							state = tf.tile(state, (1,1,1,3))
						dens_imgs.append(tf.identity(state))
					
				renderer.write_images([tf.concat(dens_imgs, axis=-2)], [name_pre + '_cmp_dens_cam{}_{:04d}'], base_path=path, use_batch_id=True, frame_id=i, format=image_format)
				
				# velocity: [orig, veldens_warp]
				if render_velocity:
					vel_imgs = []
					if args.console: progress_bar(i*2+1,len(sequence)*2, "Step {:03d}/{:03d}: {:30}".format(i+1,len(sequence), "Sequence cmp Velocity"), length=30)
					for state in states:
						if isinstance(state, State):
							vel_transform = state.get_velocity_transform()
							vel_scale = vel_transform.cell_size_world().value 
							log.debug("Render velocity frame %d with cell size %s", i, vel_scale)
							vel_centered = state.velocity.centered()*vel_scale/float(setup.data.step)*setup.rendering.velocity_scale
							vel_imgs.append(tf.concat(vel_renderer.render_density(vel_transform, [tf.abs(vel_centered)], cameras, split_cameras=split_cams), axis=0))
					vel_renderer.write_images([tf.concat(vel_imgs, axis=-2)], [name_pre + '_cmp_velA_cam{}_{:04d}'], base_path=path, use_batch_id=True, frame_id=i, format=image_format)
				
				i+=1
		
		sF_sequence = []
		def get_frame_stats(state, mask=None, cmp_scalarFlow=False):
			stats = {}
			try:
				with profiler.sample("stats"):
					dens_stats, vel_stats, tar_stats = state.stats(render_ctx=None if not args.fit else main_render_ctx, dt=1.0, order=setup.training.velocity.warp_order, clamp=setup.training.density.warp_clamp)#vel_scale)
					vel_stats['scale'] = world_scale(state.velocity.centered_shape, width=1.)
					if mask is not None:
						dens_hull_stats, vel_hull_stats, _ = state.stats(mask=mask, dt=1.0, order=setup.training.velocity.warp_order, clamp=setup.training.density.warp_clamp)
						vel_hull_stats['scale'] = world_scale(state.velocity.centered_shape, width=1.)
					
				if cmp_scalarFlow: #setup.validation.cmp_scalarFlow
					#compare to scalarFlow
					#problem: unknown offset between raw and preprocessed/reconstructed frame indices
					#	get max frame for both and use difference, assuming the end frame is the same
					#	constant 11?
					offset=setup.data.scalarFlow_frame_offset #-11
					#load sF density, scale to current grid size, normalize to current mean (i.e. same total density), report MSE
					try:
						sF_d = DensityGrid.from_scalarFlow_file(setup.data.density.scalarFlow_reconstruction.format(sim=setup.data.simulation, frame=state.frame+offset), \
							as_var=False, scale_renderer=scale_renderer, device=resource_device)
						#load sF velocity, scale to size and make staggered, report mean length of vector differences
						sF_v = VelocityGrid.from_scalarFlow_file(setup.data.velocity.scalarFlow_reconstruction.format(sim=setup.data.simulation, frame=state.frame+offset), boundary=vel_bounds, \
							as_var=False, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device)
					except:
						log.exception("Failed to load scalarFlow reconstruction of frame %d (%d) for statistics.", state.frame, state.frame+offset)
					else:
						# (IF clip and crop grid) -> always: use base SF transform to sample SF data to current grid
						dens_transform = state.get_density_transform()
						sF_d = DensityGrid(state.density.shape, d=tf.squeeze(scale_renderer._sample_transform(sF_d.d, [sF_transform], [dens_transform], fix_scale_center=True), (0,)), as_var=False, scale_renderer=scale_renderer, device=resource_device)
						
						# corrections for staggered grids needed
						vel_transform = state.get_velocity_transform()
						sF_v_x_transform = GridTransform(sF_v.x_shape, translation=[-.5, 0., 0.], parent=sF_transform)
						vel_x_transform = GridTransform(state.velocity.x_shape, translation=[-.5, 0., 0.], parent=vel_transform)
						sF_v_y_transform = GridTransform(sF_v.y_shape, translation=[0., -.5, 0.], parent=sF_transform)
						vel_y_transform = GridTransform(state.velocity.y_shape, translation=[0., -.5, 0.], parent=vel_transform)
						sF_v_z_transform = GridTransform(sF_v.z_shape, translation=[0., 0., -.5], parent=sF_transform)
						vel_z_transform = GridTransform(state.velocity.z_shape, translation=[0., 0., -.5], parent=vel_transform)
						sF_v = VelocityGrid(state.velocity.centered_shape, \
							x=tf.squeeze(scale_renderer._sample_transform(sF_v.x, [sF_v_x_transform], [vel_x_transform], fix_scale_center=True), (0,)), \
							y=tf.squeeze(scale_renderer._sample_transform(sF_v.y, [sF_v_y_transform], [vel_y_transform], fix_scale_center=True), (0,)), \
							z=tf.squeeze(scale_renderer._sample_transform(sF_v.z, [sF_v_z_transform], [vel_z_transform], fix_scale_center=True), (0,)), \
							boundary=sF_v.boundary, as_var=False, scale_renderer=scale_renderer, warp_renderer=warp_renderer, device=resource_device)
						
						#sF_d.scale(dens_stats['dMean']/sF_d.mean())
						sF_v.scale_magnitude(setup.data.step) # account for larger step sizes
						
						sF_state = State(sF_d, sF_v, frame=state.frame, transform=dens_transform.copy_no_data())
						if setup.validation.cmp_scalarFlow_render:
							sF_sequence.append(sF_state)
						#sF_state.target_cameras = setup_target_cameras(scalarFlow_cameras, train_cam_resolution, None, setup.rendering.target_cameras.crop_frustum_pad)
						sF_dens_stats, sF_vel_stats, sF_tar_stats = sF_state.stats(dt=1.0, order=setup.training.velocity.warp_order, clamp=setup.training.density.warp_clamp)#render_ctx=main_render_ctx)
						if mask is not None:
							sF_dens_hull_stats, sF_vel_hull_stats, _ = sF_state.stats(mask=mask, dt=1.0, order=setup.training.velocity.warp_order, clamp=setup.training.density.warp_clamp)
						dens_SE = (sF_state.density.d - state.density.d)**2
						vel_diffMag = (sF_state.velocity - state.velocity).magnitude()
						
						# esxclude any zero (small) velocities, as there is no valid angle
						vel_CangleRad_mask = tf.greater(sF_state.velocity.magnitude() * state.velocity.magnitude(), 1e-8)
						vel_CangleRad = tf_angle_between(sF_state.velocity.centered(), state.velocity.centered(), axis=-1, keepdims=True)
						
						if mask.dtype!=tf.bool:
							mask = tf.not_equal(mask, 0)
						
						dens_stats['_sF_SE'] = tf_tensor_stats(dens_SE, as_dict=True)
						vel_stats['_sF_vdiff_mag'] = tf_tensor_stats(vel_diffMag, as_dict=True)
						vel_stats['_sF_angleCM_rad'] = tf_tensor_stats(tf.boolean_mask(vel_CangleRad, vel_CangleRad_mask), as_dict=True)
						if mask is not None:
							dens_hull_stats['_sF_SE'] = tf_tensor_stats(tf.boolean_mask(dens_SE, mask), as_dict=True)
							vel_hull_stats['_sF_vdiff_mag'] = tf_tensor_stats(tf.boolean_mask(vel_diffMag, mask), as_dict=True)
							vel_hull_stats['_sF_angleCM_rad'] = tf_tensor_stats(tf.boolean_mask(vel_CangleRad, tf.logical_and(mask, vel_CangleRad_mask)), as_dict=True)
						
						stats["sF_density"]=sF_dens_stats
						stats["sF_velocity"]=sF_vel_stats
						#stats["sF_target"]=sF_tar_stats
						if mask is not None:
							stats["sF_density_hull"]=sF_dens_hull_stats
							stats["sF_velocity_hull"]=sF_vel_hull_stats
						
				
				stats["density"]=dens_stats
				stats["velocity"]=vel_stats
				stats["target"]=tar_stats
				if mask is not None:
					stats["density_hull"]=dens_hull_stats
					stats["velocity_hull"]=vel_hull_stats
			except:
				log.exception("Exception during reconstruction stats of frame %d", state.frame)
			return stats
		
		if setup.validation.stats:
			log.info("Data Statistics")
			stats_file = os.path.join(setup.paths.log, "stats.json")
			stats_dict = {}
			frame_keys = []
			for state in sequence:
				frame_key = "{:04d}".format(state.frame)
				frame_keys.append(frame_key)
				stats_mask = state.density.hull #tf.greater(state.density.hull, 0.5)
				stats_dict[frame_key] = get_frame_stats(state=state, mask=state.density.hull, cmp_scalarFlow=setup.validation.cmp_scalarFlow)
			# TODO seqence stats
			# mean and std of sF errors
			if setup.validation.cmp_scalarFlow:
				stats_dict["sequence"]={"density":{'_sF_SE':{
					"mean":tf.reduce_mean([stats_dict[k]["density"]['_sF_SE']["mean"] for k in frame_keys]),
					"std":tf_reduce_std([stats_dict[k]["density"]['_sF_SE']["mean"] for k in frame_keys]),
				}}}
				stats_dict["sequence"]["velocity"]={'_sF_vdiff_mag': {
					"mean":tf.reduce_mean([stats_dict[k]["velocity"]['_sF_vdiff_mag']["mean"] for k in frame_keys]),
					"std":tf_reduce_std([stats_dict[k]["velocity"]['_sF_vdiff_mag']["mean"] for k in frame_keys]),
			#	}, '_sF_anglediff_rad'
				}}
				if all(["density_hull" in stats_dict[_] for _ in frame_keys]): #all have hull stats available
					stats_dict["sequence"]["density_hull"]={'_sF_SE': {
						"mean":tf.reduce_mean([stats_dict[k]["density_hull"]['_sF_SE']["mean"] for k in frame_keys]),
						"std":tf_reduce_std([stats_dict[k]["density_hull"]['_sF_SE']["mean"] for k in frame_keys]),
					}}
				if all(["velocity_hull" in stats_dict[_] for _ in frame_keys]): #all have hull stats available
					stats_dict["sequence"]["velocity_hull"]={'_sF_vdiff_mag': {
						"mean":tf.reduce_mean([stats_dict[k]["velocity_hull"]['_sF_vdiff_mag']["mean"] for k in frame_keys]),
						"std":tf_reduce_std([stats_dict[k]["velocity_hull"]['_sF_vdiff_mag']["mean"] for k in frame_keys]),
					}}
				
				if setup.validation.cmp_scalarFlow_render and len(sequence)==len(sF_sequence):
					log.info("Render ScalarFlow comparison.")
					try:
						seq_cmp_path = os.path.join(setup.paths.data, "ScalarFlow_cmp")
						os.makedirs(seq_cmp_path, exist_ok=True)
						render_sequence_cmp(sequence, sF_sequence, cameras=cameras, path=seq_cmp_path, name_pre='sF-seq')
						#render_sequence_cmp(sequence, sF_sequence, cameras=scalarFlow_cameras, path=seq_cmp_path, name_pre='sF-cam')
						render_sequence_cmp([state.targets_raw for state in sequence], sequence, sF_sequence, \
							cameras=target_cameras, path=seq_cmp_path, name_pre='sF-tar', render_velocity=False, background=[state.bkgs for state in sequence], crop_cameras=False)
					except:
						log.exception("ScalarFlow compare render exception:")
			del sF_sequence
			
			try:
				json_dump(stats_file, stats_dict, compressed=True, default=tf_to_dict, sort_keys=True)
			except:
				log.exception("Failed to write stats:")
		
		if setup.validation.warp_test:
			log.info("Warp Test")
			with profiler.sample("warp test"):
				# no clamping used here to see error introduced by the method
				# only density warped
				warp_test_clamp_dens = 'NONE' if setup.training.density.warp_clamp=='NEGATIVE' else setup.training.density.warp_clamp
				
				
				def make_warp_test_inflow():
					# XYZ
					if_offset = Int3(25,14,35)
					if_size = Int3(40,10,40)
					if_shape = GridShape(if_size) #1DHW1
					test_inflow = tf.constant(20., shape=if_shape.as_shape, dtype=tf.float32)
					dens_shape = GridShape.from_tensor(sequence[0].density.d)
					padding_before = GridShape((0,*if_offset.as_shape,0))
					padding_after = dens_shape - if_shape - padding_before
					padding = [_ for _ in zip(padding_before, padding_after)]
					test_inflow = tf.pad(test_inflow, padding)
					return test_inflow
				
				vel_bounds = None
				def tmp_get_test_bounds(shape):
					offset = shape.value //4
					offset[0] = 0
					offset[-1] = 0
					pad = list(zip(offset, offset))
					size = shape.value - 2*offset
					levelset = tf.pad(tf.ones(size, dtype=tf.float32), pad, constant_values=-1)
					return Zeroset(levelset, outer_bounds="OPEN", as_var=False, device=resource_device)
				vel_bounds = tmp_get_test_bounds(GridShape.from_tensor(sequence[0].density._d))
				
				sequence_warped_dens_noIF = sequence.copy(as_var=False, device=resource_device)
				sequence_warped_dens_noIF[0].density.assign(make_warp_test_inflow())
				for s in sequence_warped_dens_noIF:
					s.density.restrict_to_hull=False
					s.density._inflow=None
					s.velocity.set_boundary(vel_bounds)
				sequence_warped_dens_noIF.densities_advect_fwd(order=setup.training.velocity.warp_order, clamp=warp_test_clamp_dens)
				
			#	sequence_warped_dens_noD = sequence.copy(as_var=False, device=resource_device)
			#	for s in sequence_warped_dens_noD:
			#		s.density.restrict_to_hull=False
			#		s.density.assign(tf.zeros_like(s.density._d))
			#	sequence_warped_dens_noD.densities_advect_fwd(order=setup.training.velocity.warp_order, clamp=warp_test_clamp_dens)
				
				# velocity and density warped
				sequence_warped_veldens = sequence.copy(as_var=False, device=resource_device)
				for s in sequence_warped_veldens:
					s.density.restrict_to_hull=False
					s.velocity.set_boundary(vel_bounds)
				sequence_warped_veldens.velocities_advect_fwd(order=setup.training.velocity.warp_order, clamp=setup.training.velocity.warp_clamp)
				sequence_warped_veldens.densities_advect_fwd(order=setup.training.velocity.warp_order, clamp=warp_test_clamp_dens)
				
				# warp errors
				if False:
					warp_errors = {}
					we_tmp = []
					for i in range(len(sequence)):
						state = sequence[i]
						state_d = sequence_warped_dens[i]
						state_vd = sequence_warped_veldens[i]
						e = {
							"dens_warp_MSE": tf.reduce_mean(tf.math.squared_difference(state.density.d, state_d.density.d)),
							"dens_vel_warp_MSE": tf.reduce_mean(tf.math.squared_difference(state.density.d, state_vd.density.d)),
							"velMag_warp_MSE": tf.reduce_mean(tf.math.squared_difference(state.velocity.magnitude(), state_vd.velocity.magnitude())),
							"velX_warp_MSE": tf.reduce_mean(tf.math.squared_difference(state.velocity.x, state_vd.velocity.x)),
							"velY_warp_MSE": tf.reduce_mean(tf.math.squared_difference(state.velocity.y, state_vd.velocity.y)),
							"velZ_warp_MSE": tf.reduce_mean(tf.math.squared_difference(state.velocity.z, state_vd.velocity.z)),
							"velDiv_warp_absMean": tf.reduce_mean(tf.abs(state_vd.velocity.divergence())),
						}
						e["warp_dens"] = get_frame_stats(state_d, mask=state_d.density.hull)
						e["warp_vel-dens"] = get_frame_stats(state_vd, mask=state_d.density.hull)
						
						warp_errors["{:04d}".format(state.frame)]=e
						we_tmp.append(e)
					# seqence stats, mean and std of errors
					warp_errors["sequence"] = {
						_:{ \
							"mean":tf.reduce_mean([f[_] for f in we_tmp]), \
							"std":tf_reduce_std([f[_] for f in we_tmp]) \
						} for _ in we_tmp[0] if _ not in ["warp_dens", "warp_vel-dens"]\
					}
					#with open(os.path.join(setup.paths.warp_test, "warp_error.json"), "w") as file:
					#	json.dump(warp_errors, file, default=tf_to_dict, sort_keys=True, indent=2)
					json_dump(os.path.join(setup.paths.warp_test, "warp_error.json"), warp_errors, compressed=True, default=tf_to_dict, sort_keys=True)
				
				# render warp image cmp
				if setup.validation.warp_test_render:
				#	AABB_corners_WS = []
				#	for state in sequence:
				#		dens_transform = state.get_density_transform()
				#		dens_hull = state.density.hull #state.hull if hasattr(state, "hull") else 
				#		if dens_hull is None:
				#			continue
				#		AABB_corners_WS += dens_transform.transform_AABB(*hull_AABB_OS(tf.squeeze(dens_hull, (0,-1))), True)
				#		del dens_hull
				#	if AABB_corners_WS:
				#		seq_cams = [cam.copy_clipped_to_world_coords(AABB_corners_WS)[0] for cam in cameras]
				#	else:
					seq_cams = cameras[:2]
					# synth:
					dens_grads = ("viridis", 0., 1.1) #np.percentile(state.density.d.numpy(), 95))
					#dens_grads = ("viridis", 0., 5.5) #np.percentile(state.density.d.numpy(), 95))
					# sF real:
					#dens_grads = ("viridis", 0., 2.5) #np.percentile(state.density.d.numpy(), 95))
					split_cams = True
					i=0
					if args.console: render_bar = ProgressBar(len(sequence)*2,  name="Render warp test: ")
					#for state, state_d, state_noIF, state_noD, state_vd in zip(sequence,sequence_warped_dens,sequence_warped_dens_noIF,sequence_warped_dens_noD,sequence_warped_veldens):
					for state, state_noIF, state_vd in zip(sequence,sequence_warped_dens_noIF,sequence_warped_veldens):
					#for state in sequence_warped_dens_noIF:
						log.debug("Render warp cmp frame %d (%d)", state.frame, i)
						# density: [orig, dens_warp, veldens_warp]
						if args.console: render_bar.update(i*2, desc="Frame {:03d} ({:03d}/{:03d}): {:30}".format(state.frame, i+1,len(sequence), "Warp cmp Density"))# progress_bar(i*2,len(sequence)*2, , length=30)
						
						tmp_transform = state_noIF.get_density_transform()
						tmp_transform.set_data(tf.zeros_like(state_noIF.density.d))
						imgs = tf.concat(vel_renderer.render_density(tmp_transform, [tf.concat([state_noIF.density.d]*3, axis=-1)], seq_cams, background=None, split_cameras=False), axis=0)
						#render_cycle(tmp_transform, [main_camera], [tf.concat([state_d.density.d]*3, axis=-1)], vel_renderer, state.data_path, steps=cycle_steps, steps_per_cycle=cycle_steps, name_pre='dens', img_transfer=dens_grads, img_stats=False, format="PNG")
						imgs = tf_cmap_nearest(imgs, *dens_grads)
						renderer.write_images([imgs], ['warp_dens_cam{}_{:04d}'], base_path=setup.paths.warp_test, use_batch_id=True, frame_id=i, format='PNG')
						
						# velocity: [orig, veldens_warp]
						if True:
							vel_transform = state.get_velocity_transform()
							vel_scale = vel_transform.cell_size_world().value 
							log.debug("Render velocity frame %d (%d) with cell size %s", state.frame, i, vel_scale)
							if args.console: render_bar.update(i*2, desc="Frame {:03d} ({:03d}/{:03d}): {:30}".format(state.frame, i+1,len(sequence), "Warp cmp Velocity"))# progress_bar(i*2+1,len(sequence)*2, , length=30)
							vel_render_scale = vel_scale/float(setup.data.step)*setup.rendering.velocity_scale
							vel_imgs = [
								tf.concat(vel_renderer.render_density(vel_transform, [tf.abs(state.velocity.centered()*vel_render_scale)], seq_cams, split_cameras=split_cams), axis=0),
								tf.concat(vel_renderer.render_density(state_vd.get_velocity_transform(), [tf.abs(state_vd.velocity.centered()*vel_render_scale)], seq_cams, split_cameras=split_cams), axis=0),
							]
							vel_renderer.write_images([tf.concat(vel_imgs, axis=-2)], ['warp_velA_cam{}_{:04d}'], base_path=setup.paths.warp_test, use_batch_id=True, frame_id=i, format='PNG')
						
						i+=1
					if args.console: render_bar.finish("Done")
			#del sequence_warped_dens
			del sequence_warped_dens_noIF
			#del sequence_warped_dens_noD
			del sequence_warped_veldens
		
	if args.render:
		try:
			log.info('Render final output.')
			render_sequence(sequence, z, cycle=setup.validation.render_cycle, cycle_steps=setup.validation.render_cycle_steps, \
				sF_cam=setup.validation.render_target, \
				render_density=setup.validation.render_density, render_shadow=setup.validation.render_shadow, \
				render_velocity=setup.validation.render_velocity)
		except KeyboardInterrupt:
			log.warning("Interrupted final output rendering.")
		except:
			log.exception("Error during final output rendering:")
			
		
		
		#render_sequence(sequence, vel_scale, z, cycle=True, cycle_steps=12, sF_cam=True)
			
		
	with open(os.path.join(setup.paths.log, 'profiling.txt'), 'w') as f:
		profiler.stats(f)
	#profiler.stats()
	logging.shutdown()
	sys.exit(0)