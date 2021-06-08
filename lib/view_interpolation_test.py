import os, sys, shutil, copy #, argparse, re
import cv2
import numpy as np

OPTFLOW_METHOD = "Brox" # Farneback, Brox, TODO: Horn-Schunck
USE_RENDER = True
CUDA_ID = "3"

if USE_RENDER:
	if __name__=="__main__":
		if CUDA_ID is None:
			from tools.GPUmonitor import getAvailableGPU
			gpu_available = getAvailableGPU() #active_mem_threshold=0.05
			if gpu_available:
				CUDA_ID = str(gpu_available[0])
			if CUDA_ID is None:
				print('No GPU available.')
				sys.exit(4)
		os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_ID
	
	import tensorflow as tf
	if __name__=="__main__":
		tf.enable_eager_execution()
	from phitest.render import *
	import json

# image format: RGBA, FP16, [0,1]


if OPTFLOW_METHOD=="Brox":
	sys.path.append("path/to/pyflow")
	import pyflow

def img_to_UINT8(img):
	if img.dtype==np.float64 or img.dtype==np.float32:
		return (img*255.).astype(np.uint8)
	elif img.dtype==np.uint8:
		return img
	else:
		raise TypeError("Unknown image type %s"%img.dtype)

def img_to_FP32(img):
	if img.dtype==np.float64:
		return img.astype(np.float32)
	elif img.dtype==np.float32:
		return img
	elif img.dtype==np.uint8:
		return img.astype(np.float32) / 255.
	else:
		raise TypeError("Unknown image type %s"%img.dtype)

def img_to_FP64(img):
	if img.dtype==np.float64:
		return img
	elif img.dtype==np.float32:
		return img.astype(np.float64)
	elif img.dtype==np.uint8:
		return img.astype(np.float64) / 255.
	else:
		raise TypeError("Unknown image type %s"%img.dtype)

scalarFlow_path_mask = 'data/ScalarFlow/sim_{sim:06d}/input/cam/imgsUnproc_{frame:06d}.npz'
def load_scalarFlow_images(sim, frame, cams=[0,1,2,3,4]):
	path = scalarFlow_path_mask.format(sim=sim, frame=frame)
	with np.load(path) as np_data:
		images = np_data["data"]
	for image in images:
		print("loaded image stats:", np.amin(image), np.mean(image), np.amax(image), image.dtype)
	images = [np.array(np.flip(normalize_image_shape(img_to_FP32(images[_]), "GRAY"), axis=0)) for _ in cams]
	return images

def normalize_image_shape(image, out_format="RGB"):
	image_rank = len(image.shape)
	assert image_rank<4
	
	if image_rank==2:
		image = image[..., np.newaxis]
	image_channels = image.shape[-1]
	
	if image_channels==1 and out_format=="RGB":
		image = np.repeat(image, 3, axis=-1)
	elif image_channels>1 and out_format=="GRAY":
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
		
	return image

def get_max_mip_level(image):
	assert len(image.shape)==3
	min_res = min(image.shape[:-1])
	return np.log2(min_res).astype(np.int32)

def write_images(path_mask, images):
	for i, image in enumerate(images):
		path = path_mask.format(frame=i)
		print("write image with shape", image.shape, "to", path)
		image = img_to_UINT8(image)
		if image.shape[-1]==3:
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		if image.shape[-1]==4:
			image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
		cv2.imwrite(path, image)

def flow_to_image(flow, mode="MAG_ANG"):
	flow = flow.astype(np.float32)
	print("flow stats: ", np.amin(flow), np.mean(flow), np.amax(flow))
	
	if mode=="ABS_NORM":
		flow = np.abs(flow)
		flow /= np.amax(flow)
		flow = np.pad(flow, ((0,0),(0,0),(0,1)))
	if mode=="MAG_ANG":
		hsv = np.ones(list(flow.shape[:-1])+[3], dtype=np.uint8)*255
		mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
		hsv[..., 0] = ang * (180 / (2*np.pi))
		hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
		flow = img_to_FP32(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
	
	print("flow image stats: ", np.amin(flow), np.mean(flow), np.amax(flow))
	return flow

alpha = 0.045 # 0.012
window = 30 # 20
def get_dense_optical_flow(image1, image2):
	"""compute a vector field indicating how each position in image1 flows into image2?
	"""
	if image1.shape[-1]>1:
		image1 = cv2.cvtColor(image1, cv2.COLOR_RGBA2GRAY)[...,np.newaxis]
	if image2.shape[-1]>1:
		image2 = cv2.cvtColor(image2, cv2.COLOR_RGBA2GRAY)[...,np.newaxis]
	
	if OPTFLOW_METHOD=="Farneback":
		flow = cv2.calcOpticalFlowFarneback(img_to_UINT8(image1), img_to_UINT8(image2), None, \
			pyr_scale=0.5, levels=15, #min(get_max_mip_level(image1), get_max_mip_level(image2)), \
			winsize=11, iterations=5, poly_n=5, poly_sigma=1.1, flags=0)
	
	elif OPTFLOW_METHOD=="Brox":
		u, v, _ = pyflow.coarse2fine_flow(
			img_to_FP64(image1), img_to_FP64(image2), alpha=alpha, ratio=0.875, minWidth=window, nOuterFPIterations=12, nInnerFPIterations=1,
			nSORIterations=40, colType=1) #alpha 0.012
		flow = np.concatenate((u[...,np.newaxis], v[...,np.newaxis]), axis=-1).astype(np.float32)
	
	#print("flow_shape", flow.shape, "dtype", flow.dtype)
	return flow

def warp_image(image, flow):
	"""
	flow: inverse relative lookup position for every pixel in the output
	"""
	flow = -flow
	flow[...,0] += np.arange(flow.shape[1])
	flow[...,1] += np.arange(flow.shape[0])[:, np.newaxis]
	return cv2.remap(image, flow, None, cv2.INTER_LINEAR)

def lerp_image(image1, image2, t, optical_flow=None):
	if optical_flow is None:
		optical_flow = get_dense_optical_flow(image1, image2)
	
	i1_warp = warp_image(image1, optical_flow*(t))
	i2_warp = warp_image(image2, optical_flow*(-(1.-t)))
	
	return i1_warp*(1.-t) + i2_warp*t

def lerp_image_2(image1, image2, t, optical_flow1=None, optical_flow2=None):
	if optical_flow1 is None:
		optical_flow1 = get_dense_optical_flow(image1, image2)
	if optical_flow2 is None:
		optical_flow2 = get_dense_optical_flow(image2, image1)
	
	i1_warp = warp_image(image1, optical_flow1*(t))
	i2_warp = warp_image(image2, optical_flow2*(1.-t))
	
	return i1_warp*(1.-t) + i2_warp*t

def lerp(a, b, t):
	return (1-t)*a + t*b

def lerp_vector(v1, v2, t):
	return lerp(np.asarray(v1), np.asarray(v2), t)

def slerp_vector(v1, v2, t, normalized=True):
	"""https://en.wikipedia.org/wiki/Slerp
	"""
	l1 = np.linalg.norm(v1)
	l2 = np.linalg.norm(v2)
	v1 = np.asarray(v1)/l1
	v2 = np.asarray(v2)/l2
	angle = np.dot(v1, v2)
	if np.abs(angle)==1:
		raise ValueError("Can't interpolate {} and {}".format(v1, v2))
	angle = np.arccos(angle)
	direction =  (v1 * np.sin((1-t)*angle) + v2 * np.sin(t*angle))/np.sin(angle)
	if not normalized:
		direction *= lerp(l1, l2, t)
	return direction


def interpolate_camera_calibration(cal1, cal2, t, focus_slerp=None):
	calib = {}
	calib["forward"] = slerp_vector(cal1["forward"], cal2["forward"], t, normalized=True)
	calib["up"] = slerp_vector(cal1["up"], cal2["up"], t, normalized=True)
	calib["right"] = slerp_vector(cal1["right"], cal2["right"], t, normalized=True)
	if focus_slerp is not None:
		p1 = np.subtract(cal1["position"], focus_slerp)
		p2 = np.subtract(cal2["position"], focus_slerp)
		calib["position"] = np.add(slerp_vector(p1, p2, t, normalized=False), focus_slerp)
	else:
		calib["position"] = lerp_vector(cal1["position"], cal2["position"], t)
	calib["fov_horizontal"] = lerp(cal1["fov_horizontal"], cal2["fov_horizontal"], t)
	
	return calib

if USE_RENDER:
	flip_z = lambda v: np.asarray(v)*np.asarray([1,1,-1])
	invert_v = lambda v: np.asarray(v)*(-1)
	
	def lerp_transform(T1, T2, t):
		assert isinstance(T1, Transform)
		assert isinstance(T2, Transform)
	
	def make_camera(calib, focus, focus_depth_clip=1.0):
		def pos():
			return flip_z(calib["position"])
		def fwd():
			return invert_v(flip_z(calib["forward"]))
		def up():
			return flip_z(calib["up"])
		def right():
			return flip_z(calib["right"])
		#fwd, up, right, pos
		cam_focus = flip_z(focus)
		train_cam_resolution = (256, 1920//4, 1080//4)
		aspect = train_cam_resolution[2]/train_cam_resolution[1]
		cam_dh = focus_depth_clip*0.5 #depth half
		
		position = pos()
		dist = np.linalg.norm(cam_focus-position)
		cam = Camera(MatrixTransform.from_fwd_up_right_pos(fwd(), up(), right(), position), nearFar=[dist-cam_dh,dist+cam_dh], fov=calib["fov_horizontal"], aspect=aspect, static=None)
		cam.transform.grid_size = copy.copy(train_cam_resolution)
		
		return cam

if __name__=="__main__":
	out_path = "./view_interpolation_tests/view_interpolation_synth_1-3_B_%.2e_%d_2-way-warp"%(alpha, window)
	#if os.path.exists(out_path):
	#	shutil.rmtree(out_path)
	os.makedirs(out_path, exist_ok=True)
	print("output path:", out_path)
	
	if True: #USE_RENDER:
		print("interpolation test with rendered images")
		camId_1 = 1
		camId_2 = 3
		n_subdivisions = 5
		#slerp_position = True
		print("load camera calibration for 2 cameras")
		with open("scalaFlow_cameras.json", "r") as file:
			calib = json.load(file)
		cal1 = calib[str(camId_1)]
		cal2 = calib[str(camId_2)]
		if cal1["fov_horizontal"] is None: cal1["fov_horizontal"] =calib["fov_horizontal_average"]
		if cal2["fov_horizontal"] is None: cal2["fov_horizontal"] =calib["fov_horizontal_average"]
		focus = calib["focus"]
		print("interpolate cameras")
	#	for cal in calibrations:
	#		print(cal)
	#	exit(0)
		
		print("load density")
		sf_dens_transform = GridTransform([100,178,100], translation=flip_z(calib["volume_offset"] + np.asarray([0,0,calib["marker_width"]])), scale=[calib["marker_width"]]*3, normalize='MIN')
		with np.load("data/ScalarFlow/sim_000000/reconstruction/density_000140.npz") as np_data:
			density = np_data["data"][np.newaxis,::-1,...]
		print(density.shape)
		density = tf.constant(density, dtype=tf.float32)
		sf_dens_transform.set_data(density)
		
		for slerp_position in [False, True]:
			calibrations = [cal1] + [interpolate_camera_calibration(cal1, cal2, (_+1)/(n_subdivisions+1), focus if slerp_position else None) for _ in range( n_subdivisions)] + [cal2]
			print("prepare rendering")
			cameras = [make_camera(_, focus) for _ in calibrations]
			renderer = Renderer(None,
				filter_mode="LINEAR",
				mipmapping="LINEAR",
				num_mips=3,
				blend_mode="BEER_LAMBERT",
			)
			
			print("render interpolated cameras")
			images = [renderer.render_density(sf_dens_transform, [Light(intensity=1.0)], [cam], cut_alpha=True)[0].numpy()[0] for cam in cameras]
			print(images[0].shape)
			write_images(os.path.join(out_path, "render-sub-" + ("slerp" if slerp_position else "lerp") + "_image_{frame:04d}.png"), images)
		
		#optical flow interpolation for comparison
		print("optical flow")
		flow = get_dense_optical_flow(images[0], images[-1])
		write_images(os.path.join(out_path, "warp-sub_flow_{frame:04d}.png"), [flow_to_image(flow), flow_to_image(-flow)])
		print("backwards optical flow")
		flow_b = get_dense_optical_flow(images[-1], images[0])
		write_images(os.path.join(out_path, "warp-sub_back-flow_{frame:04d}.png"), [flow_to_image(flow_b), flow_to_image(-flow_b)])
		
		print("warp")
		i_warp_sub = [images[0]]
		i_warp_sub.extend(lerp_image(images[0], images[-1], (s+1)/(n_subdivisions+1), optical_flow=flow) for s in range(n_subdivisions))
		i_warp_sub.append(images[-1])
		print("write images")
		write_images(os.path.join(out_path, "warp-sub_image_{frame:04d}.png"), i_warp_sub)
		
		print("warp 2-way")
		i_warp_sub = [images[0]]
		i_warp_sub.extend(lerp_image_2(images[0], images[-1], (s+1)/(n_subdivisions+1), optical_flow1=flow, optical_flow2=flow_b) for s in range(n_subdivisions))
		i_warp_sub.append(images[-1])
		print("write images")
		write_images(os.path.join(out_path, "warp-2-sub_image_{frame:04d}.png"), i_warp_sub)
		
	else:
		step_2 = True
		step_4 = True
		step_1_subdivsion = True
		n_subdivisions = 5
		
		print("load images")
		images = load_scalarFlow_images(0,130, cams=[2,1,0,4,3]) #
		#images = [(_*255.) for _ in images] #.astype(np.uint8)
		print("write original images")
		write_images(os.path.join(out_path, "input_image_{frame:04d}.png"), images)
		
		if step_2:
			print("step 2 optical flow (%s)"%OPTFLOW_METHOD)
			flow1_1 = get_dense_optical_flow(images[0], images[2])
			flow2_1 = get_dense_optical_flow(images[2], images[4])
			write_images(os.path.join(out_path, "warp1_flow_{frame:04d}.png"), [flow_to_image(flow1_1), flow_to_image(flow2_1)])
			print("step 2 warp")
			i_warp_1 = [
				images[0],
				lerp_image(images[0], images[2], 0.5, optical_flow=flow1_1),
				images[2],
				lerp_image(images[2], images[4], 0.5, optical_flow=flow2_1),
				images[4],
			]
			write_images(os.path.join(out_path, "warp1_image_{frame:04d}.png"), i_warp_1)
		
		if step_4:
			print("step 4 optical flow (%s)"%OPTFLOW_METHOD)
			flow_2 = get_dense_optical_flow(images[0], images[4])
			write_images(os.path.join(out_path, "warp2_flow_{frame:04d}.png"), [flow_to_image(flow_2)])
			print("step 4 warp")
			i_warp_2 = [
				images[0],
				lerp_image(images[0], images[4], 0.25, optical_flow=flow_2),
				lerp_image(images[0], images[4], 0.5 , optical_flow=flow_2),
				lerp_image(images[0], images[4], 0.75, optical_flow=flow_2),
				images[4],
			]
			write_images(os.path.join(out_path, "warp2_image_{frame:04d}.png"), i_warp_2)
		
		if step_1_subdivsion:
			print("step 1 - subdivision optical flow (%s)"%OPTFLOW_METHOD)
			flows = [get_dense_optical_flow(images[i], images[i+1]) for i in range(0, len(images)-1)]
			write_images(os.path.join(out_path, "warp1sub_flow_{frame:04d}.png"), [flow_to_image(_) for _ in flows])
			print("step 1 - subdivision warp")
			i_warp_sub = []
			for i in range(0, len(images)-1):
				i_warp_sub.append(images[i])
				i_warp_sub.extend(lerp_image(images[i], images[i+1], (s+1)/(n_subdivisions+1), optical_flow=flows[i]) for s in range(n_subdivisions))
			i_warp_sub.append(images[-1])
			write_images(os.path.join(out_path, "warp1sub_image_{frame:04d}.png"), i_warp_sub)
		
		print("Done.")