import numpy as np
from scipy.optimize import least_squares
import pandas as pd
from lib.logger import Logger
import json, math

logger = Logger('./calibration', clear_log=True)
# y,x,z
#pixel ray directions [center-center][left-center][right-center][center-bottom][center-top]
# width, height
# 539 959, 0 959, 1079 959, 539 0, 539 1919
indices = {
	'center': [(539, 959), (539, 960), (540, 959), (540, 960)],
	'left': [(0, 959), (0, 960)],
	'left_half': [(269, 959), (269, 960)],
	'right': [(1079, 959), (1079, 960)],
	'right_half': [(810, 959), (810, 960)],
	'top': [(539, 1919), (540, 1919)],
	'top_half': [(539, 1440), (540, 1440)],
	'bottom': [(539, 0), (540, 0)],
	'bottom_half': [(539, 479), (540, 479)],
	'top_left_half': [(269, 1440)],
	'top_right_half': [(810, 1440)],
	'bottom_left_half': [(269, 479)],
	'bottom_right_half': [(810, 479)],
}
idx = [(539, 959), ]#,(0, 959), (1079, 959), (539, 0), (539, 1919)]


cams = []
for i in [1,2,3,4,5]:
	rays =pd.read_csv('data/scalarFlow/calib20190813/{}_rays.txt'.format(i), sep=' ', skiprows=1, header=None, names=['pY','pX','dY','dX','dZ'], index_col=(0,1))
	cam = {}
	for key, idx in indices.items():
		tmp = []
		try:
			for id in idx:
				ray = rays.loc[id]
				tmp.append({'start': np.asarray([ray['pX'],ray['pY'],0.0]),
					'dir': np.asarray([ray['dX'],ray['dY'],ray['dZ']]),
				})
			cam[key] = tmp
		except:
			print('[W]: could not access index {} for cam {}, key {}'.format(id,i, key))
	cams.append(cam)

#https://math.stackexchange.com/questions/2598811/calculate-the-point-closest-to-multiple-rays
def to_ray(pos, start, dir):
	t = np.dot(dir, pos-start)/np.dot(dir,dir)
	return pos - (start + t*dir)

def dist_to_ray(pos, start, dir):
	dist = no.linalg.norm(to_ray(pos, start, dir))

def f(x, rays):
	e = []
	for ray in rays:
		e += list(to_ray(x, ray['start'], ray['dir']))
	return e


def angle_between(a, b):
	return np.arccos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))

#https://en.wikipedia.org/wiki/Slerp
def slerp(v1, v2, t):
	O = angle_between(v1, v2)
	sO = np.sin(O)
	return np.sin((1-t)*O)/sO * v1 + np.sin(t*O)/sO * v2

deg_to_rad = np.pi/180
rad_to_deg = 180/np.pi

cam_order = [0,1,2,3,4] #[0, 4, 3, 2, 1] #mapping of ray calibration files to recorded sequences?
#i=0
cam_rays = []
cam_params = []
cam_json = {str(_):{} for _ in cam_order}
for i in range(len(cams)):
	cam = cams[cam_order[i]]
	params = {'rotation':None, 'position':None, 'position_error':None,
		'forward':None, 'right':None, 'up':None,
		'fov_horizontal':None, 'fov_vertical':None}
	print('cam', i+1)
	if 'center' in cam:
		c_rays = [ray['dir'] for ray in cam['center']]
		c = slerp(slerp(c_rays[0],c_rays[1], 0.5), slerp(c_rays[2],c_rays[3], 0.5), 0.5)
		c /= np.linalg.norm(c)
		fwd = c
		t_y = np.arctan(c[0]/c[2])
		t_x = np.arctan(c[1]/np.linalg.norm([c[0], c[2]]))
		print('\trot:', t_x*rad_to_deg,t_y*rad_to_deg,0.0)
		print('\tfwd: {} (center ray)'.format(c))
		params['rotation']=[t_x*rad_to_deg,t_y*rad_to_deg,0.0]
		params['forward']=list(fwd)
	
	if 'left' in cam and 'right' in cam:
		l = slerp(cam['left'][0]['dir'],cam['left'][1]['dir'], 0.5)
		r = slerp(cam['right'][0]['dir'],cam['right'][1]['dir'], 0.5)
		up = np.cross(l, r)
		up /= np.linalg.norm(up)
		print('\tup: {}'.format(up))
		print('\t\tfov x: {}'.format(angle_between(l,r)*rad_to_deg))
		params['up']=list(up)
		params['fov_horizontal']=angle_between(l,r)*rad_to_deg
		
	if 'left_half' in cam and 'right_half' in cam:
		l = slerp(cam['left_half'][0]['dir'],cam['left_half'][1]['dir'], 0.5)
		r = slerp(cam['right_half'][0]['dir'],cam['right_half'][1]['dir'], 0.5)
		up = np.cross(l, r)
		up /= np.linalg.norm(up)
		print('\t[up_half: {}]'.format(up))
		if params['up'] is None:
			params['up']=list(up)
	
	if 'top' in cam and 'bottom' in cam:
		t = slerp(cam['top'][0]['dir'],cam['top'][1]['dir'], 0.5)
		b = slerp(cam['bottom'][0]['dir'],cam['bottom'][1]['dir'], 0.5)
		right = np.cross(t, b)
		right /= np.linalg.norm(right)
		print('\tright: {}'.format(right))
		print('\t\tfov y: {}'.format(angle_between(t,b)*rad_to_deg))
		params['right']=list(right)
		params['fov_vertical']=angle_between(t,b)*rad_to_deg
	if 'top_half' in cam and 'bottom_half' in cam:
		t = slerp(cam['top_half'][0]['dir'],cam['top_half'][1]['dir'], 0.5)
		b = slerp(cam['bottom_half'][0]['dir'],cam['bottom_half'][1]['dir'], 0.5)
		right = np.cross(t, b)
		right /= np.linalg.norm(right)
		print('\t[right_half: {}]'.format(right))
		if params['right'] is None:
			params['right']=list(right)
	
	rays = []
	for key, r in cam.items():
		rays += r
	#print(cam)
	#print(rays)
	pos = least_squares(f, [0,0,0], kwargs={'rays':rays})
	print('\tpos: {}, error: {}'.format(pos.x, pos.cost))
	params['position']=list(pos.x)
	params['position_error']=pos.cost
	cam_rays.append({'start': pos.x, 'dir':c})
	print('\tangles: zx={}, zy={}, xy={}'.format(angle_between(fwd,right)*rad_to_deg, angle_between(fwd,up)*rad_to_deg, angle_between(right, up)*rad_to_deg))
	# for camera transform setup. look along -z so invert, also invert fwd as it is the z+ axis of the camera
	cam_json[str(i)] = params
	flip_z = np.asarray([1,1,-1])
	cam_params.append([list(-1*(c*flip_z)), list(up*flip_z), list(right*flip_z), list(pos.x*flip_z)])
	#print('{}, {}, {}, {}'.format(list(-1*(c*flip_z)), list(up*flip_z), list(right*flip_z), list(pos.x*flip_z)))
#	if len(cam)>=5:
#		fov = (angle_between(cam[1], cam[2]),angle_between(cam[3], cam[4]))
#		print('\tfov: x', fov[0]*rad_to_deg,'y',fov[1]*rad_to_deg)
#		up = np.cross(cam[1], cam[2])
#		up /= np.linalg.norm(up)
#		print('\tup:', up, angle_between(up, cam[0])*rad_to_deg)
	
	logger.flush()
focus = least_squares(f, [0,0,0], kwargs={'rays':cam_rays})
print('focus: {}, error: {}:'.format(focus.x, focus.cost))

cam_json['fov_horizontal_average'] = np.mean([cam_json[str(i)]['fov_horizontal'] for i in cam_order if cam_json[str(i)]['fov_horizontal'] is not None])
cam_json['fov_vertical_average'] = np.mean([cam_json[str(i)]['fov_vertical'] for i in cam_order if cam_json[str(i)]['fov_vertical'] is not None])
cam_json['focus']=list(focus.x)
cam_json['focus_error']=focus.cost

# fixed parameters from scalarFlow (/source/util/ray.cpp)
cam_json['scale_y'] = 1.77
cam_json['marker_width'] = 0.4909
cam_json['volume_width'] = cam_json['marker_width']
cam_json['volume_offset'] = [cam_json['marker_width']/6., -cam_json['marker_width']/11., -cam_json['marker_width']/100.] #world-space position of the (0,0,0) corner of the volume
#cam_json['volume_size'] = [cam_json['volume_width'], math.ceil(cam_json['scale_y']*cam_json['volume_width']), cam_json['volume_width']]
cam_json['volume_size'] = [cam_json['volume_width'], cam_json['scale_y']*cam_json['volume_width'], cam_json['volume_width']]
# the volume_size here is wrong, normalize volume globally s.t. x (width) ==1.0 and scale with volume_width


with open('scalaFlow_cameras.json', 'w') as f:
	json.dump(cam_json, f, sort_keys=True, indent=2)

for cam in cam_params:
	print('{},'.format(cam))
	#i+=1