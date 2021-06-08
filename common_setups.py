import copy, munch, numbers, collections.abc

import lib.scalar_schedule as sSchedule

#https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def _CSETUPS_update_dict_recursive(d, u, deepcopy=False, new_key='KEEP'):
	if deepcopy:
		d = copy.deepcopy(d)
	for k, v in u.items():
		if not k in d:
			if new_key.upper()=='ERROR':
				raise KeyError("Update key {} does not exisit in base dictionary.".format(k))
			elif new_key.upper()=='DISCARD':
				continue
			elif new_key.upper()=='KEEP':
				pass
			else:
				raise ValueError("Unknown policy for new key: {}".format(new_key))
		if isinstance(v, collections.abc.Mapping):
			if k in d and not isinstance(d[k], collections.abc.Mapping):
				# if something that is not a Mapping is updated with a Mapping
				#e.g. a default constant schedule (int, float, list) with a complex one (dict {"type":<type>, ...})
				if isinstance(d[k], (numbers.Number, list)) and "type" in v and isinstance(v["type"], str) and v["type"].upper() in ['SCHEDULE', 'CONST','LINEAR','EXPONENTIAL','ROOT_DECAY'] and "start" in v:
					d[k] = sSchedule._get_base_schedule()
				else:
					d[k] = {}
			d[k] = _CSETUPS_update_dict_recursive(d.get(k, {}), v, deepcopy=deepcopy, new_key=new_key)
		else:
			if deepcopy:
				d[k] = copy.deepcopy(v)
			else:
				d[k] = v
	return d



################################
# --- Full Setups for Scrips ---
################################

RECONSTRUCT_SEQUENCE_SETUP = {
	'title':'seq_test',
	'desc':'sequence reconstruction run description',
	'paths':{
		'base':"./runs",
		'group':"reconstruct_sequence",#"test-and-debug",#
	},
	'rendering':{
		'monochrome':False,
		'luma':[0.2126,0.7152,0.0722], #[0.299,0.587,0.144] #https://en.wikipedia.org/wiki/Luma_(video)
		'filter_mode':'LINEAR', #NEAREST, LINEAR
		'mip':{
			'mode':'LINEAR', #NONE, NEAREST, LINEAR
			'level':4,
			'bias':0.0,
		},
		'blend_mode':'BEER_LAMBERT', #BEER_LAMBERT, ALPHA, ADDITIVE
		'sample_gradients':True,
		
		'steps_per_cycle':24,
		'num_images': 24,
		
		'main_camera':{
			'base_resolution':[256,1920,1080], #z(depth), y(height), x(width)
			'resolution_scale':1./3., # only for xy
			'fov':40,
			'near':0.3,
			'distance':0.8,
			'far':1.3,
		},
		'target_cameras':{
			'calibration_file':"scalaFlow_cameras.json",
			'camera_ids':[2,1,0,4,3],
			'crop_frustum':False, # crop frustum grid to AABB of vidual hull. for performance
			'crop_frustum_pad':0, # view space cells
		},
		
		'allow_static_cameras':False,
		'allow_fused_rendering':True,
		
		'background':{
			'type':'COLOR', #'CAM', 'COLOR', 'NONE'; only for vis-rendering, not used in optimization
			'color': [0,0.5,1.0], #[0,0.5,0],
		},
		
		'lighting':{
			'ambient_intensity':0.55,
			'initial_intensity':1.13,
			'shadow_resolution':[64,64,64], #DHW
		},
		"velocity_scale":1.0*256,
		"synthetic_target":{
			'filter_mode':'LINEAR',
			'blend_mode':'BEER_LAMBERT',
			'ambient_intensity':0.64,
			'initial_intensity':0.85,
		}
	},#rendering
	'data':{
		"rand_seed_global": 460585320,
		'run_dirs':['./runs',],
		'grid_size':128, #x and z resolution/grid size, y is ceil(this*scale_y) from callibration (scale_y ~ 1.77)
		'clip_grid':False,
		'clip_grid_pad':3,
		'crop_grid':False,
		'hull':'TARGETS', #ALL, TARGETS, ROT, [<ids>, ], empty==ALL
		
		'simulation':0,
		'start':140,
		'stop':142, #exclusive
		'step':1,
		'scalarFlow_frame_offset':-11,
		'density':{
			'scale': 0.01,
			'initial_value':'HULL',#"CONST", "ESTIMATE", "HULL" or path to an npz file '[RUNID:000000-000000]frame_{frame:06d}/density_pre-opt.npz'
			'min':0.0,
			'max':0.5,
			#'file_mask':'[RUNID:000000-000000]frame_{frame:06d}/density_fit.npz',
			#'file_mask':''
			#'render_target':False,
			#'synthetic_target':False,
			'target_type': "RAW", #PREPROC, SYNTHETIC
			'target': 'data/ScalarFlow/sim_{sim:06d}/input/cam/imgsUnproc_{frame:06d}.npz',
			'target_preproc': 'data/ScalarFlow/sim_{sim:06d}/input/postprocessed/imgs_{frame:06d}.npz',
			'target_flip_y': False,
			'target_cam_ids':[0,1,2,3,4], #which ScalarFlow camera targets to use. 0:center, 1:center-left, 2:left, 3:right, 4:center-right (CW starting from center)
			'target_threshold':4e-2,
			'target_scale': 1.5,
			'hull_image_blur_std':1.0,
			'hull_volume_blur_std':0.5,
			'hull_smooth_blur_std':0.0,
			'hull_threshold':4e-2,
			'inflow':{
				'active':True,
				'hull_height':10,
				'height':8,
			},
			'scalarFlow_reconstruction':'data/ScalarFlow/sim_{sim:06d}/reconstruction/density_{frame:06d}.npz',
			'synthetic_target_density_scale':1.,
		},
		'velocity':{
			'initial_value':'RAND',
			'load_step':1,
			'init_std':0.1,
			'init_mask':'HULL_TIGHT', #NONE HULL, HULL_TIGHT
			'boundary':'CLAMP', #BORDER (closed 0 bounds), CLAMP (open bounds)
			'scalarFlow_reconstruction':'data/ScalarFlow/sim_{sim:06d}/reconstruction/velocity_{frame:06d}.npz',
		},
		'initial_buoyancy':[0.,0.,0.],
		'discriminator':{
			'simulations':[0,6],
			'frames':[45,145, 1],
			'target_type': "RAW", #PREPROC, (SYNTHETIC not supported)
			'target': 'data/ScalarFlow/sim_{sim:06d}/input/cam/imgsUnproc_{frame:06d}.npz', # 'data/ScalarFlow/sim_{:06d}/input/cam/imgsUnproc_{:06d}.npz',
			'target_preproc': 'data/ScalarFlow/sim_{sim:06d}/input/postprocessed/imgs_{frame:06d}.npz',
			#augmentation
			'crop_size':[int(1080//8), int(1080//8)], #HW, input size to disc
			'scale_input_to_crop':False, #resize all discriminator input to its base input resolution (crop_size)
			'real_res_down': 4,
			'scale_real_to_cam':True, #scale real data resolution to current discriminator fake-sample camera (to match resolution growth)
			'scale_range':[0.52,1.0],
			'rotation_mode': "90",
		#	'resolution_fake':[256,1920/4,1080/4],
		#	'resolution_scales_fake':[1, 1/2],
		#	'resolution_loss':[256,1920/8,1080/8],
			'scale_real':[0.8, 1.8], #range for random intensity scale on real samples
			'scale_fake':[0.7, 1.4],
		#	'scale_loss':[0.8, 1.2],
			'gamma_real':[0.5,2], #range for random gamma correction on real samples (value here is inverse gamma)
			'gamma_fake':[0.5,2], #range for random gamma correction on fake samples (value here is inverse gamma)
		#	'gamma_loss':[0.5,2], #range for random gamma correction applied to input when evaluating the disc as loss (for density) (value here is inverse gamma)
			
		},
		'load_sequence':None, #only for rendering without optimization
		'load_sequence_pre_opt':False,
	},#data
	'training':{
		'iterations':5000,#30000,
		'frame_order':'FWD', #FWD, BWD, RAND
		
		'resource_device':'/cpu:0', #'/cpu:0', '/gpu:0'
		
		#'loss':'L2',
		'train_res_down':8,
		'loss_active_eps':1e-18, #minimum absolute value of a loss scale for the loss to be considered active (to prevent losses scaled to (almost) 0 from being evaluated)
		
		'density':{
			'optim_beta':0.9,
			'use_hull':True,
			'warp_clamp':"NONE",
			'camera_jitter':False,
			'scale_render_grads_sharpness':0.0,
			'error_functions':{
				'raw_target_loss':'AE',
				'preprocessed_target_loss':'AE',
				'target_depth_smoothness_loss':'SE',
				'hull':'SE',
				'negative':'SE',
				'smoothness_loss':'SE',
				'smoothness_loss_2':'SE',
				'temporal_smoothness_loss':'SE',
				'warp_loss':'AE',
			},
			'pre_optimization':True, #whether pre-optim will run for density, affects forward propagation/advection of state and optimization
			# to only have fwd advection without optimization set iterations to 0
			'pre_opt':{
				'first':{ #settings for first frame
					'iterations':5000,#30000,
					'learning_rate':{'type':'exponential', 'start':0.0002, 'base':0.5, 'scale':2/30000},
					
					'raw_target_loss':1.,
					'preprocessed_target_loss':0.,
					'target_depth_smoothness_loss':0.,
					'hull':0.,
					'negative':0.,
					'smoothness_loss':0.0,#1.0
					'smoothness_neighbours':3,
					'smoothness_loss_2':0.0,#0.2
					'temporal_smoothness_loss':0.0,#0.2
					'discriminator_loss':[0.0, 0.0, 1.2e-3, 1.2e-3/2000, 1000],#0.004,
					'warp_loss':0.0,
				},
				#settings for remaining frames
				'iterations':2400,#5000,
				'seq_init':"WARP", #WARP, COPY, BASE
				'learning_rate':{'type':'exponential', 'start':0.0002, 'base':0.5, 'scale':1/3000}, #{'type':'exponential', 'start':0.00005, 'base':0.5, 'scale':2/30000},
				
				'raw_target_loss':1.,
				'preprocessed_target_loss':0.,
				'target_depth_smoothness_loss':0.,
				'hull':0.,
				'negative':0.,
				'smoothness_loss':0.0,#1.0
				'smoothness_neighbours':3,
				'smoothness_loss_2':0.0,#0.2
				'temporal_smoothness_loss':0.0,#0.2
				'discriminator_loss':[0.0, 0.0, 1.2e-3, 1.2e-3/2000, 400],#0.004,
				'warp_loss':0.0,
				
				'inspect_gradients':1,
			},
			'learning_rate':{'type':'exponential', 'start':0.00015, 'base':0.5, 'scale':2/30000},#0.00015, #[0.00001,0.00001,0.0001, 0.00009/20000, 4000],
			
		#	'loss':'L2',
			'raw_target_loss':1.,#0.3, [float('-inf'),float('+inf'), 0.0, 0], #[min, max, step, offset]
			'preprocessed_target_loss':0.,# [float('-inf'),float('+inf'), 0.0, 0], #[min, max, step, offset]
			'target_depth_smoothness_loss':0.,
			'hull':0.,
			'negative':0.,
			
			'smoothness_loss':0.0, #1.0
			'smoothness_neighbours':3, # the kind of neighbourhood to consider in the edge filter (e.g. wether to use diagonals), NOT the kernel size.
			'smoothness_loss_2':0.0,#0.2
			'temporal_smoothness_loss':0.0,#0.2
			
			'discriminator_loss':1.2e-3,
			'warp_loss':[0.0,0.0,2., 2./20000, 0],# [float('-inf'),float('+inf'), 0.0, 0], #[min, max, step, offset] #{'type':'schedule','schedule':[0.0,0.0,1.0,1.0/6000,6000]} 4000
			
			'main_warp_fwd':False,
			'warp_gradients':{
				'weight':sSchedule.setup_constant_schedule(start=1.0),
				'active':False,
				'decay':sSchedule.setup_constant_schedule(start=0.9), #[0,1], lower is faster decay
				'update_first_only':False,
			},
			
			"view_interpolation":{
				"steps":0,
			},
			
			'grow':{ # not yet used
				"factor":1.2,#2. factor per interval, max down-scaling is factor^len(intervals)
				"pre_grow_actions":[],# "WARP", list
				"post_grow_actions":[],# "WARP", list
				#iterations for each grow step, empty to disable
				'intervals':[],#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 3200, 3800], #-> 6,14,24,36,50,66,84,104,128,156,188,226 (x100)
			},
		},
		'velocity':{
			'optim_beta':0.9,
			'warp_order':1,
			'warp_clamp':"NONE",
		#	'use_hull':False,
			'error_functions':{
				'density_warp_loss':'AE',
				'velocity_warp_loss':'AE',
				'divergence_loss':'SE',
				'magnitude_loss':'SE',
			},
			'pre_optimization':True, #whether pre-optim will run for velocity, affects forward propagation/advection of state and optimization
			'pre_opt':{
				'first':{ #settings for first frame
					'iterations':10000, #30000,
					'learning_rate':0.04, #{'type':'exponential', 'start':0.04, 'base':0.5, 'scale':1/10000},
					
					'density_warp_loss':1.0,
					'velocity_warp_loss':0.0,
					'smoothness_loss':0.0,
					'smoothness_neighbours':3,
					'cossim_loss':0.0,
					'divergence_loss':[0.0, 0.0,0.3, 0.3/14000, 1200], #[0.0, 0.0,0.05, 0.05/26000, 2600],
					#'divergence_normalize':0.0,
					#adjust according to data.step, large values can lead to NaN.
					'magnitude_loss':{'type':'exponential', 'start':0.00004, 'base':0.1, 'scale':6/20000},#[4e-05, 0.0, 4e-05, -(4e-05/1000), 0],#[1e-06, 0.0, 1e-06, -(1e-06/8000), 0]
					'grow':{
						"factor":1.2,
						"scale_magnitude":True,
						#'intervals':[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 3200, 3800], #22600
						#'intervals':[400, 520, 660, 800, 920, 1060, 1200, 1320, 1600, 1860, 2120, 2520], #7490*2 (ca. 2/3)
						'intervals':[200, 260, 330, 400, 460, 530, 600, 660, 800, 930, 1060, 1260], #7490 (ca. 1/3)
						#'intervals':[60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 380], #2260
					},
				
				},
				#settings for remaining frames
				'iterations':2000, #2000,
				'seq_init':"WARP", #WARP, COPY, BASE
				'learning_rate':0.02, #{'type':'exponential', 'start':0.0004, 'base':0.5, 'scale':3/30000},
				
				'density_warp_loss':1.0,
				'velocity_warp_loss':0.0,
				'smoothness_loss':0.0,
				'smoothness_neighbours':3,
				'cossim_loss':0.0,
				'divergence_loss':0.3,
				#'divergence_normalize':0.0,
				'magnitude_loss':0.0,
				
				'grow':{
					"factor":1.2,#2. factor per interval, max down-scaling is factor^len(intervals)
					"scale_magnitude":True,
					'intervals':[],
				},
			},
			'noise_std':sSchedule.setup_constant_schedule(start=0.0),
			'learning_rate':{'type':'exponential', 'start':0.02, 'base':0.5, 'scale':2/30000},#0.05,#[0.0,0.0,0.005,0.005/6000,6000]
		#	'lr_decay':0.00,
			
		#	'loss':'L2',
			
			'density_warp_loss':1.0,#influence of loss(A(dt, vt), dt+1) on velocity, can be a schedule
			'velocity_warp_loss':1.0,#influence of loss(A(vt, vt), vt+1) on velocity, can be a schedule
			
			'smoothness_loss':0.0,#0.005, [float('-inf'),float('+inf'), 0.0, 0], #[min, max, step, offset] [0.0,0.0002, 0.0002/5000, 25000],#
			'smoothness_neighbours':3, # the kind of neighbourhood to consider in the edge filter (e.g. wether to use diagonals), NOT the kernel size.
			'cossim_loss':0.0,
			
			'divergence_loss':0.3,#0.03, [0.0,0.0,0.05, 0.05/26000, 2600], #[float('-inf'),float('+inf'), 0.0, 0], #[min, max, step, offset] [0.0,0.05, 0.05/10000, 0] 2000
			'divergence_normalize':0.0,
			
			'magnitude_loss':0.0,#0.00001, [float('-inf'),float('+inf'), 0.0, 0], #[min, max, step, offset] [0.0,0.05, 0.05/10000, 0]
			#'warp_loss':0.1,# [float('-inf'),float('+inf'), 0.0, 0], #[min, max, step, offset] 4000
			
			'warp_gradients':{
				'weight':sSchedule.setup_constant_schedule(start=1.0), #affects warp gradients for velocity from backward dens warp, even if vel- warp gradients are inactive
				'active':False,
				'decay':sSchedule.setup_constant_schedule(start=0.9), #[0,1], lower is faster decay
			},
			
			'grow':{
				"factor":1.2,#2. factor per interval, max down-scaling is factor^len(intervals)
				"scale_magnitude":True,
				'intervals':[],
			},
		},
		'optimize_buoyancy':False,
		"light":{
			'optim_beta':0.9,
			"optimize":False,
			"min":0.01,
			"max":6.0,
			"learning_rate":{'type':'exponential', 'start':0.001, 'base':1, 'scale':0},
		},
		
		
		'discriminator':{
			'active':False,
			'model':None,#'[RUNID:000000-000000]disc_model.h5',#
			'loss_type':"SGAN", #"SGAN", "RpSGAN", "RpLSGAN", "RaSGAN", "RaLSGAN"
			'layers':[24]*6,
			'stride':1, # larger receptive field, strong effect on results
			'kernel_size':4,
			'padding':'NONE', #NONE(=valid), ZERO(=same), MIRROR(=tf.pad(REFLECT))
			'activation':'lrelu', # relu or lrelu
			'activation_alpha':0.2, # leak for lrelu
			'use_fc':False, #usually makes disc stronger
			'noise_std':0.0,
		#	'noise_schedule': [float('-inf'),float('+inf'), 0.0, 0], #[min, max, step, offset]
			'start_delay':0, #50,
			#'loss_scale':[0.0001,0.0001,0.002,0.0019/18000,4000],#0.004, #scale of gradients provided by the disc when optimizing density, to balance between L1 and disc
		#	'loss_schedule': [0.0,0.004,0.004/12000,8000],#[float('-inf'),float('+inf'), 0.0, 0], #[min, max, step, offset]
			'pre_opt':{
				'first':{
					'train':True,
					'learning_rate':{'type':'exponential', 'start':4e-4, 'base':0.5, 'scale':4/30000},#0.00025
			#		'steps':1,
					'regularization':0.0,
				},
				'train':True,
				'learning_rate':1.6e-4,#0.00002, #{'type':'exponential', 'start':0.00025, 'base':0.5, 'scale':4/30000}
			#	'steps':1,
				'regularization':0.0,
			},
			'train':True,
			'learning_rate':2e-4,#0.00008, 
			'steps':1,
			'regularization':0.0,
			'optim_beta':0.5,
			
			'grow':{ # not yet used
				"factor":2.,#2. factor per interval, max down-scaling is factor^len(intervals)
				#iterations for each grow step, empty to disable
				'intervals':[]
			},
			
			'conditional_hull':False,
			'temporal_input':{
				'active':False,
				'step_range':(-3,4,1), #-3 to 3 inclusive, 0 will be removed
			},
			'num_real':4,
			'cam_res_down':8,
			'num_fake':3,
			'fake_camera_jitter':False,
			'target_label':0.9, #0.9 for label smoothing
			#'separate':True, #separate real and fake batches
			"history":{
				'load':None,
				'samples':4, #use older samples as fake samples as well. 0 to disable
				'size':800, #
				'keep_chance':0.01, # chance to put a rendered sample in the history buffer
				'save':False,
				'sequence_reuse':True,
				'reset_on_density_grow':True,
			#	'reset_on_discriminator_grow':False,
			},
			
		#	'sequence_reuse':True,
		},#discriminator
		'summary_interval':200,
	},#training
	'validation':{
		'output_interval':100,
		'stats':True,
		'cmp_scalarFlow':False,
		'cmp_scalarFlow_render':False,
		'warp_test':False,
		'warp_test_render':False,
		'render_cycle':False,
		'render_cycle_steps':8,
		'render_density':True,
		'render_shadow':True,
		'render_target':True,
		'render_velocity':True,
	},
	'debug':{
		'disc_dump_samples':False,
	},
}

#crop and main-opt growth with new loss balaning (after scale fixes)
RECONSTRUCT_SEQUENCE_SETUP_BASE = {
	'title':'seq_test',
	'desc':'sequence reconstruction run description',
	'paths':{
		'base':"./runs",
		'group':"sequence_recon_test",
	},
	'rendering':{
		'monochrome':False,
		'luma':[0.2126,0.7152,0.0722], #[0.299,0.587,0.144] #https://en.wikipedia.org/wiki/Luma_(video)
		'filter_mode':'LINEAR', #NEAREST, LINEAR
		'mip':{
			'mode':'LINEAR', #NONE, NEAREST, LINEAR
			'level':4,
			'bias':0.0,
		},
		'blend_mode':'BEER_LAMBERT', #BEER_LAMBERT, ALPHA, ADDITIVE
		'sample_gradients':True,
		
		'steps_per_cycle':24,
		'num_images': 24,
		
		'main_camera':{
			'base_resolution':[256,1920,1080], #z(depth), y(height), x(width)
			'resolution_scale':1./3., # only for xy
			'fov':40,
			'near':0.3,
			'distance':0.8,
			'far':1.3,
		},
		'target_cameras':{
			'calibration_file':"scalaFlow_cameras.json",
			'camera_ids':[2,1,0,4,3],
			'crop_frustum':False, # crop frustum grid to AABB of vidual hull. for performance
			'crop_frustum_pad':2, # view space cells
		},
		
		'allow_static_cameras':False,
		'allow_fused_rendering':True,
		
		'background':{
			'type':'COLOR', #'CAM', 'COLOR', 'NONE'; only for vis-rendering, not used in optimization
			'color': [0,0.5,1.0], #[0,0.5,0],
		},
		
		'lighting':{
			#'scattering_ratio':1.,
			'ambient_intensity':0.55,
			'initial_intensity':1.13,
			'shadow_resolution':[256,196,196], #DHW
		},
		"velocity_scale":1024,
		"synthetic_target":{
			'filter_mode':'LINEAR',
			'blend_mode':'BEER_LAMBERT',
			'ambient_intensity':0.64,
			'initial_intensity':0.85,
		}
	},#rendering
	'data':{
		"rand_seed_global": 460585320,
		'run_dirs':['./runs/test-and-debug','./runs/sequence_recon_test'],
		'grid_size':128, #x and z resolution/grid size, y is ceil(this*scale_y) from callibration (scale_y ~ 1.77)
		'clip_grid':True,
		'clip_grid_pad':4,
		'crop_grid':True,
		'hull':'TARGETS', #ALL, TARGETS, ROT, [<ids>, ], empty==ALL
		
		'simulation':0,
		'start':140,
		'stop':142, #exclusive
		'step':1,
		'scalarFlow_frame_offset':-11,
		'density':{
			'scale': 2.5,
			'initial_value':'HULL_TIGHT',#"CONST", "ESTIMATE", "HULL" or path to an npz file '[RUNID:000000-000000]frame_{frame:06d}/density_pre-opt.npz'
			'min':0.0,
			'max':256.0,
			'target_type': "RAW", #PREPROC, SYNTHETIC
			'target': 'data/ScalarFlow/sim_{sim:06d}/input/cam/imgsUnproc_{frame:06d}.npz', # 'data/ScalarFlow/sim_{:06d}/input/cam/imgsUnproc_{:06d}.npz',
			'target_preproc': 'data/ScalarFlow/sim_{sim:06d}/input/postprocessed/imgs_{frame:06d}.npz',
			'target_flip_y': False,
			'target_cam_ids':[0,1,2,3,4], #which ScalarFlow camera targets to use. 0:center, 1:center-left, 2:left, 3:right, 4:center-right (CW starting from center)
			'target_threshold':4e-2, #only used for disc dataset
			'target_scale': 1.5,
			'hull_image_blur_std':1.0,
			'hull_volume_blur_std':0.5,
			'hull_smooth_blur_std':0.0,
			'hull_threshold':4e-2,
			'inflow':{
				'active':True,
				'hull_height':10,
				'height':'MAX',
			},
			'scalarFlow_reconstruction':'data/ScalarFlow/sim_{sim:06d}/reconstruction/density_{frame:06d}.npz',
			'synthetic_target_density_scale':1.,
		},
		'velocity':{
			'initial_value':'RAND',
			'load_step':1,
			'init_std':0.1,
			'init_mask':'HULL_TIGHT_NEXT', #NONE HULL, HULL_TIGHT
			'boundary':'CLAMP', #BORDER (closed 0 bounds), CLAMP (open bounds)
			'scalarFlow_reconstruction':'data/ScalarFlow/sim_{sim:06d}/reconstruction/velocity_{frame:06d}.npz',
		},
		'initial_buoyancy':[0.,0.,0.],
		'discriminator':{
			'simulations':[0,6],
			'frames':[45,145, 1],
			'target_type': "RAW", #PREPROC, (SYNTHETIC not supported)
			'target': 'data/ScalarFlow/sim_{sim:06d}/input/cam/imgsUnproc_{frame:06d}.npz', #
			'target_preproc': 'data/ScalarFlow/sim_{sim:06d}/input/postprocessed/imgs_{frame:06d}.npz',
			#augmentation
			'crop_size':[96,96], #HW, input size to disc
			'scale_input_to_crop':False, #resize all discriminator input to its base input resolution (crop_size) before augmentation
			'real_res_down': 4,
			'scale_real_to_cam':True, #scale real data resolution to current discriminator fake-sample camera (to match resolution growth)
			'scale_range':[0.85,1.15],
			'rotation_mode': "90",
		#	'resolution_fake':[256,1920/4,1080/4],
		#	'resolution_scales_fake':[1, 1/2],
		#	'resolution_loss':[256,1920/8,1080/8],
			'scale_real':[0.8, 1.8], #range for random intensity scale on real samples
			'scale_fake':[0.7, 1.4],
		#	'scale_loss':[0.8, 1.2],
			'gamma_real':[0.5,2], #range for random gamma correction on real samples (value here is inverse gamma)
			'gamma_fake':[0.5,2], #range for random gamma correction on fake samples (value here is inverse gamma)
		#	'gamma_loss':[0.5,2], #range for random gamma correction applied to input when evaluating the disc as loss (for density) (value here is inverse gamma)
			
		},
		'load_sequence':None, #only for rendering without optimization
		'load_sequence_pre_opt':False,
	},#data
	'training':{
		'iterations':5000,
		'frame_order':'FWD', #FWD, BWD, RAND
		
		'resource_device':'/cpu:0', #'/cpu:0', '/gpu:0'
		
		#'loss':'L2',
		'train_res_down':6,
		'loss_active_eps':1e-18, #minimum absolute value of a loss scale for the loss to be considered active (to prevent losses scaled to (almost) 0 from being evaluated)
		
		'density':{
			'optim_beta':0.9,
			'use_hull':True,
			'warp_clamp':"MC_SMOOTH",
			'camera_jitter':False,
			'scale_render_grads_sharpness':0.0,
			'error_functions':{
				'raw_target_loss':'SE',
				'preprocessed_target_loss':'SE',
				'target_depth_smoothness_loss':'SE',
				'hull':'SE',
				'negative':'SE',
				'smoothness_loss':'SE',
				'smoothness_loss_2':'SE',
				'temporal_smoothness_loss':'SE',
				'warp_loss':'SE',
			},
			'pre_optimization':True, #whether pre-optim will run for density, affects forward propagation/advection of state and optimization
			# to only have fwd advection without optimization set iterations to 0
			'pre_opt':{
				'first':{ #settings for first frame
					'iterations':5000,#30000,
					'learning_rate':{'type':'exponential', 'start':3.0, 'base':0.5, 'scale':2/30000},
					
					'raw_target_loss':8.7e-7 *20,
					'preprocessed_target_loss':0.,
					'target_depth_smoothness_loss':0.,
					'hull':0.,
					'negative':0.,
					'smoothness_loss':0.0,#6.8e-14,
					'smoothness_neighbours':3,
					'smoothness_loss_2':0.0,#8.2e-14,
					'temporal_smoothness_loss':0.0,#0.2
					'discriminator_loss':0.0,
					'warp_loss':0.0,
				},
				#settings for remaining frames
				'iterations':2400,#5000,
				'seq_init':"WARP", #WARP, COPY, BASE
				'learning_rate':{'type':'exponential', 'start':3.0, 'base':0.5, 'scale':1/3000}, #{'type':'exponential', 'start':0.00005, 'base':0.5, 'scale':2/30000},
				
				'raw_target_loss':8.7e-7 *20,
				'preprocessed_target_loss':0.,
				'target_depth_smoothness_loss':0.,
				'hull':0.,
				'negative':0.,
				'smoothness_loss':0.0,#6.8e-14,
				'smoothness_neighbours':3,
				'smoothness_loss_2':0.0,#8.2e-14,
				'temporal_smoothness_loss':0.0,#0.2
				'discriminator_loss':0.0,
				'warp_loss':0.0,
				
				'inspect_gradients':1,
			},
			'learning_rate':{'type':'exponential', 'start':2.45, 'base':0.5, 'scale':2/30000},#0.00015, #[0.00001,0.00001,0.0001, 0.00009/20000, 4000],
			
			'raw_target_loss':8.7e-7 *20,#for AE; for SE *40; for Huber *80
			'preprocessed_target_loss':0.,
			'target_depth_smoothness_loss':0.,
			'hull':0.,#1e-12,
			'negative':0.,#1e-12,
			
			'smoothness_loss':0.0, 
			'smoothness_neighbours':3, # the kind of neighbourhood to consider in the edge filter (e.g. wether to use diagonals), NOT the kernel size.
			'smoothness_loss_2':0.0,
			'temporal_smoothness_loss':0.0,#0.2
			
			'discriminator_loss':1.5e-5,
			'warp_loss':[6.7e-11 *4,6.7e-11 *4,13.4e-11 *4, 6.7e-11 *4/2000, 2000],#for AE; for SE *8, for Huber *24
			
			'main_warp_fwd':False,
			'warp_gradients':{
				'weight':sSchedule.setup_constant_schedule(start=1.0),
				'active':False,
				'decay':sSchedule.setup_constant_schedule(start=0.9), #[0,1], lower is faster decay
				'update_first_only':False,
			},
			"view_interpolation":{
				"steps":0,
			},
			
			'grow':{ 
				"factor":1.2,#2. factor per interval, max down-scaling is factor^len(intervals)
				"pre_grow_actions":[],# "WARP", list, unused
				"post_grow_actions":[],# "WARP", list
				#iterations for each grow step, empty to disable
				'intervals':[],
			},
		},
		'velocity':{
			'optim_beta':0.9,
			'warp_order':2,
			'warp_clamp':"MC_SMOOTH",
			'error_functions':{
				'density_warp_loss':'SE',
				'velocity_warp_loss':'SE',
				'divergence_loss':'SE',
				'magnitude_loss':'SE',
			},
			'pre_optimization':True, #whether pre-optim will run for velocity, affects forward propagation/advection of state and optimization
			'pre_opt':{
				'first':{ #settings for first frame
					'iterations':10000,
					'learning_rate':0.04,
					
					'density_warp_loss':8.2e-11 *5,
					'velocity_warp_loss':0.0,
					'smoothness_loss':0.0,
					'smoothness_neighbours':3,
					'cossim_loss':0.0,
					'divergence_loss':sSchedule.setup_exponential_schedule(start=4.3e-10 *2, base=1.2, scale=1/10000), #sSchedule.setup_linear_schedule_2(start=0, end=4.3e-10, steps=7000),#
					#'divergence_normalize':0.0,
					#adjust according to data.step, large values can lead to NaN.
					'magnitude_loss':0.0,#{'type':'exponential', 'start':4e-11, 'base':0.10, 'scale':1/5000},
					'grow':{
						"factor":1.2,
						"scale_magnitude":True,
						'intervals':[200, 260, 330, 400, 460, 530, 600, 660, 800, 930, 1060, 1260], #7490
					},
				
				},
				#settings for remaining frames
				'iterations':1200,
				'seq_init':"WARP", #WARP, COPY, BASE
				'learning_rate':0.02,
				
				'density_warp_loss':8.2e-11 *5,
				'velocity_warp_loss':0.0,
				'smoothness_loss':0.0,
				'smoothness_neighbours':3,
				'cossim_loss':0.0,
				'divergence_loss':4.3e-10 *6,
				#'divergence_normalize':0.0,
				'magnitude_loss':0.0,#4e-12,
				
				'grow':{
					"factor":1.2,#2. factor per interval, max down-scaling is factor^len(intervals)
					"scale_magnitude":True,
					'intervals':[],
				},
			},
			'noise_std':sSchedule.setup_constant_schedule(start=0.0),
			'learning_rate':{'type':'exponential', 'start':0.02, 'base':0.5, 'scale':2/30000},
		#	'lr_decay':0.00,
			
		#	'loss':'L2',
			
			'density_warp_loss':8.2e-11 *5,#for AE; for SE *10; for Huber *25 #influence of loss(A(dt, vt), dt+1) on velocity, can be a schedule
			'velocity_warp_loss':sSchedule.setup_linear_schedule_2(start=1.35e-11 *3, end=1.35e-11 *6, steps=5000), #2.7e-12 *5,#for AE; for SE *10; for Huber *20 #influence of loss(A(vt, vt), vt+1) on velocity, can be a schedule
			
			'smoothness_loss':0.0,
			'smoothness_neighbours':3, # the kind of neighbourhood to consider in the edge filter (e.g. wether to use diagonals), NOT the kernel size.
			'cossim_loss':0.0,
			
			'divergence_loss':sSchedule.setup_exponential_schedule(start=4.3e-10 *6, base=1.2, scale=1/500),
			'divergence_normalize':0.0,
			
			'magnitude_loss':0.0,#1e-12,
			
			'warp_gradients':{
				'weight':sSchedule.setup_constant_schedule(start=1.0), #affects warp gradients for velocity from backward dens warp, even if vel- warp gradients are inactive
				'active':False,
				'decay':sSchedule.setup_constant_schedule(start=0.9), #[0,1], lower is faster decay
			},
			
			'grow':{
				"factor":1.2,#2. factor per interval, max down-scaling is factor^len(intervals)
				"scale_magnitude":True,
				'intervals':[],
			},
		},
		'optimize_buoyancy':False,
		"light":{
			"optimize":False,
			'optim_beta':0.9,
			"min":0.01,
			"max":6.0,
			"learning_rate":{'type':'exponential', 'start':0.001, 'base':1, 'scale':0},
		},
	#	"scattering":{
	#		"optimize":False,
	#		'optim_beta':0.9,
	#		"min":0.01,
	#		"max":1.0,
	#		"learning_rate":{'type':'exponential', 'start':0.001, 'base':1, 'scale':0},
	#	},
		
		
		'discriminator':{
			'active':False,
			'model':None,#'[RUNID:000000-000000]disc_model.h5',#
			'loss_type':"RaLSGAN", #"SGAN", "RpSGAN", "RpLSGAN", "RaSGAN", "RaLSGAN"
			'target_label':1.0,#0.9, #0.9 for label smoothing, 1.0 for LS-GAN 
			# l4s
			'layers':[16,16,24,24,32,32,32,64,64,64,16, 4],
			'stride':[ 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1],
			'kernel_size':4,
			'padding':'MIRROR', #NONE(=valid), ZERO(=same), MIRROR(=tf.pad(REFLECT))
			'activation':'lrelu', # relu or lrelu
			'activation_alpha':0.2, # leak for lrelu
			'use_fc':False, #usually makes disc stronger
			'noise_std':0.0,
			'start_delay':0, #50,
			'pre_opt':{
				'first':{
					'train':False,
					'learning_rate':{'type':'exponential', 'start':4e-4, 'base':0.5, 'scale':4/30000},
			#		'steps':1,
					'regularization':0.002,
				},
				'train':False,
				'learning_rate':1.6e-4,
			#	'steps':1,
				'regularization':0.002,
			},
			'train':True,
			'learning_rate':2e-4,
			'steps':1,
			'regularization':0.002,
			'optim_beta':0.5,
			
			'grow':{ # not yet used
				"factor":2.,#2. factor per interval, max down-scaling is factor^len(intervals)
				#iterations for each grow step, empty to disable
				'intervals':[]
			},
			
			'conditional_hull':False,
			'temporal_input':{
				'active':False,
				'step_range':(-3,4,1), #-3 to 3 inclusive, 0 will be removed
			},
			'num_real':4,
			'cam_res_down':6,
			'num_fake':3,
			'fake_camera_jitter':False,
			"history":{
				'load':None,
				'samples':4, #use older samples as fake samples as well. 0 to disable
				'size':800, #
				'keep_chance':0.01, # chance to put a rendered sample in the history buffer
				'save':False,
				'sequence_reuse':True,
				'reset_on_density_grow':True,
			#	'reset_on_discriminator_grow':False,
			},
			
		#	'sequence_reuse':True,
		},#discriminator
		'summary_interval':200,
	},#training
	'validation':{
		'output_interval':100,
		'stats':True,
		'cmp_scalarFlow':True,
		'cmp_scalarFlow_render':False,
		'warp_test':True,
		'warp_test_render':True,
		'render_cycle':False,
		'render_cycle_steps':8,
		'render_density':True,
		'render_shadow':True,
		'render_target':True,
		'render_velocity':True,
	},
	'debug':{
		'disc_dump_samples':False,
	},
}

RECONSTRUCT_SEQUENCE_SETUP_GROW = {
	"training":{
		'iterations':7000,
		"density":{
			'pre_opt':{
				'first':{
					'iterations':400,
				},
				'iterations':400,
			},
			'grow':{ 
				"factor":1.2,
				"pre_grow_actions":[],
				"post_grow_actions":[],
				'intervals':[200,300,400,500,500,600,700,800], #4000
				#'intervals':[400,400,400,400,400,400,400,400], #3200
			},
		},
		'velocity':{
			'pre_opt':{
				'first':{
					'iterations':5000,
					'divergence_loss':sSchedule.setup_exponential_schedule(start=4.3e-10 *2, base=1.2, scale=1/1000),
					'grow':{
						"factor":1.2,
						"scale_magnitude":True,
						'intervals':[660, 860, 1100, 1300], #3920
					},
				},
				'iterations':400,
				'divergence_loss':4.3e-10 *6,
			},
			'divergence_loss':sSchedule.setup_exponential_schedule(start=4.3e-10 *6, base=1.2, scale=1/700),
			'grow':{
				"factor":1.2,
				"scale_magnitude":True,
				'intervals':[200,300,400,500,500,600,700,800], #4000
			},
		},
	},
}
RECONSTRUCT_SEQUENCE_SETUP_GROW = _CSETUPS_update_dict_recursive(RECONSTRUCT_SEQUENCE_SETUP_BASE, RECONSTRUCT_SEQUENCE_SETUP_GROW, deepcopy=True, new_key='ERROR')

RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP = {
	"training":{
		'frame_order':'BWD',
		"density":{
			'main_warp_fwd':True,
			'warp_gradients':{
				'weight':1.0,
				'active':True,
				'decay':0.9,
				'update_first_only':True,
			},
		},
		'velocity':{
			'warp_gradients':{
				'weight':1.0,
				'active':False,
				'decay':0.9,
			},
		},
	},
}

RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP = _CSETUPS_update_dict_recursive(RECONSTRUCT_SEQUENCE_SETUP_GROW, RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP, deepcopy=True, new_key='ERROR')

RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP_FRONT = {
	"desc":"front-loaded training. 50% longer pre-training, linear growth intervals (longer low res, shorter high res)",
	"training":{
		'iterations':4200,
		"density":{
			'pre_opt':{
				'first':{
					'iterations':600,
				},
				'iterations':600,
			},
			'grow':{ 
				"factor":1.2,
				"pre_grow_actions":[],
				"post_grow_actions":[],
				#'intervals':[200,300,400,500,500,600,700,800], #4000
				'intervals':[400,400,400,400,400,400,400,400], #3200
			},
		},
		'velocity':{
			'pre_opt':{
				'first':{
					'iterations':6000,
					'divergence_loss':sSchedule.setup_exponential_schedule(start=4.3e-10 *2, base=1.2, scale=1/1000),
					'grow':{
						"factor":1.2,
						"scale_magnitude":True,
						'intervals':[1000, 1000, 1000, 1000], #4000
					},
				},
				'iterations':600,
				'divergence_loss':4.3e-10 *6,
			},
			'divergence_loss':sSchedule.setup_exponential_schedule(start=4.3e-10 *6, base=1.2, scale=1/400),
			'grow':{
				"factor":1.2,
				"scale_magnitude":True,
				'intervals':[400,400,400,400,400,400,400,400], #3200
			},
		},
	},
}

RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP_FRONT = _CSETUPS_update_dict_recursive(RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP, RECONSTRUCT_SEQUENCE_SETUP_GROW_DGRADWARP_FRONT, deepcopy=True, new_key='ERROR')


def reconstruct_sequence_setup_compatibility(setup, log_func=lambda s, *p: None):
	'''
		backwards compat for changes in the configuration
	'''
	setup = copy.deepcopy(setup)
	setup = munch.munchify(setup)
	# changed/replaced keys
	def log_key_update(k1, v1, k2, v2):
		log_func("Update old key '%s' with value '%s' to '%s' with value '%s'", k1, v1, k2, v2)
	#	target type
	if "synthetic_target" in setup.data.density and not "target_type" in setup.data.density:
		setup.data.density.target_type = "SYNTHETIC" if setup.data.density.synthetic_target else "RAW"
		log_key_update('setup.data.density.synthetic_target', setup.data.density.synthetic_target, 'setup.data.density.target_type', setup.data.density.target_type)
		del setup.data.density.synthetic_target
	
	# adjustments for mechanical changes
	#	if using data from before the sampling step correction: multiply density with 256, warn about loss and shadow/light scaling

# old setups