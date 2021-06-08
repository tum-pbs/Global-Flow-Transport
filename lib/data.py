import os, copy
import tensorflow as tf
import numpy as np
import numbers, collections.abc

import logging
log = logging.getLogger('data')
log.setLevel(logging.DEBUG)

def load_density_grid(sim_transform, path, density=0.06, frame=0, mask='density_{:06d}.npz', input_format='DHW', reverse_dims='', array_name='arr_0'):
	out_format = 'NDHWC'
	if mask is not None and len(mask)>0:
		filename = mask.format(frame)
		p = os.path.join(path, filename)
	else:
		filename = path
		p = path
	if p.endswith('.npy'):
		d = np.load(p)
	else:
		with np.load(p) as np_data:
			try:
				d = np_data[array_name].astype(np.float32)
			except KeyError:
				raise KeyError('key \'{}\' not in archive. Available keys: {}'.format(array_name, list(np_data.keys())))
	log.info('loaded density grid \'{}\' from \'{}\' with shape {}.'.format(filename, path, d.shape))
	if len(d.shape)!=len(input_format):
		raise ValueError('Given input format {} does not match loaded data shape {}'.format(input_format, d.shape))
#	print_stats(d, filename, log)
	#d = d[:,:128]
	reverse = [input_format.index(_) for _ in reverse_dims]
	d = tf.reverse(d, reverse)
	for dim in out_format:
		if dim not in input_format:
			d = tf.expand_dims(d, out_format.index(dim))
	d = tf.constant(d*density, dtype=tf.float32)
	#d = tf_pad_to_next_pow_two(d)
	sim_transform.set_data(d)
	return d

def load_targets(pack_path_mask, simulations=[0], frames=[140], cam_ids=[0,1,2,3,4], threshold=0.0, bkg_subtract=True, flip_y=False):
	bkgs = []
	targets = []
	targets_raw = []
	for sim in simulations:
		log.debug('loading sim {}'.format(sim))
		with np.load(pack_path_mask.format(sim=sim, frame=0)) as np_data:
			bkg = np_data['data'][cam_ids].astype(np.float32)
			if flip_y:
				bkg = np.flip(bkg, axis=-3)
		bkgs.append(bkg)
		for frame in frames:
			log.debug('loading frame {}'.format(frame))
			with np.load(pack_path_mask.format(sim=sim, frame=frame)) as np_data:
				target_raw = np_data['data'][cam_ids].astype(np.float32)
				if flip_y:
					target_raw = np.flip(target_raw, axis=-3)
			if bkg_subtract:
				target = tf.maximum(target_raw-bkg, 0.0)
				if threshold>0.0: #cut off noise in background 
					condition = tf.greater_equal(target, threshold)
					target_raw = tf.where(condition, target_raw, bkg)
					target = tf.where(condition, target, tf.zeros_like(target))
				targets.append(target)
			targets_raw.append(target_raw)
	if bkg_subtract:
		return tf.concat(targets_raw, axis=0), tf.concat(targets, axis=0), tf.concat(bkgs, axis=0)
	else:
		return tf.concat(targets_raw, axis=0), tf.concat(bkgs, axis=0)

'''
def background_substract(img_bkg, bkg, threshold=1e-2, hull_blur=0.0):
	img_bsub = tf.maximum(img_bkg-bkg, 0)
	condition = tf.greater_equal(img_bsub, threshold)
	mask = 
'''

class ScalarFlowDataset(tf.data.Dataset):
	def _generator(sim_range, frame_range, cam_ids=[0,1,2,3,4]):
		for sim in range(*sim_range):
			with np.load(path_mask.format(sim=sim, frame=0)) as np_data:
				bkgs = np_data['data'][cam_ids].astype(np.float32)
			for frame in range(*frame_range):
				with np.load(path_mask.format(sim=sim, frame=frame)) as np_data:
					views = np_data['data'][cam_ids].astype(np.float32)
				log.debug('loaded frame {} of sim {}'.format(frame, sim))
				for view, bkg in zip(views, bkgs):
					yield (view, bkg)
				#yield (views, bkgs)
	
	def __new__(cls, sim_range, frame_range, cam_ids=[0,1,2,3,4]):
		return tf.data.Dataset.from_generator(
			cls._generator,
			output_types=(tf.float32, tf.float32),
			output_shapes=(tf.TensorShape([1920,1080,1]),tf.TensorShape([1920,1080,1])),
			args=(sim_range, frame_range, cam_ids)
		)

class ScalarFlowIndexDataset(tf.data.Dataset):
	def _generator(sim_range, frame_range):
		for sim in range(*sim_range):
			for frame in range(*frame_range):
				yield (sim, frame)
	
	def __new__(cls, sim_range, frame_range):
		return tf.data.Dataset.from_generator(
			cls._generator,
			output_types=(tf.int32, tf.int32),
			output_shapes=([],[]),
			args=(sim_range, frame_range)
		)

def get_scalarflow_dataset(sim_range, frame_range, path_mask, cam_ids=[0,1,2,3,4], down_scale=1, threshold=0.0, flat=True, cache=True, raw=True, preproc=True, bkg=True, hull=False, path_preproc=None, temporal_input_steps=None):
	'''
		create a dataset from scalarFlow target images.
		sim_range: the simulation ids to use. parameters for the range() object (start[, stop[, step]]).
		frame_range: the frame numbers to use from each simulation. parameters for the range() object (start[, stop[, step]]).
		path_mask: string to use with .format(sim, frame) to give the file-paths to the images.
		cam_ids: indices of the image packs to use.
		down_scale: integer. window for average pooling.
		threshold: threshodl for background and noise substraction.
		flat: bool. whether to flatten the dataset, camera id dimension.
		cache: cache the loaded dataset in RAM.
		raw, preproc, bkg, hull: bool. the data types to provide.
		path_preproc: load scalarFlow preprocessed images instad of raw images from this path if not None.
		temporal_input_steps: create prev-curr-next inputs with this frame step if not None. trncate frame_range accordingly to avoid out-of-bounds accesses.
	'''
	if not (raw or preproc or bkg or hull):
		raise ValueError("empty dataset")
	from_preproc = (path_preproc is not None)
	mask = path_preproc if from_preproc else path_mask
	
	make_temporal_input = False
	if temporal_input_steps is not None:
		make_temporal_input = True
		if isinstance(temporal_input_steps, numbers.Integral):
			temporal_input_steps = [temporal_input_steps]
		elif not isinstance(temporal_input_steps, collections.abc.Iterable):
			raise ValueError("Invalid temporal_input_steps.")
	
	def load(sim, frame):
		bkgs = None
		if bkg or preproc or hull:
			with np.load(path_mask.format(sim=sim, frame=0)) as np_data:
				bkgs = np_data['data'][cam_ids].astype(np.float32)
		with np.load(mask.format(sim=sim, frame=frame)) as np_data:
			views = np_data['data'][cam_ids].astype(np.float32)
		
		if make_temporal_input:
			t_step = np.random.choice(temporal_input_steps)
			with np.load(mask.format(sim=sim, frame=frame-t_step)) as np_data:
				views_prev = np_data['data'][cam_ids].astype(np.float32)
			with np.load(mask.format(sim=sim, frame=frame+t_step)) as np_data:
				views_next = np_data['data'][cam_ids].astype(np.float32)
			views = [views_prev, views, views_next]
		
		log.debug('loaded frame {} of sim {}'.format(frame, sim))
		return views, bkgs
	#load_func = lambda s,f: tf.py_func(load, [s,f], (tf.float32,tf.float32))
	
	def preprocess(img_view, img_bkg):
		img_view = tf.nn.avg_pool(img_view, (1,down_scale, down_scale, 1), (1,down_scale, down_scale, 1), 'VALID')
		img_bkg = tf.nn.avg_pool(img_bkg, (1,down_scale, down_scale, 1), (1,down_scale, down_scale, 1), 'VALID')
		#print(img_raw.shape)
		if from_preproc:
			img_raw = img_view + img_bkg
			view_preproc = img_view
		else:
			img_raw = img_view
			view_preproc = tf.maximum(img_raw-img_bkg, 0)
		if threshold>0.0: #cut off noise in background 
			condition = tf.greater_equal(view_preproc, threshold)
			if not from_preproc:
				img_raw = tf.where(condition, img_raw, img_bkg)
				if preproc: view_preproc = tf.where(condition, view_preproc, tf.zeros_like(view_preproc))
			if hull: img_hull = tf.cast(condition, tf.float32) #tf.where(condition, tf.ones_like(view_preproc), tf.zeros_like(view_preproc))
		ret = []
		if raw: ret.append(img_raw)
		if preproc: ret.append(view_preproc)
		if bkg: ret.append(img_bkg)
		if hull: ret.append(img_hull)
		return tuple(ret)
	#map_func = lambda v,b: tf.py_func(preprocess, [v,b], (tf.float32,tf.float32,tf.float32))
	
	def fused_load(sim, frame):
		views, bkgs = load(sim, frame)
		if make_temporal_input:
			return ( \
				*preprocess(views[0], bkgs), \
				*preprocess(views[1], bkgs), \
				*preprocess(views[2], bkgs), \
			)
		else:
			return preprocess(views, bkgs) #views_raw, views_preproc, bkgs
	ret_type = []
	types = []
	if raw:
		ret_type.append(tf.float32)
		types.append("img_raw")
	if preproc:
		ret_type.append(tf.float32)
		types.append("img_preproc")
	if bkg:
		ret_type.append(tf.float32)
		types.append("img_bkg")
	if hull:
		ret_type.append(tf.float32)
		types.append("hull")
	if make_temporal_input:
		ret_type *=3
	f_func = lambda s,f: tf.py_func(fused_load, [s,f], tuple(ret_type))
	#
	num_sims = len(range(*sim_range))
	num_frames = len(range(*frame_range))
	log.info("Initialize scalarFlow dataset:\n\ttypes: %s\n\t%d frames from %d simulations each%s", types, num_frames, num_sims, ("\n\ttemporal steps %s"%temporal_input_steps) if make_temporal_input else "")
	#,num_parallel_calls=4
	loaded_data = ScalarFlowIndexDataset(sim_range, frame_range).shuffle(num_sims*num_frames,reshuffle_each_iteration=False).map(f_func,num_parallel_calls=4).prefetch(8)
	if flat:
		#loaded_data = loaded_data.flat_map(lambda x,y,z: tf.data.Dataset.from_tensor_slices((x,y,z)))
		loaded_data = loaded_data.flat_map(lambda *d: tf.data.Dataset.from_tensor_slices(d))
#	if preproc_only:
#		loaded_data = loaded_data.map(lambda x,y,z: y)
	if cache:
		loaded_data = loaded_data.cache()
	return loaded_data.repeat().shuffle(64)
	