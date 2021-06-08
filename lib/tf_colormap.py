import numpy as np
import tensorflow as tf

import matplotlib.cm

def tf_cmap_nearest(data, map, min_val=0., max_val=1.):
	map = matplotlib.cm.get_cmap(map).colors
	map = tf.constant(map, dtype=tf.float32)
	
	data = tf.clip_by_value(data, min_val, max_val)
	#normalize to [0,1]
	data = (data - min_val) / (max_val - min_val)
	#quantize
	data *= map.get_shape().as_list()[0]-1
	data = tf.cast(data, dtype=tf.int32)
	#map
	return tf.reduce_mean(tf.gather(map, data), axis=-2)