import os, copy, json, datetime, math
import numpy as np
import collections.abc

def _get_base_schedule():
	return {'type':'CONST', 'start':0.0,
		 'min':float('-inf'), 'max':float('inf'), 'step':0, 'offset':0,
		 'base':2.0, 'scale':1.0}
def setup_constant_schedule(*, start, **kwargs):
	d = _get_base_schedule()
	d.update({'type':'CONST', 'start':start})
	return d
def setup_linear_schedule(*, start, min=float('-inf'), max=float('inf'), step=0.0, offset=0, **kwargs):
	d = _get_base_schedule()
	d.update({'type':'LINEAR', 'start':start, 'min':min, 'max':max, 'step':step, 'offset':offset})
	return d
def setup_exponential_schedule(*, start, min=float('-inf'), max=float('inf'), base=2., scale=1., offset=0, **kwargs):
	if base=='e': base = np.e
	d = _get_base_schedule()
	d.update({'type':'EXPONENTIAL', 'start':start, 'min':min, 'max':max, 'base':base, 'scale':scale, 'offset':offset})
	return d
def setup_root_decay_schedule(*, start, min=float('-inf'), max=float('inf'), base=2., scale=1., offset=0, **kwargs):
	if base=='e': base = np.e
	d = _get_base_schedule()
	d.update({'type':'ROOT_DECAY', 'start':start, 'min':min, 'max':max, 'base':base, 'scale':scale, 'offset':offset})
	return d

def setup_boolean_schedule(*, start, offset=0, **kwargs):
	d = _get_base_schedule()
	d.update({'type':'BOOLEAN', 'start':start, 'offset':offset})
	return d

def setup_linear_schedule_2(*, start, end, steps, offset=0, **kwargs):
	d = _get_base_schedule()
	d.update({'type':'LINEAR', 'start':start, 'min':start if start<end else end, 'max':end if start<end else start, 'step':(end-start)/steps, 'offset':offset})
	return d

def setup_exponential_schedule_2(*, start, min=float('-inf'), max=float('inf'), base=2., steps=1., offset=0, **kwargs):
	if base=='e': base = np.e
	d = _get_base_schedule()
	d.update({'type':'EXPONENTIAL', 'start':start, 'min':min, 'max':max, 'base':base, 'scale':1./steps, 'offset':offset})
	return d
#def setup_exponential_schedule_3(*, start, end, min=float('-inf'), max=float('inf'), base=2., steps=1., offset=0, **kwargs):
#	if base=='e': base = np.e
#	d = _get_base_schedule()
#	d.update({'type':'EXPONENTIAL', 'start':start, 'min':min, 'max':max, 'base':base, 'scale':scale, 'offset':offset})
#	return d

def make_constant_schedule(*, start, **kwargs):
	return lambda it: start
def make_linear_schedule(*, start, min=float('-inf'), max=float('inf'), step=0.0, offset=0, **kwargs):
	return lambda it: np.clip(start + step*(it-offset), min, max)
def make_exponential_schedule(*, start, min=float('-inf'), max=float('inf'), base=2., scale=1., offset=0, **kwargs):
	if base=='e': base = np.e
	return lambda it: np.clip(start *(base**((it-offset)*scale)), min, max)
def make_root_decay_schedule(*, start, min=float('-inf'), max=float('inf'), base=2., scale=1., offset=0, **kwargs):
	if base=='e': base = np.e
	return lambda it: np.clip(start /(1. + (np.maximum(0,it-offset)*scale)**(1./base)), min, max)
def make_boolean_schedule(*, start, offset=0, **kwargs):
	"""has value of start as long as iteration<offset, flips start afterwards.
	always on (no negative iterations): start=False, offset=0. (flip immediately)
	"""
	start = bool(start)
	return lambda it: ((offset<=it) ^ start)
SCHEDULE_FACTORIES = {
	'CONST':make_constant_schedule,
	'LINEAR':make_linear_schedule,
	'EXPONENTIAL':make_exponential_schedule,
	'ROOT_DECAY':make_root_decay_schedule,
	'BOOLEAN':make_boolean_schedule,
}
def make_schedule(setup):
	if isinstance(setup, collections.abc.Mapping):
		if 'type' not in setup : raise ValueError("Missing schedule type")
		return make_schedule_type(**setup)
	elif np.isscalar(setup):
		if isinstance(setup, bool):
			return make_schedule_type(type='BOOLEAN', start=not setup, offset=-1)
		else:
			return make_schedule_type(type='CONST', start=setup)
	elif isinstance(setup, collections.abc.Iterable) and len(setup)==5:
		start, min, max, step, offset = setup
		return make_schedule_type(type='LINEAR', start=start, min=min, max=max, step=step, offset=offset)

def make_schedule_type(type,**kwargs):
	if type.upper()=='SCHEDULE':
		v = copy.deepcopy(kwargs)
		v['type'] = 'SCHEDULE'
		return lambda it: scalar_schedule(v, it)
	if type.upper() not in SCHEDULE_FACTORIES:
		raise ValueError("Unsupported schedule type '{}'".format(type))
	return SCHEDULE_FACTORIES[type.upper()](**kwargs)

def linear_schedule(start, iteration, min=float('-inf'), max=float('inf'), step=0.0, offset=0, **kwargs):
	return np.clip(start + step*(iteration-offset), min, max)
def exponential_schedule(start, iteration, min=float('-inf'), max=float('inf'), base=2., scale=1., offset=0, **kwargs):
	if base=='e': base = np.e
	return np.clip(start *(base**((iteration-offset)*scale)), min, max)
def logarithmic_decay_schedule(start, iteration, min=float('-inf'), max=float('inf'), base=2., scale=1., offset=0, **kwargs):
	if base=='e': base = np.e
	return np.clip(start /(1. + math.log(np.maximum(0.,iteration-offset)*scale + 1.)), min, max)
def root_decay_schedule(start, iteration, min=float('-inf'), max=float('inf'), base=2., scale=1., offset=0, **kwargs):
	if base=='e': base = np.e
	return np.clip(start /(1. + (np.maximum(0.,iteration-offset)*scale)**(1./base)), min, max)

def boolean_schedule(start, iteration, offset=0, **kwargs):
	return ((offset<=iteration) ^ bool(start))

def scalar_schedule(v, iteration):
	if isinstance(v, collections.abc.Mapping):
		if 'type' not in v : raise ValueError("Missing schedule type")
		if v['type'].upper()=='SCHEDULE':#list of other scalar schedules: [(iterations, schedule),...]
			it=0
			for iterations, schedule in v['schedule']:
				s = schedule
				if iteration<(it + iterations):
					break
				it += iterations
			return scalar_schedule(s, iteration-it) #these are scaled relative to the end of the previous schedule
			
		if 'start' not in v: raise ValueError("Missing start value")
		if v['type'].upper()=='LINEAR':
			return linear_schedule(iteration=iteration, **v)
		elif v['type'].upper()=='CONST':
			return v['start']
		elif v['type'].upper()=='EXPONENTIAL':
			return exponential_schedule(iteration=iteration, **v)
		elif v['type'].upper()=='LOG_DECAY':
			return logarithmic_decay_schedule(iteration=iteration, **v)
		elif v['type'].upper()=='ROOT_DECAY':
			return root_decay_schedule(iteration=iteration, **v)
		else:
			raise ValueError("Unkown schedule type '{}'".format(v['type']))
	elif np.isscalar(v):
		return v
	elif isinstance(v, collections.abc.Iterable) and len(v)==5:
		start, min, max, step, offset = v
		return linear_schedule(start, iteration, min, max, step, offset)

def convert_setup_to_schedule(setup, keys):
	schedule = copy.deepcopy(keys)
	
	def _convert(setup, schedule):
		for k, v in schedule.items():
			if k not in setup:
				raise AttributeError("No schedule setup for '{}'".format(k))
			if isinstance(v, collections.abc.Mapping):
				_convert(setup[k], v)
			else:
				try:
					if isinstance(setup[k], collections.abc.Mapping):
						schedule[k] = make_schedule(**setup[k])
					elif np.isscalar(setup[k]):
						schedule[k] = make_constant_schedule(start=setup[k])
					elif len(setup[k])==5:
						schedule[k] = make_linear_schedule(start=setup[k][0], min=setup[k][1], max=setup[k][2], step=setup[k][3], offset=setup[k][4])
					else: raise ValueError("Can't parse schedule type")
				except:
					raise ValueError("Failed to make schedule for '{}' from {}".format(k, setup[k]))
	
	_convert(setup, schedule)
	return schedule

def plot_schedule(schedule, iterations, path, title="Scalar Schedule"):
	import matplotlib.pyplot as plt
	values = [scalar_schedule(schedule, _) for _ in range(iterations)]
	plt.plot(values)
	plt.xlabel('Iteration')
	plt.ylabel('Value')
	plt.title(title)
	plt.savefig(path)
	plt.clf()

def plot_schedules(schedules, iterations, path, labels, title="Scalar Schedules"):
	import matplotlib.pyplot as plt
	assert len(schedules)==len(labels)
	for schedule, label in zip(schedules, labels):
		values = [scalar_schedule(schedule, _) for _ in range(iterations)]
		plt.plot(values, label=label)
	plt.xlabel('Iteration')
	plt.ylabel('Value')
	plt.title(title)
	plt.legend()
	plt.savefig(path)
	plt.clf()