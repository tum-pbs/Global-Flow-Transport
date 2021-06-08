import numpy as np
import importlib
#import numbers
#https://medium.com/python-pandemonium/json-the-python-way-91aac95d4041
def to_dict(obj):
	if obj is None or isinstance(obj, (int, bool, float, str, list, tuple, dict)):
		return obj
	if isinstance(obj, (np.ndarray, np.number)):
		# alternative: save ndarray as .npz and put path here. might need base path
		return obj.tolist()
	d = {
		"__class__":obj.__class__.__name__,
		"__module__":obj.__module__
	}
	if hasattr(obj, "to_dict"):
		d.update(obj.to_dict())
	else:
		d.update(obj.__dict__)
	return d

def from_dict(d):
	if d is None or isinstance(d, (int, bool, float, str, list, tuple)):
		return d
	elif "__class__" in d:
		cls_name = d.pop("__class__")
		mod_name = d.pop("__module__")
		mod = importlib.import_module(mod_name)
		cls = getattr(mod, cls_name)
		if hasattr(cls, "from_dict"):
			obj = cls.from_dict(d)
		else:
			obj = cls(**d)
	else:
		obj = d
	return obj