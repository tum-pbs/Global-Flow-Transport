import os, zipfile, json
#from contextlib import contextmanager

def archive_files(archive, *files):
	with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
		for file in files:
			z.write(file)

def json_dump_archive(path, obj, **json_args):
	'''Write a compressed json object to path/name.zip/name.json, (name is without extension)'''
	dirname, basename = os.path.split(path)
	with zipfile.ZipFile(path+".zip", mode="w", compression=zipfile.ZIP_DEFLATED) as z:
		z.writestr(basename, json.dumps(obj, **json_args))
def json_load_archive(path, **json_args):
	'''Read a compressed json object from path/name.zip/name.json, (name is without extension)'''
	dirname, basename = os.path.split(path)
	with zipfile.ZipFile(path+".zip", mode="r") as z:
		with z.open(basename) as f:
			return json.load(f, **json_args)

def json_load(file, **json_args):
	'''path is without extension'''
	if isinstance(file, str):
		if os.path.isfile(file):
			with open(file, "r") as f:
				return json.load(f, **json_args)
		elif os.path.isfile(file+".zip"):
			return json_load_archive(file, **json_args)
		else:
			raise OSError("Could not find json '%s'"%file)

def json_dump(path, obj, compressed=False, makedirs=False, **json_args):
	'''path is without extension'''
	dirname, basename = os.path.split(path)
	if makedirs:
		os.makedirs(dirname, exist_ok=True)
	if not compressed:
		with open(path, "w") as f:
			json.dump(obj, f, **json_args)
	else:
		json_dump_archive(path, obj, **json_args)