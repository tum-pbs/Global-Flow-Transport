
import subprocess, datetime
import os, shutil

def compile(file_name, compile_cuda=True, create_backup=True):
	srcPath = os.path.abspath("./phitest/render/cuda/src")
	buildPath = os.path.abspath("./phitest/render/cuda/build")
	#buildPath = os.path.abspath("./phitest/render/cuda/test_build")
	now = datetime.datetime.now()
	now_str = now.strftime("%y%m%d-%H%M%S")
	bpkPath = os.path.join(os.path.abspath("./phitest/render/cuda/build_bkp"), now_str + "_" + file_name)
	print("Source Path:\t" + srcPath)
	print("Build Path:\t" + buildPath)

	# Get TF Compile/Link Flags and write to env
	import tensorflow as tf
	print('tensorflow version', tf.__version__)
	TF_CFLAGS = tf.sysconfig.get_compile_flags()
	TF_CFLAGS += ['-I'+os.path.abspath('./lib/glm')]
	TF_LFLAGS = tf.sysconfig.get_link_flags()
	
	print('compile flags:', TF_CFLAGS)
	print('link flags:', TF_LFLAGS)
	
	
	# Remove old build files
	if os.path.isdir(buildPath):
		build_dir = list(os.listdir(buildPath))
		# Backup old build files
		if create_backup:
			print("Create backup for '%s' in '%s'" %(file_name, bpkPath))
			os.makedirs(bpkPath)
			for f in build_dir:
				if file_name in f:
					shutil.copy2(os.path.join(buildPath, f), os.path.join(bpkPath, f))
		print("Removing old build files from %s" % buildPath)
		for f in build_dir:
			if file_name in f and (compile_cuda or '.cu.o' not in f):
				os.remove(os.path.join(buildPath, f))
	else:
		print("Creating build directory at %s" % buildPath)
		os.mkdir(buildPath)

	print("Compiling CUDA code...")
	# Build the Laplace Matrix Generation CUDA Kernels
	
	if compile_cuda:
		subprocess.check_call(['nvcc',
							   "-std=c++11",
							   "-c",
							   "-o",
							   os.path.join(buildPath, file_name+'.cu.o'),
							   os.path.join(srcPath, file_name+'.cu.cc'),
							   "-D GOOGLE_CUDA=1",
							   "-x",
							   "cu",
							   "-Xcompiler",
							   "-fPIC"]
							   + TF_CFLAGS)
	
	subprocess.check_call(['gcc',
						   "-std=c++11",
						   "-shared",
						   "-o",
						   os.path.join(buildPath, file_name+'.so'),
						   os.path.join(srcPath, file_name+'.cc'),
						   os.path.join(buildPath, file_name+'.cu.o'),
						   "-D GOOGLE_CUDA=1",
						   "-fPIC",]
						   + TF_CFLAGS + TF_LFLAGS)

if __name__=='__main__':
	bkp = input("Backup previous build? [Y/n]:").lower()
	if bkp in ["", "y", "yes", "t", "true", "1"]:
		bkp = True;
	elif bkp in ["n", "no", "f", "false", "0"]:
		bkp = False;
	compile('resample_grid', create_backup=bkp)
	compile('raymarch_grid', create_backup=bkp)
	compile('reduce_blend', create_backup=bkp)