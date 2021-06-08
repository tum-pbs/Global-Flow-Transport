import sys, os, time

def mkdirs(path):
	if not os.path.exists(path):
		os.makedirs(path)

def timestring():
	sec = time.time()
	ts = time.gmtime(sec)
	s = '{:04d}.{:02d}.{:02d}-{:02d}:{:02d}:{:02d}.{:03d}'.format(ts.tm_year, ts.tm_mon, ts.tm_mday, ts.tm_hour, ts.tm_min, ts.tm_sec, int((sec%1.0)*1000))
	return s

class StreamLogger(object):
	def __init__(self, file, term=sys.stdout):
		self.terminal = term
		self.log = open(file, "a") 
	def write(self, message):
		self.terminal.write(message)
		self.log.write(message) 
	def flush(self):
		self.log.flush()


class StreamCapture(object):
	def __init__(self, file, stream=sys.stdout):
		self.stream = stream
		self.log = open(file, "a") 
	def write(self, msg):
		self.stream.write(msg)
		self.log.write(msg) 
	def flush(self):
		self.stream.flush()
		self.log.flush()
	def close(self):
		self.log.close()
		return self.stream
	def __del__(self):
		self.flush()
		self.close()

def init(path, error_log=True):
	sys.stdout = StreamLogger(os.path.join(path, 'logfile.txt'), sys.stdout)
	if error_log:
		sys.stderr = StreamLogger(os.path.join(path, 'errorlog.txt'), sys.stderr)
		
		

#log levels
LOG_LEVEL_FATAL = 0
LOG_LEVEL_ERROR = 1
LOG_LEVEL_WARNING = 2
LOG_LEVEL_INFO = 3
LOG_LEVEL_PRINT = 4 # used for normal 'print()' function
LOG_LEVEL_DEBUG = 5
LOG_LEVEL_TRACE = 6

LOG_LEVEL_DEFAULT = LOG_LEVEL_PRINT #LOG_LEVEL_INFO
LOG_LEVELS = [LOG_LEVEL_FATAL, LOG_LEVEL_ERROR, LOG_LEVEL_WARNING, LOG_LEVEL_INFO, LOG_LEVEL_PRINT, LOG_LEVEL_DEBUG, LOG_LEVEL_TRACE]

class Logger(object):
	class StreamHook(object):
		def __init__(self, write_func, flush_func):
			self.write_func = write_func
			self.flush_func = flush_func
		def write(self, msg):
			self.write_func(msg, end='')
		def flush(self):
			self.flush_func()
	
	'''
		path: directory to write the log files to
		clear_log: if True, existing log files are removed. Default False.
		log_level: the maximum log level of messages to handle. Errors are always logged.
		terminal_log_level: the maximum log level of messages to print to stdout. higher than log_level has no effect.
	'''
	def __init__(self, path, clear_log=False, log_level = LOG_LEVEL_TRACE, terminal_log_level = LOG_LEVEL_PRINT, terminal_timestamp=True):
		self.log_level = log_level
		if self.log_level not in LOG_LEVELS:
			self.log_level = LOG_LEVEL_DEFAULT
		self.terminal_log_level = terminal_log_level
		if self.terminal_log_level not in LOG_LEVELS:
			self.terminal_log_level = LOG_LEVEL_PRINT
		self.terminal_timestamp = terminal_timestamp
		
		self.path = path
		mkdirs(path)
		logfile = os.path.join(path, 'logfile.txt')
		errorlogfile = os.path.join(path, 'errorlog.txt')
		if clear_log:
			if os.path.exists(logfile):
				os.remove(logfile)
			if os.path.exists(errorlogfile):
				os.remove(errorlogfile)
		self.log = open(logfile, 'a')
		self.errlog = open(errorlogfile, 'a')
		#if self.log_level>=LOG_LEVEL_DEBUG:
		#	self.dbglog = open(os.path.join(path, 'debuglog.txt'), 'a')
		
		self.stdout = sys.stdout
		self.stderr = sys.stderr
		
		self.start_time = time.time()
		self.debug('Logger starting.')
		
		sys.stdout = self.StreamHook(self._pr_func, self.flush)
		sys.stderr = self.StreamHook(self._err_func, self.flush)
		
		self.info('Logger initialized.')
		#print('Print test')
		#raise TypeError('Test error')
	
	def set_log_level(self, log_level):
		self.log_level = log_level
	
	def set_terminal_log_level(self, log_level):
		self.terminal_log_level = log_level
	
	def _write(self, level, *args, end=None, write_timestamp=True, write_log_level=True):
		if self.log_level>=level or LOG_LEVEL_ERROR>=level:
			elapsed = time.time() - self.start_time
			timestamp = '[{}] '.format(timestring()) #'[{},{: 3d}:{:02d}:{:06.03f}] '.format(timestring(), int(elapsed//3600), int(elapsed%3600)//60, elapsed%60)
			level_string = '[{}] '.format(self._level_string(level))
			msg_end = end if end is not None else '\n'
			pre = '' 
			t_pre = ''
			if write_timestamp:
				pre+=timestamp
				if self.terminal_timestamp:
					t_pre+=timestamp
			if write_log_level:
				pre+=level_string
				t_pre+=level_string
			strs = [str(arg) for arg in args]
			msg = '{}{}'.format(' '.join(strs), msg_end)
			
			self.log.write(pre+msg)
			if LOG_LEVEL_ERROR>=level: #always write errors
				self.errlog.write(pre+msg)
				self.stderr.write(t_pre+msg)
			elif self.terminal_log_level >= level:
				self.stdout.write(t_pre+msg)
	
	def _err_func(self, *args, end=None):
		self._write(LOG_LEVEL_FATAL, *args, end=end, write_timestamp=False, write_log_level=False)
	# for normal 'print' function
	def _pr_func(self, *args, end=None, timestamp=False):
		self._write(LOG_LEVEL_PRINT, *args, end=end, write_timestamp=timestamp, write_log_level=False)
	
	def fatal(self, *args, end=None):
		self._write(LOG_LEVEL_FATAL, *args, end=end)
	def error(self, *args, end=None):
		self._write(LOG_LEVEL_ERROR, *args, end=end)
	def warning(self, *args, end=None):
		self._write(LOG_LEVEL_WARNING, *args, end=end)
	def info(self, *args, end=None, timestamp=True):
		self._write(LOG_LEVEL_INFO, *args, end=end, write_timestamp=timestamp)
	def print(self, *args, end=None, timestamp=True):
		self._pr_func(*args, end=end, timestamp=timestamp)
	'''
		helpful for general troubleshooting 
	'''
	def debug(self, *args, end=None, timestamp=True):
		self._write(LOG_LEVEL_DEBUG, *args, end=end, write_timestamp=timestamp)
	'''
		very detailed information for software debugging
	'''
	def trace(self, *args, end=None, timestamp=True):
		self._write(LOG_LEVEL_TRACE, *args, end=end, write_timestamp=timestamp)
	
	def newline(self):
		self._write(LOG_LEVEL_INFO, '', write_timestamp=False, write_log_level=False)
	
	def flush(self):
		self.log.flush()
		self.errlog.flush()
	
	def _level_string(self, log_level):
		if log_level == LOG_LEVEL_FATAL: return 'F'
		if log_level == LOG_LEVEL_ERROR: return 'E'
		if log_level == LOG_LEVEL_WARNING: return 'W'
		if log_level == LOG_LEVEL_INFO: return 'I'
		if log_level == LOG_LEVEL_PRINT: return 'P'
		if log_level == LOG_LEVEL_DEBUG: return 'D'
		if log_level == LOG_LEVEL_TRACE: return 'T'
		return '?'