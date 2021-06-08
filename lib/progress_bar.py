import sys

#https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def progress_bar(curr, total, desc='', length=100, decimals=2):
	percent = ('{:.'+str(int(decimals))+'f}%').format(curr/total *100.0)
	filled = int(length * curr/total)
	bar = '='*filled + '-'*(length-filled)
	sys.stdout.write('[{}] ({}) {}\r'.format(bar, percent, desc))
	if curr>=total:
		sys.stdout.write('\n')
	sys.stdout.flush()

class ProgressBar:
	def __init__(self, max_steps, name=None, length=25, decimals=2, file=sys.stdout, *, active=True):
		self._file = file
		self._active = active
		self._max_steps = max(max_steps,1)
		self._perc_fmt = '{:.'+str(int(decimals))+'f}%'
		self._fmt = '\r{name}[{bar}] ({perc:.0'+str(int(decimals))+'f}%) {desc}'
		self._name = name if name is not None else ''
		self._bar_length = length
		
		self._closed = False
		self._last_print_length = 0
		self.update(0)
	
	def get_bar(self, perc):
		filled = int(self._bar_length * perc)
		return ''.join(['=']*filled + ['-']*(self._bar_length-filled))
	
	def update(self, curr_steps, desc=''):
		if self._closed:
			raise RuntimeError("This ProgressBar is already finished.")
		perc = curr_steps/self._max_steps
		msg = self._fmt.format(name=self._name, bar=self.get_bar(perc), perc=perc*100.0, desc=desc)
		
		print_length = len(msg)
		if print_length<self._last_print_length:
			msg += " "*(self._last_print_length - print_length)
		
		if self._active:
			self._file.write(msg)
			self._file.flush()
		self._last_print_length = len(msg)
	
	def finish(self, desc=""):
		self.update(self._max_steps, desc=desc)
		self.close()
	
	def close(self):
		if self._active:
			self._file.write('\n')
			self._file.flush()
		self._closed = True
	