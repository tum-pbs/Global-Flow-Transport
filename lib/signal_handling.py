import signal

class SignalHandler:
	def __init__(self, signalnum):
		self._signalnum = signalnum
		self._occured = False
		self._handler_fn = None
	
	def _handler(self, sig, frame):
		self._occured = True
		if self._handler_fn is not None:
			self._handler_fn(sig, frame)
	
	def set(self, *, handler_func=None):
		self._handler_fn = handler_func
		self.reset()
		signal.signal(self._signalnum, self._handler)
	
	def reset(self):
		self._occured = False
	
	def unset(self):
		if signal.getsignal(self._signalnum) is self._handler:
			signal.signal(self._signalnum, signal.SIG_DFL)
		self.reset()
	
	def close(self):
		self.unset()
	
	def check(self):
		return self._occured
	
	def wait(self, timeout=None):
		success = False
		if timeout is None:
			signal.sigwait({self._signalnum,})
			success = True
		else:
			success = (signal.sigtimedwait({self._signalnum,}, timeout) is not None)
		self._occured = success
		return success

class InterruptHandler(SignalHandler):
	def __init__(self):
		super().__init__(signal.SIGINT)