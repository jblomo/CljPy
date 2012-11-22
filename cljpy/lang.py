"""Supporting classes for Clojure language features."""

class Delay:
	"""The Delay object is used to prevent a function from running until forced.
	Upon being forced, it will cache the result for all subsequent force
	calls."""
	def __init__(self, delayed_fn, args, kwargs):
		self.delayed_fn = delayed_fn
		self.args = args
		self.kwargs = kwargs

	def force(self):
		if not self.realized_p():
			self.state = self.delayed_fn(*self.args, **self.kwargs)

		return self.state

	deref = force

	def realized_p(self):
		return hasattr(self, 'state')

class Promise:
	"""The Promise object is used to return a token to a caller and have the
	value calculated at a later time or (TODO) on another thread."""

	def deliver(self, val):
		"""Delivers the supplied value to the promise, releasing any pending
		derefs.  A subsequent call to deliver on a promise will throw an
		exception."""
		if not hasattr(self, 'state'):
			self.state = val
		else:
			raise RuntimeError("promise %r already delivered!" % self)

		# TODO unblock waiting threads
		return self

	def deref(self, *args):
		"""Returns delivered promise.  If a promise hasn't been delivered yet, block.
		
		TODO: optional args timeout_ms and timeout_val will return timeout_val
		if call is blocked for more than timeout ms."""
		try:
			return self.state
		except AttributeError:
			return NotImplementedError("Blocking not implemented yet")

