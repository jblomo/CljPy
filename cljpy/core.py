"""Implements functions in clojure.core"""
import copy
import inspect
import itertools
import operator

from collections import defaultdict
from decimal import Decimal
from functools import wraps
from numbers import Integral

from lang import Delay, Promise

# override this method to change the way nondestructive operations copy
# arguments
_copy = copy.deepcopy

__version__ = "0.0"

def accessor(s, key):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of a structmap?
	"""
	raise NotImplementedError()

def aclone(array):
	"""Returns a clone of the array/list"""
	return list(array)

def add_watch(reference, key, fn):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of a reference?
	"""
	raise NotImplementedError()

def agent(state, **options):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of an agent?
	"""
	raise NotImplementedError()

def agent_error(a):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of a structmap?
	"""
	raise NotImplementedError()

def agent_errors(a):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of a structmap?
	"""
	raise NotImplementedError()

def aget(array, idx, *idxs):
	"""Returns the value at the index/indices.
	
	Optional idxs arguement will be used to index multidimensional arrays.
	"""
	result = array[idx]
	if idxs:
		return aget(result, *idxs)
	else:
		return result

alength = len

def alias(name, namespace_sym):
	"""NOT IMPLEMENTED

	TODO: not sure if this function is possible, learn more about modules.
	"""
	raise NotImplementedError()

def all_ns():
	"""NOT IMPLEMENTED

	TODO: Not sure if you can inspect the calling environment. Could do
	something like
	ifilter(lambda m: inspect.ismodule(eval(m)), dir())
	"""
	raise NotImplementedError()

def alter(ref, fn, *args):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of a ref?
	"""
	raise NotImplementedError()

def alter_meta__(ref, f, args):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of a ref?
	"""
	raise NotImplementedError()
	ref.__meta__ = apply(f, [ref.__meta__] + args)

def alter_var_root(v, f, *args):
	"""NOT IMPLEMENTED

	TODO: can you change calling environment?
	"""
	raise NotImplementedError()

def amap(array, idx, ref, expr):
	"""NOT IMPLEMENTED

	TODO: I'm not even sure what this does in Clojure
	"""
	raise NotImplementedError()

def ancestors(h, tag):
	"""Returns the immediate and indirect parents of tag, either via a Python
	class inheritance relationship or a relationship established via derive. h
	must be a hierarchy obtained from make-hierarchy, if not supplied defaults
	to the global hierarchy."""

	if tag is None:
		# shift args over
		tag = h
		h = derive.global_ns
	
	mro = inspect.getmro(tag) if inspect.isclass(tag) else set()
	return frozenset(h['ancestors'][tag] | mro)

def and_(*args):
	"""Return True if all ags are logically true.

	Note: and_ is not lazy and arguments will be evaluated before they are processed."""
	return all(args)

apply = apply

def areduce(array, idx, result, init, expr):
	"""NOT IMPLEMENTED
	"""
	raise NotImplementedError()

array_map = dict

def aset(array, idx, *idxv):
	"""Sets the value at the index/indices.  Returns val.

	Optional idxs arguement will be used to index multidimensional arrays.
	"""
	if len(idxv) == 1:
		array[idx] = idxv[0]
		return idxv[0]
	elif idxv:
		return aset(array[idx], *idxv)
	else:
		raise ValueError("Must supply value to set")

aset_boolean = aset_byte = aset_char = aset_double = aset_float = aset_int = aset_long = aset_short = aset

def assert_(x, *message):
	"""Evaluates expr and throws an exception if it does not evaluate to logical
	true.
	
	Note: varies from Python's assert in that it does not check __debug__ and
	has already evaluated the expression.
	"""
	if not x: raise AssertionError(message and message[0] or '')

def assoc(amap, key, value, *kvs):
	"""assoc[iate] new keys to values.
	
	When applied to a dict, returns a new dict of the same type, that contains
	the maping of key(s) to val(s).
	
	When applied to a vector, returns a new vector that contains val at index.
	Note: index must be <= (count vector)."""
	# TODO avoid _copy of amap[key] since we're replacing it anyway
	result = _copy(amap)
	return assoc__(result, key, value, *kvs)

def assoc__(amap, key, value, *kvs):
	"""Destructively mutate assoc[iate] new keys to values.

	Returns amap.
	"""
	for k,v in [(key, value)] + zip(kvs[::2],kvs[1::2]):
		amap[k] = v

	return amap

def assoc_in(m, ks, v):
	"""Associates a value in a nested associative structure, where ks is a
	sequence of keys and v is the new value and returns a new nested structure.
	If any levels do not exist, hash-maps will be created."""

	k, keys = (ks[0], ks[1:])

	#TODO use parrent type for new level?
	#TODO update with generic get function to support vectors
	if keys:
		return assoc(m, k, assoc_in(m.get(k, {}), keys, v))
	else:
		return assoc(m, k, v)

def associative_p(coll):
	"""Returns true if coll implements."""
	return hasattr(coll, '__getitem__')

def atom(x, *options):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of an atom?
	"""
	raise NotImplementedError()

def await(*agents):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of an agent?
	"""
	raise NotImplementedError()

def await_for(timeout_ms, *agents):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of an agent?
	"""
	raise NotImplementedError()

def bases(c):
	"""Returns the immediate superclass and direct interfaces of c, if any"""
	return c.__bases__

def bean(x):
	"""NOT IMPLEMENTED

	TODO: no beans
	"""
	raise NotImplementedError()

def bigdec(x):
	"""Coerce to Decimal"""
	try:
		return Decimal(x)
	except TypeError: # Cannot convert float to Decimal.  First convert the float to a string
		return Decimal(str(x))

# should be Integer instead?
biginteger = bigint = long

def binding(bindings, fn):
	"""NOT IMPLEMENTED

	TODO: can't figure out how to access caller's or fn's globals
	globals()?
	"""
	raise NotImplementedError()

def bit_and(x, *ys):
	"""Bitwise and"""
	return reduce(operator.and_, ys, x)

def bit_and_not(x, *ys):
	"""Bitwise and with complement"""
	return reduce(lambda a,b: a & ~b, ys, x)

def bit_clear(x, n):
	"""Clear bit at index n"""
	return bit_and_not(x, 1<<n)

def bit_flip(x, n):
	"""Flip bit at index n"""
	return x ^ (1<<n)

bit_not = operator.invert

def bit_or(x, *ys):
	"""Bitwise or"""
	return reduce(operator.or_, ys, x)

bit_shift_left = operator.lshift
bit_shift_right = operator.rshift

def bit_test(x, n):
	"""Test bit at index n"""
	return bool(x & (1<<n))

def bit_xor(x, *ys):
	"""Bitwise exclusive or"""
	return reduce(operator.xor, ys, x)

boolean = bool

def _type_array(fn, size_or_seq, init_val_or_seq=None):
	if isinstance(size_or_seq, Integral):
		try:
			return map(fn, itertools.islice(init_val_or_seq, size_or_seq))
		except TypeError:
			return [fn(init_val_or_seq)]*size_or_seq
	else:
		try:
			return map(fn, iter(size_or_seq))
		except TypeError, e:
			raise ValueError("fnean_array accepts a size or an iterable: %r" % e)

# TODO use wraps or partial
def boolean_array(size_or_seq, init_val_or_seq=None):
	"""Creates an array of booleans

	Either provide:
	- a size
	- a size and initial value
	- a size and sequence
	- a sequence
	"""
	return _type_array(bool, size_or_seq, init_val_or_seq)

def booleans(xs):
	"""Casts to list of bool"""
	return boolean_array(xs)

def bound_fn(*fntail):
	"""NOT IMPLEMENTED

	TODO: macro support
	"""
	raise NotImplementedError()

def bound_fn_(f):
	"""Returns a function, which will install the same bindings in effect as in
	the thread at the time bound-fn* was called and then call f with any given
	arguments. This may be used to define a helper function which runs on a
	different thread, but needs the same bindings in place.

	Note: Python doesn't support threads, so this is currently a noop"""
	return f

def bound_p(*vars):
	"""NOT IMPLEMENTED

	TODO: Don't have access to globals() of caller
	"""
	raise NotImplementedError()

def butlast(coll):
	"""Return a generator of all but the last item in coll, in linear time"""
	icoll = iter(coll)

	try:
		last = icoll.next()
	except StopIteration:
		#TODO return None for empty collections?
		return

	while(True):
		try:
			nextlast = icoll.next()
		except StopIteration:
			return

		yield last
		last = nextlast

def byte(x):
	"""Coerce to byte"""
	return bytearray([x])[0]

def byte_array(size_or_seq, init_val_or_seq=None):
	"""Creates an array of bytes

	Either provide:
	- a size
	- a size and initial value
	- a size and sequence
	- a sequence
	"""
	# custom implementation to use bytearray
	if isinstance(size_or_seq, Integral):
		try:
			return bytearray(int(e) for e in itertools.islice(init_val_or_seq, size_or_seq))
		except TypeError: # init_val_or_seq is not iterable
			if init_val_or_seq is None:
				return bytearray(size_or_seq)
			else:
				return bytearray([int(init_val_or_seq)]*size_or_seq)
	else:
		# size_or_seq is a sequence of something
		return bytearray(size_or_seq)

bytes_ = bytes

def case(e, *clauses):
	"""Takes an expression, and a set of clauses.

	Each clause can take the form of either:

	test-constant fn

	(test-constant1, ..., test-constantN)  fn

	If the expression is equal to a test-constant, the corresponding fn() result
	is returned. A single default fn can follow the clauses, and its result will
	be returned if no clause matches. If no default expression is provided and
	no clause matches, a ValueError is thrown.
	"""
	for test, fn in zip(clauses[::2], clauses[1::2]):
		# TODO limit to tuples or avoid lookups in strings?
		try:
			if e == test or e in test:
				return fn()
		except TypeError:
			pass

	if len(clauses) % 2:
		return clauses[-1]()

	raise ValueError("No matching cases")

def cast(c, x):
	"""Throws a TypeError if x is not a c, else returns x."""
	if isinstance(x, c):
		return x
	else:
		raise TypeError("%r cannot be cast to %r" % (x, c))

def char(x):
	"""Coerce to char"""
	if char_p(x):
		return x

	if isinstance(x, Integral):
		return unichr(x)

	raise ValueError("Can't coerce %r to char" % x)

# TODO use wraps or partial
def char_array(size_or_seq, init_val_or_seq=0):
	"""Creates an array of chars fron integers

	Either provide:
	- a size
	- a size and initial value
	- a size and sequence
	- a sequence
	"""
	return _type_array(char, size_or_seq, init_val_or_seq)

def char_escape_string(c):
	"""Returns escape string for char or None"""
	return {"\n": r'\n',
			"\t": r'\t',
			"\r": r'\r',
			'"' : r'\"',
			"\\": r'\\',
			"\f": r'\f',
			"\b": r'\b'}.get(c)

def char_name_string(c):
	"""Returns escape string for char or None"""
	return {"\n": "newline",
			"\t": "tab",
			"\r": "return",
			"\f": "formfeed",
			"\b": "backspace"}.get(c)

def char_p(c):
	"""Return true if x is a Character"""
	try:
		return bool(ord(c))
	except TypeError:
		return False

def chars(xs):
	"""NOT IMPLEMENTED

	TODO: Not sure how this should varry from char-array given no Python type.
	"""
	raise NotImplementedError()

def class_(x):
	"""Returns the Class of x"""
	return x.__class__

class_p = inspect.isclass

def clear_agent_errors(a):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of an agent?
	"""
	raise NotImplementedError()

def clojure_version():
	"""Returns clojure *API* version as a printable string.

	Includes version of this module."""
	return "1.3.0 - CljPy %s" % __version__

def coll_p(x):
	"""Returns true if x implements __len__ and __iter__"""
	# Key qualities of IPersistentCollection:
	# - cons TODO
	# - count __len__
	# - seq __iter__
	return hasattr(x, '__len__') and hasattr(x, '__iter__')

def comment(*body, **kwargs):
	"""Ignores body, returns None"""
	return None

def commute(ref, fun, *args):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of a ref?
	"""
	raise NotImplementedError()

def comp(*fns):
	"""Takes a set of functions and returns a fn that is the composition of
	those fns.  The returned fn takes a variable number of args, applies the
	rightmost of fns to the args, the next fn (right-to-left) to the result,
	etc.
	
	If no functions are provided, returns a tuple of args and kwargs.
	"""
	def result(*args, **kwargs):
		if fns:
			fnlist = list(reversed(fns))
			ret = fnlist[0](*args, **kwargs)
			for fn in fnlist[1:]:
				ret = fn(ret)
			return ret
		else:
			return (args, kwargs)

	return result

def comparator(pred):
	"""NOT IMPLEMENTED

	TODO: Not sure there's an equiv in Python
	"""
	raise NotImplementedError()

def compare_and_set__(atom, oldval, newval):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of an atom?
	"""
	raise NotImplementedError()

def compile(lib):
	"""NOT IMPLEMENTED

	TODO: perhaps just load the library, which will produce pyc files?
	"""
	raise NotImplementedError()

def complement(f):
	"""Takes a fn f and returns a fn that takes the same arguments as f, has the
	same effects, if any, and returns the opposite truth value."""
	#TODO use wraps?
	return lambda *args, **kwargs: not f(*args, **kwargs)

concat = itertools.chain

def cond(*clauses):
	"""Takes a set of test/funcion pairs. It evaluates each test function one at
	a time.  If a test returns logical true, cond evaluates and returns the
	value of the corresponding expr and doesn't evaluate any of the other tests
	or exprs.  cond() returns None."""
	# TODO default case?
	
	for test, expr in zip(clauses[::2], clauses[1::2]):
		if test():
			return expr()

	return None

def condp(pred, expr, *clauses):
	""" Takes a binary predicate, an expression, and a set of clauses.  Each
	clause can take the form of either:

	test-expr result-expr

	test-expr condp.to result-fn

	For each clause, (pred test-expr expr) is evaluated. If it returns logical
	true, the clause is a match. If a binary clause matches, the result-expr is
	returned, if a ternary clause matches, its result-fn, which must be a unary
	function, is called with the result of the predicate as its argument, the
	result of that call being the return value of condp. A single default
	expression can follow the clauses, and its value will be returned if no
	clause matches. If no default expression is provided and no clause matches,
	an ValueError is thrown."""

	iter_c = iter(clauses)
	while(iter_c):
		try:
			test = next(iter_c)
		except StopIteration:
			return ValueError("No matching cases")

		try:
			to = next(iter_c)
			if to == condp.to:
				result = next(iter_c)
			else:
				result = to
		except StopIteration:
			# test holds the last expression, the default
			return test
		
		p = pred(test, expr)
		if p:
			return (result(p) if to == condp.to else result())
condp.to = ":>>"

def conj(coll, *xs):
	"""conj[oin]. Returns a new collection with the xs 'added'. (conj nil item)
	returns (item).  The 'addition' may happen at different 'places' depending
	on the concrete type."""
	#TODO multimethod?

	add = _copy(xs)

	if not coll:
		return add

	# immutible types
	if isinstance(coll, frozenset):
		return coll | frozenset(add)

	if isinstance(coll, tuple):
		return coll + tuple(add)

	# mutable
	result = _copy(coll)

	return conj__(result, *xs)

def conj__(coll, *xs):
	"""Adds x to the collection by mutation, and return coll. The 'addition' may
	happen at different 'places' depending on the concrete type."""

	if isinstance(coll, (dict, set)):
		coll.update(xs)
		return coll

	if isinstance(coll, list):
		coll.extend(xs)
		return coll

	try:
		return itertools.chain(coll, xs)
	except TypeError:
		raise ValueError("Cannot conj onto %r" % coll)

def cons(x, seq):
	"""Returns a new iterable where x is the first element and seq is the rest.
	
	Note: as in Clojure, if seq is a dict it will iterate over items
	"""
	try:
		return itertools.chain((x,), seq.iteritems())
	except AttributeError: # not a dict
		return itertools.chain((x,), seq)

def constantly(x):
	"""Returns a function that takes any number of arguments and returns x."""
	return lambda *args,**kwargs: x

def construct_proxy(c, *ctor_args):
	"""NOT IMPLEMENTED

	TODO: not sure what Clojure version does.
	"""
	raise NotImplementedError()

contains_p = operator.contains

def count(coll):
	"""Returns the number of items in the collection. count(None) returns 0.
	
	Note: if necessary, this will iterate through a collection
	"""
	try:
		return len(coll)
	except TypeError: # has no len()
		return sum(1 for _ in coll)

def counted_p(coll):
	"""Returns true if coll implements count in via __len__ function."""
	return hasattr(coll, '__len__')

def create_ns(sym):
	"""NOT IMPLEMENTED

	TODO: perhaps creates a module?
	"""
	raise NotImplementedError()

def create_struct(*keys):
	"""NOT IMPLEMENTED

	TODO: maybe a named tuple with a __dict__? These are not recommended in Clojure anyway
	"""
	raise NotImplementedError()

cycle = itertools.cycle

def dec(x):
	"""Returns a number one less than x."""
	return x-1

def decimal_p(n):
	"""Returns true if n is a Decimal."""
	return isinstance(n, Decimal)

def declare(*names):
	"""NOT IMPLEMENTED

	TODO: change callers globals()
	"""
	raise NotImplementedError()

def definline(name, *decl):
	"""NOT IMPLEMENTED

	TODO: hint to compile to expand function?
	"""
	raise NotImplementedError()

def defmacro(name, doc_string, attr_map, *params_body):
	"""NOT IMPLEMENTED

	probably not going to happen
	"""
	raise NotImplementedError()

def defmethod(dispatch_val):
	"""Creates and installs a new method of multimethod associated with dispatch-value.
	
	Note: instead of a Clojure macro, this is a Python decorator.  Use it to
	decorate the implementation for dispatch_val.
	
	TODO: support custom hierarchies."""

	def decorator(fn):
		name = fn.__name__
		dispatch = defmulti.dispatch.get(name)
		if dispatch:
			defmulti.methods[name][dispatch_val] = fn
		else:
			raise ValueError("No dispatch function for %s. Must define with defmulti first." % name)
		return dispatch

	return decorator

def defmulti(**options):
	"""Decorator creates a new multimethod with the associated dispatch function.

	Options are key-value pairs and may be one of:
	:default    the default dispatch value, defaults to :default
	
	Note: instead of a Clojure macro, this is a Python decorator.  Use it to
	decorate the dispatch function.
	"""
	default = options.get('default')

	def decorator(dispatch_fn):
		name = dispatch_fn.__name__
		@wraps(dispatch_fn)
		def wrapped(*args, **kwargs):
			lookup = dispatch_fn(*args, **kwargs)
			method = defmulti.methods[name].get(lookup) or defmulti.methods[name].get(default)
			if method:
				return method(*args, **kwargs)
			else:
				raise RuntimeError("No method defined for dispatch value %r" % lookup)

		defmulti.dispatch[name] = wrapped
		return wrapped

	return decorator
defmulti.methods = defaultdict(dict)
defmulti.dispatch = {}

def defn(name, docstring, attr_map, dispatch_fn, **options):
	"""NOT IMPLEMENTED

	Probably will not implement: use Python def
	"""
	raise NotImplementedError()

def defn_(name, *decls):
	"""NOT IMPLEMENTED

	Probably will not implement: use Python def and limit exports
	"""
	raise NotImplementedError()

def defonce(name, expr):
	"""NOT IMPLEMENTED

	TODO: namespace manipulation
	"""
	raise NotImplementedError()

def defprotocol(name, *args):
	"""NOT IMPLEMENTED

	TODO: need to think about this
	"""
	raise NotImplementedError()

def defrecord(name, fields, *specs, **opts):
	"""NOT IMPLEMENTED

	TODO: maybe a namedtuple with multimethods?
	"""
	raise NotImplementedError()

def defstruct(name, *keys):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of a structmap?
	see create_structmap
	"""
	raise NotImplementedError()

def deftype(name, fields, *specs, **options):
	"""NOT IMPLEMENTED

	TODO: what is the equivilant of a type?
	"""
	raise NotImplementedError()

def delay(fn, *args, **kwargs):
	"""Takes a function and arguements to that function, returns a Delay object
	that will invoke the body only the first time it is forced (with force or
	deref), and will cache the result and return it on all subsequent force
	calls. See also - realized_p"""

	return Delay(fn, args, kwargs)

def delay_p(x):
	"""Returns true if x is a Delay created with delay"""
	return isinstance(x, Delay)

def force(x):
	"""If x is a Delay, returns the (possibly cached) value of its expression, else returns x"""
	if delay_p(x):
		return x.force()
	else:
		return x

def deliver(promise, val):
	"""Delivers the supplied value to the promise, releasing any pending derefs.
	A subsequent call to deliver on a promise will throw an exception."""

	return promise.deliver(val)

def deref(ref, *args):
	"""Within a transaction, returns the in-transaction-value of ref, else
	returns the most-recently-committed value of ref. When applied to a var,
	agent or atom, returns its current state. When applied to a delay, forces it
	if not already forced. When applied to a future, will block if computation
	not complete. When applied to a promise, will block until a value is
	delivered.  
	
	TODO:
	Extra args are timeout_ms and timeout_val.  The variant taking a
	timeout can be used for blocking references (futures and promises), and will
	return timeout-val if the timeout (in milliseconds) is reached before a
	value is available. See also - realized_p."""

	return ref.deref(*args)

def promise():
	"""Returns a promise object that can be read with deref, and set, once only,
	with deliver. Calls to deref prior to delivery will block, unless the
	variant of deref with timeout is used. All subsequent derefs will return the
	same delivered value without blocking. See also - realized_p."""

	return Promise()

def make_hierarchy():
	"""Creates a hierarchy object for use with derive, isa? etc."""
	# consider using frozenset
	return {'parents': defaultdict(set),
			'ancestors': defaultdict(set),
			'descendants': defaultdict(set)}

def derive(h, tag, parent=None):
	"""Establishes a parent/child relationship between parent and tag. Parent
	and tag must be hashable objects.  h must be a hierarchy obtained from
	make-hierarchy, if not supplied defaults to, and modifies, the global
	hierarchy.

	The two ways to call this function are:
	derive(tag, parent)
	derive(h, tag, parent)
	"""

	if parent is None:
		# shift everything over
		parent = tag
		tag = h
		h = derive.global_ns
	else:
		h = _copy(h)

	tp = h['parents']
	td = h['descendants']
	ta = h['ancestors']

	if parent in ta[tag]:
		raise ValueError("%r already has %r as ancestor" % (tag, parent))
	if tag in ta[parent]:
		raise ValueError("Cyclic derivation: %r has %r as ancestor" % (parent, tag))

	tp[tag].add(parent)
	td[parent].update(td[tag] | set([tag]))
	ta[tag].update(ta[parent] | set([parent]))

	for anc in ta[parent]:
		td[anc].add(tag)

	for desc in td[tag]:
		ta[desc].add(parent)

	return h
derive.global_ns = make_hierarchy()

def descendants(h, tag=None):
	"""Returns the immediate and indirect children of tag, through a
	relationship established via derive. h must be a hierarchy obtained from
	make-hierarchy, if not supplied defaults to the global hierarchy.
	
	The two ways of calling this function are:
	descendants(tag)
	descendants(h, tag)
	"""

	if tag is None:
		# shift args over
		tag = h
		h = derive.global_ns
	
	return frozenset(h['descendants'][tag])

def disj(coll, *ks):
	"""disj[oin]. Returns a new set of the same (hashed/sorted) type, that does
	not contain key(s)."""
	# should work with tuples?
	return coll - frozenset(ks)

def disj__(coll, *ks):
	"""Destructively disj[oin]. Returns coll with ks removed."""
	for key in ks:
		coll.discard(key)
	return coll

def dissoc(d, *ks):
	"""dissoc[iate]. Returns a new map of the same (hashed/sorted) type, that
	does not contain a mapping for key(s)."""

	keep = set(d.keys()) - set(ks)
	return dict(zip(keep, [d[k] for k in keep]))

def distinct(coll):
	"""Returns a generator of the elements of coll with duplicates removed"""
	seen = set()
	for e in coll:
		if e not in seen:
			yield e
			seen.add(e)

def distinct_p(*args):
	"""Returns true if no two of the arguments are =="""
	for i, e in enumerate(args):
		if e in args[i+1:]:
			return False
	return True


def parents(h, tag=None):
	"""Returns the immediate parents of tag, either via a Java type inheritance
	relationship or a relationship established via derive. h must be a hierarchy
	obtained from make-hierarchy, if not supplied defaults to the global
	hierarchy."""

	if tag is None:
		# shift args over
		tag = h
		h = derive.global_ns
	
	mro = tag.__bases__ if hasattr(tag, '__bases__') else set()
	return frozenset(h['parents'][tag] | mro)

def partition(n, coll, step=None, pad=None):
	"""Returns a generator of lists of n items each, at offsets step apart.
	If step is not supplied, defaults to n, i.e. the partitions do not overlap.
	If a pad collection is supplied, use its elements as necessary to complete
	last partition upto n items. In case there are not enough padding elements,
	return a partition with less than n items.

	Note: positional arguments differ from Clojure"""
	step = step or n
	seq = iter(coll)
	result = ()

	while True:
		part = tuple(itertools.islice(seq, step))
		result += part
		if len(result) >= n:
			yield result[:n]
			result = result[step:]
		elif len(part) < step:
			if pad:
				yield result + tuple(itertools.islice(pad, n-len(result)))
			return

def merge_with(f, *maps):
	"""Returns a map that consists of the rest of the maps conj-ed onto the
	first.  If a key occurs in more than one map, the mapping(s) from the latter
	(left-to-right) will be combined with the mapping in the result by calling
	f(result[k], map[k])."""

	result = {}
	for d in maps:
		for k in d:
			if k in result:
				result[k] = f(result[k], d[k])
			else:
				result[k] = _copy(d[k])
	return result

