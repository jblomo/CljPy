"""Implements functions in clojure.core"""
import copy
from decimal import Decimal
import operator

# override this method to change the way nondestructive operations copy
# arguments
_copy = copy.deepcopy


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
	"""NOT IMPLEMENTED

	TODO: implement module local hierarchy
	"""
	raise NotImplementedError()

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
	return Decimal(x)

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

