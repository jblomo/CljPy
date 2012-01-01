import pytest
import operator
from decimal import Decimal
from datetime import datetime
from itertools import islice

from cljpy.core import *
from lang import Promise

def test_aget():
	single = [1,2,3]
	nested = [[1],[2],[3]]

	assert aget(single, 1) == 2
	assert aget(nested, 1, 0) == 2

def test_aset():
	single = [1,2,3]
	nested = [[1],[2],[3]]

	aset(single, 1, 42)
	aset(nested, 1, 0, 42)

	assert single[1] == 42
	assert nested[1][0] == 42

	with pytest.raises(ValueError):
		aset(single, 1)
	with pytest.raises(ValueError):
		aset(nested, 0)

def test_assert_():
	with pytest.raises(AssertionError):
		assert_(False, "problem")
	with pytest.raises(AssertionError):
		assert_(False)

	assert_(1, "no problem")
	assert_(True)

def test_assoc():
	amap = {'one': 1, 'two': 2, 'three': 3}
	vector = [1, 2, 3]

	modified = assoc(amap, 'four', 4)
	assert amap != modified
	assert modified['four'] == 4

	modified = assoc(vector, 2, 42)
	assert vector != modified 
	assert modified[2] == 42

	modified = assoc(amap, 'four', 4, 'two', 42)
	assert amap != modified
	assert modified['four'] == 4
	assert modified['two'] == 42

	modified = assoc(vector, 2, 42, 0, 42)
	assert vector != modified 
	assert modified[2] == 42
	assert modified[0] == 42

def test_ancestors():
	h = make_hierarchy()

	h = derive(h, 'daughter', 'parent')
	h = derive(h, 'parent', 'grandparent')
	h = derive(h, 'son', 'parent')

	assert 'grandparent' in ancestors(h, 'son')
	assert 'grandparent' in ancestors(h, 'daughter')
	
	assert 'parent' in ancestors(h, 'son')
	assert 'parent' in ancestors(h, 'daughter')

def test_assoc_in_single():
	amap = {'one': 1, 'two': 2, 'three': 3}
	vector = [1, 2, 3]

	modified = assoc_in(amap, ['four'], 4)
	assert amap != modified
	assert modified['four'] == 4

	modified = assoc_in(vector, [2], 42)
	assert vector != modified 
	assert modified[2] == 42

	modified = assoc_in(amap, ['four', 'two'], 42)
	assert amap != modified
	assert modified['four']['two'] == 42

	###
	# TODO include test when assoc_in supports vecors
	# modified = assoc_in(vector, [2, 0], 42)
	# assert vector != modified 
	# assert modified[2][0] == 42
	##/

def test_assoc_in_nested():
	amap = {'one': 1, 'two': 2, 'three': 3, 'four': {'old': True}}
	vector = [1, 2, ['old']]

	modified = assoc_in(amap, ['four', 'old'], False)
	assert amap != modified
	assert modified['four']['old'] == False

	###
	# TODO include test when assoc_in supports vecors
	# modified = assoc_in(vector, [2, 0], False)
	# assert vector != modified 
	# assert modified[2][0] == False
	##?

	modified = assoc_in(amap, ['four', 'two'], 42)
	assert amap != modified
	assert modified['four']['two'] == 42

def test_associative_p():
	assert associative_p({}) == True
	assert associative_p([]) == True
	
	assert associative_p(1) == False
	assert associative_p(False) == False

def test_bigdec():
	for test in [-1.2, ".00000000000000000000000000000000000001", 2.1, 50]:
		bigd = bigdec(test)
		assert isinstance(bigd, Decimal)
		assert bigd != 0

	assert bigdec(0) == 0

def test_bit_and():
	x1 = 1<<0
	x2 = 1<<1
	x3 = 1<<2
	x4 = 1<<3
	all_x = x1 | x2 | x3 | x4

	assert bit_and(all_x, x1) == x1
	assert bit_and(all_x, x1, x2) == 0
	assert bit_and(all_x, x1|x2, x2) == x2

def test_bit_and_not():
	x1 = 1<<0
	x2 = 1<<1
	x3 = 1<<2
	x4 = 1<<3
	all_x = x1 | x2 | x3 | x4

	assert bit_and_not(all_x, x1) == x2 | x3 | x4
	assert bit_and_not(all_x, x1, x2) == x3 | x4
	assert bit_and_not(all_x, x1|x3, x2) == x4

def test_bit_clear():
	x1 = 1<<0
	x2 = 1<<1
	x3 = 1<<2
	x4 = 1<<3
	all_x = x1 | x2 | x3 | x4

	assert bit_clear(all_x, 0) == x2 | x3 | x4
	assert bit_clear(all_x, 2) == x1 | x2 | x4

def test_bit_flip():
	x1 = 1<<0
	x2 = 1<<1
	x3 = 1<<2
	x4 = 1<<3
	all_x = x1 | x2 | x3 | x4
	x2_up = x2 | x3 | x4

	assert bit_flip(all_x, 0) == x2 | x3 | x4
	assert bit_flip(x2_up, 0) == all_x

def test_bit_or():
	x1 = 1<<0
	x2 = 1<<1
	x3 = 1<<2
	x4 = 1<<3
	all_x = x1 | x2 | x3 | x4

	assert bit_or(x1) == x1
	assert bit_or(x1, x2) == x1 | x2
	assert bit_or(x1, x2, x3, x4) == all_x

def test_bit_test():
	x1 = 1<<0
	x2 = 1<<1
	x3 = 1<<2
	x4 = 1<<3
	all_x = x1 | x2 | x3 | x4

	assert bit_test(all_x, 0) == True
	assert bit_test(all_x, 1) == True
	assert bit_test(x1|x2|x4, 2) == False

def test_bit_xor():
	x1 = 1<<0
	x2 = 1<<1
	x3 = 1<<2
	x4 = 1<<3
	all_x = x1 | x2 | x3 | x4

	assert bit_xor(all_x, x1) == x2 | x3 | x4
	assert bit_xor(all_x, x1, x2) == x3 | x4
	assert bit_xor(x1, x2, x3, x4) == all_x

def test_boolean_array():
	size = 5
	seq = [False, True, True, True, True, True]
	init_val = True

	limited_f = boolean_array(size)
	limited_t = boolean_array(size, init_val)
	limited_s = boolean_array(size, seq)

	assert len(limited_f) == size
	assert len(limited_t) == size
	assert len(limited_s) == size

	assert not any(limited_f)
	assert all(limited_t)
	assert map(bool, limited_s) == [False, True, True, True, True]

	bool_seq = boolean_array(seq)
	assert len(bool_seq) == len(seq)
	assert map(bool, bool_seq) == [False, True, True, True, True, True]

def test_butlast():
	seq = [0,1,2,3,4,5,6]
	assert list(butlast(seq)) == seq[:-1]
	
	seq = xrange(10)
	assert list(butlast(seq)) == list(seq)[:-1]

	assert list(butlast(i for i in [1,2,3])) == [1,2]

def test_byte():
	assert byte(5) == 5
	assert byte('s') == 115

	with pytest.raises(ValueError):
		byte(-5)
	with pytest.raises(ValueError):
		byte(500)

def test_byte_array():
	size = 5
	seq    = [0,1,2,3,4,5,'s']
	init_val = 1

	limited_f = byte_array(size)
	limited_t = byte_array(size, 1)
	limited_s = byte_array(size, seq)

	assert len(limited_f) == size
	assert len(limited_t) == size
	assert len(limited_s) == size

	assert not any(limited_f)
	assert all(limited_t)
	assert list(limited_s) == seq[:size]

	byte_seq = byte_array(seq)
	assert len(byte_seq) == len(seq)
	assert list(byte_seq) == [0,1,2,3,4,5,115]

def test_case():
	def one():
		return 1

	def two():
		return 2

	three = lambda: 3

	default = lambda: 'default'

	for test in [1,2,3]:
		assert case(test,
				1, one,
				2, two,
				3, three,
				default) == test

	for test in [1,2,3]:
		assert case(test,
				('one', 1), one,
				2, two,
				(3, 'three'), three,
				default) == test

	assert case('string',
			1, one,
			2, two,
			3, three,
			default) == 'default'

	with pytest.raises(ValueError):
		case('string',
				1, one,
				2, two,
				3, three)

def test_char():
	assert char('s') == 's'
	assert char(115) == 's'
	assert char(True) == u'\x01'

	with pytest.raises(ValueError):
		char("string")

	with pytest.raises(ValueError):
		char([1,2,3])

def test_char_array():
	size = 5
	seq = "abcdef"
	init_val = 's'

	limited_f = char_array(size)
	limited_t = char_array(size, init_val)
	limited_s = char_array(size, seq)

	assert len(limited_f) == size
	assert len(limited_t) == size
	assert len(limited_s) == size

	assert list(limited_s) == list("abcde")

	char_seq = char_array(seq)
	assert len(char_seq) == len(seq)
	assert list(char_seq) == list("abcdef")

	assert char_array(c for c in "abcdef") == char_seq

def test_char_escape_string():
	assert char_escape_string("\n") == r'\n'
	assert char_escape_string("\\") == r'\\'
	assert char_escape_string("s") == None

def test_char_name_string():
	assert char_name_string("\n") == "newline"
	assert char_name_string("\f") == "formfeed"
	assert char_name_string("s") == None

def test_char_p():
	assert map(char_p, [1, 't', 'three', False]) == [False, True, False, False]

def test_class_():
	assert class_(1) == type(1)
	assert class_([1,2,3]) == type([])
	assert class_(class_.__class__) == type

def test_clojure_version():
	assert clojure_version().startswith("1.3.0")

def test_coll_p():
	assert coll_p([1,2,3])
	assert coll_p({'one': 1})
	assert coll_p(set([]))

	assert coll_p("string") == False
	assert coll_p(False) == False
	assert coll_p(1) == False

def test_comment():
	for item in [1, "string", list]:
		assert comment(item) == None

	for items in [ [1,2,3], "string"]:
		assert comment(*items) == None

	assert comment(1, "two", three=3) == None

def test_comp():
	string = "string"
	alist = map(ord, string)

	all_chars_fn = comp(all, list)
	assert all_chars_fn(string) == True
	assert all_chars_fn(alist) == True

	combine_char = comp(unichr, operator.add)
	assert combine_char(100, 15) == 's'

	kw_to_set = comp(set, dict)
	assert kw_to_set(one=1, two=2) == set(['one', 'two'])

	assert comp()(1, 2, three=3) == ((1,2), {'three': 3})

def test_complement():
	c_all = complement(all)
	c_any = complement(any)

	assert c_all([False, True]) == True
	assert c_any([False, True]) == False

	assert complement(lambda: True)() == False
	assert complement(dict)(one=1) == False

def test_condp():
	assert condp(operator.eq, 3,
			1, lambda: "one",
			2, lambda: "two",
			3, lambda: "three",
			"default") == "three"

	assert condp(operator.sub, 3,
			1, condp.to, operator.abs,
			2, lambda: "two",
			3, condp.to, lambda p: "three" if p else "wrong",
			) == 2

	assert condp(operator.eq, 4,
			1, lambda: "one",
			2, lambda: "two",
			3, condp.to, lambda p: "three" if p else "wrong",
			"default") == "default"

def test_conj():
	coll = {'one': 1}
	add = ('two', 2)
	conjed = conj(coll, add)
	assert conjed != coll
	assert conjed == {'one': 1, 'two': 2}

	coll = [1,2]
	add = [3,4]
	conjed = conj(coll, *add)
	assert conjed != coll
	assert conjed == [1,2,3,4]

	coll = set([1,2])
	add = [3,4]
	conjed = conj(coll, *add)
	assert conjed != coll
	assert conjed == set([1,2,3,4])

	coll = frozenset([1,2])
	add = [3,4]
	conjed = conj(coll, *add)
	assert conjed != coll
	assert conjed == frozenset([1,2,3,4])

	coll = (1,2)
	add = [3,4]
	conjed = conj(coll, *add)
	assert conjed != coll
	assert conjed == (1,2,3,4)

	coll = xrange(3)
	add = [3,4]
	conjed = conj(coll, *add)
	assert conjed != coll
	assert list(conjed) == [0,1,2,3,4]

	coll = None
	add = [3,4]
	conjed = conj(coll, *add)
	assert conjed != coll
	assert conjed == (3,4)

def test_cons():
	assert list(cons(1, [1,2,3])) == [1,1,2,3]
	assert list(cons([1,2], [1,2,3])) == [[1,2], 1,2,3]

	assert list(cons('a', "abc")) == "a a b c".split()
	fromset = list(cons('a', frozenset(['z', 'b', 'c'])))
	assert fromset[0] == 'a'
	assert len(fromset) == 4

def test_constantly():
	truef = constantly(True)

	assert truef()
	assert truef(False)
	assert truef(correct=True)
	assert truef(1,2,3)

def test_count():
	coll = [1,2,3]
	for coll_type in [list, set, frozenset, tuple]:
		assert count(coll_type(coll)) == 3

	assert count(i for i in coll) == 3
	assert count(xrange(3)) == 3

	assert count(dict(zip(coll,coll))) == 3

	assert count("string") == 6

def test_counted_p():
	coll = [1,2,3]
	for coll_type in [list, set, frozenset, tuple]:
		assert counted_p(coll_type(coll)) == True

	assert counted_p(xrange(3)) == True
	assert counted_p(dict(zip(coll,coll))) == True
	assert counted_p("string") == True

	assert counted_p(i for i in coll) == False

def test_decimal_p():
	assert decimal_p(Decimal(0))
	assert not decimal_p(0)

	assert decimal_p(Decimal("1.2"))
	assert not decimal_p(1.2)

def test_defmethod_defmulti_simple():
	@defmulti()
	def gt_one(num):
		"""Multifunction to state if a number is greater than one"""
		return num > 1

	@defmethod(True)
	def gt_one(num):
		"""states yes!"""
		return "%s greater than" % num

	assert gt_one(2) == "2 greater than"

	with pytest.raises(RuntimeError):
		gt_one(0)

	@defmethod(False)
	def gt_one(num):
		"""states no!"""
		return "%s less than" % num

	assert gt_one(0) == "0 less than"

def test_defmethod_defmulti_types():
	@defmulti(default=None)
	def type_depend(*args):
		return tuple(map(type, args))

	@defmethod((int, int))
	def type_depend(a, b):
		return ("two ints: %s %s" % (a,b))

	@defmethod((str,))
	def type_depend(s):
		return ("one string: %s" % s)

	@defmethod(None)
	def type_depend(*args):
		return "default"

	assert type_depend(1,2) == "two ints: 1 2"
	assert type_depend("s") == "one string: s"
	assert type_depend(True) == "default"

def test_delay_force():
	def microsecond_n(n):
		return (datetime.now().microsecond, n)

	delayed = delay(microsecond_n, 5)

	assert delay_p(delayed)

	(micro, n) = force(delayed)

	assert n == 5
	
	for _ in xrange(4):
		# ensure that all calls to force return the same value
		assert (micro, n) == force(delayed)

def test_derive():
	# also see tests for ancestors, parents, descendants
	h = make_hierarchy()

	h = derive(h, 'daughter', 'parent')
	h = derive(h, 'parent', 'grandparent')
	h = derive(h, 'son', 'parent')

	assert derive('global-daughter', 'global-parent')
	assert derive('global-parent', 'global-grandparent')
	assert derive('global-son', 'global-parent')

	with pytest.raises(ValueError): # Cyclic
		derive(h, 'grandparent', 'daughter')
		derive('global-grandparent', 'global-daughter')

	with pytest.raises(ValueError): # already has ancestor
		derive(h, 'daughter', 'grandparent')
		derive('global-daughter', 'global-grandparent')

def test_descendants():
	h = make_hierarchy()

	h = derive(h, 'daughter', 'parent')
	h = derive(h, 'parent', 'grandparent')
	h = derive(h, 'son', 'parent')

	assert descendants(h, 'daughter') == frozenset([])
	assert descendants(h, 'son') == frozenset([])
	assert descendants(h, 'parent') == frozenset(['daughter', 'son'])
	assert descendants(h, 'grandparent') == frozenset(['parent', 'daughter', 'son'])

def test_disj():
	coll = set([1,2,3,4])

	assert disj(coll, 1) == set([2,3,4])
	assert disj(coll, 1, 2) == set([3,4])
	assert disj(coll, 1, 2, 11, 12) == set([3,4])

	coll = frozenset([1,2,3,4])
	assert disj(coll, 4) == frozenset([1,2,3])
	assert disj(coll, 4, 3, -4, -3) == frozenset([1,2])

def test_disj__():
	coll = set([1,2,3,4])

	disj__(coll, 1)
	assert coll == set([2,3,4])
	
	disj__(coll, 2, 3)
	assert coll == set([4])

	disj__(coll, 2, 3)
	assert coll == set([4])

	disj__(coll, 4)
	assert coll == set([])

def test_distinct():
	coll = [1,1,2,2,3,3,4,4,5,5]
	concat = coll + coll
	dist = [1,2,3,4,5]

	assert list(distinct(coll)) == dist
	assert list(distinct(concat)) == dist
	assert list(distinct(dist)) == dist
	assert list(distinct(xrange(1,6))) == dist

def test_distinct_p():
	assert distinct_p(1,2,3)
	assert distinct_p(*xrange(5))

	assert not distinct_p(1,2,3,2)
	assert not distinct_p(*([1,2,3] + [1,2,3]))

def test_doall():
	def inc_times_run():
		max_runs = 100
		while max_runs:
			max_runs -= 1
			inc_times_run.count += 1
			yield inc_times_run.count
	inc_times_run.count = 0
	
	realize10 = doall(inc_times_run(), 10)

	assert inc_times_run.count == 10
	assert list(islice(realize10, 10)) == range(1, 11)
	assert inc_times_run.count == 10

	inc_times_run.count = 0
	realize_all = doall(inc_times_run())
	assert inc_times_run.count == 100
	assert list(realize_all) == range(1, 101)
	assert inc_times_run.count == 100

def test_dorun():
	def inc_times_run():
		max_runs = 100
		while max_runs:
			max_runs -= 1
			inc_times_run.count += 1
			yield inc_times_run.count
	inc_times_run.count = 0
	
	assert dorun(inc_times_run(), 10) == None
	assert inc_times_run.count == 10

	inc_times_run.count = 0
	assert dorun(inc_times_run()) == None
	assert inc_times_run.count == 100

def test_doseq():
	def running(inc=0, dec=0):
		running.total = running.total + inc - dec
		return running.total
	running.total = 0

	assert doseq(((2, 1) for _ in xrange(10)), running) == None
	assert running.total == 10

	running.total = 0
	assert doseq((dict(dec=1, inc=3) for _ in xrange(10)), running) == None
	assert running.total == 20

def test_dotimes():
	def running(inc=0, dec=0):
		running.total = running.total + inc - dec
		return running.total
	running.total = 0

	assert dotimes('inc', 10, running) == None
	assert running.total == 45

	running.total = 0
	assert dotimes('dec', 10, running) == None
	assert running.total == -45

def test_double_array():
	size = 5
	devisor = float(3)
	seq = map(lambda i: i/devisor, [1,2,3,4,5,6])
	init_val = 1.0

	limited_f = double_array(size)
	limited_t = double_array(size, init_val)
	limited_s = double_array(size, seq)

	assert len(limited_f) == size
	assert len(limited_t) == size
	assert len(limited_s) == size

	assert list(limited_s) == seq[:5]

	double_seq = double_array(seq)
	assert len(double_seq) == len(seq)
	assert list(double_seq) == seq

	assert double_array(i/devisor for i in [1,2,3,4,5,6]) == double_seq

def test_drop():
	correct = [3,4,5]

	assert list(drop(2, [1,2,3,4,5])) == correct
	assert list(drop(2, xrange(1,6))) == correct
	assert list(drop(2, (i for i in [1,2,3,4,5]))) == correct
	assert len(list(drop(2, set([1,2,3,4,5])))) == 3

def test_drop_last():
	correct = [1,2,3]

	assert list(drop_last(2, [1,2,3,4,5])) == correct
	assert list(drop_last(2, xrange(1,6))) == correct
	assert list(drop_last(2, (i for i in [1,2,3,4,5]))) == correct
	assert len(list(drop_last(2, set([1,2,3,4,5])))) == 3

	assert list(drop_last(coll=[1,2,3])) == [1,2]

def test_empty():
	tests = [set([1, 2, 3]),
			frozenset([1, 2, 3]),
			{'one': 1, 'two': 2},
			[1,2,3]]

	for t in tests:
		emp = empty(t)
		assert type(emp) == type(t)
		assert len(emp) == 0

	none_tests = [(i for i in [1,2,3]),
			1,
			1.0,
			"string",
			xrange(5)]

	for t in none_tests:
		assert empty(t) == None

def test_empty_p():
	tests = [{}, [], set(), frozenset(), (i for i in [])]

	for t in tests:
		assert empty_p(t)

	tests_full = [{'one': 1}, [1,2], set([1,2,3]), frozenset([1]), (i for i in [1,2])]
	for t in tests_full:
		assert empty_p(t) == False

def test_eval():
	form = (int,)
	assert eval(form) == 0

	form = (max, 1,-1,0)
	assert eval(form) == 1

	form = (sum, [1, 2, 3], 0)
	assert eval(form) == 6

def test_even_p():
	for even, odd in zip(xrange(0, 10, 2), xrange(1,10,2)):
		assert even_p(even)
		assert not even_p(odd)

	with pytest.raises(TypeError): # unsupported operand type for &
		assert even_p("string")

def test_every_pred():
	assert every_pred(int)(1)

	non_empty = every_pred(len, bool, unicode)
	assert non_empty("string", "not", "empty")
	assert not non_empty("string", "", "empty")

def test_every_p():
	assert every_p(int, [1])

	assert every_p(len, ["string", "not", "empty"])
	assert not every_p(len, ["string", "", "empty"])

def test_false_p():
	assert false_p(False)

	for t in ["", [], "string", 4, set([7])]:
		assert not false_p(t)

def test_partition():
	string = "hello world"
	tries = [
			dict(n=1),
			dict(n=1, step=2),
			dict(n=3),
			dict(n=3, step=1),
			dict(n=3, step=1, pad='!!')]
	results = [
			[tuple(c) for c in string],
			[tuple(c) for c in string[::2]],
			[('h', 'e', 'l'), ('l', 'o', ' '), ('w', 'o', 'r')],
			[('h', 'e', 'l'),
				('e', 'l', 'l'),
				('l', 'l', 'o'),
				('l', 'o', ' '),
				('o', ' ', 'w'),
				(' ', 'w', 'o'),
				('w', 'o', 'r'),
				('o', 'r', 'l'),
				('r', 'l', 'd')],
			[('h', 'e', 'l'),
				('e', 'l', 'l'),
				('l', 'l', 'o'),
				('l', 'o', ' '),
				('o', ' ', 'w'),
				(' ', 'w', 'o'),
				('w', 'o', 'r'),
				('o', 'r', 'l'),
				('r', 'l', 'd'),
				('l', 'd', '!')]]

	for kwargs, result in zip(tries, results):
		assert list(partition(coll=string, **kwargs)) == result


def test_promise_deliver_simple():
	p = promise()

	assert isinstance(p, Promise)

	deliver(p, 1)

	assert deref(p) == 1

	with pytest.raises(RuntimeError):
		deliver(p, 0)


def test_promise_deliver_blocking():
	#TODO
	pass

def test_merge_with():
	d1 = {'one': 1, 'two': 2, 'seven': 3}
	d2 = {'three': 3, 'four': 4, 'seven': 4}

	merged = merge_with(operator.add, d1, d2)
	assert all(key in merged for key in d1.keys()+d2.keys())
	assert merged['seven'] == 7

	d1 = {'one': [1], 'two': [2], 'seven': [3]}
	d2 = {'three': [3], 'four': [4], 'seven': [4]}

	merged = merge_with(operator.concat, d1, d2)
	assert all(key in merged for key in d1.keys()+d2.keys())
	assert merged['seven'] == [3, 4]

