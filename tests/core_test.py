import pytest
import operator

from cljpy.core import *

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
	seq = [0,1,2,3,4,5,6]
	init_val = 1

	limited_f = boolean_array(size)
	limited_t = boolean_array(size, 1)
	limited_s = boolean_array(size, seq)

	assert len(limited_f) == size
	assert len(limited_t) == size
	assert len(limited_s) == size

	assert not any(limited_f)
	assert all(limited_t)
	assert limited_s == [False, True, True, True, True]

	bool_seq = boolean_array(seq)
	assert len(bool_seq) == len(seq)
	assert bool_seq == [False, True, True, True, True, True, True]

def test_booleans():
	seq = [0,1,2,3,4,5,6]
	bool_seq = boolean_array(seq)
	assert len(bool_seq) == len(seq)
	assert bool_seq == [False, True, True, True, True, True, True]

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

def test_char_array():
	size = 5
	seq = map(ord, "abcdef")
	init_val = 115

	limited_f = char_array(size)
	limited_t = char_array(size, 1)
	limited_s = char_array(size, seq)

	assert len(limited_f) == size
	assert len(limited_t) == size
	assert len(limited_s) == size

	assert limited_s == list("abcde")

	char_seq = char_array(seq)
	assert len(char_seq) == len(seq)
	assert char_seq == list("abcdef")

def test_char_escape_string():
	assert char_escape_string("\n") == r'\n'
	assert char_escape_string("\\") == r'\\'
	assert char_escape_string("s") == None

def test_char_name_string():
	assert char_name_string("\n") == "newline"
	assert char_name_string("\f") == "formfeed"
	assert char_name_string("s") == None

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

