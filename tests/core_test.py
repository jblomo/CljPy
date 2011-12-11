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

