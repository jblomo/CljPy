import pytest

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
