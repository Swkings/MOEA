#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : NDSetRankSwk.py
@Date  : 2019/01/10
@Desc  : 
"""
"""==================================="""
"""
高维还可改进：可参考NDSetSwk中求第一等级的算法，
在比较时引入最小比较集，剪去不必要比较的元素，
并且设置每个等级比较的最小元素组合，缩小比较范围。
"""
"""==================================="""
import numpy as np
import math

def NDSetRankSwk(data):
	if type(data) is list:
		data = np.array(data)
	if data.shape[1] == 2:
		return _NDSetRankLD(data)
	else:
		return _NDSetRankHD(data)

def _NDSetRankLD(data):
	"""
	LD：低维（2维）
	精简第一版算法，直接给每一个等级设定一个最小值
	"""
	# 排序
	index = np.lexsort([data[:, 1], data[:, 0]])
	compareSet = data[index, :]
	# 结果数组
	rankSet = []
	# 维持每个等级的最小值
	minRankValueSet = []
	# 将第一个值加入第一等级序列
	rankSet.append([index[0]])
	minRankValueSet.append(compareSet[0])
	# 记录比较元素前一个元素的等级
	prevValueRank = 0
	for i in range(1, data.shape[0]):
		compareValue = compareSet[i]
		if compareValue[1] < minRankValueSet[prevValueRank][1]: # 如果小于前一个数等级中的最小值
			maxRank = prevValueRank
			minRank = 0
		elif compareValue[1] == minRankValueSet[prevValueRank][1] and compareValue[0] == minRankValueSet[prevValueRank][0]: # 如果等于前一个元素
			minRank = prevValueRank
			maxRank = prevValueRank
		else:
			minRank = prevValueRank + 1
			maxRank = len(rankSet)
		iRank = _findRankLD(rankSet, minRankValueSet, compareValue, minRank, maxRank)
		_addToRankSetLD(rankSet, minRankValueSet, index[i], compareValue, iRank)
		prevValueRank = iRank
	return rankSet

def _findRankLD(rankSet, minRankValueSet, compareValue, minRank, maxRank):
	"""
	寻找nextValue的rank
	:param rankSet:
	:param minRankValueSet:
	:param compareValue:
	:param minRank: 寻找等级范围：最小等级
	:param maxRank: 寻找等级范围：最大等级
	:return: rank
	"""
	while minRank < maxRank:
		midRank = math.floor((maxRank+minRank)/2)  # 二分查找
		midRankMinValue = minRankValueSet[midRank]
		if compareValue[1] < midRankMinValue[1]:
			maxRank = midRank
		elif compareValue[1] > midRankMinValue[1]:
			minRank = midRank + 1
		else:
			return midRank + 1
	return minRank

def _addToRankSetLD(rankSet, minRankValueSet, index, compareValue, iRank):
	# 如果等级超过上限，用处理方式处理
	if iRank >= len(rankSet):
		rankSet.append([index])
		minRankValueSet.append(compareValue)
	else:
		rankSet[iRank].append(index)
		if compareValue[1] < minRankValueSet[iRank][1]:
			minRankValueSet[iRank] = compareValue

def _NDSetRankHD(data):
	"""
	HD：高维（2~n维）
	遍历一遍得出全部等级
	"""
	# 对data排序
	dimension = data.shape[1]
	sortCmp = [data[:, i] for i in reversed(range(dimension))]
	index = np.lexsort(sortCmp)
	compareSet = data[index, :]
	# 结果数组
	rankSet = []
	# 维持每个等级中与最小值都无关的元素，相当于比较集合
	minRankValueSet = []
	# 第一个值一定是第一等级序列的
	rankSet.append([index[0]])
	minRankValueSet.append([compareSet[0]])
	prevValueRank = 0
	for i in range(1, data.shape[0]):
		prevValue = compareSet[i-1]
		nextValue = compareSet[i]
		relevantValue = _relevantHD(prevValue, nextValue)
		if (prevValue == nextValue).all():  # 如果比较元素和前一个元素相同，则为相同等级
			minRank = prevValueRank
			maxRank = prevValueRank
		elif relevantValue == 0:   # 如果比较元素和前一个元素除第一维值都相同，则必为前一个元素等级加1
			minRank = prevValueRank + 1
			maxRank = prevValueRank + 1
		elif relevantValue == 1:   # 如果前一个元素支配比较元素，则比较元素最小等级为前一元素等级加1
			minRank = prevValueRank + 1
			maxRank = len(rankSet)
		elif relevantValue == 2:   # 如果前一个元素和比较元素互不支配，则不能确定等级
			minRank = 0
			maxRank = len(rankSet)
		elif relevantValue == 3:   # 如果比较元素支配前一个元素（除第一维外的支配），则比较元素最大等级为前一元素等级
			minRank = 0
			maxRank = prevValueRank
		iRank = _findRankHD(rankSet, minRankValueSet, nextValue, minRank, maxRank)
		_addToRankSetHD(rankSet, minRankValueSet, index[i], nextValue, iRank)
		prevValueRank = iRank
	return rankSet

def _addToRankSetHD(rankSet, minRankValueSet, index, compareValue, iRank):
	# 如果等级超过上限，用处理方式处理
	if iRank >= len(rankSet):
		rankSet.append([index])
		minRankValueSet.append([compareValue])
	else:
		rankSet[iRank].append(index)
		minRankValueSet[iRank].append(compareValue)

# def _findRankHD(rankSet, minRankValueSet, compareValue, minRank, maxRank):
# 	while minRank < maxRank:
# 		midRank = math.floor((maxRank + minRank) / 2)
# 		midRankMinValueSet = minRankValueSet[midRank]
# 		isNonDominated = True
# 		for i in reversed(range(len(midRankMinValueSet))):
# 			relevantValue = _relevantHD(midRankMinValueSet[i], compareValue)
# 			if relevantValue < 2:
# 				minRank = midRank + 1
# 				isNonDominated = False
# 				break
# 			elif relevantValue == 2:
# 				continue
# 			elif relevantValue == 3:
# 				maxRank = midRank
# 				isNonDominated = False
# 				break
# 		if isNonDominated:
# 			maxRank = midRank
# 	return minRank

def _findRankHD(rankSet, minRankValueSet, compareValue, minRank, maxRank):
	while minRank < maxRank:
		midRank = math.floor((maxRank + minRank) / 2)
		midRankMinValueSet = minRankValueSet[midRank]
		isNonDominated = True
		for i in reversed(range(len(midRankMinValueSet))):
			if not _isDominated(midRankMinValueSet[i], compareValue):
				continue
			else:
				minRank = midRank + 1
				isNonDominated = False
				break
		if isNonDominated:
			maxRank = midRank
	return minRank

def _relevantHD(prevValue, nextValue):
	"""返回三个值：[第一维不做比较]
	0：值相等
	1：prevValue 支配 nextValue
	2：无关系，互不支配
	3：nextValue 支配 prevValue
	"""
	p = prevValue[1:]
	n = nextValue[1:]
	if (p == n).all():
		return 0
	elif (p <= n).all():
		return 1
	elif (n <= p).all():
		return 3
	else:
		return 2

def _isDominated(p, q):
	if (p <= q).all() and (p != q).any():
		return True
	return False

if __name__ == '__main__':
	from pylab import *
	data = np.array(
		[[9, 1], [7, 2], [5, 4], [4, 5], [3, 6], [2, 7], [1, 9], [10, 1], [8, 5], [7, 6], [5, 7], [4, 8], [3, 9],
		 [10, 5], [9, 6], [8, 7], [7, 9], [10, 6], [9, 7], [8, 9]])
	import random
	def randData(size, dimension=2, span=(0, 1)):
		data = np.empty((size, dimension))
		for i in range(size):
			X = random.sample(range(span[0], span[1]), dimension)
			data[i] = np.array(X)
		return data


	data = np.array([[1, 17, 5], [2, 1, 7], [2, 7, 17], [2, 11, 12], [2, 19, 11], [14, 4, 9], [14, 12, 10], [15, 3, 13],
	                 [16, 12, 1], [17, 7, 6]])

	data = randData(1000, dimension=3, span=(1, 200))
	start = time.clock()
	pereto = NDSetRankSwk(data)
	end = time.clock()
	print('NDSetRankSwk耗时：', end - start)
	print(pereto[0])

	# import MOEA.ndomin.NDSetRankDelete as NDD

	# start = time.clock()
	# indX = NDD.NDSetRankDelete(data)
	# end = time.clock()
	# print('NDSetDelete耗时：', end - start)

	import MOEA.ndomin.NDSetRankArena as NDA
	start = time.clock()
	indX = NDA.NDSetRankArena(data)
	end = time.clock()
	print('NDSetRankArena耗时：', end - start)
	print(indX[0])