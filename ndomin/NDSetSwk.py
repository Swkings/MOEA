#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : NDSetSwk.py
@Date  : 2019/01/10
@Desc  : 
"""

"""
1：：先将数组排序
2：记录每一等级的最小值（针对二维的数据）
"""
import numpy as np

def NDSetSwk(data):
	if type(data) is list:
		data = np.array(data)
	if data.shape[1] == 2:
		return _NDSetLD(data)
	else:
		return _NDSetHD(data)

def _NDSetLD(data):
	"""
	LD：低维（2维）
	有该算法特性可知，排好序的第一个元素定为第一等级，
	之后的元素只有小于第一等级中最小元素，才属于第一等级，否则必定不属于，则删之
	"""
	# 对data排序
	index = np.lexsort([data[:, 1], data[:, 0]])
	compareSet = data[index, :]
	# 结果数组
	firstRankSet = []
	# 第一个值一定是第一等级序列的
	firstRankSet.append(index[0])
	# 记录第一等级中最小值元素
	minRankValue = compareSet[0]
	for i in range(1, data.shape[0]):
		compareValue = compareSet[i]
		if compareValue[1] < minRankValue[1]:
			# 比较元素比最小元素小，则肯定为第一等级，并替换最小值
			firstRankSet.append(index[i])
			minRankValue = compareSet[i]
		elif compareValue[1] == minRankValue[1] and compareValue[0] == minRankValue[0]:
			# 和最小值元素相等
			firstRankSet.append(index[i])
	return firstRankSet

def _NDSetHD(data):
	"""
	HD：高维（2~n维）
	"""
	dimension = data.shape[1]
	# 排序
	sortCmp = [data[:, i] for i in reversed(range(dimension))]
	index = np.lexsort(sortCmp)
	compareSet = data[index, :]
	# 结果数组
	firstRankSet = []
	# 第一个值一定是第一等级序列的
	firstRankSet.append(index[0])
	# 比较数组，应该尽量减少该数组中无关元素，减少比较次数
	tempSet = [compareSet[0]]
	# 组合元素，由各维度最小值组成
	combineElem = compareSet[0]
	# 第一等级中暂时最后一个元素
	prevValue = compareSet[0]
	for i in range(1, data.shape[0]):
		compareValue = compareSet[i]
		# 首先和组合元素比较
		relevantValue = _relevant(combineElem, compareValue)
		if relevantValue > 0:  # 若组合元素不能支配比较元素，则该比较元素为第一等级中的元素
			firstRankSet.append(index[i])
			prevValue = compareValue
			# 更新组合元素最小值
			combineElem = np.array([combineElem[i] if combineElem[i] < compareValue[i] else compareValue[i] for i in
			                        range(combineElem.shape[0])])
			if relevantValue == 1:  # 如果互不支配，比较元素加入到比较集合中
				tempSet.append(compareValue)
			else:  # 如果比较元素支配组合元素，则清空比较集合，因为该比较元素最优
				tempSet = []
				tempSet.append(compareValue)
		elif (compareValue == prevValue).all():  # 若比较元素和第一等级的最后一个元素相等时，肯定为第一等级
			firstRankSet.append(index[i])
		elif _relevant(prevValue, compareValue) == 0:  # 若第一等级的最后一个元素支配比较元素，则比较元素必不为第一等级
			continue
		elif _relevant(prevValue, compareValue) == 2:  # 若第一等级的最后一个元素被比较元素支配，则比较元素必为第一等级
			firstRankSet.append(index[i])
			prevValue = compareValue
		else:  # 若前面条件都不符合，则和比较集合中元素比较
			isNonDominated = True
			for j in reversed(range(len(tempSet))):  # 从后比到前，保持有序，一旦支配比较的元素则必和前面的支配，或互不支配
				relevantValue = _relevant(tempSet[j], compareValue)
				if relevantValue == 0:  # 比较元素一旦被比较集合中某个元素支配，在比较元素必不为第一等级， 直接退出
					isNonDominated = False
					break
				elif relevantValue == 2:  # 比较元素一旦支配比较集合中某个元素，在比较元素必为第一等级， 直接退出
					tempSet = tempSet[:j] + tempSet[j + 1:]
					# tempSet = tempSet[:j]
					break
			if isNonDominated:
				tempSet.append(prevValue)
				tempSet.append(compareValue)
				firstRankSet.append(index[i])
				prevValue = compareValue
	return firstRankSet

def _relevant(prevValue, nextValue):
	"""返回三个值：[第一维不做比较]
	0：prevValue 支配 nextValue 或值相等
	1：无关系，互不支配
	2：nextValue 支配 prevValue
	"""
	p = prevValue[1:]
	n = nextValue[1:]
	if (p <= n).all():
		return 0
	elif (n <= p).all():
		return 2
	else:
		return 1

if __name__ == '__main__':
	data = np.array(
		[[9, 1], [7, 2], [5, 4], [4, 5], [3, 6], [2, 7], [1, 9], [10, 1], [8, 5], [7, 6], [5, 7], [4, 8], [3, 9],
		 [10, 5], [9, 6], [8, 7], [7, 9], [10, 6], [9, 7], [8, 9]])
	data = np.array([[ 1,17, 5],
					 [ 2, 1, 7],
					 [ 2, 7,17],
					 [ 2,11,12],
					 [ 2,19,11],
					 [14, 4, 9],
					 [14,12,10],
					 [15, 3,13],
					 [16,12, 1],
					 [17, 7, 6]])

	from MOEA.sample.LatinHyperCube import LatinHyperCube
	from pylab import *
	from MOEA.ndomin.NDSetDelete import NDSetDelete
	data = LatinHyperCube(1000, 9)

	start = time.clock()
	pereto = NDSetDelete(data)
	end = time.clock()
	print('NDSetDelete耗时：', end - start)
	print(len(pereto))

	start = time.clock()
	pereto = NDSetSwk(data)
	end = time.clock()
	print('NDSetSwk耗时：', end - start)
	print(len(pereto))