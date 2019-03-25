#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : NDSetRankENS_SS.py
@Date  : 2019/03/22
@Desc  : 
"""
import numpy as np
import math

def NDSetRankENS_SS(data):
	if type(data) is list:
		data = np.array(data)
	return _NDSetRank(data)

def _NDSetRank(data):
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
	for i in range(1, data.shape[0]):
		compareValue = compareSet[i]
		minRank = 0
		maxRank = len(rankSet)
		iRank = _findRank(rankSet, minRankValueSet, compareValue, minRank, maxRank)
		_addToRankSet(rankSet, minRankValueSet, index[i], compareValue, iRank)
	return rankSet

def _findRank(rankSet, minRankValueSet, compareValue, minRank, maxRank):
	while True:
		if minRank >= maxRank:
			return maxRank
		else:
			isNonDominated = True
			minRankMinValueSet = minRankValueSet[minRank]
			for i in reversed(range(len(minRankMinValueSet))):
				if not _isDominated(minRankMinValueSet[i], compareValue):
					continue
				else:
					minRank += 1
					isNonDominated = False
					break
			if isNonDominated:
				return minRank
	return minRank

def _addToRankSet(rankSet, minRankValueSet, index, compareValue, iRank):
	# 如果等级超过上限，用处理方式处理
	if iRank >= len(rankSet):
		rankSet.append([index])
		minRankValueSet.append([compareValue])
	else:
		rankSet[iRank].append(index)
		minRankValueSet[iRank].append(compareValue)

def _isDominated(p, q):
	if (p <= q).all() and (p != q).any():
		return True
	return False
