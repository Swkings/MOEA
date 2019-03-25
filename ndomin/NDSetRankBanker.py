#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : NDSetRankBanker.py
@Date  : 2019/01/10
@Desc  : 庄家法求非支配排序集
		 时间复杂度：O(rmN)
"""
import numpy as np
import copy

def NDSetBankerForRank(data, index=[]):
	if len(index) == 0:
		if type(data) == np.ndarray:
			size = data.shape[0]
		else:
			size = len(data)
		indexList = [i for i in range(size)]
	else:
		indexList = index
	NDSet = []
	deleteSet = []
	while len(indexList) > 0:
		DSet = []
		bankerIndex = indexList.pop(0)
		banker = data[bankerIndex]
		isNonDominated = True
		for item in indexList:
			sampler = data[item]
			if _isDominated(sampler, banker):
				isNonDominated = False
				DSet.append(item)
			elif not _isDominated(banker, sampler):
				DSet.append(item)
			else:
				deleteSet.append(item)
		if isNonDominated:
			NDSet.append(bankerIndex)
		else:
			deleteSet.append(bankerIndex)
		indexList = DSet
	return NDSet, deleteSet

def NDSetRankBanker(data, maxLevel=None):
	peretoRank = []
	NDSet, deleteSet = NDSetBankerForRank(data)
	peretoRank.append(NDSet)
	while len(deleteSet) > 0:
		if maxLevel is None:
			pass
		else:
			if len(peretoRank) >= maxLevel:
				peretoRank.append(deleteSet)
				break
		NDSet, deleteSet = NDSetBankerForRank(data, deleteSet)
		peretoRank.append(NDSet)
	return peretoRank

def _isDominated(p, q):
	if (p <= q).all() and (p != q).any():
		return True
	return False

if __name__ == '__main__':
	data = np.array(
		[[9, 1], [7, 2], [5, 4], [4, 5], [3, 6], [2, 7], [1, 9], [10, 1], [8, 5], [7, 6], [5, 7], [4, 8], [3, 9],
		 [10, 5], [9, 6], [8, 7], [7, 9], [10, 6], [9, 7], [8, 9]])
	pereto = NDSetRankBanker(data)
	print(pereto)