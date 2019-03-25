#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : NDSetRankArena.py
@Date  : 2019/01/10
@Desc  : 擂台法求非支配排序集
		 时间复杂度：O(rmN)
"""
import numpy as np
import copy

def NDSetArenaForRank(data, index=[]):
	"""擂台法 O(rmN)"""
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
		arenaIndex = indexList.pop(0)
		arena = data[arenaIndex]
		SignCount = 0
		for item in indexList:
			competitor = data[item]
			if _isDominated(competitor, arena):
				deleteSet.append(arenaIndex)
				arena, arenaIndex = competitor, item
				SignCount = len(DSet)
			elif not _isDominated(arena, competitor):
				DSet.append(item)
			else:
				deleteSet.append(item)
		NDSet.append(arenaIndex)
		for i in reversed(range(SignCount)):
			competitor = data[DSet[i]]
			if _isDominated(arena, competitor):
				deleteSet.append(DSet[i])
				DSet.remove(DSet[i])
		indexList = DSet
	return NDSet, deleteSet

def NDSetRankArena(data, maxLevel=None):
	peretoRank = []
	NDSet,  deleteSet = NDSetArenaForRank(data)
	peretoRank.append(NDSet)
	while len(deleteSet) > 0:
		if maxLevel is None:
			pass
		else:
			if len(peretoRank) >= maxLevel:
				peretoRank.append(deleteSet)
				break
		NDSet, deleteSet = NDSetArenaForRank(data, deleteSet)
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
	pereto = NDSetRankArena(data)
	print(pereto)