#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : NDSetArena.py
@Date  : 2019/01/10
@Desc  : 擂台法求非支配集
		 时间复杂度：O(rmN)
"""
"""
1：：从list中取出一个个体i, 作为擂主
2：将i与list中其他个体比较：
					被i支配的，从list中删除
					支配i的，取代i成为擂主，并记录好换擂主的位置SignCount
					互不支配的加入DSet, 进入下一轮守擂
3：进行完一轮比较后，最后一个擂主还需要与前SignCount个个体比较，排除个体
"""
import numpy as np
import copy
def NDSetArena(data):
	"""擂台法 O(rmN)"""
	if type(data) == np.ndarray:
		size = data.shape[0]
	else:
		size = len(data)
	indexList = [i for i in range(size)]
	NDSet = []
	while len(indexList) > 0:
		DSet = []
		arenaIndex = indexList.pop(0)
		arena = data[arenaIndex]
		SignCount = 0
		for item in indexList:
			competitor = data[item]
			if _isDominated(competitor, arena):
				arena, arenaIndex = competitor, item
				SignCount = len(DSet)
			elif not _isDominated(arena, competitor):
				DSet.append(item)
		NDSet.append(arenaIndex)
		for i in reversed(range(SignCount)):
			competitor = data[DSet[i]]
			if _isDominated(arena, competitor):
				DSet.remove(DSet[i])
		indexList = DSet
	return NDSet

def _isDominated(p, q):
	if (p <= q).all() and (p != q).any():
		return True
	return False

if __name__ == '__main__':
	data = np.array(
		[[9, 1], [7, 2], [5, 4], [4, 5], [3, 6], [2, 7], [1, 9], [10, 1], [8, 5], [7, 6], [5, 7], [4, 8], [3, 9],
		 [10, 5], [9, 6], [8, 7], [7, 9], [10, 6], [9, 7], [8, 9]])
	pereto = NDSetArena(data)
	print(pereto)