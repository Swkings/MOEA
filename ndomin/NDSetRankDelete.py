#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : NDSetRankDelete.py
@Date  : 2019/01/10
@Desc  : 删除法求非支配排序集
"""
import numpy as np
import copy
def NDSetRankDelete(data, maxLevel=None):
	"""可求排序等级
	本质为删除法
	"""
	if type(data) == np.ndarray:
		size = data.shape[0]
	else:
		size = len(data)
	peretoRank = []
	indexList = [i for i in range(size)]
	while True:
		NDSet = []
		deleteSet = []
		while len(indexList) > 0:
			DSet = []
			tempIndividual = indexList.pop(0)
			p = data[tempIndividual]
			isNonDominated = True
			for item in indexList:
				q = data[item]
				if _isDominated(p, q):
					deleteSet.append(item)
				else:
					DSet.append(item)
					if _isDominated(q, p):
						isNonDominated = False
			indexList = DSet
			if isNonDominated:
				NDSet.append(tempIndividual)
			else:
				deleteSet.append(tempIndividual)
		if len(deleteSet) == 0:
			peretoRank.append(NDSet)
			break
		indexList = deleteSet
		peretoRank.append(NDSet)
		if maxLevel is None:
			continue
		else:
			if len(peretoRank) >= maxLevel:
				peretoRank.append(deleteSet)
				break
	return peretoRank

def _isDominated(p, q):
	if (p <= q).all() and (p != q).any():
		return True
	return False

if __name__ == '__main__':
	data = np.array(
		[[9, 1], [7, 2], [5, 4], [4, 5], [3, 6], [2, 7], [1, 9], [10, 1], [8, 5], [7, 6], [5, 7], [4, 8], [3, 9],
		 [10, 5], [9, 6], [8, 7], [7, 9], [10, 6], [9, 7], [8, 9]])
	pereto = NDSetRankDelete(data, 2)
	print(pereto)