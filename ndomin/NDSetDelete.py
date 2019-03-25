#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : NDSetDelete.py
@Date  : 2019/01/10
@Desc  : 删除法求非支配集
		 时间复杂度：O(rNlogN)
"""
"""
1：：从list中取出一个个体i
2：将i与list中其他个体比较，被i支配的，从list中删除
3：若i不被任何个体支配，则加入NDSet，否则弃之
3：若list为空则结束，否则执行1~3
"""
import numpy as np
import copy
def NDSetDelete(data):
	"""求DNSet O(rNlogN)
	删除法，将被支配的个体删除集合，直至为空
	"""
	if type(data) == np.ndarray:
		size = data.shape[0]
	else:
		size = len(data)
	indexList = [i for i in range(size)]
	NDSet = []
	while len(indexList) > 0:
		DSet = []
		tempIndividual = indexList.pop(0)
		p = data[tempIndividual]
		isNonDominated = True
		for item in indexList:
			q = data[item]
			if not _isDominated(p, q):
				DSet.append(item)
				if _isDominated(q, p):
					isNonDominated = False
		indexList = DSet
		if isNonDominated:
			NDSet.append(tempIndividual)
	return NDSet

def _isDominated(p, q):
	if (p <= q).all() and (p != q).any():
		return True
	return False

if __name__ == '__main__':
	data = np.array(
		[[9, 1], [7, 2], [5, 4], [4, 5], [3, 6], [2, 7], [1, 9], [10, 1], [8, 5], [7, 6], [5, 7], [4, 8], [3, 9],
		 [10, 5], [9, 6], [8, 7], [7, 9], [10, 6], [9, 7], [8, 9]])
	pereto = NDSetDelete(data)
	print(pereto)