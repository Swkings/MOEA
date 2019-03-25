#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : NDSetBanker.py
@Date  : 2019/01/10
@Desc  : 庄家法求非支配集
		 时间复杂度：O(rmN)
"""
"""
1：：从list中取出一个个体i, 作为庄家
2：将i与list中其他个体比较：
					被i支配的，从list中删除
					一旦有支配i的，将标记修改isNonDominated
					互不支配的加入DSet, 进入下一轮比较
3：和擂台它的区别就是，庄家每一轮只有一个，中途不换
"""
import numpy as np
import copy

def NDSetBanker(data):
	"""庄家法"""
	if type(data) == np.ndarray:
		size = data.shape[0]
	else:
		size = len(data)
	indexList = [i for i in range(size)]
	NDSet = []
	while len(indexList) > 0:
		DSet = []
		bankerIndex = indexList.pop(0)
		banker = data[bankerIndex]
		isNonDominated = True
		for item in indexList:
			sampler = data[item]
			if not _isDominated(banker, sampler):
				DSet.append(item)
				if _isDominated(sampler, banker):
					isNonDominated = False
		if isNonDominated:
			NDSet.append(bankerIndex)
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
	pereto = NDSetBanker(data)
	print(pereto)