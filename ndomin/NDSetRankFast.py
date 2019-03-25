#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : NDSetRankFast.py
@Date  : 2019/01/10
@Desc  : 快速排序法求非支配集
		 时间复杂度：O(N^2)
"""
"""
主要是维持 Np（支配p的个体数），Sp（被p支配的个体集合）
"""
import numpy as np
import copy

def NDSetRankFast(data):
	"""
	O(N^2)
	:param data: 一个目标函数值数组
	:return: 目标序号
	"""
	if type(data) == np.ndarray:
		size = data.shape[0]
	else:
		size = len(data)
	# 第一部分：计算Np（支配p的个体数），Sp（被p支配的个体集合）
	Np = []
	Sp = []
	peretoRankSet = []
	P1 = []
	for i in range(size):
		p = data[i]
		spSet = []
		ni = 0
		for j in range(size):
			q = data[j]
			if _isDominated(p, q):
				spSet.append(j)
			elif _isDominated(q, p):
				ni += 1
		Np.append(ni)
		Sp.append(spSet)
		if ni == 0:
			P1.append(i)
	peretoRankSet.append(P1)
	# 第二部分：计算其他种群层
	Pi = P1
	while len(Pi) > 0:
		H = []
		# 注意：p和q都是序号
		for p in Pi:
			for q in Sp[p]:
				Np[q] -= 1
				if Np[q] == 0:
					H.append(q)
		if len(H) > 0:
			peretoRankSet.append(H)
		Pi = copy.deepcopy(H)
	return peretoRankSet

def _isDominated(p, q):
	if (p <= q).all() and (p != q).any():
		return True
	return False

if __name__ == '__main__':
	data = np.array(
		[[9, 1], [7, 2], [5, 4], [4, 5], [3, 6], [2, 7], [1, 9], [10, 1], [8, 5], [7, 6], [5, 7], [4, 8], [3, 9],
		 [10, 5], [9, 6], [8, 7], [7, 9], [10, 6], [9, 7], [8, 9]])
	pereto = NDSetRankFast(data)
	print(pereto)