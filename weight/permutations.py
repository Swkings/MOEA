#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : permutations.py
@Date  : 2019/01/12
@Desc  : 全排列
"""
import numpy as np
import copy
def permutations(seq):
	seq = np.sort(seq)
	size = seq.shape[0]
	isVisited = [False for i in range(size)]
	temp = [None for i in range(size)]
	result = []
	def dfs(pos):
		if pos == size:
			result.append(copy.deepcopy(temp))
			return
		for i in range(size):
			if i>0 and seq[i]==seq[i-1] and not isVisited[i-1]:
				continue
			if not isVisited[i]:
				temp[pos] = seq[i]
				isVisited[i] = True
				dfs(pos+1)
				isVisited[i] = False  # 回溯
	dfs(0)
	return np.array(result)

if __name__ == '__main__':
	seq = [1, 1, 0, 0, 0, 0]
	result = permutations(seq)
	print(result)