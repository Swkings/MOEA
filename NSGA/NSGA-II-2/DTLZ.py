#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : DTLZ.py
@Date  : 2019/01/03
@Desc  : 
"""
from functools import reduce
import numpy as np
class DTLZ1:
	def __init__(self, M=3):
		"""定义有7个自变量"""
		self.dimension = 7
		self.objFuncNum = M
		self.isMin = True
		self.min = np.zeros(self.dimension)
		self.max = np.zeros(self.dimension) + 1
		self.span = (self.min, self.max)

	def Func(self, X):
		M = self.objFuncNum
		k = X.shape[0] - M + 1
		XM = X[M-1:]
		g = 100 * (k + np.sum((XM-0.5)**2 - np.cos(20*np.pi*(XM-0.5))))
		Y = np.empty(M)
		Y[0] = 0.5 * self.Mul(X[:M-1]) * (g + 1)
		for i in range(1, M-1):
			Y[i] = 0.5 * self.Mul(X[:M-i-1]) * (1 - X[M-i-1]) * (g + 1)
		Y[-1] = 0.5 * (1 - X[0]) * (g + 1)
		return Y

	def Mul(self, X):
		"""X数组内的元素求积"""
		Y = reduce(lambda x1, x2: x1*x2, X)
		return Y

if __name__ == '__main__':
	DTLZ = DTLZ1()
	X = np.zeros(7)
	X[2:] += 0.5
	X = np.array([1, 0, 0., 0.72872384, 0.36045774, 0.55786712, 0.77502622])
	# print(np.cos(X))
	print(DTLZ.Func(X))
