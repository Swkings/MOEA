#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : ZDT.py
@Date  : 2019/02/01
@Desc  : 
"""
import numpy as np
class ZDT1:
	def __init__(self):
		self.dimension = 30
		self.objFuncNum = 2
		self.isMin = True
		self.min = np.zeros(self.dimension)
		self.max = np.zeros(self.dimension) + 1
		self.span = (self.min, self.max)

	def Func(self, X):
		Y = np.zeros(2)
		Y[0] = X[0]
		g = 1 + (9 / (self.dimension - 1)) * np.sum(X[1:-1])
		Y[1] = g * (1 - ((Y[0] / g)**0.5))
		return Y

class ZDT2:
	def __init__(self):
		self.dimension = 30
		self.objFuncNum = 2
		self.isMin = True
		self.min = np.zeros(self.dimension)
		self.max = np.zeros(self.dimension) + 1
		self.span = (self.min, self.max)

	def Func(self, X):
		Y = np.zeros(2)
		Y[0] = X[0]
		g = 1 + (9 / (self.dimension - 1)) * np.sum(X[1:-1])
		Y[1] = g * (1 - ((Y[0] / g) ** 2))
		return Y


class ZDT3:
	def __init__(self):
		self.dimension = 10
		self.objFuncNum = 2
		self.isMin = True
		self.min = np.zeros(self.dimension)
		self.max = np.zeros(self.dimension) + 1
		self.span = (self.min, self.max)

	def Func(self, X):
		Y = np.zeros(2)
		Y[0] = X[0]
		g = 1 + (9 / (self.dimension - 1)) * np.sum(X[1:-1])
		Y[1] = g * (1 - (np.sqrt(Y[0] / g)) - (Y[0] / g) * np.sin(10 * np.pi * Y[0]))
		return Y


class ZDT4:
	def __init__(self):
		self.dimension = 10
		self.objFuncNum = 2
		self.isMin = True
		self.min = np.zeros(self.dimension) - 5
		self.min[0] = 0
		self.max = np.zeros(self.dimension) + 5
		self.max[0] = 1
		self.span = (self.min, self.max)

	def Func(self, X):
		Y = np.zeros(2)
		Y[0] = X[0]
		g = 1 + 10 * (self.dimension - 1) + np.sum(np.power(X[1:-1], 2) - 10 * np.cos(4 * np.pi * X[1:-1]))
		Y[1] = g * (1 - (np.sqrt(Y[0] / g)))
		return Y


class ZDT6:
	def __init__(self):
		self.dimension = 10
		self.objFuncNum = 2
		self.isMin = True
		self.min = np.zeros(self.dimension)
		self.max = np.zeros(self.dimension) + 1
		self.span = (self.min, self.max)

	def Func(self, X):
		Y = np.zeros(2)
		Y[0] = 1 - np.exp(-4 * X[0]) * (np.sin(6 * np.pi * X[0]) ** 6)
		g = 1 + 9 * (np.sum(X[1:-1] / (self.dimension - 1)) ** 0.25)
		Y[1] = g * (1 - (Y[0] / g) ** 2)
		return Y

if __name__ == '__main__':
	zdt = ZDT1()
	print(zdt.Func(np.ones(zdt.dimension)))