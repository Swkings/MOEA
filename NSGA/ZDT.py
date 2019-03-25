#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : ZDT.py
@Date  : 2018/12/28
@Desc  : 
"""


class ZDT:
	def __init__(self):
		self.ZDT1_m = 30
		self.ZDT1_dimension = 2
		self.ZDT1_X1_SPAN = (0, 1)
		self.ZDT1_Xm_SPAN = (0, 1)
		self.ZDT1_isMin = True

		self.ZDT2_m = 30
		self.ZDT2_dimension = 2
		self.ZDT2_X1_SPAN = (0, 1)
		self.ZDT2_Xm_SPAN = (0, 1)
		self.ZDT2_isMin = True

		self.ZDT3_m = 10
		self.ZDT3_dimension = 2
		self.ZDT3_X1_SPAN = (0, 1)
		self.ZDT3_Xm_SPAN = (0, 1)
		self.ZDT3_isMin = True

		self.ZDT4_m = 10
		self.ZDT4_dimension = 2
		self.ZDT4_X1_SPAN = (0, 1)
		self.ZDT4_Xm_SPAN = (-5, 5)
		self.ZDT4_isMin = True

		self.ZDT6_m = 10
		self.ZDT6_dimension = 2
		self.ZDT6_X1_SPAN = (0, 1)
		self.ZDT6_Xm_SPAN = (0, 1)
		self.ZDT6_isMin = True

	def ZDT1(self, X):
		Y = np.zeros(2)
		Y[0] = X[0]
		g = 1 + (9 / (self.ZDT1_m - 1)) * np.sum(X[1:-1])
		Y[1] = g * (1 - ((Y[0] / g)**0.5))
		return Y

	def ZDT2(self, X):
		Y = np.zeros(2)
		Y[0] = X[0]
		g = 1 + (9 / (self.ZDT2_m - 1)) * np.sum(X[1:-1])
		Y[1] = g * (1 - (Y[0] / g) ** 2)
		return Y

	def ZDT3(self, X):
		Y = np.zeros(2)
		Y[0] = X[0]
		g = 1 + (9 / (self.ZDT2_m - 1)) * np.sum(X[1:-1])
		Y[1] = g * (1 - (np.sqrt(Y[0]/g)) - (Y[0]/g)*np.sin(10*np.pi*Y[0]))
		return Y

	def ZDT4(self, X):
		Y = np.zeros(2)
		Y[0] = X[0]
		g = 1 + 10*(self.ZDT4_m-1) + np.sum(np.power(X[1:-1], 2) - 10*np.cos(4*np.pi*X[1:-1]))
		Y[1] = g * (1 - (np.sqrt(Y[0] / g)))
		return Y

	def ZDT6(self, X):
		Y = np.zeros(2)
		Y[0] = 1 - np.exp(-4*X[0]) * (np.sin(6*np.pi*X[0])**6)
		g = 1 + 9 * (np.sum(X[1:-1]/(self.ZDT6_m-1))**0.25)
		Y[1] = g * (1 - (Y[0] / g) ** 2)
		return Y

if __name__ == '__main__':
	zdt = ZDT()
	print(zdt.ZDT1(np.zeros(zdt.ZDT1_m)))