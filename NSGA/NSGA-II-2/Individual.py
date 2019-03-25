#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : Individual.py
@Date  : 2019/01/01
@Desc  : 
"""
import numpy as np
from ZDT import *
class Individual:
	def __init__(self, X=None, Span=None, Func=None):
		"""
		当指定个体X值时，直接复制，否则根据Span（min，max）随机取值
		:param X:
		:param Span:
		"""
		if X is None:
			dimension = Span[0].shape[0]
			randX = np.random.uniform(0, 1, dimension)
			self.X = (Span[1] - Span[0]) * randX + Span[0]
		else:
			self.X = X
		self.Func = Func
		self.rank = -1
		self.crowdingDistance = 0
		# 被当前个体支配的个体集合
		self.dominateSet = []
		# 支配当前个体的个体个数
		self.dominatedNum = 0
	#
	# def calculateFuncValue(self, Func):
	# 	self.Y = Func(self.X)

	@property
	def Y(self):
		return self.Func(self.X)

	# 重写大于号、小于号和等于号
	def __gt__(self, other):
		if self.rank > other.rank:
			return True
		elif self.rank == other.rank:
			if self.crowdingDistance < other.crowdingDistance:
				return True
		return False

	def __lt__(self, other):
		if self.rank < other.rank:
			return True
		elif self.rank == other.rank:
			if self.crowdingDistance > other.crowdingDistance:
				return True
		return False

	def __eq__(self, other):
		if self is other:
			return True
		else:
			return False
	# 重写深拷贝
	def __deepcopy__(self):
		newIndividual = Individual(X=self.X, Func=self.Func)
		newIndividual.rank = self.rank
		newIndividual.crowdingDistance = self.crowdingDistance
		newIndividual.dominateSet = self.dominateSet
		newIndividual.dominatedNum = self.dominatedNum
		return newIndividual

	def isDominate(self, other):
		if (self.Y<=other.Y).all() and (self.Y!=other.Y).any():
			return True
		return False

	def reset(self):
		self.rank = -1
		self.crowdingDistance = 0
		self.dominateSet = []
		self.dominatedNum = 0

if __name__ == '__main__':
	fun = ZDT1()
	ind1 = Individual(Span=fun.span, Func=fun.Func)
	ind1.rank = 1
	ind1.crowdingDistance = 10
	ind2 = Individual(Span=fun.span, Func=fun.Func)
	ind2.rank = 1
	ind2.crowdingDistance = 20
	# print(ind1.Y, ind2.Y)
	# print(ind1.isDominate(ind2))
	# print(ind2.isDominate(ind1))
	# print(ind1.isDominate(ind1))
	print(ind1 > ind2)
	print(ind1 < ind2)
	print(ind1 == ind2)