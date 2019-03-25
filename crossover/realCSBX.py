#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : realCSBX.py
@Date  : 2019/01/12
@Desc  : 实数模拟二进制交叉
"""
import numpy as np
import copy
import random


def realCSBX(data, cp, fixElement=None):
	"""
	实数模拟二进制交叉
	:param data:
	:param cp:
	:param fixElement: 修正函数，修正超出约束范围的值
	:return:
	"""
	if type(data) is list:
		data = np.array(data)
	populationSize = data.shape[0]
	dimension = data.shape[1]
	newPopulation = np.empty((populationSize, dimension))
	cpSet = []
	count = 0
	mu = 1  # SBX方式的分布指数
	for i in range(int(populationSize / 2)):
		cpSet.append(random.random())
	for cpi in cpSet:
		if cpi < cp:
			randList = np.random.random(dimension)
			alpha = np.zeros(dimension)
			for i in range(len(randList)):
				if randList[i] <= 0.5:
					alpha[i] = (2.0 * randList[i]) ** (1.0 / (mu + 1))
				else:
					alpha[i] = (1 / (2 * (1 - randList[i]))) ** (1.0 / (mu + 1))
			child1 = 0.5 * ((1 + alpha) * data[2 * count] + (1 - alpha) * data[2 * count + 1])
			child2 = 0.5 * ((1 - alpha) * data[2 * count] + (1 + alpha) * data[2 * count + 1])
			if fixElement is None:
				pass
			else:
				fixElement(child1)
				fixElement(child2)
			newPopulation[2 * count] = child1
			newPopulation[2 * count + 1] = child2
		else:
			newPopulation[2 * count] = data[2 * count]
			newPopulation[2 * count + 1] = data[2 * count + 1]
		count += 1
	return newPopulation

def realCSBXIndividual(individualA, individualB, fixElement=None):
	if type(individualA) is list:
		individualA = np.array(individualA)
		individualB = np.array(individualB)
	dimension = individualA.shape[0]
	mu = 2  # SBX方式的分布指数
	randList = np.random.random(dimension)
	alpha = np.zeros(dimension)
	for i in range(len(randList)):
		if randList[i] <= 0.5:
			alpha[i] = (2.0 * randList[i]) ** (1.0 / (mu + 1))
		else:
			alpha[i] = (1 / (2 * (1 - randList[i]))) ** (1.0 / (mu + 1))
	child1 = 0.5 * ((1 + alpha) * individualA + (1 - alpha) * individualB)
	child2 = 0.5 * ((1 - alpha) * individualA + (1 + alpha) * individualB)
	if fixElement is None:
		pass
	else:
		fixElement(child1)
		fixElement(child2)
	return child1, child2

if __name__ == '__main__':
	from MOEA.sample import LatinHyperCube as lh
	p = lh.LatinHyperCube(10, 3)
	print(p)
	newP = realCSBX(p, 0.9)
	print(newP)
	# a = np.array([1, 2])
	# print(np.where(a == 2)[0][0])