#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : realMPN.py
@Date  : 2019/01/12
@Desc  : 多项式变异
"""
import numpy as np
import random


def realMPN(data, mp, fixElement=None):
	"""
	多项式变异
	:return:
	"""
	if type(data) is list:
		data = np.array(data)
	populationSize = data.shape[0]
	dimension = data.shape[1]
	newPopulation = np.empty((populationSize, dimension))
	mpSet = []
	yita = 20
	for i in range(populationSize):
		mpSet.append(random.random())
	for i in range(populationSize):
		if mpSet[i] < mp:
			delta = np.zeros(dimension)
			randAi = np.random.random(dimension)
			for k in range(dimension):
				if randAi[k] <= 0.5:
					delta[k] = (2 * randAi[k]) ** (1 / (yita + 1)) - 1
				else:
					delta[k] = 1 - ((2 * (1 - randAi[k])) ** (1 / (yita + 1)))
			child = data[i] + delta
			if fixElement is None:
				pass
			else:
				fixElement(child)
			newPopulation[i] = child
		else:
			newPopulation[i] = data[i]
	return newPopulation

def realMPNIndividual(individual, fixElement=None):
	if type(individual) is list:
		individual = np.array(individual)
	dimension = individual.shape[0]
	yita = 5
	delta = np.zeros(dimension)
	randAi = np.random.random(dimension)
	for k in range(dimension):
		if randAi[k] <= 0.5:
			delta[k] = (2 * randAi[k]) ** (1 / (yita + 1)) - 1
		else:
			delta[k] = 1 - ((2 * (1 - randAi[k])) ** (1 / (yita + 1)))
	child = individual + delta
	if fixElement is None:
		pass
	else:
		fixElement(child)
	return child



if __name__ == '__main__':
	from MOEA.sample import LatinHyperCube as lh
	def fixElement(individual):
		dimension = individual.shape[0]
		for i in range(dimension):
			if individual[i] < 0:
				individual[i] = 0
			elif individual[i] > 1:
				individual[i] = 1
	p = lh.LatinHyperCube(100, 3)
	print(p)
	newP = realMPN(p, 0.9, fixElement)
	print(newP)