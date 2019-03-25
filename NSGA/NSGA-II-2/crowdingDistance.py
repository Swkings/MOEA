#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : crowdingDistance.py
@Date  : 2019/01/01
@Desc  : 
"""


def crowdingDistance(population):
	populationSize = len(population)
	objFuncNum = population[0].Y.shape[0]
	for item in population:
		item.crowdingDistance = 0
	for i in range(objFuncNum):
		population.sort(key=lambda ind: ind.Y[i])
		for j in range(populationSize):
			if j == 0 or j == populationSize-1:
				population[j].crowdingDistance += 1000000
			else:
				population[j].crowdingDistance += (population[j+1].Y[i] - population[j-1].Y[i])
	return population