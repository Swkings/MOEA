#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : binMSP.py
@Date  : 2019/01/12
@Desc  : 二进制单点变异
"""
import numpy as np
import random


def binMSP(data, mp):
	"""
	变异
	:return:
	"""
	if type(data) is list:
		data = np.array(data)
	populationSize = data.shape[0]
	chromosomeLength = data.shape[1]
	newPopulation = np.empty((populationSize, chromosomeLength))
	mpSet = []
	for i in range(populationSize):
		mpSet.append(random.random())
	for i in range(populationSize):
		if mpSet[i] < mp:
			mutationPos = random.randint(0, chromosomeLength-1)
			data[i][mutationPos] = int(data[i][mutationPos]) ^ 1
		newPopulation[i] = data[i]
	return newPopulation

def binMSPIndividual(individual):
	if type(individual) is list:
		individual = np.array(individual)
	chromosomeLength = individual.shape[0]
	mutationPos = random.randint(0, chromosomeLength-1)
	individual[mutationPos] = int(individual[mutationPos]) ^ 1
	return individual


if __name__ == '__main__':
	from MOEA.sample import binSample as ss
	p = ss.binSample(10, 10)
	print(p)
	newP = binMSP(p, 0.9)
	print(newP)