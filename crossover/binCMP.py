#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : binCMP.py
@Date  : 2019/01/12
@Desc  : 二进制多点交叉
"""
import numpy as np
import copy
import random

def binCMP(data, cp, posNum=2):
	"""
	二进制多点交叉
	:param data: 种群数据
	:param cp: 交叉概率
	:return: newPopulation
	"""
	if type(data) is list:
		data = np.array(data)
	populationSize = data.shape[0]
	chromosomeLength = data.shape[1]
	newPopulation = np.empty((populationSize, chromosomeLength))
	cpSet = []
	count = 0
	for i in range(int(populationSize / 2)):
		cpSet.append(random.random())
	for cpi in cpSet:
		if cpi <= cp:
			cPos = np.sort(random.sample(range(1, chromosomeLength-1), posNum))
			cPos = np.append(cPos, chromosomeLength)
			cPos = np.insert(cPos, 0, 0)
			child1 = np.empty(chromosomeLength)
			child2 = np.empty(chromosomeLength)
			for i in range(1, cPos.shape[0]):
				if i % 2 == 1:
					child1[cPos[i-1]:cPos[i]] = data[2 * count + 1, cPos[i-1]:cPos[i]]
					child2[cPos[i-1]:cPos[i]] = data[2 * count, cPos[i-1]:cPos[i]]
				else:
					child1[cPos[i-1]:cPos[i]] = data[2 * count, cPos[i-1]:cPos[i]]
					child2[cPos[i-1]:cPos[i]] = data[2 * count + 1, cPos[i-1]:cPos[i]]
			newPopulation[2 * count] = child1
			newPopulation[2 * count + 1] = child2
		else:
			newPopulation[2 * count] = data[2 * count]
			newPopulation[2 * count + 1] = data[2 * count + 1]
		count += 1
	return newPopulation

def binCMPIndividual(individualA, individualB, posNum=2):
	if type(individualA) is list:
		individualA = np.array(individualA)
		individualB = np.array(individualB)
	chromosomeLength = individualA.shape[0]
	cPos = np.sort(random.sample(range(1, chromosomeLength - 1), posNum))
	cPos = np.append(cPos, chromosomeLength)
	cPos = np.insert(cPos, 0, 0)
	child1 = np.empty(chromosomeLength)
	child2 = np.empty(chromosomeLength)
	for i in range(1, cPos.shape[0]):
		if i % 2 == 1:
			child1[cPos[i - 1]:cPos[i]] = individualB[cPos[i - 1]:cPos[i]]
			child2[cPos[i - 1]:cPos[i]] = individualA[cPos[i - 1]:cPos[i]]
		else:
			child1[cPos[i - 1]:cPos[i]] = individualA[cPos[i - 1]:cPos[i]]
			child2[cPos[i - 1]:cPos[i]] = individualB[cPos[i - 1]:cPos[i]]
	return child1, child2



if __name__ == '__main__':
	from MOEA.sample import binSample as bs
	p = bs.binSample(10, 10)
	print(p)
	newP = binCMP(p, 0.9)
	print(newP)
