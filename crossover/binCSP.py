#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : binCSP.py
@Date  : 2019/01/12
@Desc  : 二进制单点交叉
"""
import numpy as np
import copy
import random

def binCSP(data, cp):
	"""
	二进制单点交叉
	:param data: 种群数据
	:param cp: 交叉概率
	:param populationSize: 需要的种群大小
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
			cPos = random.randint(1, chromosomeLength-2)
			individual1Part1 = data[2 * count, 0:cPos]
			individual1Part2 = data[2 * count, cPos:chromosomeLength]
			individual2Part1 = data[2 * count + 1, 0:cPos]
			individual2Part2 = data[2 * count + 1, cPos:chromosomeLength]
			child1 = np.concatenate((individual1Part1, individual2Part2))
			child2 = np.concatenate((individual2Part1, individual1Part2))
			newPopulation[2 * count] = child1
			newPopulation[2 * count + 1] = child2
		else:
			newPopulation[2 * count] = data[2 * count]
			newPopulation[2 * count + 1] = data[2 * count + 1]
		count += 1
	return newPopulation

def binCSPIndividual(individualA, individualB):
	if type(individualA) is list:
		individualA = np.array(individualA)
		individualB = np.array(individualB)
	chromosomeLength = individualA.shape[0]
	cPos = random.randint(1, chromosomeLength - 2)
	individualAPart1 = individualA[0:cPos]
	individualAPart2 = individualA[cPos:chromosomeLength]
	individualBPart1 = individualB[0:cPos]
	individualBPart2 = individualB[cPos:chromosomeLength]
	child1 = np.concatenate((individualAPart1, individualBPart2))
	child2 = np.concatenate((individualBPart1, individualAPart2))
	return child1, child2


if __name__ == '__main__':
	from MOEA.sample import binSample as bs
	p = bs.binSample(10, 10)
	print(p)
	newP = binCSP(p, 0.9)
	print(newP)

