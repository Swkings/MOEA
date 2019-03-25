#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : seqCPMX.py
@Date  : 2019/01/12
@Desc  : 序列部分匹配交叉
"""
import numpy as np
import random
import copy

def seqCPMX(data, cp):
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
			crossoverPos = random.sample(range(chromosomeLength - 1), 2)
			(crossoverPosStart, crossoverPosEnd) = (min(crossoverPos), max(crossoverPos))
			newPopulation[2 * count], newPopulation[2 * count + 1] = PMX(data[2 * count], data[2 * count + 1],
			                                                             chromosomeLength, crossoverPosStart, crossoverPosEnd)
		else:
			newPopulation[2 * count], newPopulation[2 * count + 1] = data[2 * count], data[2 * count + 1]
		count += 1
	return newPopulation

def seqCPMXIndividual(individualA, individualB):
	if type(individualA) is list:
		individualA = np.array(individualA)
		individualB = np.array(individualB)
	chromosomeLength = individualA.shape[0]
	crossoverPos = random.sample(range(chromosomeLength - 1), 2)
	(crossoverPosStart, crossoverPosEnd) = (min(crossoverPos), max(crossoverPos))
	child1, child2 = PMX(individualA, individualB, chromosomeLength, crossoverPosStart, crossoverPosEnd)
	return child1, child2

def PMX(individual1, individual2, chromosomeLength, crossoverPosStart, crossoverPosEnd):
	individual1Part = copy.deepcopy(individual1[crossoverPosStart:crossoverPosEnd + 1])  # 注意拷贝
	individual2Part = copy.deepcopy(individual2[crossoverPosStart:crossoverPosEnd + 1])
	# 交叉的片段直接交换
	individual1[crossoverPosStart:crossoverPosEnd + 1] = individual2Part
	individual2[crossoverPosStart:crossoverPosEnd + 1] = individual1Part
	# 调整顺序，解决冲突
	for i in range(chromosomeLength):
		if i in range(crossoverPosStart, crossoverPosEnd + 1):
			continue
		while (individual2Part == individual1[i]).any():
			index = np.where(individual2Part == individual1[i])[0][0]
			individual1[i] = individual1Part[index]
		while (individual1Part == individual2[i]).any():
			index = np.where(individual1Part == individual2[i])[0][0]
			individual2[i] = individual2Part[index]
	return individual1, individual2

if __name__ == '__main__':
	from MOEA.sample import seqSample as ss
	p = ss.seqSample(10, 10)
	print(p)
	newP = seqCPMX(p, 0.9)
	print(newP)
	# a = np.array([1, 2])
	# print(np.where(a == 2)[0][0])