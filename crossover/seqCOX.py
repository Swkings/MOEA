#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : seqCOX.py
@Date  : 2019/01/12
@Desc  : 序列顺序交叉
"""
import numpy as np
import random
import copy

def seqCOX(data, cp):
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
			newPopulation[2 * count], newPopulation[2 * count + 1] = OX(data[2 * count], data[2 * count + 1],
			                                                             chromosomeLength, crossoverPosStart, crossoverPosEnd)
		else:
			newPopulation[2 * count], newPopulation[2 * count + 1] = data[2 * count], data[2 * count + 1]
		count += 1
	return newPopulation

def seqCOXIndividual(individualA, individualB):
	if type(individualA) is list:
		individualA = np.array(individualA)
		individualB = np.array(individualB)
	chromosomeLength = individualA.shape[0]
	crossoverPos = random.sample(range(chromosomeLength - 1), 2)
	(crossoverPosStart, crossoverPosEnd) = (min(crossoverPos), max(crossoverPos))
	child1, child2 = OX(individualA, individualB, chromosomeLength, crossoverPosStart, crossoverPosEnd)
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

def OX(individual1, individual2, chromosomeLength, crossoverPosStart, crossoverPosEnd):
	"""
	OX算法 顺序交叉算法
	:param individual:
	:param crossoverPosStart:
	:param crossoverPosEnd:
	:return: child1, child2
	"""
	individual1Part = (copy.deepcopy(individual1[crossoverPosStart:crossoverPosEnd + 1])).tolist()
	individual2Part = (copy.deepcopy(individual2[crossoverPosStart:crossoverPosEnd + 1])).tolist()
	p1, p2 = (copy.deepcopy(individual1)).tolist(), (copy.deepcopy(individual2)).tolist()
	p1 = p1[crossoverPosEnd + 1:] + p1[0:crossoverPosEnd + 1]
	p2 = p2[crossoverPosEnd + 1:] + p2[0:crossoverPosEnd + 1]
	for i in range(len(individual1Part)):
		p1.remove(individual2Part[i])
		p2.remove(individual1Part[i])
	child1 = p2[(chromosomeLength - crossoverPosEnd):] + individual1Part + p2[0:(
				chromosomeLength - crossoverPosEnd)]
	child2 = p1[(chromosomeLength - crossoverPosEnd):] + individual2Part + p1[0:(
				chromosomeLength - crossoverPosEnd)]
	return (child1, child2)

if __name__ == '__main__':
	from MOEA.sample import seqSample as ss
	p = ss.seqSample(10, 10)
	print(p)
	newP = seqCOX(p, 0.9)
	print(newP)
	# a = np.array([1, 2])
	# print(np.where(a == 2)[0][0])