#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : fastNonDominatedSort.py
@Date  : 2019/01/01
@Desc  : 
"""
import copy

def fastNonDominatedSort(population):
	"""
	非支配排序
	:param population:
	:return: 一个分层的列表peretoSort，每一层都是Individual
	"""
	populationSize = len(population)
	peretoSort = []
	pereto = []
	# 第一部分：计算Np（支配p的个体数），Sp（被p支配的个体集合）
	for i in range(populationSize-1):
		for j in range(i+1, populationSize):
			if population[i].isDominate(population[j]):
				population[i].dominateSet.append(population[j])
				population[j].dominatedNum += 1
			elif population[j].isDominate(population[i]):
				population[j].dominateSet.append(population[i])
				population[i].dominatedNum += 1
		if population[i].dominatedNum == 0:
			population[i].rank = 0
			pereto.append(population[i])
	peretoSort.append(pereto)
	rank = 1
	while len(pereto) > 0:
		H = []
		for ind in pereto:
			for indDominate in ind.dominateSet:
				indDominate.dominatedNum -= 1
				if indDominate.dominatedNum == 0:
					indDominate.rank = rank
					H.append(indDominate)
		pereto = H
		if len(pereto) > 0:
			peretoSort.append(pereto)
		rank += 1
	return peretoSort

def fastNonDominated(population):
	"""
	为了方便计算，支配排序各集合都储存对应的序号（下标）
	:param population: 一个目标函数值数组
	:return: 目标序号
	"""
	funcValue = population
	size = funcValue.shape[0]
	# 第一部分：计算Np（支配p的个体数），Sp（被p支配的个体集合）

	Np = []
	Sp = []
	peretoSortPop = []
	P1 = []
	for i in range(size):
		p = funcValue[i]
		spSet = []
		ni = 0
		for j in range(size):
			q = funcValue[j]
			if isDominated(p, q):
				spSet.append(j)
			elif isDominated(q, p):
				ni += 1
		Np.append(ni)
		Sp.append(spSet)
		if ni == 0:
			P1.append(i)
	peretoSortPop.append(P1)
	# 第二部分：计算其他种群层
	Pi = P1
	while len(Pi) > 0:
		H = []
		# 注意：p和q都是序号
		for p in Pi:
			for q in Sp[p]:
				Np[q] -= 1
				if Np[q] == 0:
					H.append(q)
		if len(H) > 0:
			peretoSortPop.append(H)
		Pi = copy.deepcopy(H)
	return peretoSortPop

def isDominated(p, q):
	if (p<=q).all() and (p!=q).any():
		return True
	return False

if __name__ == '__main__':
	pass
