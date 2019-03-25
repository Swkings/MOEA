#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : NSGA-II.py
@Date  : 2018/12/30
@Desc  : 
"""
from pylab import *
import numpy as np
import pandas as pd
import copy
import random
from operator import itemgetter
import matplotlib.pyplot as plt
from ZDT import ZDT

class NSGAII:
	def __init__(self, objFunc, populationSize=10, maxGeneration=5, cp=0.9, mp=0.05, isMin=True):
		"""
		初始化各参数
		:param objFunc: 目标函数对象
		:param genoNum: 每条染色体基因个数
		:param populationSize: 种群大小
		:param maxGeneration: 迭代次数
		:param cp: 交叉概率
		:param mp: 变异概率
		:param isMin: True最小化问题， False最大化问题
		"""
		self.population = []
		self.objFunc = objFunc.ZDT1
		self.m = objFunc.ZDT1_m  # 染色体长度（变量的个数）
		self.dimension = objFunc.ZDT1_dimension
		self.X1_SPAN = objFunc.ZDT1_X1_SPAN
		self.Xm_SPAN = objFunc.ZDT1_Xm_SPAN
		self.populationSize = populationSize
		self.maxGeneration = maxGeneration
		self.isMin = isMin
		self.cp = cp
		self.mp = mp

	def populationInit(self):
		"""
		初始化种群
		:return: population
		"""
		population = []
		for i in range(self.populationSize):
			# 随机生成单条染色体
			chromosome = []
			# 生成X1
			X1 = random.uniform(self.X1_SPAN[0], self.X1_SPAN[1])
			# X1 = random.random()*(self.X1_SPAN[1]-self.X1_SPAN[0]) + self.X1_SPAN[0]
			chromosome.append(X1)
			# 生成Xm
			for j in range(self.m-1):
				Xm = random.uniform(self.Xm_SPAN[0], self.Xm_SPAN[1])
				# Xm = random.random() * (self.Xm_SPAN[1] - self.Xm_SPAN[0]) + self.Xm_SPAN[0]
				chromosome.append(Xm)
			population.append(np.array(chromosome))
		self.population = copy.deepcopy(population)
		return population

	def selection(self):
		newPopulation = []
		funcValue = self.calculateFuncValue(self.population)
		distanceSort, distance = self.crowdingDistance(range(self.populationSize), funcValue)
		for i in range(self.populationSize):
			randPosX, randPosY = random.sample(range(0, self.populationSize), 2)
			p = self.population[randPosX]
			q = self.population[randPosY]
			if self.isDominated(p, q):
				newPopulation.append(p)
			elif self.isDominated(q, p):
				newPopulation.append(q)
			else:
				if distance[randPosX] > distance[randPosY]:
					newPopulation.append(p)
				else:
					newPopulation.append(q)
		return newPopulation

	def crossover(self, population):
		"""
		SBX算法
		:return:
		"""
		newPopulation = []
		cp = []
		count = 0
		mu = 1  # SBX方式的分布指数
		for i in range(int(self.populationSize/2)):
			cp.append(random.random())
		for cpi in cp:
			if cpi < self.cp:
				randList = np.random.random(self.m)
				alpha = np.zeros(self.m)
				for i in range(len(randList)):
					if randList[i] <= 0.5:
						alpha[i] = (2.0*randList[i])**(1.0/(mu+1))
					else:
						alpha[i] = (1/(2*(1-randList[i])))**(1.0/(mu+1))
				newIndividual1 = 0.5 * ((1 + alpha) * population[count] + (1 - alpha) * population[count + 1])
				newIndividual2 = 0.5 * ((1 - alpha) * population[count] + (1 + alpha) * population[count + 1])
				self.fixElement(newIndividual1)
				self.fixElement(newIndividual2)
				newPopulation.append(newIndividual1)
				newPopulation.append(newIndividual2)
			else:
				newPopulation.append(population[count])
				newPopulation.append(population[count + 1])
			count += 1
		return newPopulation

	def mutation(self, population):
		newPopulation = []
		mp = []
		m = 20
		for i in range(self.populationSize):
			mp.append(random.random())
		for i in range(self.populationSize):
			if mp[i] < self.mp:
				delta = 0
				randAi = np.random.random(m)
				for k in range(m):
					if randAi[k] <= (1/m):
						delta += (1/(2**k))
				randList = np.random.random(self.m)
				for j in range(self.m):
					L = 0
					if j == 0:
						L = np.abs(self.X1_SPAN[1] - self.X1_SPAN[0])
					else:
						L = np.abs(self.Xm_SPAN[1] - self.Xm_SPAN[0])
					if randList[j] <= 0.5:
						population[i][j] += (0.5 * L * delta)
					else:
						population[i][j] -= (0.5 * L * delta)
			self.fixElement(population[i])
			newPopulation.append(population[i])
		return newPopulation

	def fixElement(self, individual):
		if individual[0] < self.X1_SPAN[0]:
			individual[0] = self.X1_SPAN[0]
			# individual[0] = random.random() * (self.X1_SPAN[1] - self.X1_SPAN[0]) + self.X1_SPAN[0]
		elif individual[0] > self.X1_SPAN[1]:
			individual[0] = self.X1_SPAN[1]
			# individual[0] = random.random() * (self.X1_SPAN[1] - self.X1_SPAN[0]) + self.X1_SPAN[0]
		for i in range(1, self.m):
			if individual[i] < self.Xm_SPAN[0]:
				individual[i] = self.Xm_SPAN[0]
				# individual[i] = random.random() * (self.Xm_SPAN[1] - self.Xm_SPAN[0]) + self.Xm_SPAN[0]
			elif individual[i] > self.Xm_SPAN[1]:
				individual[i] = self.Xm_SPAN[1]
				# individual[i] = random.random() * (self.Xm_SPAN[1] - self.Xm_SPAN[0]) + self.Xm_SPAN[0]
		# fixIndividual = copy.deepcopy(individual)
		# return fixIndividual

	def evolution(self):
		self.populationInit()
		for i in range(self.maxGeneration):
			print("第%d代："%(i+1))
			selectPopulation = self.selection()
			crossoverPopulation = self.crossover(selectPopulation)
			mutationPopulation  = self.mutation(crossoverPopulation)

			tempPopulation = self.population + mutationPopulation

			start = time.clock()
			self.selectNextPopulation(tempPopulation)
			end = time.clock()
			print("selectNextPopulation耗时：", end - start)
			print('\n================================================\n')

	def selectNextPopulation(self, population):
		funcValue, peretoSortPop = self.fastNonDominatedSort(population)
		self.pereto = np.array([funcValue[item] for item in peretoSortPop[0]])
		peretoSize = len(peretoSortPop)
		populationSize = self.populationSize
		newPopulation = []
		for i in range(peretoSize):
			peretoISize = len(peretoSortPop[i])
			if populationSize - peretoISize >= 0:
				populationSize -= peretoISize
				for item in peretoSortPop[i]:
					newPopulation.append(population[item])
			else:
				distanceSort, distance = self.crowdingDistance(peretoSortPop[i], funcValue)
				for item in distanceSort[peretoISize-1:peretoISize-populationSize-1:-1]:
					newPopulation.append(population[peretoSortPop[i][item]])
				populationSize = 0
			if populationSize == 0:
				break
		self.population = copy.deepcopy(newPopulation)
		return newPopulation

	def calculateFuncValue(self, population):
		size = len(population)
		dimension = self.dimension
		funcValue = np.zeros((size, dimension))
		for i in range(size):
			X = population[i]
			funcValue[i] = self.objFunc(X)
		return funcValue

	def isDominated(self, p, q):
		if (p<=q).all() and (p!=q).any():
			return True
		return False

	def fastNonDominatedSort(self, population):
		"""
		为了方便计算，支配排序各集合都储存对应的序号（下标）
		:param population:
		:return:
		"""
		funcValue = self.calculateFuncValue(population)
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
				if self.isDominated(p, q):
					spSet.append(j)
				elif self.isDominated(q, p):
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
		return funcValue, peretoSortPop

	def crowdingDistance(self, population, funcValue):
		"""
		该处population为个体序号
		:param population:
		:param funcValue:
		:return:
		"""
		dimension = funcValue.shape[1]
		size = len(population)
		funcV = np.zeros((size, dimension))
		distance = np.zeros(size)
		for i in range(size):
			funcV[i] = (funcValue[population[i]])
		for di in range(dimension):
			pop = np.argsort(funcV[:, di])
			for i in range(1, size-1):
				if i == 0 or i == size-1:
					distance[pop[i]] = np.Inf
				else:
					distance[pop[i]] += (funcV[pop[i+1]][di]-funcV[pop[i-1]][di])/(np.abs(funcV[pop[0]][di]-funcV[pop[size-1]][di])+1)
		distanceSort = np.argsort(distance)
		return distanceSort, distance

def draw(fv, pop):
	arr = np.array(pop)
	for i in range(len(arr)):
		funcV = []
		for item in arr[i]:
			funcV.append(fv[item])
		if i == 0:
			color = '#00FF00'
		elif i == 1:
			color = '#FF1111'
		elif i == 2:
			color = '#8888FF'
		elif i == 3:
			color = '#AFA0AF'
		elif i == 4:
			color = '#FF88FF'
		elif i == 5:
			color = '#11FF70'
		elif i == 6:
			color = '#88FF22'
		elif i == 7:
			color = '#F558FF'
		elif i == 8:
			color = '#02FAA0'
		elif i == 9:
			color = '#0AAF88'
		elif i == 10:
			color = '#0AFFA0'
		elif i == 11:
			color = '#AF33AF'
		elif i == 12:
			color = '#129900'
		else:
			color = '#FFFFFF'
		funcV = sorted(funcV, key=itemgetter(0))
		funcV = np.array(funcV)
		plt.scatter(funcV[:, 0], funcV[:, 1], color=color)
	plt.xlabel('f1')
	plt.ylabel('f2')
	# plt.show()
def drawPereto(fv, pop):
	arr = np.array(pop)
	funcV = []
	for item in arr:
		funcV.append(fv[item])
	color = '#00FF00'
	funcV = sorted(funcV, key=itemgetter(0))
	funcV = np.array(funcV)
	plt.scatter(funcV[:, 0], funcV[:, 1],marker='o', color=color, s=15)
	plt.xlabel('f1')
	plt.ylabel('f2')
	# plt.show()

if __name__ == '__main__':
	Fun = ZDT()
	# populationSize = 100, maxGeneration = 500
	# populationSize = 50, maxGeneration = 500
	NSGA = NSGAII(Fun, populationSize=100, maxGeneration=500)
	start = time.clock()
	NSGA.evolution()
	end = time.clock()
	print("求解耗时：", end - start)

	funcValue, peretoSortPop = NSGA.fastNonDominatedSort(NSGA.population)
	# draw(funcValue, peretoSortPop)

	data = pd.read_csv('./NSGA-II-2\PF\ZDT1.dat', header=None, sep='\s+')
	X, Y = data[0].values.astype(np.float), data[1].values.astype(np.float)
	plt.scatter(X, Y, color='#8888FF', marker='o', s=15)

	# drawPereto(funcValue, peretoSortPop[0])
	# drawPereto(funcValue, peretoSortPop[1])
	# drawPereto(funcValue, peretoSortPop[2])
	plt.scatter(NSGA.pereto[:, 0], NSGA.pereto[:, 1], color='#00FF00', marker='o', s=15)

	plt.show()