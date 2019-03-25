#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : NSGA-II-3.py
@Date  : 2019/01/09
@Desc  : 
"""
from pylab import *
import numpy as np
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ZDT import *
from DTLZ import *
from crowdingDistance import *
from LatinHyperCube import *
import geatpy
from MOEA.nondominated import HNDSetRank as fast

class NSGAII:
	def __init__(self, FuncObj, populationSize=10, maxGeneration=5, cp=0.9, mp=0.1):
		self.objFunc = FuncObj.Func
		self.dimension = FuncObj.dimension
		self.objFuncNum = FuncObj.objFuncNum
		self.isMin = FuncObj.isMin
		self.span = FuncObj.span
		self.populationSize = populationSize
		self.maxGeneration = maxGeneration
		self.cp = cp
		self.mp = mp
		self.pereto = []

	def populationInit(self):
		self.populationX = LatinHyperCube(self.populationSize, self.dimension, self.span)
		self.populationY = self.funcValue(self.populationX)

	def selection(self):
		newPopulationX = []
		peretoSort = fast.HNDSetRank(self.populationY)
		rank = np.empty(self.populationSize)
		for iRank in range(len(peretoSort)):
			for item in peretoSort[iRank]:
				rank[item] = iRank
		distance = crowdingDistance(self.populationY)
		for i in range(self.populationSize):
			rand1, rand2 = random.sample(range(self.populationSize), 2)
			try:
				if rank[rand1] == rank[rand2]:
					if distance[rand1] > distance[rand2]:
						newPopulationX.append(self.populationX[rand1])
					else:
						newPopulationX.append(self.populationX[rand2])
				else:
					if rank[rand1] < rank[rand2]:
						newPopulationX.append(self.populationX[rand1])
					else:
						newPopulationX.append(self.populationX[rand2])
			except IndexError:
				# print(rand1, rand2)
				pass
		return newPopulationX

	def crossover(self, populationX):
		newPopulationX = []
		cp = []
		count = 0
		mu = 1  # SBX方式的分布指数
		for i in range(int(self.populationSize / 2)):
			cp.append(random.random())
		for cpi in cp:
			if cpi < self.cp:
				randList = np.random.random(self.dimension)
				alpha = np.zeros(self.dimension)
				for i in range(len(randList)):
					if randList[i] <= 0.5:
						alpha[i] = (2.0 * randList[i]) ** (1.0 / (mu + 1))
					else:
						alpha[i] = (1 / (2 * (1 - randList[i]))) ** (1.0 / (mu + 1))

				child1 = 0.5 * ((1 + alpha) * populationX[2 * count] + (1 - alpha) * populationX[2 * count + 1])
				child2 = 0.5 * ((1 - alpha) * populationX[2 * count] + (1 + alpha) * populationX[2 * count + 1])
				self.fixElement(child1)
				self.fixElement(child2)
				newPopulationX.append(child1)
				newPopulationX.append(child2)
			else:
				newPopulationX.append(populationX[2 * count])
				newPopulationX.append(populationX[2 * count + 1])
			count += 1
		return newPopulationX

	def mutation(self, populationX):
		newPopulationX = []
		mp = []
		yita = 10
		for i in range(self.populationSize):
			mp.append(random.random())
		for i in range(self.populationSize):
			if mp[i] < self.mp:
				delta = np.zeros(self.dimension)
				randAi = np.random.random(self.dimension)
				for k in range(self.dimension):
					if randAi[k] < 0.5:
						delta[k] = (2*randAi[k])**(1/(yita+1))-1
					else:
						delta[k] = 1-((2*(1-randAi[k]))**(1/(yita+1)))
				child = populationX[i] + delta
				self.fixElement(child)
				newPopulationX.append(child)
			else:
				newPopulationX.append(populationX[i])
		return newPopulationX

	def mutation1(self, populationX):
		newPopulationX = []
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
				randList = np.random.random(self.dimension)
				L = np.abs(self.span[1] - self.span[0])
				for j in range(self.dimension):
					if randList[j] <= 0.5:
						populationX[i][j] += (0.5 * L[j] * delta)
					else:
						populationX[i][j] -= (0.5 * L[j] * delta)
				self.fixElement(populationX[i])
				newPopulationX.append(populationX[i])
			else:
				newPopulationX.append(populationX[i])
		return newPopulationX

	def selectNextPopulation(self, populationX):
		newPopulationY = self.funcValue(np.array(populationX))
		allPopulationX = np.concatenate((self.populationX, np.array(populationX)))
		allPopulationY = np.concatenate((self.populationY, newPopulationY))
		peretoSortPop = fast.HNDSetRank(allPopulationY)
		self.pereto = np.array([allPopulationY[item] for item in peretoSortPop[0]])
		# self.ind = np.array([item.X for item in peretoSortPop[1]])
		# CD = crowdingDistance(allPopulation)

		peretoSize = len(peretoSortPop)
		populationSize = self.populationSize
		newPopulationX = []
		for i in range(peretoSize):
			peretoISize = len(peretoSortPop[i])
			if populationSize - peretoISize >= 0:
				populationSize -= peretoISize
				for item in peretoSortPop[i]:
					newPopulationX.append(allPopulationX[item])
			else:
				PopulationY = np.array([allPopulationY[item] for item in peretoSortPop[i]])
				distance = crowdingDistance(PopulationY)
				index = np.argsort(-distance)
				for item in index:
					newPopulationX.append(allPopulationX[peretoSortPop[i][item]])
					populationSize -= 1
					if populationSize == 0:
						break
			if populationSize == 0:
				break
		self.populationX = np.array(newPopulationX)
		self.populationY = self.funcValue(self.populationX)

	def evolution(self):
		self.populationInit()
		for i in range(self.maxGeneration):
			if (i+1) % 10 == 0 or i == self.maxGeneration-1:
				print("第{0}代".format(i+1))
			newPopulationX = self.selection()
			crossoverPop = self.crossover(newPopulationX)
			mutationPop = self.mutation(crossoverPop)
			start = time.clock()
			self.selectNextPopulation(mutationPop)
			end = time.clock()
			# print("selectNextPopulation耗时：", end - start)
			# print('\n================================================\n')
		# peretoSortPop = fast.HNDSetRank(self.populationY)
		# self.pereto = np.array([self.populationY[item] for item in peretoSortPop[0]])

	def fixElement(self, individual):
		for i in range(self.dimension):
			if individual[i] < self.span[0][i]:
				individual[i] = self.span[0][i]
			elif individual[i] > self.span[1][i]:
				individual[i] = self.span[1][i]

	def funcValue(self, populationX):
		size = populationX.shape[0]
		populationY = np.empty((size, self.objFuncNum))
		for i in range(size):
			populationY[i] = self.objFunc(populationX[i])
		return populationY

if __name__ == '__main__':
	DTLZ = DTLZ1()
	NSGA = NSGAII(DTLZ, populationSize=250, maxGeneration=300)
	start = time.clock()
	NSGA.evolution()
	end = time.clock()
	print('计算耗时：', end - start)
	pereto = NSGA.pereto
	data = pd.read_csv('.\PF\/' + DTLZ.__class__.__name__ + '.dat', header=None, sep='\s+').values.astype(np.float)
	X, Y, Z = data[:, 0], data[:, 1], data[:, 2]
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X, Y, Z, marker='x', color='red')
	ax.scatter(pereto[:, 0], pereto[:, 1], pereto[:, 2], marker='x', color='blue')
	ax.set_xlabel('f1')
	ax.set_ylabel('f2')
	ax.set_zlabel('f3')
	plt.show()
