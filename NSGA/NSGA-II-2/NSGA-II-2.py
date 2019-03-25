#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : NSGA-II-2.py
@Date  : 2019/01/01
@Desc  : 
"""
from pylab import *
import numpy as np
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Individual import *
from ZDT import *
from DTLZ import *
from fastNonDominatedSort import *
from crowdingDistance import *
from LatinHypercube import *
import geatpy

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
		self.population = []
		self.pereto = []

	def populationInit(self):
		popX = LatinHypercube(self.populationSize, self.dimension, self.span)
		for X in popX:
			self.population.append(Individual(X=X, Func=self.objFunc))

	def selection(self, population):
		newPopulation = []
		peretoSort = fastNonDominatedSort(population)
		CD = crowdingDistance(population)
		for i in range(self.populationSize):
			rand1, rand2 = random.sample(range(self.populationSize), 2)
			try:
				if population[rand1] < population[rand2]:
					newPopulation.append(population[rand1].__deepcopy__())
				else:
					newPopulation.append(population[rand2].__deepcopy__())
			except IndexError:
				# print(rand1, rand2)
				pass
		return newPopulation

	def crossover(self, population):
		newPopulation = []
		cp = []
		count = 0
		mu = 1  # SBX方式的分布指数
		for i in range(int(len(population) / 2)):
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

				population[count].X = 0.5 * ((1 + alpha) * population[count].X + (1 - alpha) * population[count + 1].X)
				population[count].X = 0.5 * ((1 - alpha) * population[count].X + (1 + alpha) * population[count + 1].X)
				self.fixElement(population[count])
				self.fixElement(population[count + 1])
				newPopulation.append(population[count].__deepcopy__())
				newPopulation.append(population[count + 1].__deepcopy__())
			else:
				newPopulation.append(population[count].__deepcopy__())
				newPopulation.append(population[count + 1].__deepcopy__())
			count += 1
		return newPopulation

	def mutation(self, population):
		newPopulation = []
		mp = []
		m = 20
		for i in range(len(population)):
			mp.append(random.random())
		for i in range(len(population)):
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
						population[i].X[j] += (0.5 * L[j] * delta)
					else:
						population[i].X[j] -= (0.5 * L[j] * delta)
			self.fixElement(population[i])
			newPopulation.append(population[i].__deepcopy__())
		return newPopulation

	def mutation2(self, population):
		newPopulation = []
		mp = []
		yita = 10
		for i in range(len(population)):
			mp.append(random.random())
		for i in range(len(population)):
			if mp[i] < self.mp:
				delta = np.zeros(self.dimension)
				randAi = np.random.random(self.dimension)
				for k in range(self.dimension):
					if randAi[k] < 0.5:
						delta[k] = (2*randAi[k])**(1/(yita+1))-1
					else:
						delta[k] = (1-(2*(1-randAi[k])))**(1/(yita+1))
				population[i].X = population[i].X + delta
			self.fixElement(population[i])
			newPopulation.append(population[i].__deepcopy__())
		return newPopulation

	def fixElement(self, individual):
		for i in range(self.dimension):
			x = individual.X
			if x[i] < self.span[0][i]:
				x[i] = self.span[0][i]
			elif x[i] > self.span[1][i]:
				x[i] = self.span[1][i]

	def selectNextPopulation(self, population):
		allPopulation = self.population + population
		peretoSortPop = fastNonDominatedSort(allPopulation)
		self.pereto = np.array([item.Y for item in peretoSortPop[0]])
		# self.ind = np.array([item.X for item in peretoSortPop[1]])
		# CD = crowdingDistance(allPopulation)

		peretoSize = len(peretoSortPop)
		populationSize = self.populationSize
		newPopulation = []
		for i in range(peretoSize):
			peretoISize = len(peretoSortPop[i])
			if populationSize - peretoISize >= 0:
				populationSize -= peretoISize
				for item in peretoSortPop[i]:
					newPopulation.append(item.__deepcopy__())
			else:
				distancePop = crowdingDistance(peretoSortPop[i])
				distancePop.sort(key=lambda x:x.crowdingDistance, reverse=True)
				for item in distancePop:
					newPopulation.append(item.__deepcopy__())
					populationSize -= 1
					if populationSize == 0:
						break
			if populationSize == 0:
				break
		self.population = newPopulation

	def evolution(self):
		self.populationInit()
		for i in range(self.maxGeneration):
			print("第{0}代".format(i+1))
			newPopulation = self.selection(self.population)
			crossoverPop = self.crossover(newPopulation)
			mutationPop = self.mutation2(crossoverPop)
			start = time.clock()
			self.selectNextPopulation(mutationPop)
			end = time.clock()
			print("selectNextPopulation耗时：", end - start)
			print('\n================================================\n')
			for item in self.population:
				item.reset()
		peretoSortPop = fastNonDominatedSort(self.population)

def getValue(population):
	funcValue = [ind.Y for ind in population]
	return np.array(funcValue)

if __name__ == '__main__':
	# ZDT = ZDT1()
	DTLZ = DTLZ1()
	# populationSize = 50, maxGeneration = 500
	# populationSize = 50, maxGeneration = 200(效果一般)
	# populationSize = 100, maxGeneration = 200
	nsga = NSGAII(FuncObj=DTLZ, populationSize=50, maxGeneration=500)
	start = time.clock()
	nsga.evolution()
	end = time.clock()
	print('计算耗时：', end - start)
	pereto = nsga.pereto

	# data = pd.read_csv('.\PF\/'+ZDT.__class__.__name__+'.dat', header=None, sep='\s+')
	# X, Y= data[0].values.astype(np.float), data[1].values.astype(np.float)
	# plt.scatter(X, Y, color='#8888FF', marker='o', s=15)
	#
	# plt.scatter(pereto[:, 0], pereto[:, 1], color='#00FF00', marker='o', s=15)
	# plt.show()

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


