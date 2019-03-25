#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : MOEAD3.py
@Date  : 2019/03/03
@Desc  : 
"""

from pylab import *
import numpy as np
import pandas as pd
import math
import random
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MOEA.sample.LatinHyperCube import LatinHyperCube
from MOEA.weight.produceWeight import produceWeight
from MOEA.testFunction import DTLZ
from MOEA.testFunction import ZDT
from MOEA.mutation import realMPN as mut
from MOEA.crossover import realCSBX as cro
import MOEA.ndomin.NDSetSwk as NDSet

class MOEAD:
	def __init__(self, ObjFunc, stepLength=10, T=20, maxGeneration=20):
		"""
		初始化
		:param populationSize: 种群大小
		:param T: 最近的T个权重向量
		"""
		self.objFunc = ObjFunc.Func
		self.dimension = ObjFunc.dimension
		self.objFuncNum = ObjFunc.objFuncNum
		self.isMin = ObjFunc.isMin
		self.span = ObjFunc.span
		self.stepLength = stepLength
		self.T = T
		self.maxGeneration = maxGeneration
		self.Init()

	def Init(self):
		self.weights = produceWeight(self.objFuncNum, self.stepLength)
		self.populationSize = self.weights.shape[0]
		self.population = LatinHyperCube(self.populationSize, self.dimension, self.span)

	def isDominated(self, p, q):
		if (p <= q).all() and (p != q).any():
			return True
		return False

	def Tchebycheff(self, Y, Lambda, Z):
		"""
		切比雪夫公式（值都为向量）
		:param Y: 目标函数值
		:param Lambda: 权重
		:param Z: 最小值集合
		:return:
		"""
		Lambda = [0.01 if item==0 else item for item in Lambda]
		tchebycheff = np.max(np.abs(Y - Z) * Lambda)
		return tchebycheff

	def neighbor(self, Lambda):
		B = []
		Lambda_copy = copy.deepcopy(Lambda)
		for i in range(Lambda.shape[0]):
			distance = (np.sum((Lambda - Lambda_copy[i])**2, axis=1))
			indexSort = np.argsort(distance)
			B.append(list(indexSort[:self.T]))
		# 	print("neighbor:", distance)
		# print(B)
		# print(Lambda)
		return B

	def getZ(self):
		Z = self.getFuncValue(self.population[0])
		for i in range(1, self.populationSize):
			iValue = self.getFuncValue(self.population[i])
			for j in range(Z.shape[0]):
				if iValue[j] < Z[j]:
					Z[j] = iValue[j]
		return Z

	def updateZ(self, Z, individualY):
		for i in range(Z.shape[0]):
			if individualY[i] < Z[i]:
				Z[i] = individualY[i]
		return Z

	def geneOp(self, individualA, individualB):
		child1, child2 = cro.realCSBXIndividual(individualA, individualB, self.fixElement)
		if random.random() > 0.5:
			newChild = copy.deepcopy(child1)
		else:
			newChild = copy.deepcopy(child2)
		if random.random() > 0.5:
			return mut.realMPNIndividual(individualA, self.fixElement), newChild
		else:
			return mut.realMPNIndividual(individualB, self.fixElement), newChild

	def getFuncValue(self, X):
		return self.objFunc(X)

	def fixElement(self, individual):
		for i in range(self.dimension):
			if individual[i] < self.span[0][i]:
				individual[i] = self.span[0][i]
			elif individual[i] > self.span[1][i]:
				individual[i] = self.span[1][i]

	def run(self):
		# self.Init()
		B = self.neighbor(self.weights)
		Z = self.getZ()
		EP = []
		for geneNum in range(self.maxGeneration):
			for i in range(self.populationSize):
				k, l = random.sample(B[i], 2)
				childA, childB = self.population[k], self.population[l]
				childA, childB = self.geneOp(childA, childB)
				YA, YB = self.getFuncValue(childA), self.getFuncValue(childB)
				if self.isDominated(YA, YB):
					betterIndividual = childA
					betterY = YA
					Z = self.updateZ(Z, YA)
				else:
					betterIndividual = childB
					betterY = YB
					Z = self.updateZ(Z, YB)
				for j in range(self.T):
					individualJValue = self.getFuncValue(self.population[B[i][j]])
					TBJ = self.Tchebycheff(individualJValue, self.weights[B[i][j]], Z)
					TBB = self.Tchebycheff(betterY, self.weights[B[i][j]], Z)
					if TBB < TBJ:
						self.population[B[i][j]] = betterIndividual
				if geneNum == 0:
					EP.append([betterY])
				else:
					isDominated = True
					reList = []
					for j in range(len(EP[i])):
						if self.isDominated(EP[i][j], betterY) or (EP[i][j] == betterY).all():
							isDominated = False
							break
						elif not self.isDominated(betterY, EP[i][j]):
							reList.append(EP[i][j])
					if isDominated:
						EP[i] = copy.deepcopy(reList)
						EP[i].append(betterY)
			if (geneNum+1) % 10 == 0:
				print("迭代次数:", (geneNum+1))
		EP = np.array([np.array(v) for item in EP for v in item])
		EP = EP[NDSet.NDSetSwk(EP),]
		return EP

if __name__ == '__main__':
	DTLZ = DTLZ.DTLZ3()
	ZDT = ZDT.ZDT2()

	MOEA = MOEAD(ObjFunc=DTLZ, stepLength=23, T=20, maxGeneration=100)
	print(MOEA.populationSize)
	EP = MOEA.run()
	X, Y, Z = EP[:, 0], EP[:, 1], EP[:, 2]
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X, Y, Z, marker='x', color='blue')
	ax.set_xlabel('f1')
	ax.set_ylabel('f2')
	ax.set_zlabel('f3')

	data = pd.read_csv('.\PF\/' + DTLZ.__class__.__name__ + '.dat', header=None, sep='\s+').values.astype(np.float)
	X, Y, Z = data[:, 0], data[:, 1], data[:, 2]
	ax.scatter(X, Y, Z, marker='x', color='red', s=10)

	# MOEA = MOEAD(ObjFunc=ZDT, stepLength=99, T=20, maxGeneration=100)
	# print(MOEA.populationSize)
	# EP = MOEA.run()
	# X, Y, = EP[:, 0], EP[:, 1]
	# plt.scatter(X, Y, marker='x')

	# data = pd.read_csv('.\PF\/' + ZDT.__class__.__name__ + '.dat', header=None, sep='\s+').values.astype(np.float)
	# X, Y= data[:, 0], data[:, 1]
	# plt.scatter(X, Y, marker='x', color='red', s=10)

	plt.show()


