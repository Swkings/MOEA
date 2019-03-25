import numpy as np
import math
import copy
import random

s = [1, 2, 3, 4, 5, 6, 7, 0]
t = s[4:0:-1]
print(t)
print(s.pop())
print(s)
a = [0, 0, 0]
s[4:7] = a
print(4 in range(1, 4))
total = sum(s)
t = np.array(s)
avg = t.mean()
a = avg / (avg-t.min())
b = (-avg*min(t))/(avg-min(t))
print(t.min())
print(t+(math.fabs(avg))+math.fabs(min(t)))
# s.sort()
print(-t)
c = [2, 3]
g = []
# g = s.extend(c)
chromosome = list()
chromosome.extend([0])
chromosome.extend(random.sample(s, len(s)))
print(chromosome)



p1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
p2 = [4, 5, 2, 1, 8, 7, 6, 9, 3]
print(p1, p2)
p1[2], p1[3] = p1[3], p1[2]
print(p1, p2)
print(p1[3:7], p2[3:7])
print(p1, p2)


import matplotlib.pyplot as plt

x_values=[1,2,3,4,5]
y_values=[1,4,9,16,25]
# s为点的大小
plt.scatter(x_values,y_values,s=100)

# # 设置图表标题并给坐标轴加上标签
# plt.title("Scatter pic",fontsize=24)
# plt.xlabel("Value",fontsize=14)
# plt.ylabel("Scatter of Value",fontsize=14)
#
# # 设置刻度标记的大小
# plt.tick_params(axis='both',which='major',labelsize=14)

plt.show()
print(type(x_values))

p1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
p2 = [0, 4, 5, 2, 1, 8, 7, 6, 9, 3]
def OX(individual, crossoverPosStart, crossoverPosEnd):
	"""
	OX算法 顺序交叉算法
	:param individual:
	:param crossoverPosStart:
	:param crossoverPosEnd:
	:return:
	"""
	father = copy.deepcopy(individual)
	individual1Part = individual[0][crossoverPosStart:crossoverPosEnd + 1]
	individual2Part = individual[1][crossoverPosStart:crossoverPosEnd + 1]
	p1, p2 = copy.deepcopy(individual)
	p1 = p1[crossoverPosEnd + 1:] + p1[1:crossoverPosEnd + 1]
	p2 = p2[crossoverPosEnd + 1:] + p2[1:crossoverPosEnd + 1]
	for i in range(len(individual1Part)):
		p1.remove(individual2Part[i])
		p2.remove(individual1Part[i])
	child1 = [0] + p2[(9 - crossoverPosEnd):] + individual1Part + p2[0:(9 - crossoverPosEnd)]
	child2 = [0] + p1[(9 - crossoverPosEnd):] + individual2Part + p1[0:(9 - crossoverPosEnd)]
	return (child1, child2), father
(child1, child2), father=OX((p1,p2), 4, 7)
print(child1)
print(child2)