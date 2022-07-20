"""
** Step: **
- Read file -> Task (graph)
- Generate trees from graph
- Encode tree
- Run genetic algorithm
- Return last generation
"""

from datetime import datetime
from random import randrange
import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
N = 200
Tx = 2
Rx = 1
q = 4
NE = 25
deltaT = 10000
INF = 9999999999

random.seed(100)
"""
Decode individual from genotype to pheotype for each task
Param: individual (2darray)
Output: adjacent List (dictionary of (int, list))
"""
def decode(individual):
    cnt = 0
    individual = individual.tolist()
    tree = {}
    for i in range(len(individual[0])-1,0,-1):
        deepth = individual[1][i]
        idxParentNode = i-1
        while True:
            if(individual[1][idxParentNode] == deepth - 1):
                node = individual[0][i]
                parentNode = individual[0][idxParentNode]
                if(node not in tree):
                    tree[node] = list([parentNode])
                else:
                    tree[node].append(parentNode)
                if(parentNode not in tree):
                    tree[parentNode] = list([node])
                else:
                    tree[parentNode].append(node)
                break
            idxParentNode -= 1
            # except:
            #     exit(1)
            #     return {83: [0], 0: [83, 18], 87: [78], 78: [87, 51, 31, 18], 62: [88], 88: [62, 51], 
            #     51: [88, 55, 78], 55: [51], 64: [76], 76: [64, 26, 71], 15: [26], 26: [15, 76], 
            #     71: [76, 31], 31: [71, 1, 78], 11: [59], 59: [11, 9], 9: [59, 29], 29: [9, 12], 12: [29, 1], 
            #     1: [12, 31], 18: [78, 72, 57, 0], 92: [65], 65: [92, 49, 38], 49: [65], 38: [65, 58, 30, 45], 
            #     91: [58], 58: [91, 75, 32, 8, 38], 90: [68], 68: [90, 24, 52], 10: [24], 24: [10, 68], 
            #     52: [68, 96], 96: [52, 48, 67], 42: [47], 47: [42, 27], 27: [47, 48], 48: [27, 96], 
            #     67: [96, 93, 75], 28: [93], 93: [28, 67], 75: [67, 43, 23, 58], 43: [75], 69: [23], 
            #     23: [69, 75], 32: [58], 8: [58], 36: [33], 33: [36, 98], 98: [33, 99], 99: [98, 53, 30], 
            #     53: [99], 30: [99, 38], 45: [38, 72], 72: [45, 7, 18], 73: [16], 16: [73, 3, 79], 82: [50], 
            #     50: [82, 2, 61], 66: [2], 2: [66, 5, 50], 74: [5], 5: [74, 46, 2], 84: [85], 85: [84, 94], 
            #     94: [85, 60, 6, 77], 95: [60], 60: [95, 94], 89: [6], 6: [89, 94], 77: [94, 46], 46: [77, 5], 
            #     61: [50, 97], 97: [61, 63], 63: [97, 39, 25, 3], 39: [63], 80: [25], 25: [80, 35, 63], 35: [25], 
            #     3: [63, 14, 16], 86: [14], 14: [86, 81, 3], 70: [81], 81: [70, 34, 4, 14], 19: [22], 22: [19, 34], 
            #     34: [22, 20, 17, 81], 54: [20], 20: [54, 34], 17: [34], 37: [40], 40: [37, 4], 4: [40, 13, 81], 
            #     41: [13], 13: [41, 4], 79: [16, 7], 7: [79, 44, 72], 44: [7], 56: [21], 21: [56, 57], 57: [21, 18]}
    return tree

"""
Define task for an input/data. Class Task will have some fields: 
n (number of sensor), 
m (number of relay node), 
numberOfEdges, 
s (list of report size),
listOfRelayNode (numpy array of relay node),
adjList (adjacent list of graph) (dictionary of (int, list))
"""
class Task:
    def __init__(self, n, m, numberOfEdges, s, listOfRelayNode, adjList):
        self.n = n
        self.m = m
        self.numberOfEdges = numberOfEdges
        self.s = s
        self.listOfRelayNode = listOfRelayNode
        self.adjList = adjList
        self.huyanh = s.copy()

    """
    Evaluate factorial cost of each individual
    Param: task (self), individual (in genotype)
    Output: a number (used energy)
    """
    # def evaluateIndividualFactorialCost(self, individual):
    #     result = 0
    #     pheotype = decode(individual) # adjacent list
    #                                         # (dictionary of (int, list))
    #     z = np.array([0]*(self.n+self.m+1))
    #     while(len(pheotype) != 1):
    #         for key in list(pheotype):
    #             if(key != 0 and len(pheotype[key]) == 1):
    #                 z[key] += self.s[key]
    #                 adjVertice = pheotype[key][0]
    #                 z[adjVertice] += z[key]
    #                 pheotype.pop(key, None)
    #                 pheotype[adjVertice].remove(key)

    #     for i in range(self.n + self.m + 1):
    #         if(i == 0):
    #             continue
    #         result += math.ceil(z[i]/q)
    #     return (Tx + Rx)*result

    def evaluateIndividualFactorialCost(self, individual):
        individual = decode(individual)
        return self.energy_cost(self.n+self.m+1, self.dic_to_list(individual),Tx, Rx,q, self.huyanh.copy(), self.level(self.dic_to_list(individual)))
        # return self.energy_cost(self.n+self.m+1, individual.tolist(),Tx, Rx,q, self.s, self.level(individual.tolist()))

    def level(self, tree):
        level = {}
        for i in range(self.n+self.m+1):
            level[i] = len(tree[i])
        # level = sorted(level.items(),key = lambda x:x[1])
        return level

    def dic_to_list(self, dic):
        res = [[] for i in range(len(dic))]
        # res = [[]]
        for key, value in dic.items():
            res[key] = value
        return res

    def energy_cost(self, num_node,tree,h,r,q,report_size,level):
        z = report_size.copy()
    
        res = 0
        checked = [False for i in range(num_node)]
        
        level.pop(0)

        while len(level.keys()) != 0:
            key = min(level, key=level.get)
            level.pop(key)
            checked[key] = True
            for neighbor in tree[key]:
                if neighbor in level:
                    level[neighbor] -= 1
                if checked[neighbor] == True:
                    z[key] +=z[neighbor]
            res += math.ceil(z[key]/q)
        return res*(h+r)


"""
Get input/data for each task from file.
Param: path of file
Output: class Task, this Task will have some fields: 
n (number of sensor), 
m (number of relay node),
numberOfEdges, 
s (list of report size),
listOfRelayNode (numpy array of relay node),
adjList (adjacent list of graph) (dictionary of (int, list))
"""
def getInputFromFile(filePath):
    # We don't need q, Tx, Rx, coordinate, communication radius, initial energy
    file = open(filePath,"r")
    file.readline()
    file.readline()
    numberOfAllNode = int(file.readline())

    m = 0
    s = list()
    listOfRelayNode = list()
    for i in range(numberOfAllNode):
        reportSize = int(file.readline().split()[-1])
        s.append(reportSize)
        if(reportSize == 0 and i!=0):
            m+=1
            listOfRelayNode.append(i)
    
    n = numberOfAllNode - m - 1

    numberOfEdges = int(file.readline())
    tmpNumberOfEdges = 0
    adjList = dict()
    for i in range(numberOfAllNode):
        adjList[i] = list([])
    for i in range(numberOfEdges):
        edge = file.readline().split()
        edge = [int(vertice) for vertice in edge]
        if(edge[0] not in adjList[edge[1]]):
            adjList[edge[1]].append(edge[0])
            adjList[edge[0]].append(edge[1])
            tmpNumberOfEdges += 1
    numberOfEdges = tmpNumberOfEdges
    
    file.close()
    return Task(n,m,numberOfEdges,s,listOfRelayNode,adjList)

"""
Evaluate factorial cost of population for all of tasks
Param: population, tasks
Output: 2-D array, arr[i][j] is factorial cost of individual pi on task Tj
"""
def evaluatePopulationFactorialCost(population, tasks):
    res = np.array([0]*len(tasks))
    for individual in population:
        costOfTask = np.array([])
        for task in tasks:
            cost = task.evaluateIndividualFactorialCost(individual)
            costOfTask = np.append(costOfTask, cost)
        res = np.vstack([res, costOfTask])
    res = np.delete(res, 0, axis=0)
    return res

"""
Evaluate factorial rank
Param: factorial cost of population (2-D array)
Output: 1-D array, arr[i][j] is the rank(index) of pi on task Tj in the list of
population members sorted in ascending order
"""
def evaluateFactorialRank(populationFactorialCost):
    return np.argsort(np.argsort(populationFactorialCost, axis=0), axis=0)

"""
Evaluate scalar fitness of individual is its best rank over all tasks
= 1/min{r[i][j], j=1...k}
Param: factorial rank of population
Output: 1-D array
"""
def evaluateScalarFitness(factorialRank):
    return 1/(np.min(factorialRank, axis=1)+1)

"""
Evaluate skill factor of individual is argmin{r[i][j], j=1...k}
Param: factorial rank of population
Output: 1-D array
"""
def evaluateSkillFactor(factorialRank):
    # print(len(factorialRank))
    res = [i % 2 for i in range(N*2)]
    return np.array(res)
    # exit(1)
    # return np.argmin(factorialRank, axis=1)

"""
Tournament selection to select the fittest individuals
Param: size of population, k (number of selection in population), scalar fitness
Output: index of fittest individual
"""
def tournamentSelectionIndividual(sizeOfPopulation, k, scalarFitness):
    selected = np.array(random.sample(range(sizeOfPopulation),k))
    idxOfResult = np.argmax(scalarFitness[selected])  
    return int(selected[idxOfResult])

"""
Mutation for individual
Param: individual (2darray), list
Output: individual
"""
def EPO_withPhenotype(individual, edge):
    tree = decode(individual)
    if(edge[1] in tree[edge[0]]):
        return individual
    individual = individual.tolist()
    idxRoot = individual[1].index(0)
    idxNode = list()
    idxNode.append(individual[0].index(edge[0])) 
    idxNode.append(individual[0].index(edge[1]))
    isReversed = False
    # edge[1] will be ancestor of edge[0] in tree
    if(individual[1][idxNode[0]] < individual[1][idxNode[1]]):
        edge.reverse()
        idxNode.reverse()
        isReversed = True
    # check ancestor of edge[0]
    cloneIdxNode0 = idxNode[0]
    for _ in range(individual[1][idxNode[0]] - individual[1][idxNode[1]]):
        deepth = individual[1][cloneIdxNode0]
        idxParentNode = cloneIdxNode0 - 1
        while True:
            if(individual[1][idxParentNode] == deepth -1):
                cloneIdxNode0 = idxParentNode
                break
            idxParentNode -= 1
    if (cloneIdxNode0 != idxNode[1]):
        if(isReversed == True):
            edge.reverse()
            idxNode.reverse()
    # find parent of edge[0]
    deepth = individual[1][idxNode[0]]
    idxParentNode = idxNode[0]-1
    while True:
        if(individual[1][idxParentNode] == deepth -1):
            node = edge[0]
            parentNode = individual[0][idxParentNode]
            tree[node].remove(parentNode)
            tree[parentNode].remove(node)
            break
        idxParentNode -= 1
    
    # add new edge to tree
    tree[edge[0]].append(edge[1])
    tree[edge[1]].append(edge[0])
    newIndividual = encode(tree, individual[0][idxRoot])
    return newIndividual

"""
Crossover 2 parents individuals
Param: 2 individuals
Output: child individual
"""
def ECO_withPhenotype(individuals):
    A = individuals[0]
    A = A.tolist()
    B = individuals[1]
    B = B.tolist()
    Fab = A
    n = len(A[0])
    i = random.randint(n//4, 3*n//4)
    vr = np.array(random.sample(range(n),i))
    for node in vr:
        idxNodeInB = B[0].index(node)
        if(B[1][idxNodeInB] != 0):
            # find parent of node in B
            deepth = B[1][idxNodeInB]
            idxParentNode = idxNodeInB - 1
            while True:
                if(B[1][idxParentNode] == deepth -1):
                    parentNode = B[0][idxParentNode]
                    Fab = EPO_withPhenotype(np.array(Fab),list([node, parentNode]))
                    break
                idxParentNode -=1
        
    return np.array(Fab)

"""
Evaluate factorial cost for offspring population
Param: offspring population, offspring skill-factor, tasks
Output: 1-D array
"""
def evaluateOffspringCost(offspringPopulation, offspringSkillFactor, tasks):
    return np.array([tasks[int(idxOfTask)].evaluateIndividualFactorialCost(offspringPopulation[int(idxOfIndividual)]) for idxOfIndividual, idxOfTask in enumerate(offspringSkillFactor)])

"""
Evaluate rank base on skill-factor 
Param: individual best factorial cost, skill-factor, number of tasks
Output: 1-D array
"""
def evaluateRankBaseOnSkillFactor(individualBestCost, skillFactor, numOfTasks):
    result = np.zeros_like(skillFactor)
    for i in range(numOfTasks):
        idxs = np.where(skillFactor == i)
        result[idxs] = np.argsort(np.argsort(individualBestCost[idxs]))
    return result

"""
Encode from tree to genotype
Param: tree (adj list (int, list)), root of tree
Output: individual (2darray)
"""
def encode(tree, root):
    tmpTree = {k:v for k,v in sorted(list(tree.items()))}
    tree = tmpTree
    for _,v in tree.items():
        v = v.sort()
    individual = [[],[]]
    numberOfEdges = len(tree)
    visited = [False for i in range(numberOfEdges)]
    deepth = 0
    def dfs(tree, root,deepth):
        if visited[root] == False:
            visited[root] = True
            individual[0].append(root)
            individual[1].append(deepth)
            for value in tree[root]:
                dfs(tree, value, deepth+1)
    dfs(tree, root, deepth)
    return np.array(individual)

# def eco2(gen1, gen2):
#     num_node = len(gen1[0])
#     i = random.randint(num_node//4, 3*num_node//4)
#     vr = random.sample(range(num_node), i)
#     # vr = [5, 6, 8, 3]
#     childGen = (gen2[0].copy(), gen2[1].copy())
#     for k in vr:
#         idxK = gen1[0].index(k)
#         if gen1[1][idxK] > 0:
#             for index in range(idxK-1, -1, -1):
#                 if gen1[1][index] < gen1[1][idxK]:
#                     childGenResult = np.copy(epo(childGen, gen1[0][idxK], gen1[0][index]))
#                     childGen = (childGenResult[0].tolist().copy(), childGenResult[1].tolist().copy())
#                     break
#     return np.array(childGen)

def eco3(individual1, individual2):
    A = individual1
    A = A.tolist()
    B = individual2
    B = B.tolist()
    Fab = A
    n = len(A[0])
    i = random.randint(n//4, 3*n//4)
    vr = np.array(random.sample(range(n),i))
    for node in vr:
        idxNodeInB = B[0].index(node)
        if(B[1][idxNodeInB] != 0):
            # find parent of node in B
            deepth = B[1][idxNodeInB]
            idxParentNode = idxNodeInB - 1
            while True:
                if(B[1][idxParentNode] == deepth -1):
                    parentNode = B[0][idxParentNode]
                    Fab = epo(np.array(Fab), node, parentNode)  
                    break
                idxParentNode -=1
        
    return np.array(Fab)

import copy

def epo(gen, edge1, edge2):
    # gen = [gen[0], gen[1]]
    gen = gen.tolist()
    num_node = len(gen[0])
    childGen = gen
    # childGen = (gen[0].copy(), gen[1].copy())

    u = gen[0].index(edge1)
    v = gen[0].index(edge2)
    gg = [u,v,edge1,edge2]
    # check if edge(edge1-edge2) is existed in gen
    # check if edge1 is parent of edge2
    if(u>v):
        depth = childGen[1][u]
        idxParentNode = u-1
        while (idxParentNode>-1):
            if(childGen[1][idxParentNode] == depth - 1):
                if(idxParentNode == v):
                    return np.array(childGen)
            idxParentNode -= 1
    elif(u<v):
        depth = childGen[1][v]
        idxParentNode = v-1
        while (idxParentNode>-1):
            if(childGen[1][idxParentNode] == depth - 1):
                if(idxParentNode == u):
                    return np.array(childGen)
            idxParentNode -= 1
    tree = decode(np.array(childGen))
    if(edge1 in tree[edge2]):
        return np.array(childGen)

    idxNode = list()
    idxNode.append(u) 
    idxNode.append(v)
    isReversed = False
    if(childGen[1][idxNode[0]] < childGen[1][idxNode[1]]):
        # edge.reverse()
        idxNode.reverse()
        isReversed = True
    # check ancestor of edge[0]
    cloneIdxNode0 = idxNode[0]
    for cnt in range(childGen[1][idxNode[0]] - childGen[1][idxNode[1]]):
        deepth = childGen[1][cloneIdxNode0]
        idxParentNode = cloneIdxNode0 - 1
        while True:
            if(childGen[1][idxParentNode] == deepth -1):
                cloneIdxNode0 = idxParentNode
                break
            idxParentNode -= 1
            
    if (cloneIdxNode0 != idxNode[1]):
        if(isReversed == True):
            # edge.reverse()
            idxNode.reverse()
    # if childGen[1][u]>childGen[1][v]:
    #     tmp = u
    #     u = v
    #     v = tmp
    v = idxNode[0]
    u = idxNode[1]

    # if childGen[1][u] > childGen[1][v]:
    #     u,v = v,u
    for k in range(v+1, num_node+1):
        if (k == num_node) or (childGen[1][k] <= childGen[1][v]):
            tmplist0 = childGen[0][v:k].copy()
            tmplist1 = childGen[1][v:k].copy()
            depthChange = tmplist1[0] - childGen[1][u] - 1
            del childGen[0][v:k]
            del childGen[1][v:k]
            for i in range(len(tmplist1)):
                tmplist1[i] -= depthChange
            if v>u:
                childGen[0][u+1:u+1] = tmplist0.copy()
                childGen[1][u+1:u+1] = tmplist1.copy()
            elif v<u:
                childGen[0][u+1-k+v:u+1-k+v] = tmplist0.copy()
                childGen[1][u+1-k+v:u+1-k+v] = tmplist1.copy()             
            break
    # try:
    #     abc = decode(childGen)
    # except:
    #     print(u,v)
    #     print(edge1, edge2)
    #     print(gg)
    #     print(gen)
    #     print(childGen)
    #     print(gen == childGen)
    #     exit(1)
    return np.array(childGen)

def find(u, parent):
    if parent[u] == u:
        return u
    return find(parent[u], parent)


def union(x, y, parent, rank):
    rootX = find(x, parent)
    rootY = find(y, parent)
    if rank[rootX] > rank[rootY]:
        parent[rootX] = parent[rootY]

    elif rank[rootY] > rank[rootX]:
        parent[rootY] = parent[rootX]

    else:
        parent[rootX] = parent[rootY]
        rank[rootX] += 1

    return parent, rank
"""
Generate a random tree from graph
Param: Graph
Output: Tree, root of tree
"""
def genTree(graph):
    edgeList = []
    res = {}
    parent = [i for i in range(len(graph))]
    rank = [0 for i in range(len(graph))]

    for i in range(len(graph)):
        res[i] = []

    for node, adj_list in graph.items():
        for v in adj_list:
            if(node < v):
                edgeList.append((node, v))

    random.shuffle(edgeList)
    cnt = 0
    e = 0
    while cnt < len(graph) - 1:
        x, y = edgeList[e]
        e += 1
        if find(x, parent) != find(y, parent):
            res[x].append(y)
            res[y].append(x)
            parent, rank = union(x, y, parent, rank)
            cnt += 1

    for i in range(len(graph)):
        res[i] = sorted(res[i])

    root = randrange(len(graph))

    return root, res

"""
Create potential population
Param: tree, adjList, numberOfEdges, maximumNumberOfEdges, numberOfTasks
Output: K*NE potential individuals
"""
def createPotentialPopulation(tree, adjList, numberOfEdges, lengthOfGen, numberOfTasks):
    # population = np.empty((0, maximumNumberOfEdges), float)
    population = np.array([[[0]*lengthOfGen,[0]*lengthOfGen]])
    numberOfEdgesOfTree = 0
    for key in tree:
        tree[key] = sorted(tree[key])
        numberOfEdgesOfTree += len(tree[key])
    numberOfEdgesOfTree //= 2
    for key in adjList:
        adjList[key] = sorted(adjList[key])
    for _ in range(numberOfTasks*NE):
        # individual = encode(tree, adjList, numberOfEdges, numberOfEdgesOfTree)
        population = np.vstack([population, [encode(tree,0)]])
        # population = np.vstack([population, representInCommonSpace(individual, maximumNumberOfEdges)])
    population = np.delete(population, 0, axis=0)
    return population

def updateRmp(rmp, parentPopulationFactorialCost, offspringPopulationFactorialCost):
    random.seed(9)
    parent = parentPopulationFactorialCost
    offspring = offspringPopulationFactorialCost

    parentSum = 0
    offspringSum = 0
    # res = random.random()
    for i in range(len(parent)):
        for j in range(len(parent[0])):
            parentSum += parent[i][j]
            offspringSum += offspring[i][j]
    
    if(offspringSum <= parentSum):
        res = rmp - 0.01
        if res > 0:
            return res
        else:
            return rmp
    else:
        res = rmp + 0.01
        if res < 1:
            return res
        else:
            return rmp

def SPT(numberOfNodes, graph):
    tmpGraph = {}
    for key in graph:
        vertices = graph[key]
        tmpGraph[key] = sorted(vertices)
    isVisited = [False for i in range(numberOfNodes)]
    queue = []
    queue.append(0)
    isVisited[0] = True
    count = 1
    tree = [[]for i in range(numberOfNodes)]
    while(len(queue) != 0):
        h = queue[0]
        queue.pop(0)
        for i in tmpGraph[h]:
            if isVisited[i] == False:
                isVisited[i] = True
                queue.append(i)
                count += 1
                tree[h].append(i)
                tree[i].append(h)
                if(count == numberOfNodes):
                    result = {}
                    for i in range(len(tree)):
                        result[i] = tree[i]
                    return result
    result = {}
    for i in range(len(tree)):
        result[i] = tree[i]
    return result

"""
7 approximation
"""
class graph_hat:
    def __init__(self,graph,report_size,num_node):
        self.graph = graph
        self.report_size = report_size 
        self.num_node = num_node
        self.all_pair_shortest_path = {}
        self.graph_hat = [[] for i in range(self.num_node)]

    def BFS(self,source_node):
        queue = []
        check = [False for i in range(self.num_node)]
        check[source_node] = True
        queue.append((source_node,1))
        count = 1
        while (len(queue)!= 0):
            current_node, height = queue[0]
            queue.pop(0)
            for v in self.graph[current_node]:
                if(check[v] == False):
                    check[v] = True
                    if (source_node,v) not in self.all_pair_shortest_path:
                        self.all_pair_shortest_path[(source_node,v)] = height
                        self.all_pair_shortest_path[(v,source_node)] = height
                    queue.append((v,height+1))

                    count+=1
                    if(count == self.num_node):
                        return 
        # return all_pair_shortest_path 

    def run(self):
        for source_node in range(self.num_node):
            self.BFS(source_node)

        key_list = list(self.all_pair_shortest_path.keys())

        for u,v in key_list:
            if(self.report_size[u] !=0 and self.report_size[v]!=0) or (u ==0 and self.report_size[v]!=0) or (v == 0 and self.report_size[u]!=0):
                self.graph_hat[u].append(v)
            else:
                del self.all_pair_shortest_path[(u,v)]  
        return self.graph_hat,self.all_pair_shortest_path

class LAST:
    def __init__(self,graph_hat,edge,report_size,num_relay,num_node):
        self.graph_hat = graph_hat
        self.edge = edge 
        self.report_size = report_size
        self.num_relay = num_relay
        self.num_node = num_node
        self.Tm = [[] for i in range(self.num_node)]
        self.Ts = {}
        self.parent_Ts = [-1 for i in range(self.num_node)]
        self.dist_Ts = [False for i in range(self.num_node)]
        self.tree = [[] for i in range(self.num_node)]
    def find(self,parent,i):
        if parent[i] == i:
            return i
        return self.find(parent,parent[i])

    def union(self,parent ,rank,x,y):
        xroot = self.find(parent,x)
        yroot = self.find(parent,y)
            
        if rank[xroot]< rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot]>rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] +=1

    def kruskal(self):
        res = {}
        e = 0
        parent = [i for i in range(self.num_node)]
        rank = [0 for i in range(self.num_node)]
        edge = dict(sorted(self.edge.items(),key = lambda item : item[1]))
        number_of_edge = self.num_node * 2 - 2
        for node,weight in edge.items():
            x = self.find(parent,node[0])
            y = self.find(parent,node[1])
            if x!=y:
                e+=2
                res[(node[0],node[1])] = weight
                res[(node[1],node[0])] = weight
                self.union(parent,rank,x,y)
                if e == number_of_edge:
                    for u,v in res.keys():
                        self.Tm[u].append(v)

                    for i in self.Tm:
                        i.sort()
                    return 

        for u,v in res.keys():
            self.Tm[u].append(v)

        for i in self.Tm:
            i.sort()
        return                  

    def spt_with_weight(self,source_node):
        dist = [INF for i in range(self.num_node)]
        dist[source_node] = 0
        check = [False for i in range(self.num_node)]
        self.parent_Ts[source_node] = -2
        
        cnt = 0
        while cnt < self.num_node - self.num_relay:
            index = dist.index(min(dist))
            self.dist_Ts[index] = min(dist)
            check[index] = True
            cnt+=1

            for item in self.graph_hat[index]:
                if(check[item] == False):
                    if(dist[item]>dist[index] + self.edge[(index,item)]):
                        dist[item] = dist[index] + self.edge[(index,item)]
                        self.parent_Ts[item] = index
            dist[index] = INF

            for i in range(1,self.num_node):
                if self.report_size[i] != 0:
                    self.Ts[(i,self.parent_Ts[i])] = self.edge[(i,self.parent_Ts[i])]                                   
                    self.Ts[(self.parent_Ts[i],i)] = self.edge[(self.parent_Ts[i],i)]

            
    def LAST_TREE(self,alpha = 3):
        # initialize
        dis = [INF for i in range(self.num_node)]
        dis[0] = 0
        parent = [-1 for i in range(self.num_node)]
        check = [False for i in range(self.num_node)]                   

        def relax(self,u,v,dis,parent):
            if(dis[v] > dis[u] + self.edge[(u,v)]):
                dis[v] = dis[u] + self.edge[(u,v)]
                parent[v] = u
            return dis,parent

        def add_path(self,dis,parent,node):
            if(dis[node]>self.dist_Ts[node]):
                dis,parent = add_path(self,dis,parent,self.parent_Ts[node])
                dis,parent = relax(self,self.parent_Ts[node],node,dis,parent)
            return dis,parent

        def dfs(self,node,dis,parent,check):
            check[node] = True
            if(dis[node] > alpha*self.dist_Ts[node]):
                dis,parent = add_path(self,dis,parent,node)
            for v in self.Tm[node]:
                if check[v] == False:
                    dis,parent = relax(self,node,v,dis,parent)
                    dis,parent = dfs(self,v,dis,parent,check)
                    dis,parent = relax(self,v,node,dis,parent)
            return dis,parent   

        dis,parent = dfs(self,0,dis,parent,check)
        
        for i in range(1,self.num_node):
            if self.report_size[i] != 0:
                self.tree[i].append(parent[i])
                self.tree[parent[i]].append(i)

    def run(self):
        self.kruskal()
        self.spt_with_weight(0)
        self.LAST_TREE(3)
        return self.tree

class graph_hat_hat:
    def __init__(self,graph,last_tree,report_size,num_node):
        self.graph = graph
        self.last_tree = last_tree
        self.report_size = report_size
        self.num_node = num_node
        # self.adjacent_list = []
        self.graph_hat_hat_edge = []
        self.graph_hat_hat = [[] for i in range(self.num_node)]

    def find_path(self,adjacent_list,node):
        queue = []
        check = [False for i in range(self.num_node)]
        father = [-1 for i in range(self.num_node)]
        queue.append(node)
        check[node] = True

        while len(queue) != 0:
            # if_break = True
            current = queue[0]
            queue.pop(0)
            for i in self.graph[current]:
                if check[i] == False:
                    check[i] = True
                    father[i] = current
                    queue.append(i)
        
        for it in adjacent_list:
            item = it

            while father[item] != -1:
                if(father[item],item) not in self.graph_hat_hat_edge:
                    self.graph_hat_hat_edge.append((father[item],item))
                    self.graph_hat_hat_edge.append((item,father[item]))

                if father[item] not in self.graph_hat_hat[item]:
                    self.graph_hat_hat[item].append(father[item])

                if item not in self.graph_hat_hat[father[item]]:
                    self.graph_hat_hat[father[item]].append(item)
                item = father[item]

    def run(self):
        
        for index,adjacent_list in enumerate(self.last_tree):               
            if len(adjacent_list) > 0:
                self.find_path(adjacent_list,index)

        return self.graph_hat_hat

"""
Multi-factorial Evolutionary Algorithm
Param: tasks (array of class Task), rmp, number of generation
Output: best individual for each task
"""
def mfea(databaseName, tasks, rmp=0.3, generation=100):
    # Initial population with N individuals for each task
    mecat = []
    metcat_rn = []
    lengthOfGen = len(tasks[0].adjList)
    K = len(tasks)
    size = N*K  # size of population

    population = np.array([[[0]*lengthOfGen,[0]*lengthOfGen]])
    for task in tasks:
        for i in range(N):
            tree = genTree(task.adjList)[1]
            # randTmp = random.randint(0, lengthOfGen-1)
            population = np.vstack([population, [encode(tree,0)]])
            # population = np.vstack([population, [encode(tree,randTmp)]])
    population = np.delete(population, 0, axis=0)
    t = 0

    # Evaluate factorial cost of population for all the tasks
    populationFactorialCost = evaluatePopulationFactorialCost(population, tasks)
    factorialRank = evaluateFactorialRank(populationFactorialCost)
    skillFactor = evaluateSkillFactor(factorialRank)
    scalarFitness = evaluateScalarFitness(factorialRank)
    individualBestCost = np.array([populationFactorialCost[idx][skillFactor[idx]] for idx in range(size)])

    # 2 approximation
    shortestPathTree = SPT(tasks[0].n+tasks[0].m+1, tasks[0].adjList)
    # print(shortestPathTree)
    # print("----------")

    # 7 approximation
    rn_num_node = tasks[1].n+tasks[1].m+1
    mecat_rn_adjList = [[] for i in range(rn_num_node)]
    for key,value in tasks[1].adjList.items():
        mecat_rn_adjList[key] = value.copy() 
    
    make_graph_hat = graph_hat(mecat_rn_adjList,tasks[1].s,rn_num_node)
    graph_hat_found,all_pair_shortest_path  = make_graph_hat.run() 
    
    make_last_tree = LAST(graph_hat_found,all_pair_shortest_path,tasks[1].s,tasks[1].m,rn_num_node)
    last_tree = make_last_tree.run()

    make_graph_hat_hat = graph_hat_hat(mecat_rn_adjList,last_tree,tasks[1].s,rn_num_node)
    graph_hat_hat_found = make_graph_hat_hat.run()

    mecat_rn_graph = {}
    for index,adj_list in enumerate(graph_hat_hat_found):
        mecat_rn_graph[index] = adj_list.copy()

    MECATRN_shortestPathTree = SPT(rn_num_node, mecat_rn_graph)
    # print(MECATRN_shortestPathTree)
    # exit(1)
    # Loops
    for i in range(generation):
        # offspringPopulation = np.empty((0, maximumNumberOfEdges), float)
        offspringPopulation = np.array([[[0]*lengthOfGen,[0]*lengthOfGen]])
        offspringSkillFactor = np.empty((0, 1), float)
        # potentialPopulation = np.empty((0, maximumNumberOfEdges), float)
        while(len(offspringPopulation) < size):
            # idxP1, idxP2 = tournamentSelectionParents(population.shape[0], 4, scalarFitness)
            idxP1 = tournamentSelectionIndividual(population.shape[0],5,scalarFitness)
            idxP2 = tournamentSelectionIndividual(population.shape[0],5,scalarFitness)
            rand = np.random.random()
            if(skillFactor[idxP1] == skillFactor[idxP2] or rand < rmp):
            # if rand < rmp:
            # if(True):
            # if False:
                # o1 = ECO_withPhenotype([population[idxP1], population[idxP2]])
                # o2 = ECO_withPhenotype([population[idxP2], population[idxP1]])
                # o1 = eco2((population[idxP1][0].tolist(), population[idxP1][1].tolist()),(population[idxP2][0].tolist(), population[idxP2][1].tolist()))
                # o2 = eco2((population[idxP2][0].tolist(), population[idxP2][1].tolist()),(population[idxP1][0].tolist(), population[idxP1][1].tolist()))
                o1 = eco3(population[idxP1], population[idxP2])
                o2 = eco3(population[idxP2], population[idxP1])

                edge0 = randrange(len(tasks[0].adjList))
                idxEdge1 = randrange(len(tasks[0].adjList[edge0]))
                edge1 = tasks[0].adjList[edge0][idxEdge1]

                edge0x = randrange(len(tasks[0].adjList))
                idxEdge1x = randrange(len(tasks[0].adjList[edge0x]))
                edge1x = tasks[0].adjList[edge0x][idxEdge1x]
                # offspringSkillFactor = np.append(offspringSkillFactor,[np.random.choice([skillFactor[idxP1], skillFactor[idxP2]]) for i in range(2)])
                o1 = epo(o1, edge0, edge1)
                o2 = epo(o2, edge0x, edge1x)
                offspringSkillFactor = np.append(offspringSkillFactor,[skillFactor[idxP1], skillFactor[idxP2]])
            else:
            # if(True):
                edge0 = randrange(len(tasks[0].adjList))
                idxEdge1 = randrange(len(tasks[0].adjList[edge0]))
                edge1 = tasks[0].adjList[edge0][idxEdge1]

                edge0x = randrange(len(tasks[0].adjList))
                idxEdge1x = randrange(len(tasks[0].adjList[edge0x]))
                edge1x = tasks[0].adjList[edge0x][idxEdge1x]
                # o1 = EPO_withPhenotype(population[idxP1], list([edge0, edge1]))
                # o2 = EPO_withPhenotype(population[idxP2], list([edge0, edge1]))
                # o1 = epo((population[idxP1][0].tolist(), population[idxP1][1].tolist()), edge0, edge1)
                # o2 = epo((population[idxP2][0].tolist(), population[idxP2][1].tolist()), edge0, edge1)
                o1 = epo(population[idxP1], edge0, edge1)
                o2 = epo(population[idxP2], edge0x, edge1x)
                offspringSkillFactor = np.append(offspringSkillFactor,[skillFactor[idxP1], skillFactor[idxP2]])
            offspringPopulation = np.vstack([offspringPopulation, [o1]])
            offspringPopulation = np.vstack([offspringPopulation, [o2]])

        offspringPopulation = np.delete(offspringPopulation, 0, axis = 0)

        # # if(t>deltaT and t%deltaT == 0):
        if True:
            potentialPopulation = createPotentialPopulation(shortestPathTree, tasks[0].adjList, tasks[0].numberOfEdges, lengthOfGen, len(tasks))
            potentialSkillFactor = np.array([0]*(K*NE))
            potentialCost = evaluatePopulationFactorialCost(potentialPopulation, list([tasks[0]]))

            # MECATRN_potentialPopulation = createPotentialPopulation(MECATRN_shortestPathTree, tasks[1].adjList, tasks[1].numberOfEdges, lengthOfGen, len(tasks))
            # MECATRN_potentialSkillFactor = np.array([1]*(K*NE))
            # MECATRN_potentialCost = evaluatePopulationFactorialCost(MECATRN_potentialPopulation, list([tasks[1]]))

        # update rmp
        # parentPopulationFactorialCost = evaluatePopulationFactorialCost(population, tasks)
        # offspringPopulationFactorialCost = evaluatePopulationFactorialCost(offspringPopulation, tasks)
        # rmp = updateRmp(rmp, parentPopulationFactorialCost, offspringPopulationFactorialCost)
        
        # Factorial cost of offspring population
        offspringCost = evaluateOffspringCost(offspringPopulation, offspringSkillFactor, tasks)

        # Update population
        population = np.vstack([population, offspringPopulation])
        if(t>=deltaT and t%deltaT == 0):
            population = np.vstack([population, potentialPopulation])
            # population = np.vstack([population, MECATRN_potentialPopulation])
        
        # Update scalar fitness and skill factor for each individual
        skillFactor = np.append(skillFactor, offspringSkillFactor)
        if(t>=deltaT and t%deltaT == 0):
            skillFactor = np.append(skillFactor, potentialSkillFactor)
            # skillFactor = np.append(skillFactor, MECATRN_potentialSkillFactor)
        
        individualBestCost = np.append(individualBestCost, offspringCost)
        if(t>=deltaT and t%deltaT == 0):
            individualBestCost = np.append(individualBestCost, potentialCost)
            # individualBestCost = np.append(individualBestCost, MECATRN_potentialCost)

        scalarFitness = 1 / (evaluateRankBaseOnSkillFactor(individualBestCost, skillFactor, len(tasks))+1)

        # choose fittest individual by tournament selection
        idxFittestPopulation = list()
        # for _ in range(size):
        #     idxFittestIndividual = tournamentSelectionIndividual(population.shape[0], 4, scalarFitness)
        #     idxFittestPopulation.append(idxFittestIndividual)
        # tmpSize = size*4//5
        # tmpSize = 2
        tmpSize = size // 2
        idxFittestPopulation = np.argsort(-scalarFitness)[:tmpSize]
        # for _ in range(size//5):
        for _ in range(size//2):
            idxFittestIndividual = random.randint(size, size*2-1)
            # idxFittestPopulation.append(idxFittestIndividual)
            idxFittestPopulation = np.append(idxFittestPopulation, idxFittestIndividual)
        
        population = population[idxFittestPopulation]
        skillFactor = skillFactor[idxFittestPopulation]
        individualBestCost = individualBestCost[idxFittestPopulation]

        t += 1

        nextHistory = np.empty((0, len(tasks)), float)
        for idx in range(len(tasks)):
            try:
                bestCostForTask = np.min(individualBestCost[np.where(skillFactor == idx)[0]])
            except:
                populationFactorialCost = evaluatePopulationFactorialCost(population, list([tasks[idx]]))
                bestCostForTask = np.min(populationFactorialCost)
            nextHistory = np.append(nextHistory, bestCostForTask)
        
        # history = np.vstack([history, nextHistory])
        print('Epoch [{}/{}], Best Cost: {}'.format(i + 1, generation, nextHistory))
        mecat.append(nextHistory[0])
        metcat_rn.append(nextHistory[1])
        # print(datetime.now())

    # Result
    sol_idx = [np.argmin(individualBestCost[np.where(skillFactor == idx)]) for idx in range (len(tasks))]
    # return [population[np.where(skillFactor == idx)[0]][sol_idx[idx]] for idx in range(len(tasks))], history
    plt.plot(mecat)
    plt.plot(metcat_rn,color='red')
    # plt.show()
    plotFileName = 'Plot/' + str(databaseName) + '.png'
    plt.savefig(plotFileName)
    plt.clf()
    return [population[np.where(skillFactor == idx)[0]][sol_idx[idx]] for idx in range(len(tasks))]


# main
mecatDataPath = os.getcwd()+'/dataset4mecat/mecat'

mecatDataFiles = os.listdir(mecatDataPath)

mecatDataFiles = sorted(mecatDataFiles,reverse=False)

# mecatDataFiles = mecatDataFiles[10:]

allResultCost = np.array([[0]*2])

FileName = "Record/NDE2-" + str(datetime.now())

f = open(FileName,"a")

cntDatabase = 0

for i in range(len(mecatDataFiles)):
    task1 = getInputFromFile(mecatDataPath+'/'+mecatDataFiles[i])
    task2 = getInputFromFile(mecatDataPath+'_rn/rn_'+mecatDataFiles[i])
    tasks = list([task1, task2])
    print('Task 1 and 2 is from file: ', mecatDataFiles[i])
    resultPopulation = mfea(mecatDataFiles[i], tasks, 0.3, 150)

    print("-----")
    resultPopulationCost = list()
    for j in range(len(resultPopulation)):
        print("Task", j+1)
        print(tasks[j].evaluateIndividualFactorialCost(resultPopulation[j]))
        print()
        resultPopulationCost.append(tasks[j].evaluateIndividualFactorialCost(resultPopulation[j]))
    allResultCost = np.vstack([allResultCost, np.array(resultPopulationCost)])
    print("-----")
    print()
    resultSentence = "File: "+str(mecatDataFiles[i])+" ["+str(allResultCost[i+1][0])+" "+str(allResultCost[i+1][1])+"]"+"\n"
    f.write(resultSentence)


allResultCost = np.delete(allResultCost, 0, axis = 0)

for i in range(len(mecatDataFiles)):
    print("File: ", mecatDataFiles[i],end=' ')
    print(allResultCost[i])

f.close()