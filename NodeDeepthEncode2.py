"""
** Step: **
- Read file -> Task (graph)
- Generate trees from graph
- Encode tree
- Run genetic algorithm
- Return last generation
"""

from random import randrange
import numpy as np
import random
import math
import os

N = 200
Tx = 2
Rx = 1
q = 4

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
            try:
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
            except:
                exit(1)
                return {83: [0], 0: [83, 18], 87: [78], 78: [87, 51, 31, 18], 62: [88], 88: [62, 51], 
                51: [88, 55, 78], 55: [51], 64: [76], 76: [64, 26, 71], 15: [26], 26: [15, 76], 
                71: [76, 31], 31: [71, 1, 78], 11: [59], 59: [11, 9], 9: [59, 29], 29: [9, 12], 12: [29, 1], 
                1: [12, 31], 18: [78, 72, 57, 0], 92: [65], 65: [92, 49, 38], 49: [65], 38: [65, 58, 30, 45], 
                91: [58], 58: [91, 75, 32, 8, 38], 90: [68], 68: [90, 24, 52], 10: [24], 24: [10, 68], 
                52: [68, 96], 96: [52, 48, 67], 42: [47], 47: [42, 27], 27: [47, 48], 48: [27, 96], 
                67: [96, 93, 75], 28: [93], 93: [28, 67], 75: [67, 43, 23, 58], 43: [75], 69: [23], 
                23: [69, 75], 32: [58], 8: [58], 36: [33], 33: [36, 98], 98: [33, 99], 99: [98, 53, 30], 
                53: [99], 30: [99, 38], 45: [38, 72], 72: [45, 7, 18], 73: [16], 16: [73, 3, 79], 82: [50], 
                50: [82, 2, 61], 66: [2], 2: [66, 5, 50], 74: [5], 5: [74, 46, 2], 84: [85], 85: [84, 94], 
                94: [85, 60, 6, 77], 95: [60], 60: [95, 94], 89: [6], 6: [89, 94], 77: [94, 46], 46: [77, 5], 
                61: [50, 97], 97: [61, 63], 63: [97, 39, 25, 3], 39: [63], 80: [25], 25: [80, 35, 63], 35: [25], 
                3: [63, 14, 16], 86: [14], 14: [86, 81, 3], 70: [81], 81: [70, 34, 4, 14], 19: [22], 22: [19, 34], 
                34: [22, 20, 17, 81], 54: [20], 20: [54, 34], 17: [34], 37: [40], 40: [37, 4], 4: [40, 13, 81], 
                41: [13], 13: [41, 4], 79: [16, 7], 7: [79, 44, 72], 44: [7], 56: [21], 21: [56, 57], 57: [21, 18]}
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

    """
    Evaluate factorial cost of each individual
    Param: task (self), individual (in genotype)
    Output: a number (used energy)
    """
    def evaluateIndividualFactorialCost(self, individual):
        result = 0
        pheotype = decode(individual) # adjacent list
                                            # (dictionary of (int, list))
        z = np.array([0]*(self.n+self.m+1))
        while(len(pheotype) != 1):
            for key in list(pheotype):
                if(key != 0 and len(pheotype[key]) == 1):
                    z[key] += self.s[key]
                    adjVertice = pheotype[key][0]
                    z[adjVertice] += z[key]
                    pheotype.pop(key, None)
                    pheotype[adjVertice].remove(key)

        for i in range(self.n + self.m + 1):
            if(i == 0):
                continue
            result += math.ceil(z[i]/q)
        return (Tx + Rx)*result

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
    return np.argmin(factorialRank, axis=1)

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

def eco2(gen1, gen2):
    num_node = len(gen1[0])
    i = random.randint(num_node//4, 3*num_node//4)
    vr = random.sample(range(num_node), i)
    # vr = [5, 6, 8, 3]
    childGen = (gen2[0].copy(), gen2[1].copy())
    for k in vr:
        idxK = gen1[0].index(k)
        if gen1[1][idxK] > 0:
            for index in range(idxK-1, -1, -1):
                if gen1[1][index] < gen1[1][idxK]:
                    childGenResult = np.copy(epo(childGen, gen1[0][idxK], gen1[0][index]))
                    childGen = (childGenResult[0].tolist().copy(), childGenResult[1].tolist().copy())
                    break
    return np.array(childGen)

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
Multi-factorial Evolutionary Algorithm
Param: tasks (array of class Task), rmp, number of generation
Output: best individual for each task
"""
def mfea(tasks, rmp=0.3, generation=100):
    # Initial population with N individuals for each task
    lengthOfGen = len(tasks[0].adjList)
    K = len(tasks)
    size = N*K  # size of population

    population = np.array([[[0]*lengthOfGen,[0]*lengthOfGen]])
    for task in tasks:
        for i in range(N):
            tree = genTree(task.adjList)[1]
            population = np.vstack([population, [encode(tree,0)]])
    population = np.delete(population, 0, axis=0)
    t = 0

    # Evaluate factorial cost of population for all the tasks
    populationFactorialCost = evaluatePopulationFactorialCost(population, tasks)
    factorialRank = evaluateFactorialRank(populationFactorialCost)
    skillFactor = evaluateSkillFactor(factorialRank)
    scalarFitness = evaluateScalarFitness(factorialRank)
    individualBestCost = np.array([populationFactorialCost[idx][skillFactor[idx]] for idx in range(size)])

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
            # if(skillFactor[idxP1] == skillFactor[idxP2] or rand < rmp):
            if False:
                # o1 = ECO_withPhenotype([population[idxP1], population[idxP2]])
                # o2 = ECO_withPhenotype([population[idxP2], population[idxP1]])
                # o1 = eco2((population[idxP1][0].tolist(), population[idxP1][1].tolist()),(population[idxP2][0].tolist(), population[idxP2][1].tolist()))
                # o2 = eco2((population[idxP2][0].tolist(), population[idxP2][1].tolist()),(population[idxP1][0].tolist(), population[idxP1][1].tolist()))
                o1 = eco3(population[idxP1], population[idxP2])
                o2 = eco3(population[idxP2], population[idxP1])
                offspringSkillFactor = np.append(offspringSkillFactor,[np.random.choice([skillFactor[idxP1], skillFactor[idxP2]]) for i in range(2)])
            else:
                edge0 = randrange(len(tasks[0].adjList))
                idxEdge1 = randrange(len(tasks[0].adjList[edge0]))
                edge1 = tasks[0].adjList[edge0][idxEdge1]
                # o1 = EPO_withPhenotype(population[idxP1], list([edge0, edge1]))
                # o2 = EPO_withPhenotype(population[idxP2], list([edge0, edge1]))
                # o1 = epo((population[idxP1][0].tolist(), population[idxP1][1].tolist()), edge0, edge1)
                # o2 = epo((population[idxP2][0].tolist(), population[idxP2][1].tolist()), edge0, edge1)
                o1 = epo(population[idxP1], edge0, edge1)
                o2 = epo(population[idxP2], edge0, edge1)
                offspringSkillFactor = np.append(offspringSkillFactor,[skillFactor[idxP1], skillFactor[idxP2]])
            offspringPopulation = np.vstack([offspringPopulation, [o1]])
            offspringPopulation = np.vstack([offspringPopulation, [o2]])

        offspringPopulation = np.delete(offspringPopulation, 0, axis = 0)

        # Factorial cost of offspring population
        offspringCost = evaluateOffspringCost(offspringPopulation, offspringSkillFactor, tasks)

        # Update population
        population = np.vstack([population, offspringPopulation])

        # Update scalar fitness and skill factor for each individual
        skillFactor = np.append(skillFactor, offspringSkillFactor)

        individualBestCost = np.append(individualBestCost, offspringCost)

        scalarFitness = 1 / (evaluateRankBaseOnSkillFactor(individualBestCost, skillFactor, len(tasks))+1)

        # choose fittest individual by tournament selection
        idxFittestPopulation = list()
        for _ in range(size):
            idxFittestIndividual = tournamentSelectionIndividual(population.shape[0], 4, scalarFitness)
            idxFittestPopulation.append(idxFittestIndividual)

        # idxFittestPopulation = np.argsort(-scalarFitness)[:size]

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

    # Result
    sol_idx = [np.argmin(individualBestCost[np.where(skillFactor == idx)]) for idx in range (len(tasks))]
    # return [population[np.where(skillFactor == idx)[0]][sol_idx[idx]] for idx in range(len(tasks))], history
    return [population[np.where(skillFactor == idx)[0]][sol_idx[idx]] for idx in range(len(tasks))]


# main
mecatDataPath = os.getcwd()+'/dataset4mecat/mecat'

mecatDataFiles = os.listdir(mecatDataPath)

mecatDataFiles = sorted(mecatDataFiles,reverse=False)

allResultCost = np.array([[0]*2])

for i in range(len(mecatDataFiles)):
    task1 = getInputFromFile(mecatDataPath+'/'+mecatDataFiles[i])
    task2 = getInputFromFile(mecatDataPath+'_rn/rn_'+mecatDataFiles[i])
    tasks = list([task1, task2])
    print('Task 1 and 2 is from file: ', mecatDataFiles[i])
    resultPopulation = mfea(tasks, 0.3, 1000)

    print("-----")
    resultPopulationCost = list()
    for i in range(len(resultPopulation)):
        print("Task", i+1)
        print(tasks[i].evaluateIndividualFactorialCost(resultPopulation[i]))
        print()
        resultPopulationCost.append(tasks[i].evaluateIndividualFactorialCost(resultPopulation[i]))
    allResultCost = np.vstack([allResultCost, np.array(resultPopulationCost)])
    print("-----")
    print()

allResultCost = np.delete(allResultCost, 0, axis = 0)

for i in range(len(mecatDataFiles)):
    print("File: ", mecatDataFiles[i],end=' ')
    print(allResultCost[i])