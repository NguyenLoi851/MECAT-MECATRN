"""
** Step: **
- Read file -> Task (graph)
- Generate trees from graph
- Encode tree
- Run genetic algorithm
- Return last generation
"""

import numpy as np
import random
import math
import os

Tx = 2
Rx = 1
q = 4

"""
Decode individual from genotype to pheotype for each task
Param: individual (2darray)
Output: adjacent List (dictionary of (int, list))
"""
def decode(individual):
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
        pheotype = decode(self, individual) # adjacent list
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
    return np.array([np.apply_along_axis(task.evaluateIndividualFactorialCost,1, population) for task in tasks]).T

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
Param: individual (2darray, list)
Output: individual
"""
def EPO_withPhenotype(individual, edge):
    tree = decode(individual)
    if(edge[1] in tree[edge[0]]):
        return tree
    individual = individual.tolist()
    idxRoot = individual[1].index(0)
    idxNode = list()
    idxNode.append(individual[0].index(edge[0])) 
    idxNode.append(individual[0].index(edge[1]))
    # edge[1] will be ancestor of edge[0] in tree
    if(individual[1][idxNode[0]] < individual[1][idxNode[1]]):
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
def ECO_withPhenotype():

    pass

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

"""
Generate a random tree from graph
Param: Graph
Output: Tree, root of tree
"""
def generateRandomTree():
    pass

"""
Multi-factorial Evolutionary Algorithm
Param: tasks (array of class Task), rmp, number of generation
Output: best individual for each task
"""
def mfea():
    pass

# main
mecatDataPath = os.getcwd()+'/dataset4mecat/mecat'

mecatDataFiles = os.listdir(mecatDataPath)

mecatDataFiles = sorted(mecatDataFiles,reverse=True)

for i in range(len(mecatDataFiles)):
    task1 = getInputFromFile(mecatDataPath+'/'+mecatDataFiles[i])
    task2 = getInputFromFile(mecatDataPath+'_rn/rn_'+mecatDataFiles[i])
    tasks = list([task1, task2])
    print('Task 1 and 2 is from file: ', mecatDataFiles[i])
    resultPopulation = mfea(tasks, 0.3, 1000)
    print("-----")
    for i in range(len(resultPopulation)):
        print("Task", i+1)
        print(tasks[i].evaluateIndividualFactorialCost(resultPopulation[i]))
        print()
    print("-----")
    print()