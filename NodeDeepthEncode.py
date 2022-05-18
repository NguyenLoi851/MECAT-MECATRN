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
    individual = individual.tolist()
    tree = {}
    for i in range(len(individual[0])-1,0,-1):
        print("30",i)
        deepth = individual[1][i]
        print("32",deepth)
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
    # print(population[0])
    # return np.array([np.apply_along_axis(task.evaluateIndividualFactorialCost,2, population) for task in tasks]).T
    
    # for task in tasks:
    #     costOfTask = np.array([])
    #     for individual in population:
    #         cost = task.evaluateIndividualFactorialCost(individual)
    #         costOfTask = np.append(costOfTask, cost)
    #     res = np.vstack([res, costOfTask])
    # res = np.delete(res, 0, axis=0)
    # return res
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
    isReversed = False
    # edge[1] will be ancestor of edge[0] in tree
    if(individual[1][idxNode[0]] < individual[1][idxNode[1]]):
        edge.reverse()
        idxNode.reverse()
        isReversed = True
    # chieck ancestor of edge[0]
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
    print("Edge: ", edge)
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
    # print(len(edgeList))
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
    # history = np.empty((0, len(tasks)), float)
    lengthOfGen = len(tasks[0].adjList)
    K = len(tasks)
    size = N*K  # size of population
    # maximumNumberOfEdges = 0
    # for task in tasks:
    #     maximumNumberOfEdges = max(task.numberOfEdges, maximumNumberOfEdges)
    # population = np.empty((0, maximumNumberOfEdges), float)
    # for task in tasks:
    #     populationOfTask = np.empty((0, task.numberOfEdges), float)
    #     for i in range(N):
    #         populationOfTask = np.vstack([populationOfTask,f(task.numberOfEdges)])
    #     for individual in populationOfTask:
    #         population = np.vstack([population,representInCommonSpace(individual, maximumNumberOfEdges)])


    population = np.array([[[0]*lengthOfGen,[0]*lengthOfGen]])
    for task in tasks:
        for i in range(N):
            tree = genTree(task.adjList)[1]
            population = np.vstack([population, [encode(tree,0)]])
    population = np.delete(population, 0, axis=0)
    print(population)
    t = 0
    # tree = genTree(tasks[0].adjList)[1]
    # print(encode(tree,0))
    # print("-------")
    
    # tree = genTree(tasks[0].adjList)[1]
    # population = np.array([encode(tree,0)])
    # population = np.vstack([population,[ encode(genTree(tasks[0].adjList)[1],0)]])
    # population = np.delete(population, 0, axis=0)
    # population = np.array([[[0]*3,[0]*3]])
    # population = np.vstack([population,[[[1,2,3],[4,5,9]]]])
    # # population = np.array([[2,3,4],[4,5,6]])
    # # population = np.append(population,[[1,2,3],[4,6,9]])
    # population = np.vstack([population,[[[1,2,3],[4,6,9]]]])
    # population = np.delete(population,[0,1],0)
    # print(population)
    # exit(1)

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
            if(skillFactor[idxP1] == skillFactor[idxP2] or rand < rmp):
                # o1, o2 = crossover(population[idxP1], population[idxP2], uc)
                # offspringSkillFactor = np.append(offspringSkillFactor,[np.random.choice([skillFactor[idxP1], skillFactor[idxP2]]) for i in range(2)])
                o1 = ECO_withPhenotype([population[idxP1], population[idxP2]])
                o2 = ECO_withPhenotype([population[idxP2], population[idxP1]])
                offspringSkillFactor = np.append(offspringSkillFactor,[np.random.choice([skillFactor[idxP1], skillFactor[idxP2]]) for i in range(2)])
            else:
                # o1 = polynomialMutate(population[idxP1], um)
                # o2 = polynomialMutate(population[idxP2], um)
                # offspringSkillFactor = np.append(offspringSkillFactor,[skillFactor[idxP1], skillFactor[idxP2]])
                o1 = EPO_withPhenotype(population[idxP1])
                o2 = EPO_withPhenotype(population[idxP2])
                offspringSkillFactor = np.append(offspringSkillFactor,[skillFactor[idxP1], skillFactor[idxP2]])
            offspringPopulation = np.vstack([offspringPopulation, [o1]])
            offspringPopulation = np.vstack([offspringPopulation, [o2]])

        # if(t>deltaT and t%deltaT == 0):
        #     potentialPopulation = createPotentialPopulation(shortestPathTree, tasks[0].adjList, tasks[0].numberOfEdges, maximumNumberOfEdges, len(tasks))
        #     potentialSkillFactor = np.array([0]*(K*NE))
        #     potentialCost = evaluatePopulationFactorialCost(potentialPopulation, list([tasks[0]]))

        #     MECATRN_potentialPopulation = createPotentialPopulation(MECATRN_shortestPathTree, tasks[1].adjList, tasks[1].numberOfEdges, maximumNumberOfEdges, len(tasks))
        #     MECATRN_potentialSkillFactor = np.array([1]*(K*NE))
        #     MECATRN_potentialCost = evaluatePopulationFactorialCost(MECATRN_potentialPopulation, list([tasks[1]]))

        # Factorial cost of offspring population
        offspringCost = evaluateOffspringCost(offspringPopulation, offspringSkillFactor, tasks)

        # Update population
        population = np.vstack([population, offspringPopulation])
        # if(t>deltaT and t%deltaT == 0):
        #     population = np.vstack([population, potentialPopulation])
        #     population = np.vstack([population, MECATRN_potentialPopulation])

        # Update scalar fitness and skill factor for each individual
        skillFactor = np.append(skillFactor, offspringSkillFactor)
        # if(t>deltaT and t%deltaT == 0):
        #     skillFactor = np.append(skillFactor, potentialSkillFactor)
        #     skillFactor = np.append(skillFactor, MECATRN_potentialSkillFactor)

        individualBestCost = np.append(individualBestCost, offspringCost)
        # if(t>deltaT and t%deltaT == 0):
        #     individualBestCost = np.append(individualBestCost, potentialCost)
        #     individualBestCost = np.append(individualBestCost, MECATRN_potentialCost)
            
        scalarFitness = 1 / (evaluateRankBaseOnSkillFactor(individualBestCost, skillFactor, len(tasks))+1)

        # choose fittest individual by tournament selection
        idxFittestPopulation = list()
        for _ in range(size):
            idxFittestIndividual = tournamentSelectionIndividual(population.shape[0], 8, scalarFitness)
            idxFittestPopulation.append(idxFittestIndividual)

        # for _ in range(size//4*3):
        #     idxFittestPopulation.append(int(np.random.randint(0, population.shape[0],1)[0]))

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
        
        history = np.vstack([history, nextHistory])
        print('Epoch [{}/{}], Best Cost: {}'.format(i + 1, generation, nextHistory))

    # Result
    sol_idx = [np.argmin(individualBestCost[np.where(skillFactor == idx)]) for idx in range (len(tasks))]
    return [population[np.where(skillFactor == idx)[0]][sol_idx[idx]] for idx in range(len(tasks))], history


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