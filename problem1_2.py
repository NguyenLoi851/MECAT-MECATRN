# -*- coding: utf-8 -*-
"""Problem1_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O4GiSdzbVAucQaWI88f9ZIkNye7ccKa_

**Step:**
- Get input from file
 - Create class task is corresponding to this input
- Initial population with each task
 - Count number of edges and random vector with length is number of edges with a distributed statistic
- Tranfer all population to Unified Search Space 
 - Find maximum of number of edges of all tasks and add padding to individual if it is short.
- Run genetic algorithm
 - Evaluate factorial cost of population
 - Evaluate factorial rank of population
 - Evaluate scalar fitness
 - Compute skill-factor for each individual in population
 - Loops:
 - Create offspring population
 - Get potential population
 - Add potential population to offspring population
 - Update scalar fitness and skill factor for each individual
 - Select the fittest individuals
 - Check conditions of loop
- Convert results to type of vector of each task
- Convert task to output
"""

import numpy as np
import math
import os

"""
Initial constant variable (from paper):
"""
R = 20
q = 4
Tx = 2
Rx = 1
rlpb = 0.3  # probability of being relay nodes
N = 200
NE = 75
uc = 2
um = 5
deltaT = 150

"""
Data structure checks if cycle exists in graph when add edges.
"""


class UnionFind:
    def __init__(self, parent):
        self.parent = parent

    def find(self, x):
        if(self.parent[x] == x):
            return x
        else:
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

    def unite(self, x, y):
        self.parent[self.find(x)] = self.find(y)


"""
Decode individual from genotype to pheotype for each task
Param: task (class Task), individual (array)
Output: adjacent List (dictionary of (int, array))
"""


def decode(task, individual):
    result = {}
    argSort = np.argsort(individual)
    edgesOfTask = list()
    degreeOfVertice = list()
    for key, value in task.adjList.items():
        for i in value:
            if(i > key):
                edgesOfTask.append((key, i))

    parent = np.arange(0, task.n+task.m+1, 1, dtype=int)
    UF = UnionFind(parent)
    count = 0
    for i in argSort:
        u = edgesOfTask[i][0]
        v = edgesOfTask[i][1]
        if(UF.find(u) != UF.find(v)):
            count+=1
            UF.unite(u, v)
            if(u not in result):
                result[u] = list([v])
            else:
                result[u].append(v)
            if(v not in result):
                result[v] = list([u])
            else:
                result[v].append(u)
            if(count == task.n+task.m):
                break

    tmpResult = {k: v for k, v in sorted(list(result.items()))}
    result = tmpResult

    for key, value in result.items():
        degreeOfVertice.append(len(value))
    for i in range(len(degreeOfVertice)):
        if(degreeOfVertice[i] == 1 and i in task.listOfRelayNode):
            adjVertice = result[i][0]
            result[adjVertice].remove(i)
            result.pop(i, None)
    return result


"""
Define task for an input/data. Class Task will have some fields: 
n (number of sensor), 
m (number of relay node), 
numberOfEdges, 
s (array of report size),
listOfRelayNode (numpy array of relay node),
adjList (adjacent list of graph) (dictionary of (int, array))
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
        pheotype = decode(self, individual)  # adjacent list
        # (dictionary of (int, array))
        z = np.array([0]*(self.n+self.m+1))
        while(len(pheotype) != 1):
            for key in list(pheotype):
                if(key != 0 and len(pheotype[key]) == 1):
                    z[key] += self.s[key]
                    adjVertice = pheotype[key][0]
                    z[adjVertice] += math.ceil(z[key]/q)
                    pheotype.pop(key, None)
                    pheotype[adjVertice].remove(key)

        for i in range(self.n + self.m + 1):
            if(i == 0):
                continue
            result += (Tx + Rx)*math.ceil(z[i]/q)
        return result


"""
Get input/data for each task from file.
Param: path of file
Output: class Task, this Task will have some fields: 
n (number of sensor), 
m (number of relay node), 
numberOfEdges, 
s (array of report size),
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
    adjList = dict()
    for i in range(numberOfEdges):
        edge = file.readline().split()
        edge = [int(vertice) for vertice in edge]
        if(edge[0] not in adjList):
            adjList[edge[0]] = list([edge[1]])
        else:
            adjList[edge[0]].append(edge[1])
        if(edge[1] not in adjList):
            adjList[edge[1]] = list([edge[0]])
        else:
            adjList[edge[1]].append(edge[0])

    return Task(n,m,numberOfEdges,s,listOfRelayNode,adjList)

"""
Initialize individual of each task
Param: number of edges (of task)
Output: row vector with uniform distributed U(0,1)
"""


def initializeIndividual(numberOfEdges):
    return np.random.uniform(0, 1, numberOfEdges)


"""
Represent individual to Unified Search Space.
Param: individual (array of U(0,1)), maximum of number of edges
Output: individual
"""


def representInCommonSpace(individual, maximumNumberOfEdges):
    if(individual.shape[0] < maximumNumberOfEdges):
        pad = maximumNumberOfEdges - individual.shape[0]
        individual = np.append(individual, [0]*pad)
    return individual


"""
Represent individual in commmon space to each task
Param: task (class Task), individual (array)
Output: individual
"""


def representForEachTask(task, individual):
    if(individual.shape[0] > task.numberOfEdges):
        pad = individual.shape[0] - task.numberOfEdges
        return individual[:-pad]


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
Tournament selection to select parents
Param: population, k (number of selection in population), tasks
Output: 2 index of best parents
"""


def tournamentSelectionParents(population, k, tasks):
    selected = np.random.randint(low=0, high=population.shape[0], size=k)
    while(True):
        size = selected.shape[0]
        if(size == 2):
            break
        if(size % 2 == 1):
            selected = np.append(selected, selected[-1])
            size += 1
        newSelected = np.array([])
        for i in range(size//2):
            factorialCost = evaluatePopulationFactorialCost(np.array([population[int(selected[int(i*2)])], population[int(selected[int(i*2+1)])]]), tasks)
            factorialRank = evaluateFactorialRank(factorialCost)
            scalarFitness = evaluateScalarFitness(factorialRank)
            if(scalarFitness[0] > scalarFitness[1]):
                newSelected = np.append(newSelected, i*2)
            else:
                newSelected = np.append(newSelected, i*2+1)
        selected = newSelected
    return int(selected[0]), int(selected[1])


"""
Tournament selection to select the fittest individuals
Param: size of population, k (number of selection in population), scalar fitness
Output: index of fittest individual
"""


def tournamentSelectionIndividual(sizeOfPopulation, k, scalarFitness):
    selected = np.random.randint(low=0, high=sizeOfPopulation, size=k)
    maxScalarFitness = np.max(scalarFitness[selected])
    result = np.random.choice(np.where(scalarFitness[selected] == maxScalarFitness)[0])
    return int(result)


"""
SBX-crossover 2 parents individual
Param: 2 parents individual, uc
Output: 2 children individual
"""


def sbxCrossover(p1, p2, uc=8):
    u = np.random.random()
    if(u <= 0.5):
        beta = (2*u)**(1/(uc+1))
    else:
        beta = (1/(2*(1-u)))**(1/(uc+1))
    c1 = 0.5*((1+beta)*p1 + (1-beta)*p2)
    c2 = 0.5*((1-beta)*p1 + (1+beta)*p2)
    return c1, c2


"""
Crossover 2 parents individual
Param: 2 individuals, uc
Output: 2 children individual
"""


def crossover(p1, p2, uc=8):
    size = p1.shape[0]
    while(True):
        arr = np.random.randint(low=1, high=size, size=2)
        if(arr[0] != arr[1]):
            break
    if(arr[0] < arr[1]):
        i, j = arr[0], arr[1]
    else:
        i, j = arr[1], arr[0]
    p1Segment = p1[i:j]
    p2Segment = p2[i:j]
    c1, c2 = sbxCrossover(p1Segment, p2Segment, uc)
    o1 = p1[0:i]
    o1 = np.append(o1, c1)
    o1 = np.append(o1, p1[j:])
    o2 = p2[0:i]
    o2 = np.append(o2, c2)
    o2 = np.append(o2, p2[j:])
    return o1, o2


"""
Polynomial mutation for individual
Param: individual, um
Output: individual
"""


def polynomialMutate(p, um=5):
    u = np.random.random()
    if(u <= 0.5):
        deltaL = (2*u)**(1/(um+1)) - 1
        newP = p + deltaL*p
    else:
        deltaL = 1 - (2*(1-u))**(1/(um+1))
        newP = p + deltaL*(1-p)
    return newP


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
Multi-factorial Evolutionary Algorithm
Param: tasks (array of class Task), rmp, number of generation
Output: best individual for each task
"""


def mfea(tasks, rmp=0.3, generation=100):
    # Initial population with N individuals for each task
    history = np.empty((0, len(tasks)), float)
    K = len(tasks)
    size = N*K  # size of population
    maximumNumberOfEdges = 0
    for task in tasks:
        maximumNumberOfEdges = max(task.numberOfEdges, maximumNumberOfEdges)
    population = np.empty((0, maximumNumberOfEdges), float)
    for task in tasks:
        populationOfTask = np.empty((0, task.numberOfEdges), float)
        for i in range(N):
            populationOfTask = np.vstack([populationOfTask,initializeIndividual(task.numberOfEdges)])
        for individual in populationOfTask:
            population = np.vstack([population,representInCommonSpace(individual, maximumNumberOfEdges)])

    t = 0

    # Evaluate factorial cost of population for all the tasks
    populationFactorialCost = evaluatePopulationFactorialCost(population, tasks)
    factorialRank = evaluateFactorialRank(populationFactorialCost)
    skillFactor = evaluateSkillFactor(factorialRank)
    individualBestCost = np.array([populationFactorialCost[idx][skillFactor[idx]] for idx in range(size)])

    # Loops
    for i in range(generation):
        offspringPopulation = np.empty((0, maximumNumberOfEdges), float)
        offspringSkillFactor = np.empty((0, 1), float)
        # potentialPopulation = np.empty((0, maximumNumberOfEdges), float)
        while(len(offspringPopulation) < size):
            idxP1, idxP2 = tournamentSelectionParents(population, 4, tasks)
            rand = np.random.random()
            if(skillFactor[idxP1] == skillFactor[idxP2] or rand < rmp):
                o1, o2 = crossover(population[idxP1], population[idxP2], uc)
                offspringSkillFactor = np.append(offspringSkillFactor,[np.random.choice([skillFactor[idxP1], skillFactor[idxP2]]) for i in range(2)])
            else:
                o1 = polynomialMutate(population[idxP1], um)
                o2 = polynomialMutate(population[idxP2], um)
                offspringSkillFactor = np.append(offspringSkillFactor,[skillFactor[idxP1], skillFactor[idxP2]])
            offspringPopulation = np.vstack([offspringPopulation, o1])
            offspringPopulation = np.vstack([offspringPopulation, o2])

        # Factorial cost of offspring population
        offspringCost = evaluateOffspringCost(offspringPopulation, offspringSkillFactor, tasks)

        # Update population
        population = np.vstack([population, offspringPopulation])
        # population = np.vstack([population, potentialPopulation])

        # Update scalar fitness and skill factor for each individual
        skillFactor = np.append(skillFactor, offspringSkillFactor)
        individualBestCost = np.append(individualBestCost, offspringCost)
        scalarFitness = 1 / (evaluateRankBaseOnSkillFactor(individualBestCost, skillFactor, len(tasks))+1)

        # choose fittest individual by tournament selection
        idxFittestPopulation = list()
        for _ in range(size):
            idxFittestIndividual = tournamentSelectionIndividual(population.shape[0], population.shape[0], scalarFitness)
            idxFittestPopulation.append(idxFittestIndividual)
        population = population[idxFittestPopulation]
        skillFactor = skillFactor[idxFittestPopulation]
        individualBestCost = individualBestCost[idxFittestPopulation]

        t += 1
        nextHistory = np.empty((0, len(tasks)), float)
        for idx in range(len(tasks)):
            try:
                bestCostForTask = np.min(individualBestCost[np.where(skillFactor == idx)[0]])
            except:
                populationFactorialCost = evaluatePopulationFactorialCost(population, tasks[idx])
                bestCostForTask = np.min(populationFactorialCost)
            nextHistory = np.append(nextHistory, bestCostForTask)
        
        history = np.append(history, nextHistory)
        print('Epoch [{}/{}], Best Cost: {}'.format(i + 1, generation, nextHistory))

    # Result
    sol_idx = [np.argmin(individualBestCost[np.where(skillFactor == idx)]) for idx in range (len(tasks))]
    return [population[np.where(skillFactor == idx)[0]][sol_idx[idx]] for idx in range(len(tasks))], history


mecatDataPath = os.getcwd()+'/dataset4mecat/mecat'

mecatDataFiles = os.listdir(mecatDataPath)

number = 4

task1 = getInputFromFile(mecatDataPath+'/'+mecatDataFiles[number])

task2 = getInputFromFile(mecatDataPath+'_rn/rn_'+mecatDataFiles[number])

tasks = list([task1, task2])

print('Task 1 and 2 is from file: ', mecatDataFiles[number])

resultPopulation,_ = mfea(tasks, 0.3, 10)

for i in range(len(resultPopulation)):
    print("Task", i+1)
    print(tasks[i].evaluateIndividualFactorialCost(resultPopulation[i]))
    print("-----")
    print()
