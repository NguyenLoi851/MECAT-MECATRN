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

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
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
deltaT = 1500000
INF = 9999999999
MaxNumberOfFuncEvaluate = 100000 * 2

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
Param: task (class Task), individual (np.array)
Output: adjacent List (dictionary of (int, list))
"""
def decode(task, individual):
    result = {}
    # tmpIndividual = (-1)*individual
    argSort = np.argsort(-individual)
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
        global cntNumberOfFuncEvaluate
        cntNumberOfFuncEvaluate += 1
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
    # return np.array([np.apply_along_axis(task.evaluateIndividualFactorialCost,1, population) for task in tasks]).T
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
Tournament selection to select parents
Param: population, k (number of selection in population), scalar fitness
Output: 2 index of best parents
"""
def tournamentSelectionParents(sizeOfPopulation, k, scalarFitness):
    selected = np.array(random.sample(range(sizeOfPopulation),k))
    while(True):
        size = selected.shape[0]
        if(size == 2):
            break
        if(size % 2 == 1):
            selected = np.append(selected, selected[-1])
            size += 1
        newSelected = np.array([])
        for i in range(size//2):
            if(scalarFitness[int(selected[i*2])] > scalarFitness[int(selected[i*2+1])]):
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
    selected = np.array(random.sample(range(sizeOfPopulation),k))
    maxScalarFitness = np.max(scalarFitness[selected])
    # result = np.random.choice(np.where(scalarFitness == maxScalarFitness)[0])
    idxOfResult = np.argmax(scalarFitness[selected])   
    return int(selected[idxOfResult])


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
Crossover 2 parents individuals
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
Encode from tree to genotype
Param: tree, adjList, numberOfEdges, numberOfEdgesOfTree
Output: individual
"""
def encode(tree, adjList, numberOfEdges, numberOfEdgesOfTree):
    randArr = initializeIndividual(numberOfEdges)
    result = list()
    randArr = sorted(randArr, reverse=True)

    cnt1 = 0
    cnt2 = numberOfEdgesOfTree

    for key, value in adjList.items():
        for vertice in value:
            if(vertice > key):
                if(vertice in tree[key]):
                    result.append(randArr[cnt1])
                    cnt1 += 1
                else:
                    result.append(randArr[cnt2])
                    cnt2 += 1
    return np.array(result)


"""
Create potential population
Param: tree, adjList, numberOfEdges, maximumNumberOfEdges, numberOfTasks
Output: K*NE potential individuals
"""
def createPotentialPopulation(tree, adjList, numberOfEdges, maximumNumberOfEdges,numberOfTasks):
    population = np.empty((0, maximumNumberOfEdges), float)
    numberOfEdgesOfTree = 0
    for key in tree:
        tree[key] = sorted(tree[key])
        numberOfEdgesOfTree += len(tree[key])
    numberOfEdgesOfTree //= 2
    for key in adjList:
        adjList[key] = sorted(adjList[key])
    for _ in range(numberOfTasks*NE):
        individual = encode(tree, adjList, numberOfEdges, numberOfEdgesOfTree)
        population = np.vstack([population, representInCommonSpace(individual, maximumNumberOfEdges)])
    return population


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
Create shortest path tree
Param: number of node, s, adjList
Output: tree
"""
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
Multi-factorial Evolutionary Algorithm
Param: tasks (array of class Task), rmp, number of generation
Output: best individual for each task
"""
def mfea(databaseName, tasks, rmp=0.3, generation=100):
    global cntNumberOfFuncEvaluate
    # Initial population with N individuals for each task
    mecat = []
    metcat_rn = []
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
    scalarFitness = evaluateScalarFitness(factorialRank)
    individualBestCost = np.array([populationFactorialCost[idx][skillFactor[idx]] for idx in range(size)])

    # 2 approximation
    
    shortestPathTree = SPT(tasks[0].n+tasks[0].m+1, tasks[0].adjList)

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
    # Loops
    for i in range(generation):
        if(cntNumberOfFuncEvaluate >= MaxNumberOfFuncEvaluate):
            break
        offspringPopulation = np.empty((0, maximumNumberOfEdges), float)
        offspringSkillFactor = np.empty((0, 1), float)
        # potentialPopulation = np.empty((0, maximumNumberOfEdges), float)
        while(len(offspringPopulation) < size):
            # idxP1, idxP2 = tournamentSelectionParents(population.shape[0], 4, scalarFitness)
            idxP1 = tournamentSelectionIndividual(population.shape[0],5,scalarFitness)
            idxP2 = tournamentSelectionIndividual(population.shape[0],5,scalarFitness)
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

        if(t>deltaT and t%deltaT == 0):
            potentialPopulation = createPotentialPopulation(shortestPathTree, tasks[0].adjList, tasks[0].numberOfEdges, maximumNumberOfEdges, len(tasks))
            potentialSkillFactor = np.array([0]*(K*NE))
            potentialCost = evaluatePopulationFactorialCost(potentialPopulation, list([tasks[0]]))

            MECATRN_potentialPopulation = createPotentialPopulation(MECATRN_shortestPathTree, tasks[1].adjList, tasks[1].numberOfEdges, maximumNumberOfEdges, len(tasks))
            MECATRN_potentialSkillFactor = np.array([1]*(K*NE))
            MECATRN_potentialCost = evaluatePopulationFactorialCost(MECATRN_potentialPopulation, list([tasks[1]]))

        # Factorial cost of offspring population
        offspringCost = evaluateOffspringCost(offspringPopulation, offspringSkillFactor, tasks)

        # Update population
        population = np.vstack([population, offspringPopulation])
        if(t>deltaT and t%deltaT == 0):
            population = np.vstack([population, potentialPopulation])
            population = np.vstack([population, MECATRN_potentialPopulation])

        # Update scalar fitness and skill factor for each individual
        skillFactor = np.append(skillFactor, offspringSkillFactor)
        if(t>deltaT and t%deltaT == 0):
            skillFactor = np.append(skillFactor, potentialSkillFactor)
            skillFactor = np.append(skillFactor, MECATRN_potentialSkillFactor)

        individualBestCost = np.append(individualBestCost, offspringCost)
        if(t>deltaT and t%deltaT == 0):
            individualBestCost = np.append(individualBestCost, potentialCost)
            individualBestCost = np.append(individualBestCost, MECATRN_potentialCost)
            
        scalarFitness = 1 / (evaluateRankBaseOnSkillFactor(individualBestCost, skillFactor, len(tasks))+1)

        # choose fittest individual by tournament selection
        idxFittestPopulation = list()
        for _ in range(size//4):
            idxFittestIndividual = tournamentSelectionIndividual(population.shape[0], 8, scalarFitness)
            idxFittestPopulation.append(idxFittestIndividual)

        for _ in range(size//4*3):
            idxFittestPopulation.append(int(np.random.randint(0, population.shape[0],1)[0]))

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
        mecat.append(nextHistory[0])
        metcat_rn.append(nextHistory[1])

    # Result
    sol_idx = [np.argmin(individualBestCost[np.where(skillFactor == idx)]) for idx in range (len(tasks))]
    plt.plot(mecat)
    plt.plot(metcat_rn,color='red')
    # plt.show()
    plotFileName = 'Plot/' + str(databaseName) + '.png'
    plt.savefig(plotFileName)
    plt.clf()
    return [population[np.where(skillFactor == idx)[0]][sol_idx[idx]] for idx in range(len(tasks))], history


# main
# mecatDataPath = os.getcwd()+'/dataset4mecat/mecat'

# mecatDataFiles = os.listdir(mecatDataPath)

# mecatDataFiles = sorted(mecatDataFiles,reverse=False)

# allResultCost = np.array([[0]*2])

# FileName = "Record/NDE2-" + str(datetime.now())

# f = open(FileName,"a")

# cntDatabase = 0

# for i in range(len(mecatDataFiles)):
#     task1 = getInputFromFile(mecatDataPath+'/'+mecatDataFiles[i])
#     task2 = getInputFromFile(mecatDataPath+'_rn/rn_'+mecatDataFiles[i])
#     tasks = list([task1, task2])
#     print('Task 1 and 2 is from file: ', mecatDataFiles[i])
#     resultPopulation,history = mfea(tasks, 0.3, 500)
#     print("-----")
#     for i in range(len(resultPopulation)):
#         print("Task", i+1)
#         print(tasks[i].evaluateIndividualFactorialCost(resultPopulation[i]))
#         print()
#     print("-----")
#     print()
#     for i in range (history.shape[1]):
#         plt.plot(np.arange(len(history)), history[:, i], "blue")
#         plt.title(tasks[i].__class__.__name__)
#         plt.xlabel("Epoch")
#         plt.ylabel("Best Factorial Cost")
#         plt.show()

mecatDataPath = os.getcwd()+'/dataset4mecat/mecat'

mecatDataFiles = os.listdir(mecatDataPath)

mecatDataFiles = sorted(mecatDataFiles,reverse=False)

# mecatDataFiles = mecatDataFiles[10:]

allResultCost = np.array([[0]*2])

FileName = "Record/RVE-" + str(datetime.now())

f = open(FileName,"a")

cntDatabase = 0

for i in range(len(mecatDataFiles)):
    cntNumberOfFuncEvaluate = 0
    task1 = getInputFromFile(mecatDataPath+'/'+mecatDataFiles[i])
    task2 = getInputFromFile(mecatDataPath+'_rn/rn_'+mecatDataFiles[i])
    tasks = list([task1, task2])
    print('Task 1 and 2 is from file: ', mecatDataFiles[i])
    resultPopulation, history = mfea(mecatDataFiles[i], tasks, 0.3, 1000)

    print("-----")
    resultPopulationCost = list()
    for j in range(len(resultPopulation)):
        print("Task", j+1)
        print(tasks[j].evaluateIndividualFactorialCost(resultPopulation[j]))
        print()
        print(cntNumberOfFuncEvaluate)
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