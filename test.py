import numpy as np
import random

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

tree = {0:[4],4:[0,8,6,5],8:[4,9],9:[8,1,2],
1:[9,3],3:[1],2:[9],6:[4,7],7:[6],5:[4]}
tmp = {k:v for k,v in sorted(list(tree.items()))}
for k,v in tmp.items():
    v = v.sort()
tree = tmp
print("HH",tree)

individual = encode(tree, 0)
# print(individual)

# string =[]
# for number in individual[0]:
#     char = "a"
#     i = ord(char[0])
#     i += number
#     char = chr(i)
#     string.append(char)
# individual[0] = string
# print(np.array(individual))

def decode(individual):
    individual = individual.tolist()
    tree = {}
    for i in range(len(individual[0])-1,0,-1):
        deepth = individual[1][i]
        idxParentNode = i-1
        while True:
            node = individual[0][i]
            parentNode = individual[0][idxParentNode]
            if(individual[1][idxParentNode] == deepth - 1):
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

newTree = decode(individual)
tmp = {k:v for k,v in sorted(list(newTree.items()))}
for k, v in tmp.items():
    v = v.sort()
newTree = tmp
print("GG",newTree)

print()
print("-------------")
print()
tmp = individual
individual = individual.tolist()

string =[]
for number in individual[0]:
    char = "a"
    i = ord(char[0])
    i += number
    char = chr(i)
    string.append(char)
individual[0] = string
print(np.array(individual))
individual = tmp

print()

"""
individual: 2darray, edge: list
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

newIndividual = EPO_withPhenotype(individual,[4,9])

# print(newIndividual)

newIndividual = newIndividual.tolist()

string =[]
for number in newIndividual[0]:
    char = "a"
    i = ord(char[0])
    i += number
    char = chr(i)
    string.append(char)
newIndividual[0] = string
print(np.array(newIndividual))

print()
print("++++++++++++++++")
print()

"""
list of individuals
"""
def ECO_withPhenotype(individuals):
    A = individuals[0]
    B = individuals[1]
    Fab = A
    n = len(A[0])
    i = random.randint(n//4, 3*n//4)
    vr = np.array(random.sample(range(n),i))
    vr = np.sort(vr)
    for node in vr:
        idxNodeInB = B[0].index(node)
        if(B[1][idxNodeInB] != 0):
            # find parent of node in B
            deepth = B[1][idxNodeInB]
            idxParentNode = idxNodeInB - 1
            while True:
                if(B[1][idxParentNode] == deepth -1):
                    parentNode = B[0][idxParentNode]
                    EPO_withPhenotype(Fab,list([parentNode, node]))
        
    return Fab


treeA = {0:[4],4:[0,8,6,5],8:[4,9],9:[8,1,2],
1:[9,3],3:[1],2:[9],6:[4,7],7:[6],5:[4]}
treeB = {0:[9,6,1],9:[0,5],5:[9,7],7:[5,4],4:[7],
6:[0],1:[0,8,2],8:[1,3],3:[8],2:[1]}

individualA = encode(treeA,0)
individualB = encode(treeB,0)

print(individualA)
print(individualB)
