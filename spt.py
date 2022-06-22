import math
# CONSTANT 
# 1) Every minimum descendant tree is a shortest path tree
path = 'C:/Users/huyan/OneDrive/Desktop/data for lstm/dataset4mecat/mecat/m100_4_20.test' 

def inputToGraph(path):		
	f = open(path,'r')
	
	Q = int(f.readline())

	h_and_r = f.readline()[:-1].split(' ')
	H = int(h_and_r[0])
	R = int(h_and_r[1])
	NUM_NODE = int(f.readline())
	
	graph = [[] for i in range(NUM_NODE)]
	report_size = [0 for i in range(NUM_NODE)]

	for i in range(NUM_NODE):
		line = f.readline()[:-1].split()
		report_size[int(line[0])] = int(line[-1])

	num_edge = int(f.readline())

	for i in range(num_edge):
		line = f.readline().split()
		u = int(line[0])
		v = int(line[1])
		if u not in graph[v]:
			graph[u].append(v)
			graph[v].append(u)
		
	f.close()
	for v in graph:
		v = v.sort
	return graph,report_size,Q,H,R,NUM_NODE,num_edge

def SPT(NUM_NODE,s,graph):
	is_in_tree = [False for i in range(NUM_NODE)]
	queue = []
	queue.append(0)
	is_in_tree[0] = True
	count = 1

	tree = [[] for i in range(NUM_NODE)]

	while len(queue) != 0 :
		h = queue[0]
		queue.pop(0)
		for i in graph[h]:
			if is_in_tree[i] == False:
				is_in_tree[i] = True
				queue.append(i)
				count += 1
				tree[h].append(i)
				tree[i].append(h)
				if (count == NUM_NODE):
					return tree
	return tree

def level(tree):
	level = {}
	for i in range(NUM_NODE):
		level[i] = len(tree[i])
	# level = sorted(level.items(),key = lambda x:x[1])
	return level

def energy_cost(num_node,tree,h,r,q,report_size,level):
	z = report_size
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

graph,report_size,Q,H,R,NUM_NODE,num_edge = inputToGraph(path)
tree = SPT(NUM_NODE,report_size,graph)
lev = level(tree)
print(energy_cost(NUM_NODE,tree,H,R,Q,report_size,lev)) 