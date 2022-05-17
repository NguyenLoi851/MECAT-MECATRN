import os
import math
# CONSTANT
INF = 9999999999
# NUM_NODES = 8
# graph = [[] for i in range(8)]
# graph[0] = [1,2]
# graph[1] = [0,3,4]
# graph[2] = [0,4,5]
# graph[3] = [1,4]
# graph[4] = [3,5,6,7]
# graph[5] = [2,4]
# graph[6] = [4]
# graph[7] = [4]
# report_size = [0,0,0,1,0,1,1,1]
# num_relay = report_size.count(0)
# if report_size[0] ==0:
# 	num_relay+=1

# Import data


def inputToGraph(path):
    f = open(path, 'r')

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
    num_relay = report_size.count(0)
    if report_size[0] == 0:
        num_relay += 1
    return graph, report_size, Q, H, R, NUM_NODE, num_edge, num_relay

# Create graph_hat: a completer graph with out relay node which each edge's weight equals to the min distant from u to v in original graph


def create_graph_hat(graph, report_size, NUM_NODE):
    def BFS(all_pair_shortest_path, source_node, num_nodes, graph):
        queue = []
        check = [False for i in range(num_nodes)]
        check[source_node] = True
        queue.append((source_node, 1))
        count = 1
        while (len(queue) != 0):
            current_node, height = queue[0]
            queue.pop(0)
            for v in graph[current_node]:
                if(check[v] == False):
                    check[v] = True
                    if (source_node, v) not in all_pair_shortest_path:
                        all_pair_shortest_path[(source_node, v)] = height
                        all_pair_shortest_path[(v, source_node)] = height
                    queue.append((v, height+1))

                    count += 1
                    if(count == num_nodes):
                        return all_pair_shortest_path
        return all_pair_shortest_path

    graph_hat = [[] for i in range(NUM_NODE)]
    all_pair_shortest_path = {}
    for source_node in range(NUM_NODE):
        all_pair_shortest_path = BFS(
            all_pair_shortest_path, source_node, NUM_NODE, graph)

    key_list = list(all_pair_shortest_path.keys())

    for u, v in key_list:
        if(report_size[u] != 0 and report_size[v] != 0) or (u == 0 and report_size[v] != 0) or (v == 0 and report_size[u] != 0):
            graph_hat[u].append(v)
        else:
            del all_pair_shortest_path[(u, v)]
    for i in graph_hat:
        i = i.sort()
    # print(graph_hat[0])
    # print()
    return graph_hat, all_pair_shortest_path

# Find (2,3)-LAST tree from G_hat
# (2,3) light approximate shortest-path tree


def LAST(graph, edge, report_size, num_relay, NUM_NODE):
    def find(parent, i):
        if parent[i] == i:
            return i
        return find(parent, parent[i])

    def union(parent, rank, x, y):
        xroot = find(parent, x)
        yroot = find(parent, y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskal(edge):
        res = {}
        e = 0
        parent = [i for i in range(NUM_NODE)]
        rank = [0 for i in range(NUM_NODE)]
        edge = dict(sorted(edge.items(), key=lambda item: item[1]))

        number_of_edge = NUM_NODE*2 - 2
        for node, weight in edge.items():
            x = find(parent, node[0])
            y = find(parent, node[1])
            if x != y:
                e += 2
                res[(node[0], node[1])] = weight
                res[(node[1], node[0])] = weight
                union(parent, rank, x, y)
                if e == number_of_edge:
                    return res

        return res

    # find spt tree
    def SPT_with_weight(graph, source_node, edge, num_relay):
        dist = [INF for i in range(NUM_NODE)]
        dist[source_node] = 0
        check = [False for i in range(NUM_NODE)]
        parent = [-1 for i in range(NUM_NODE)]
        parent[source_node] = -2
        new_edge = {}
        dist_ts = [0 for i in range(NUM_NODE)]
        cnt = 0
        while cnt < NUM_NODE-num_relay:
            index = dist.index(min(dist))
            dist_ts[index] = min(dist)
            check[index] = True
            cnt += 1

            for item in graph[index]:
                if(check[item] == False):
                    if (dist[item] > dist[index] + edge[(index, item)]):
                        dist[item] = dist[index] + edge[(index, item)]
                        parent[item] = index

            dist[index] = INF

        for i in range(1, NUM_NODE):
            if report_size[i] != 0:
                new_edge[(i, parent[i])] = edge[(parent[i], i)]
                new_edge[(parent[i], i)] = edge[(parent[i], i)]

        return new_edge, parent, dist_ts

    # FIND LAST
    def LAST_TREE():
        Tm = [[] for i in range(NUM_NODE)]

        for u, v in kruskal(edge).keys():
            Tm[u].append(v)

        for i in Tm:
            i.sort()

        Ts, parent_Ts, dis_Ts = SPT_with_weight(graph, 0, edge, 0)

        alpha = 2

        def initialize():
            dis = [INF for i in range(NUM_NODE)]
            dis[0] = 0
            parent = [-1 for i in range(NUM_NODE)]
            parent[0] = -1
            check = [False for i in range(NUM_NODE)]
            return dis, parent, check

        def relax(u, v, dis, parent):
            if dis[v] > dis[u] + edge[(u, v)]:
                dis[v] = dis[u] + edge[(u, v)]
                parent[v] = u
            return dis, parent

        def add_path(dis, parent, node):
            if(dis[node] > dis_Ts[node]):
                dis, parent = add_path(dis, parent, parent_Ts[node])
                dis, parent = relax(parent_Ts[node], node, dis, parent)
            return dis, parent

        def dfs(node, dis, parent, check):
            check[node] = True
            if(dis[node] > alpha*dis_Ts[node]):
                dis, parent = add_path(dis, parent, node)
            for v in Tm[node]:
                if check[v] == False:
                    dis, parent = relax(node, v, dis, parent)
                    dis, parent = dfs(v, dis, parent, check)
                    dis, parent = relax(v, node, dis, parent)

            return dis, parent
        dis, parent, check = initialize()
        dis, parent = dfs(0, dis, parent, check)
        tree = [[] for i in range(NUM_NODE)]
        for i in range(1, NUM_NODE):
            if report_size[i] != 0:
                tree[i].append(parent[i])
                tree[parent[i]].append(i)
            # tree[parent[i]].append(i)
        for i in tree:
            i = i.sort()
        return tree
    return LAST_TREE()


def create_graph_hat_hat(graph, last_tree, report_size, NUM_NODE):

    def find_path(graph, adjacent_list, node, graph_hat_hat, graph_hat_hat_edge):
        queue = []
        check = [False for i in range(NUM_NODE)]
        father = [-1 for i in range(NUM_NODE)]
        queue.append(node)
        check[node] = True

        while len(queue) != 0:
            if_break = True
            current = queue[0]
            queue.pop(0)
            for i in graph[current]:
                if check[i] == False:
                    check[i] = True
                    father[i] = current
                    queue.append(i)
            for i in adjacent_list:
                if check[i] == False:
                    if_break = False
                    break
            if if_break:
                break

        for it in adjacent_list:
            item = it
            while True:
                if father[item] != -1:
                    if (father[item], item) not in graph_hat_hat_edge:
                        graph_hat_hat_edge.append((father[item], item))
                        graph_hat_hat_edge.append((item, father[item]))
                        graph_hat_hat[item].append(father[item])
                        graph_hat_hat[father[item]].append(item)
                        item = father[item]
                    else:
                        break
                else:
                    break
        return graph_hat_hat, graph_hat_hat_edge

    graph_hat_hat = [[] for i in range(NUM_NODE)]
    graph_hat_hat_edge = []

    for index, adjacent_list in enumerate(last_tree):
        if len(adjacent_list) > 0:
            graph_hat_hat, graph_hat_hat_edge = find_path(
                graph, adjacent_list, index, graph_hat_hat, graph_hat_hat_edge)
  
    return graph_hat_hat


def SPT(NUM_NODE, graph):
    is_in_tree = [False for i in range(NUM_NODE)]
    queue = []
    queue.append(0)
    is_in_tree[0] = True
    count = 1

    tree = [[] for i in range(NUM_NODE)]

    while len(queue) != 0:
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
    return level


def energy_cost(num_node, tree, t, r, q, report_size, level):
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
                z[key] += z[neighbor]
        res += math.ceil(z[key]/q)

    return res*(t+r)


mecatDataPath = './dataset4mecat/mecat_rn/'

mecatDataFiles = os.listdir(mecatDataPath)

mecatDataFiles = sorted(mecatDataFiles, reverse=False)

for i in mecatDataFiles:
    graph, report_size, Q, T, R, NUM_NODE, num_edge, num_relay = inputToGraph(
        mecatDataPath + i)
    graph_hat, all_pair_shortest_path = create_graph_hat(
        graph, report_size, NUM_NODE)
    last_tree = LAST(graph_hat, all_pair_shortest_path,
                     report_size, num_relay, NUM_NODE)
    graph_hat_hat = create_graph_hat_hat(
        graph, last_tree, report_size, NUM_NODE)
    tree = SPT(NUM_NODE, graph_hat_hat)
    lev = level(tree)
    print(i, energy_cost(NUM_NODE, tree, T, R, Q, report_size, lev))

    # tree1 = SPT(NUM_NODE,graph)
    # lev1 = level(tree1)
    # print(i,energy_cost(NUM_NODE,tree1,T,R,Q,report_size,lev1))
    print()
# graph,report_size,Q,H,R,NUM_NODE,num_edge = inputToGraph(mecatDataPath + mecatDataFiles[0])
# print(mecatDataFiles[0],graph)
