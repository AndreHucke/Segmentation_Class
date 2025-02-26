from Project3 import *
# Assuming 'Project3' contains lwedge, graphSearchLW classes as demo'd
# in class as well as your base graphSearch (part 1)

edge = lwedge(child=0, parent=-1, cost=1)
gs = graphSearch(node_type=lwedge)
#these should instantiate without error

# Creating small test search with correct answer given below
gs = graphSearchLW()
edges = [[] for i in range(5)]
edges[0].append(lwedge(child=1, parent=0, cost=1))
edges[0].append(lwedge(child=2, parent=0, cost=4))

edges[1].append(lwedge(child=2, parent=1, cost=3))
edges[1].append(lwedge(child=3, parent=1, cost=0.5))

edges[2].append(lwedge(child=3, parent=2, cost=4))
edges[2].append(lwedge(child=4, parent=2, cost=1))

edges[3].append(lwedge(child=4, parent=3, cost=3))
edges[3].append(lwedge(child=2, parent=3, cost=1))

edges[4].append(lwedge(child=1, parent=4, cost=1))
edges[4].append(lwedge(child=3, parent=4, cost=1))

pth, cost = gs.run(edges,0, 3)
print(pth)
# [3, 1, 0]
print(cost)
# 1.5

gs.run(edges,0)
pth = gs.trace(4,0)
print(pth)
# [4, 2, 3, 1, 0]
