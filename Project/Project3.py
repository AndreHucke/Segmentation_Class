import numpy as np
import heapq as hq
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backend_bases import MouseButton  # Add this import
import numpy as np
from Project2 import *
from myVTKWin import *
import scipy.ndimage as ndi
from skimage import feature
import copy

class graphSearch:
    def __init__(self, node_type):
        self.node_type = node_type
        self.heap = []
        self.lastedge = None
    
    def run(self, edges, seed, endnode=None):
        self.edges = edges
        self.marked = np.zeros(len(edges), dtype=bool)
        self.pointers = -np.ones(len(edges), dtype=np.longlong)
        
        # Add a maximum iteration count to prevent infinite loops
        max_iterations = 100000
        iteration_count = 0
        
        # Initialize priority queue with seed node
        self.heap = []
        hq.heappush(self.heap, self.node_type(child=seed, parent=-1, cost=0))
        
        # Continue until queue is empty or end node is found
        while self.heap and iteration_count < max_iterations:
            iteration_count += 1
            # Get node with lowest cost
            edge = hq.heappop(self.heap)
            
            # Mark the node as visited
            self.mark(edge)
            
            # Store this edge as the last processed edge
            self.lastedge = edge
            
            # If we've reached our target, we're done
            if endnode is not None and edge.child == endnode:
                break
            
            # Find neighbors and add them to priority queue if not visited
            neighbors = self.findNeibs(edge)
            for neib in neighbors:
                if self.isNotMarked(neib):
                    # Set pointer for backtracking
                    self.setPointer(neib, edge.child)
                    # Add to priority queue
                    hq.heappush(self.heap, neib)
        
        # If we reached max iterations, print a warning
        if iteration_count >= max_iterations:
            print("Warning: Graph search reached maximum iterations")
        
        # If we have an endnode, return path to that
        if endnode is not None:
            return self.trace(endnode, seed), self.lastedge.cost
        # Otherwise return path to last processed node
        return self.trace(self.lastedge.child, seed), self.lastedge.cost
    
    def trace(self, endnode, seednode):
        # Start at the end node
        path = [endnode]
        current = endnode
        
        # Follow pointers back to seed node
        while current != seednode:
            current = self.getPointer(current)
            if current == -1:  # Path couldn't be found
                return []
            path.append(current)
        
        return path

class graphSearchLW(graphSearch):
    def __init__(self):
        super().__init__(lwedge)
        self.marked = None
        self.pointers = None
        self.cost = None

    def run(self, edges, seed, endnode=None):
        self.edges = edges
        self.marked = np.zeros(len(edges), dtype=bool)
        self.pointers = -np.ones(len(edges), dtype=np.longlong)
        return super().run(edges, seed, endnode)  # Fixed: passing edges parameter

    def isNotMarked(self, edge):
        return self.marked[edge.child]==False

    def getPointer(self, node):
        return self.pointers[node]

    def findNeibs(self, edge):
        neibs = copy.deepcopy(self.edges[edge.child])
        for n in neibs:
            n.cost += edge.cost
        return neibs

    def setPointer(self, edge, parent):
        self.pointers[edge.child] = parent

    def mark(self, edge):
        self.marked[edge.child] = True

# Add the lwedge class here to ensure everything works together
class lwedge:
    def __init__(self, child=-1, parent=-1, cost=0):
        self.parent = parent
        self.child = child
        self.cost = cost

    def __lt__(self, rhs):
        return self.cost < rhs.cost


class liveWire():
    def __init__(self, img, alpha=0.5, beta=0.1, edges=None):
        self.img = img
        self.alpha = alpha
        self.beta = beta
        self.r, self.c = np.shape(img)
        self.lastseed = None
        self.gs = graphSearchLW()
        self.cntrcur = np.array([])  # Changed to start as empty array instead of list
        self.exit = 0
        self.temporary_line = None
        self.first_point = None  # Track the first point to close the contour
        
        # Create edges if not provided
        if edges is None:
            self.edges = self.compute_edges()
        else:
            self.edges = edges
            
        # Display the image and connect callbacks
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img.T, 'gray')
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        
        # Start interactive loop
        plt.ion()
        plt.show()
        while self.exit == 0:
            plt.pause(0.05)
        plt.ioff()
    
    def compute_edges(self):
        # Create edge weights based on image gradients
        edges = [[] for i in range(self.r * self.c)]
        
        # Compute edge features (Sobel and Canny)
        sobel = np.zeros((self.r, self.c, 4))
        sobel[:,:,0] = ndi.convolve(self.img, np.array([[0,1,2], [-1,0,1], [-2,-1,0]]))  # x-y-/x+y+
        sobel[:,:,1] = ndi.convolve(self.img, np.array([[2,1,0], [1,0,-1], [0,-1,-2]]))  # x-y+/x+y-
        sobel[:,:,2] = ndi.convolve(self.img, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))  # y+/-
        sobel[:,:,3] = ndi.convolve(self.img, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]).T)  # x+/-
        
        # Normalize Sobel features
        sobel = np.abs(sobel)
        for i in range(4):
            sobel[:,:,i] = 1 - sobel[:,:,i]/np.amax(sobel[:,:,i])
        
        # Add Canny edge detection
        canny = 1 - feature.canny(self.img, sigma=1)
        
        # Build edge list
        sq2 = np.sqrt(2)
        for x in range(self.r):
            for y in range(self.c):
                nd = x + y * self.r
                if x > 0:
                    if y > 0:
                        # diagonal x-y- neighbor
                        neib = x - 1 + (y - 1) * self.r
                        edges[nd].append(lwedge(child=neib, parent=nd,
                                            cost=sq2 * (self.beta + self.alpha * canny[x, y] + (1 - self.alpha) * sobel[x, y, 0])))
                    if y < self.c - 1:
                        # diagonal x-y+ neighbor
                        neib = x - 1 + (y + 1) * self.r
                        edges[nd].append(lwedge(child=neib, parent=nd,
                                            cost=sq2 * (self.beta + self.alpha * canny[x, y] + (1 - self.alpha) * sobel[x, y, 1])))
                    # x- neighbor
                    neib = (x - 1 + y * self.r)
                    edges[nd].append(lwedge(child=neib, parent=nd,
                                        cost=self.beta + self.alpha * canny[x, y] + (1 - self.alpha) * sobel[x, y, 3]))
                if x < self.r - 1:
                    if y > 0:
                        # diagonal x+y- neighbor
                        neib = x + 1 + (y - 1) * self.r
                        edges[nd].append(lwedge(child=neib, parent=nd,
                                            cost=sq2 * (self.beta + self.alpha * canny[x, y] + (1 - self.alpha) * sobel[x, y, 1])))
                    if y < self.c - 1:
                        # diagonal x+y+ neighbor
                        neib = x + 1 + (y + 1) * self.r
                        edges[nd].append(lwedge(child=neib, parent=nd,
                                            cost=sq2 * (self.beta + self.alpha * canny[x, y] + (1 - self.alpha) * sobel[x, y, 0])))
                    # x+ neighbor
                    neib = (x + 1 + y * self.r)
                    edges[nd].append(lwedge(child=neib, parent=nd,
                                       cost=self.beta + self.alpha * canny[x, y] + (1 - self.alpha) * sobel[x, y, 3]))
                if y > 0:
                    # y- neighbor
                    neib = (x + (y - 1) * self.r)
                    edges[nd].append(lwedge(child=neib, parent=nd,
                                       cost=self.beta + self.alpha * canny[x, y] + (1 - self.alpha) * sobel[x, y, 2]))
                if y < self.c - 1:
                    # y+ neighbor
                    neib = (x + (y + 1) * self.r)
                    edges[nd].append(lwedge(child=neib, parent=nd,
                                       cost=self.beta + self.alpha * canny[x, y] + (1 - self.alpha) * sobel[x, y, 2]))
        
        return edges

    def display(self):
        # Clear previous plot elements
        if self.ax is not None:
            self.ax.clear()
            self.ax.imshow(self.img.T, 'gray')
            
            # Draw existing contour segments as an open contour in green
            if len(self.cntrcur) > 0:
                self.ax.plot(self.cntrcur[:, 0], self.cntrcur[:, 1], 'g-', linewidth=2)
                
                # If this is after the final right-click, we'll have exited
                if self.exit == 1:
                    # Mark contour as closed with a different color
                    self.ax.plot(self.cntrcur[:, 0], self.cntrcur[:, 1], 'y-', linewidth=2.5)
            
            # Draw current seed point in red
            if self.lastseed is not None:
                x = self.lastseed % self.r
                y = self.lastseed // self.r
                self.ax.plot(x, y, 'ro')
                
            # Draw first point with a different marker
            if self.first_point is not None:
                self.ax.plot(self.first_point[0], self.first_point[1], 'bo', markersize=8)
                
            # Update the canvas
            self.fig.canvas.draw_idle()

    def on_mouse_move(self, event):
        # Check if mouse is in axes and we have a seed point
        if event.inaxes != self.ax or self.lastseed is None:
            return
        
        try:
            # Process mouse movement only if coordinates are valid
            if event.xdata is not None and event.ydata is not None:
                # Get current mouse position
                x = np.round(event.xdata).astype(np.longlong)
                y = np.round(event.ydata).astype(np.longlong)
                
                # Check if mouse position is within image boundaries
                if 0 <= x < self.r and 0 <= y < self.c:
                    currentpos = x + self.r * y
                    
                    # Find path from last seed to current position
                    path_nodes, _ = self.gs.run(self.edges, self.lastseed, currentpos)
                    
                    # Remove previous temporary line if exists
                    if hasattr(self, 'temporary_line') and self.temporary_line is not None:
                        self.temporary_line.remove()
                        self.temporary_line = None
                    
                    # Draw new temporary line if path exists
                    if len(path_nodes) > 0:
                        # Convert path nodes to x,y coordinates
                        path = np.zeros((len(path_nodes), 2))
                        for i in range(len(path_nodes)):
                            path[i, 0] = path_nodes[i] % self.r
                            path[i, 1] = path_nodes[i] // self.r
                            
                        self.temporary_line, = self.ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=1.5)
                        self.fig.canvas.draw_idle()
                        plt.pause(0.05)
        except Exception as e:
            print(f"Error in mouse move: {e}")

    def on_mouse_click(self, event):
        # Using the existing button code (MouseButton.LEFT/RIGHT)
        if event.inaxes != self.ax:
            return
            
        try:
            if event.button == 1:  # Left click - add seed point
                x = np.round(event.xdata).astype(np.longlong)
                y = np.round(event.ydata).astype(np.longlong)
                
                # Check if within image boundaries
                if 0 <= x < self.r and 0 <= y < self.c:
                    current_point = x + self.r*y
                    
                    if self.lastseed is None:
                        # First click - initialize contour
                        self.lastseed = current_point
                        self.first_point = (x, y)
                        # Initialize first point in contour
                        self.cntrcur = np.array([[x, y]])
                        self.gs = graphSearchLW()
                        self.gs.run(self.edges, self.lastseed)
                    else:
                        # Add new segment to contour
                        self.addPath(current_point)
                        self.lastseed = current_point
                        self.gs = graphSearchLW()
                        self.gs.run(self.edges, self.lastseed)
                    
                    self.display()
                    plt.pause(0.05)

            elif event.button == 3:  # Right click - finish contour
                if len(self.cntrcur) > 0 and self.first_point is not None and self.lastseed is not None:
                    # Close the contour by connecting current point to the first point
                    first_seed = self.first_point[0] + self.r * self.first_point[1]
                    self.addPath(first_seed)
                    self.exit = 1
                
                self.display()
                plt.pause(0.05)
                
        except Exception as e:
            print(f"Error in mouse click: {e}")

    def addPath(self, endpoint):
        try:
            # Get the path from current lastseed to endpoint
            path_nodes, _ = self.gs.run(self.edges, self.lastseed, endpoint)
            
            if len(path_nodes) > 0:
                # Convert path nodes to x,y coordinates
                path = np.zeros((len(path_nodes), 2))
                for i in range(len(path_nodes)):
                    path[i, 0] = path_nodes[i] % self.r
                    path[i, 1] = path_nodes[i] // self.r
                
                # Add to contour
                if len(self.cntrcur) == 1:
                    # First segment - append the path excluding the first point
                    # (which is already in cntrcur as the first point)
                    if len(path) > 1:
                        self.cntrcur = np.concatenate((self.cntrcur, path[1:]), axis=0)
                else:
                    # Subsequent segments - append path excluding first point
                    if len(path) > 1:
                        self.cntrcur = np.concatenate((self.cntrcur, path[1:]), axis=0)
                    
        except Exception as e:
            print(f"Error adding path: {e}")
