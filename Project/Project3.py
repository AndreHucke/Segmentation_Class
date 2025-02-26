import heapq
import numpy as np
import matplotlib.pyplot as plt

class lwedge:
    """Class representing an edge with parent, child, and cost attributes."""
    def __init__(self, child, parent, cost):
        self.child = child
        self.parent = parent
        self.cost = cost

class graphSearch:
    """Base class for graph search algorithms."""
    def __init__(self, node_type=None):
        """Initialize graph search with optional node type."""
        self.node_type = node_type
        self.parents = {}
        self.costs = {}
        
    def run(self, seed, endnode=None):
        """
        Run graph search algorithm from seed to endnode (if specified).
        Returns path and cost to endnode.
        """
        # This should be implemented by subclasses
        pass
        
    def trace(self, nd, seed):
        """Trace path from nd back to seed using stored parents."""
        path = [nd]
        while nd != seed:
            if nd not in self.parents:
                return None  # No path exists
            nd = self.parents[nd]
            path.append(nd)
        return path

class graphSearchLW(graphSearch):
    """Graph search implementation for lightweight edge representation."""
    def __init__(self):
        """Initialize graph search for lightweight edges."""
        super().__init__(node_type=lwedge)
        
    def run(self, edges, seed, endnode=None):
        """
        Run Dijkstra's algorithm from seed to endnode (if specified).
        Returns path and cost to endnode or to all nodes.
        """
        # Initialize data structures
        self.parents = {}
        self.costs = {seed: 0}
        visited = set()
        priority_queue = [(0, seed)]
        
        while priority_queue:
            # Get node with lowest cost
            current_cost, current_node = heapq.heappop(priority_queue)
            
            # Skip if we've already processed this node
            if current_node in visited:
                continue
                
            # Mark as visited
            visited.add(current_node)
            
            # If we reached the target node, we're done
            if endnode is not None and current_node == endnode:
                break
                
            # Explore neighbors
            for edge in edges[current_node]:
                child = edge.child
                new_cost = current_cost + edge.cost
                
                # Update cost if we found a better path
                if child not in self.costs or new_cost < self.costs[child]:
                    self.costs[child] = new_cost
                    self.parents[child] = current_node
                    heapq.heappush(priority_queue, (new_cost, child))
        
        # If endnode is specified, return path and cost to it
        if endnode is not None:
            if endnode in self.costs:
                path = self.trace(endnode, seed)
                return path, self.costs[endnode]
            return None, float('inf')  # No path found
        
        # Return empty lists if no endnode was specified
        return [], 0

class liveWire:
    """Interactive contour drawing class using graphSearch."""
    def __init__(self, image, edges=None):
        """
        Initialize liveWire with image and edge weights.
        
        Args:
            image: 2D numpy array representing the image
            edges: List of edge lists for graph search
        """
        self.image = image
        self.edges = edges
        self.r, self.c = np.shape(image)
        self.gs = graphSearchLW()
        
        # Initialize variables for tracking contour
        self.seed = None
        self.last_point = None
        self.cntrcur = []  # Current contour segments
        self.cntrpts = []  # List of confirmed contour points
        self.active = False
        self.complete = False
        
        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.img_display = self.ax.imshow(self.image.T, cmap='gray')
        self.contour_line, = self.ax.plot([], [], 'r-')
        self.points_line, = self.ax.plot([], [], 'go')
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)
        
        plt.title("Left click to add points, right click to complete")
        plt.show()
        
    def onclick(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # Ensure clicks are within image boundaries
        if x < 0 or x >= self.r or y < 0 or y >= self.c:
            return
            
        # Convert 2D coordinates to 1D node index
        node = x + y * self.r
        
        # Left click - add point to contour
        if event.button == 1:
            if not self.active:
                # First click - initialize contour
                self.seed = node
                self.last_point = node
                self.cntrpts.append((x, y))
                self.active = True
            else:
                # Add point and segment to contour
                path, cost = self.gs.run(self.edges, self.last_point, node)
                if path:
                    # Reverse the path so it goes from start to end
                    path.reverse()
                    # Convert path nodes to 2D coordinates
                    path_coords = np.array([(n % self.r, n // self.r) for n in path])
                    self.cntrcur.append(path_coords)
                    self.last_point = node
                    self.cntrpts.append((x, y))
                    self.update_display()
                    
        # Right click - complete contour
        elif event.button == 3 and self.active:
            # Close the contour by connecting back to the first point
            if self.seed != self.last_point:
                path, cost = self.gs.run(self.edges, self.last_point, self.seed)
                if path:
                    path.reverse()  # Reverse the path
                    path_coords = np.array([(n % self.r, n // self.r) for n in path])
                    self.cntrcur.append(path_coords)
                    self.update_display()
            
            self.active = False
            self.complete = True
            print("Contour completed")
            
    def onmove(self, event):
        """Handle mouse movement to show potential contour segment."""
        if not self.active or event.inaxes != self.ax:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # Ensure coordinates are within image boundaries
        if x < 0 or x >= self.r or y < 0 or y >= self.c:
            return
            
        # Calculate potential path to current mouse position
        node = x + y * self.r
        path, _ = self.gs.run(self.edges, self.last_point, node)
        
        if path:
            # Reverse the path
            path.reverse()
            # Create temporary display of potential path
            temp_path = np.array([(n % self.r, n // self.r) for n in path])
            self.update_display(temp_path)
            
    def update_display(self, temp_path=None):
        """Update the display with current contour and temporary path."""
        # Plot confirmed contour segments
        x_points = []
        y_points = []
        
        for segment in self.cntrcur:
            x_points.extend(segment[:, 0])
            y_points.extend(segment[:, 1])
            
        # Add temporary path if available
        if temp_path is not None:
            x_points.extend(temp_path[:, 0])
            y_points.extend(temp_path[:, 1])
            
        self.contour_line.set_data(x_points, y_points)
        
        # Plot anchor points
        x_anchors = [p[0] for p in self.cntrpts]
        y_anchors = [p[1] for p in self.cntrpts]
        self.points_line.set_data(x_anchors, y_anchors)
        
        self.fig.canvas.draw_idle()
