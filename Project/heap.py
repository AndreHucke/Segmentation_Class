# % Class to wrap python's heap functions
# % ECE 8396: Medical Image Segmentation
# % Spring 2025
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

import heapq as hq
class heap:
    def __init__(self):
        self.h = []

    def pop(self):
        if len(self.h)>0:
            return hq.heappop(self.h)
        else:
            return None

    def insert(self, x):
        return hq.heappush(self.h, x)

    def isEmpty(self):
        return len(self.h)==0

    def size(self):
        return len(self.h)