import numpy as np
import math

class TreeLauncher:
    
    def __init__(self, bfr, l_max, level, id):
        
        # Params
        self.bfr            = bfr
        self.l_max          = l_max
        self.level          = level        
        self.id             = id
                
        # Other
        self.js             = None
        self.num_allocs     = None
        
        if self.id == -1:                            # Root Node
            self.num_allocs = int(self.bfr * ( (1-self.bfr**self.l_max)/(1-self.bfr) ))
            self.js         = math.ceil(np.divide(self.num_allocs, self.bfr))
            print('ROOT NODE   - BFR ', self.bfr, ' JS ', self.js, ' TOTAL ALLOCATORS ', self.num_allocs, ' ID ', self.id)

        elif (self.l_max - self.level) >= 1:         # Branch Node Node
            self.num_allocs = int(self.bfr * ( (1-self.bfr**self.l_max)/(1-self.bfr) ))
            p_js = math.ceil(np.divide(self.num_allocs, self.bfr))
            for lev in range(self.level):
                self.js         = math.ceil((p_js - 1) / self.bfr)
                p_js            = self.js
            print('BRANCH NODE - LEVEL ', self.level, ' JS ', self.js, ' ID ', self.id)
        
        else:                                       # Leaf node
            self.js = -1
            print('LEAF NODE   - LEVEL ', self.level, ' JS ', self.js, ' ID ', self.id)
            
    def node_generator(self):
        if self.level == self.l_max:
            return
        for i in range(self.bfr):
            node_id = self.id + (i * self.js) + 1
            node = TreeLauncher(bfr=self.bfr, l_max=self.l_max, id=node_id, level=self.level+1)
            yield node
            
# End of Class Definition 
