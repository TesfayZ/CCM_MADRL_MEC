import numpy


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]
    # check if data exista
    def existornot(self, instance, nextinstance):
        collectall = []
        #print("self.n_entries",self.n_entries,self.data )
        if self.n_entries==0:
            return [],[]   
        sz = self.n_entries
        for i in range(sz):
            collectall.append(self.data[i])
        #print("collectall",collectall)
        d = numpy.array(collectall, dtype=object).transpose()
        #print("d",d)
        thestate = numpy.vstack(d[0])
        theNstate = numpy.vstack(d[2])
        #print("thestate",thestate)
        indexofin = numpy.where((thestate==instance).all(axis=1))
        indexofNin = numpy.where((theNstate==nextinstance).all(axis=1))
        #if indexofin[0].size>1:
            #print("indexofin",indexofin, thestate)
            #quit()
        return indexofin,indexofNin
    
    # store priority and sample
    def add(self, p, data, dataindex, update):
        if update == 0:
            idx = self.write + self.capacity - 1

            self.data[self.write] = data
            self.update(idx, p)

            self.write += 1
            if self.write >= self.capacity:
                self.write = 0

            if self.n_entries < self.capacity:
                self.n_entries += 1
        else:
            #dataindx = idx - self.capacity + 1
            dcheck = self.data[dataindex]
            dcheck = numpy.array(dcheck, dtype=object).transpose()
            #print("dd1", data[1])
            #print("dd2", dcheck[0][1])
            if  data[1] > dcheck[0][1]: # reward update
                #print("dd", dcheck[0][1],dataindex[0][0],self.write)
                self.data[dataindex] = data #data  [0][0]
                dcheck = self.data[dataindex]
                dcheck = numpy.array(dcheck, dtype=object).transpose()
                #print("dd2after", dcheck[0][1])
                idx = dataindex + self.capacity - 1
                self.tree[idx] = 0 #reset priority
                self.update(idx, p)
            '''   
            #elif data[1] == dcheck[0][1]:# if they have the same reward, then check if their next state is different
                #print("reward", data[1], dcheck[0][1])
            if not (data[3] == dcheck[0][3]).all():# save it if their next states are different
                #print("different next items/ states", data[3], dcheck[0][3])
                idx = self.write + self.capacity - 1

                self.data[self.write] = data
                self.update(idx, p)

                self.write += 1
                if self.write >= self.capacity:
                    self.write = 0

                if self.n_entries < self.capacity:
                    self.n_entries += 1
            '''            
    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
