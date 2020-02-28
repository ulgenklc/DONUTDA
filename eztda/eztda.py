# Copyright 2018 Michael Vaiana
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
import matplotlib.pyplot as plt
try:
    import cv2
except:
    has_cv2 = False

"""
eztda is a python package for easy computation of persistent homology.

eztda does the following:
1. Compute the persistent homology of a filter complex
2. Returns persistence pairs and cycle representatives
3. Plot persitence diagrams
4. Build a complex from an Image*s3
5. Build a complex from 2D or 3D point clouds*

*The complex that is built is not the traditional Veitoris-Rips complex.  In the case of an image the complex that is built is a cubical complex in which each pixel is represented as a vertex and its degree is max(image) - pixel_value. Thus bright pixels enter the filtration first.  In the case of 2D or 3D point clouds the complex that is built is the LDA filtered complex which is a filtration of the Delaunay complex of the points.

eztda does NOT do the following:
1. Build a complex for you, if you have a point cloud you need to build your
    own complex from this point cloud
2. Use the fastest/latest algorithms.  eztda uses the original persistence
    algorithm found in Zomordian 2004.  This allows for easy tracking of the cycle representatives.
"""


class Cell:
    """
    Simplicial cell in a filtered complex

    Cell is an abstract representation of a cell in a filtered complex. It has a dimension and degree and boundary.  Typically the boundary will be set by the filtered complex object since it is the filtration which determines how the complex is built and thus what the boundary of a given cell is.


    NOTE: This class should be subclassed for specific applications since the hash function and equality are based only on degree and dimension.  This means that cells of the same dimension can not enter the filtration at the same degree.  Either a linear order needs to be put on the cells and used for degree before initializing the complex or other data needs to be used in a subclass to determine equality and hashing.  For example, if using a point cloud then coordinates can be used for each 0-cell to determine equality, edges and higher dimensional faces can use a strategy based on the coordinates of the verticies.
    """
    def __init__(self,dimension,degree, boundary=None, bnd_index=None):
        """
        Initialize new cell

        Args:
            dimension: dim of cell
            degree: parameter value at which cell enters the filtration
            boundary: list of cells which constitute this cells boundary
            bnd_index: absolute index of boundary cells in filtration
        """
        self.dimension = dimension
        self.degree = degree
        self.boundary = boundary
        self.bnd_index = bnd_index


    def __eq__(self,other):
        return self.dimension==other.dimension and self.degree==other.degree

    def __lt__(a,b):
        if a.degree < b.degree: return True
        if a.degree > b.degree: return False
        return a.dimension < b.dimension

    def __hash__(self):
        return hash((self.dimension, self.degree))

    def __str__(self):
        return "Cell Object\nDim: " +str(self.dimension) +"\nDegree: " + str(self.degree)


class FilteredComplex:
    """
    A filtered complex of cells ordered by degree

    A filtered complex is a list of cells sorted so that the lowest degree is first.  This class is meant to be a template to subclass for specific applications.  See filtrations.py for some subclasses.

    Attributes:
        cells: sorted list of cell objects
        size: number of cells of each dimension
        cell_index: dict which returns position of cell in filtration
        boundary: list of boundary indicies for each cell in filtration
    Methods:
        persistence
        number_of_cells
        cells_by_dim
    """
    def __init__(self, cells):
        """
        Create a new FilteredComplex

        Args:
            cells: list of cells in the complex
        """

        self.cells = sorted(cells) # sort cells for linear filtration
        self.cell_index = dict((self.cells[i],i) for i in range(len(cells)))
        self.size = len(cells)
        #self.boundary = self._getboundary()
        self.max_dim = self.cells[-1].dimension
        self.convert_boundary()

    def boundary2index(self,cell):
        """
        Convert boundary from list of cells to list of indices
        """
        bnd_index = [self.cell_index[c] for c in cell.boundary]
        return sorted(bnd_index,reverse=True)

    def convert_boundary(self):
        """
        Convert all boundaries from cells to indices
        """
        for cell in self.cells:
            cell.bnd_index = self.boundary2index(cell)

    def subcomplex(self,dim):
        """
        Return the subcomplex of all cells up to dimension=dim

        Args:
            dim: maximum dimension of subcomplex
        """
        cells = [cell for cell in self.cells if cell.dimension <=dim]
        return FilteredComplex(cells)
    
    def sub_complex(self,degree):
        """
        Return the subcomplex of all cells greater than the given degree=deg
        
        Args:
            degree: minimum degree of subcomplex
        """
        
        cells = [cell for cell in self.cells if cell.degree>=degree]
        return FilteredComplex(cells)

    def to_degree(self,pairs):
        """
        Convert pairs of indices to pairs of degree of the corresponding cell in
        the filtration
        """
        deg_pairs = np.zeros(pairs.shape)
        cells = self.cells
        for i,pair in enumerate(pairs):
            deg_pairs[i,0] = pair[0]
            deg_pairs[i,1:] = cells[pair[1]].degree, cells[pair[2]].degree
        return deg_pairs

    def sort_persistence(self,pairs,cycles=None, key=None):
        """
        Sort persistence so most persistent is first
        """
        if key is None:
            key = pairs[:,2] - pairs[:,1]
        i_sort = np.argsort(key[::-1])
        pairs = pairs[i_sort]
        if cycles is not None:
            cycles = np.array(cycles)[i_sort]
        return pairs,cycles



    def persistence(self, cyclereps=True, degree=False, nonzero=True):
        """
        Vanilla persistence algorithm (Zomorodian 2004).

        Compute the persistent homology of the filtration. If cycles is True then also track cycle represntatives.  Only compute the homology up to given dimension, if dimension is None compute persistent homology for all dimensions of given complex.  Note the complex should have dimension atleast the given dimension+1 since we need 1 dimension larger to compute dim-1 homology.  For example, if dimension=2 then H_0,H_1,H_2 will be returned but this means that the complex must have dimension = 3 otherwise we can not compute H_2.

        If degree is set to true then the pair (bval,dval) correspond to the degree of the cell when the cycle was born and the degree of the cell where the cycle dies.  If false then (bval,dval) correspond to the index of the cell where the cycle was born and the index of the cell where the cycle dies.  The pairs that are ussually of interest are the degree pairs corresponding to the default, however when degree is set to False (1) the degree pairs can still be recovered and (2) the index of the birth and death cell allows the user to query those cells for specific attributes, for example, what is the exact cell that enters the filtration to start a cycle and which cell kills it.

        If sort is set to true then the results are sorted so that the most persistent pairs are first.

        TODO: Alter for option to only compute H_0

        Args:
            filtration: FilteredComplex object
            degree: (optional boolean) Defaults to True.
            sort: (optional boolean) Defaults to True
        Returns:
            pairs: dictionary of persistence pairs with keys the dimension
            cycles: dictionary of cycle representatives with keys the dimension
        """
        #boundary = self.boundary
        # if dimension is not None and self.max_dim < dimension:
        #     subcomp = self.subcomplex(dimension+1)
        #     pairs,cycles = subcomp.persistence(dimension=dimension,
        #                                         cyclereps=cyclereps,
        #                                         nonzero=nonzero,degree=degree,
        #                                         sort=sort)
        # else:
        dimension = self.max_dim
        k = self.max_dim
        cells = self.cells
        ncell = len(cells)
        pairs = {i:[] for i in range(k)}
        T = np.zeros(ncell,dtype=np.uint64)
        marked = np.zeros(ncell, dtype=bool)
        cycles = {i:[] for i in range(k)}
        cycles_tmp = dict()
        maxdegree= cells[-1].degree

        #find finitely persistent cycles
        for j in range(ncell):
            d,c = self.remove_pivot_rows(cells[j], T, cyclereps=cyclereps)
            if not d:
                marked[j] = True
                if cyclereps:
                    cycles_tmp[j] = c #for trakcing cycle reps
            else:
                i = d[0]
                dim = cells[i].dimension
                if degree:
                    lower = cells[i].degree
                    upper = cells[j].degree
                else:
                    lower = i
                    upper = j

                T[i] = j
                if cells[j].degree-cells[i].degree >0:
                    pairs[dim].append((lower,upper))
                    if cyclereps:
                        cycles[dim].append(cycles_tmp[i]) #track cycle reps

        # find infinitely persistent cycles
        for j in range(ncell):
            if marked[j] and not T[j]:
                dim = cells[j].dimension
                if degree:
                    lower = cells[j].degree
                    upper = maxdegree
                else:
                    lower = j
                    upper = ncell-1

                if ((maxdegree-cells[j].degree>0)
                    and dim <= dimension):
                    pairs[dim].append((lower,upper))
                    if cyclereps:
                        cycles[dim].append(cycles_tmp[j])

            # if sort:
            #     pairs, cycles = sort_persistence(pairs,cycles)

        num_pairs = np.sum([len(pairs[i]) for i in pairs])
        array_pairs = np.zeros((num_pairs,3), dtype=np.uint)
        index = 0
        final_cycles= []
        for i in pairs:
            cycle_index=0
            for pair in pairs[i]:
                array_pairs[index,0] = i
                array_pairs[index,1:] = pair[0],pair[1]
                if cyclereps:
                    final_cycles.append(cycles[i][cycle_index])
                    cycle_index+=1
                index+=1

        #array_cycles = np.array(final_cycles)
        array_cycles = np.array(final_cycles)
        return array_pairs,array_cycles



    def remove_pivot_rows(self, cell, T,cyclereps):
        """
        Helper function for reducing the boundary 'matrix' in the persistence
        computation.  See Zomordian 2004. Our version here also keeps track of the
        cycle representatives.

        Args:
            boundary: list of list of cells sorted by degree and dimension
            j: index to current cells
            T: list of cell indicies (see Zomordian 2004)

        Returns:
            bnd: empty if simplex j adds cycle otherwise gives remaining boundary
                after reducing marked simplicies
            cycle: if current simplex adds cycle then this gives a cycle
                representative otherwise contains just index j
        """
        bnd = cell.bnd_index
        #cycle = []
        cycle = []
        if cyclereps:
            cycle.append(self.cell_index[cell])
        while bnd:
            i = bnd[0]
            if not T[i]: break
            # note: unsorting, converting to set, symmetric difference, converting
            #to list, then resorting.  This is probably not efficient and we should
            #profile this piece of the code
            #UPDATE: stackoverflow seems to think this is as efficient as it gets
            #https://stackoverflow.com/questions/46609962/python-symmetric-difference-sorted-list
            other_bnd = set(self.cells[T[i]].bnd_index)
            bnd = list(set(bnd).symmetric_difference(other_bnd))
            bnd.sort(reverse=True)
            if cyclereps:
                cycle.append(T[i]) #store cycle representative

        return bnd, cycle



###############
# Post Process
###############
# def pairs2degree(cells,pairs):
#     """
#     Convert pairs of indices to pairs of degree of the corresponding cell in
#     the filtration
#     """
#     new_pairs = dict()
#     for k in pairs.keys():
#         new_pairs[k] = [(cells[pair[0]].degree, cells[pair[1]].degree)
#                         for pair in pairs]
#     return new_pairs
#
#
# def persistence_length(pairs):
#     """
#     Return the persistence length for each pair.
#     """
#     length = dict()
#     for k in pairs.keys():
#         length[k] = [pair[1] -pair[0] for pair in pairs[k]]
#     return length

# def remove_zero(pairs,cycles,length=None):
#     """
#     Remove cycles and pairs which have zero persistence
#     """
#     if length is None:
#         length = persistence_length(pairs)
#     filt_pairs = dict()
#     filt_cycles = dict()
#     for k in pairs.keys():
#         filt_pairs[k] = [pair for i,pair in enumerate(pairs[k]) if length[k][i]]
#         filt_cycles[k] = [cycle for i,cycle in enumerate(cycles[k])
#                             if length[k][i]]
#     return filt_pairs,filt_cycles

# def sort_persistence(pairs,cycles,length=None):
#     assert(len(pairs)) == len(cycles)
#     ndims = len(pairs)
#     if length is None:
#         length = persistence_length(pairs)
#     sorted_pairs = dict()
#     sorted_cycles= dict()
#     for i in range(ndims):
#         sort_i = np.argsort(-np.array(length[i]))
#         sorted_pairs[i] = [pairs[i][k] for k in sort_i]
#         sorted_cycles[i] = [cycles[i][k] for k in sort_i]
#     return sorted_pairs,sorted_cycles

# def sort_pairs(pairs,length=None):
#     """
#     Sort the pairs so largest persitence is first
#
#     If length is given then use length as the key for sorting
#     """
#     ndims = len(pairs)
#     if length is None:
#         length = persistence_length(pairs)
#     sorted_pairs = dict()
#     for i in range(ndims):
#         sort_i = np.argsort(-np.array(length[i]))
#         sorted_pairs[i] = [pairs[i][k] for k in sort_i]
#     return sorted_pairs


# def homology(data,dim):
#     return data[k]
#
#
#
# def plot_cycles(img, cycles, **kwargs):
#     if 'ax' in kwargs:
#         ax = kwargs.pop('ax')
#     else:
#         _,ax = plt.subplots()
#
#     ax.imshow(img,**kwargs)
#     overlay = np.zeros((img.shape[0],img.shape[1],4)) #rgba overlay
#     for cycle in cycles:
#         for cell in cycle:
#             overlay[cell[1],cell[0],[0,3]] = 1
#     ax.imshow(overlay)
#     return ax



def homology(pairs,cycles,dim):
    return pairs[pairs[:,0]==dim], cycles[pairs[:,0]==dim]

################
# Plot pdiagram
################
def pdiagram(pairs,dim, **kwargs):
    """
    Plot persistence diagram

    Args:
        pairs: a list of peristence pairs

    """
    inf_pairs = pairs[~np.isfinite(pairs[:,2]),:]
    finite_pairs= pairs[np.isfinite(pairs[:,2]),:]
    max_val = np.max(finite_pairs[:,2]) + 1

    inf_pairs = inf_pairs[inf_pairs[:,0]==dim]
    finite_pairs = finite_pairs[finite_pairs[:,0] ==dim]
    inf_pairs[:,2] = max_val

    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        _,ax = plt.subplots()
    if finite_pairs.size:
        ax.scatter(*zip(*finite_pairs[:,1:]), c='b')
    if inf_pairs.size:
        ax.scatter(*zip(*inf_pairs[:,1:]), c='r')
    _plotdiag(ax)
    return ax


def imread(fn):
    return cv2.imread(fn,0)


# def pdiagram(pairs, **kwargs):
#     """
#     Plot persistence diagram
#
#     Args:
#         pairs: a list of peristence pairs
#
#     """
#     if 'ax' in kwargs:
#         ax = kwargs['ax']
#     else:
#         _,ax = plt.subplots()
#     ax.scatter(*zip(*pairs))
#     _plotdiag(ax)
#     return ax

def _plotdiag(ax):
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims,lims, c='black')

if __name__=='__main__':
    # test simple triangle example for persistence
    # expect to see a 1 dimensional hole with cycle reps ab, bc, ac, aka cycles
    # 5,4,3 in the filtration
    s = """
    a
    |  \\
    |    \\
    |  D   \\
    b ------ c

    degrees:
    a  = 0
    b  = 1
    c  = 2
    ab = 3
    bc = 4
    ac = 5  <---- hole forms
    D  = 10 <---- hole dies
    """
    a = Cell(0,0)
    b = Cell(0,1)
    c = Cell(0,2)
    ab = Cell(1,3)
    bc = Cell(1,4)
    ac = Cell(1,5)
    abc = Cell(2,10)

    cells = [a,b,c,ab,bc,ac,abc]

    # normally we'd let the complex determine the boundary but in this case its
    # easier just to tag each cell with its boundary.
    a.boundary = []
    b.boundary = []
    c.boundary = []
    ab.boundary = [a,b]
    bc.boundary = [b,c]
    ac.boundary = [a,c]
    abc.boundary = [ab,bc,ac]


    f = FilteredComplex(cells)
    pairs,cycles= f.persistence(cyclereps=True)
    print(s)
    print('Persistence pairs:', pairs)
    print('Cycle Representative:',cycles)
    pdiagram(pairs[1]) #h1 diagram
    plt.show()
