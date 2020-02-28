import numpy as np
#import cv2
import matplotlib.pyplot as plt


def twist_persistence(filtration):
    """
    Find the persistence pairs of a filtered complex

    The algorithm is the 'twist' algorithm in Chen and Kerber, Persistent
    Homology Computation with a Twist.

    Args:
        filtration: filtered complex class
    Returns:
        pairs: pair of birth-death pairs for each dimension
    """
    dim = filtration.dimension
    num_cells = filtration.num_cells()
    max_dim = filtration.max_dim()
    lookup = {}
    pair_lookup = {}

    # twist algorithm
    # Chen and Kerber, Persistent Homology Computation with a Twist
    num_thrown = 0
    columns = set([i for i in range(num_cells)])
    for d in range(max_dim, 0,-1):
        visted = []
        for cur_col in columns:
            if dim[cur_col] == d:
                visted.append(cur_col)
                low = filtration.get_max_index(cur_col)
                while low and low in lookup:
                    filtration.add_to(lookup[low], cur_col)
                    low = filtration.get_max_index(cur_col)
                if low:
                    lookup[low] = cur_col
                    pair_lookup[cur_col] = low
                    filtration.clear(low)
                    num_thrown +=1
        for col in visted:
            columns.remove(col)


    pairs = np.zeros((len(filtration.matrix)-num_thrown,3), dtype=np.uint)
    values = set(lookup.values())
    #pairs = np.zeros((filtration.num_cells(),3), dtype=np.uint)
    i = 0
    for index in range(num_cells):
        if not filtration.matrix[ index ] and index not in lookup:
    #        # essential classes
            sigma_i = index
            pairs[i,:] = dim[sigma_i], sigma_i, num_cells # max value
            i+=1
        if filtration.matrix[ index ] and index in values:
            # finite classes
            sigma_i = pair_lookup[index]
            sigma_j = index
            pairs[i,:] = dim[sigma_i], sigma_i, sigma_j
            i+=1

    return pairs




class Filtration():
    """
    A filtered complex

    This is the input to the persistence algorithm.
    """

    def __init__(self, matrix, dimension,degree=None):
        """
        Initialize filtration object

        Args:
            matrix: dict of sets of boundary indicies
            dimension: numpy array of dimension
        """
        self.matrix = matrix
        self.dimension = dimension
        if degree is None:
            degree = range(len(matrix))
        self.degree = degree
        self.maxval = np.max(degree)

    def num_cells(self,dim=None):
        """
        Return the number of cells in the complex

        Args:
            dim (optional): if set returns number of cells of this dimension
        """
        if dim is None:
            return matrix.shape[0]
        else:
            return np.sum(self.dimension==dim)

    # function for reduction algorithm
    def get_max_index(self,index):
        if self.matrix[ index ]:
            return max(self.matrix[ index ])
        return []

    def max_dim(self):
        return np.max(self.dimension)

    def add_to(self,source,target):
        #column reduction
        s = self.matrix[ source ]
        t = self.matrix[ target ]
        self.matrix[ target ] = t.symmetric_difference(s)

    def clear(self, index):
        self.matrix[ index ] = []

    # post processing functions
    def pairs_to_degree(self,pairs, maxval=np.inf):
        """
        Convert pairs from natural index to degree

        Args:
            pairs: Nx3 numpy array, dim, birth index, death index
        Returns:
            deg_pairs: Nx3 numpy array, dim, birth degree, death degree
        """
        deg_pairs = np.zeros(pairs.shape)
        for i, pair in enumerate(pairs):
            deg_pairs[i,0] = pair[0]
            deg_pairs[i,1] = self.degree[pair[1]]
            if pair[2]< self.num_cells():
                deg_pairs[i,2] = self.degree[pair[2]]
            else:
                deg_pairs[i,2] = maxval #self.maxval
        return deg_pairs




    def persistence(self,pairs,mode='degree', **kwargs):
        """
        Find persistence length of pairs
        """
        if mode=='degree':
            pairs = self.pairs_to_degree(pairs)
        return persistence(pairs, **kwargs)



    def finite_pairs(self,pairs,dim=None, mode='degree',**kwargs):
        """
        Return nonzero finitely persistent pairs
        """

        if dim is not None:
            pairs = pairs[pairs[:,0]==dim]
        if mode=='degree':
            deg_pairs = self.pairs_to_degree(pairs)
            key = persistence(deg_pairs, **kwargs)
        else:
            key = persistence(pairs, **kwargs)

        index  = key !=0 & np.isfinite(key)
        return pairs[index,:]


def dimension(pairs,dim):
    return pairs[pairs[:,0]==dim]



def persistence(pairs, dim=None, norm=False, maxval=None):
    """
    Persistence length of all pairs
    """

    if dim is not None:
        pairs = pairs[pairs[:,0]==dim]
    pers = pairs[:,2] - pairs[:,1]
    if norm:
        if maxval is None:
            maxval = np.max(pers[np.isfinite(pers)])
        pers[~np.isfinite(pers)] = maxval
        pers = pers/maxval
    return pers

def sort_pairs(pairs, key=None):
    """
    Sort the pairs by their persistence length or by another key

    The key may be useful when you want to sort index pairs by their degree persistence.
    """
    if key is None:
        key = pairs[:,2] - pairs[:,1]
    index = np.argsort(-key)
    return pairs[index]

def remove_zero(pairs, key=None):
    """
    Remove pairs with zero persistence
    """
    if key is None:
        key = persistence(pairs)
    return pairs[key!=0]


class ImageFiltration(Filtration):
    """
    Data structure for superlevel filtration of a grayscale image
    """
    def __init__(self,img, maxval=None):
        """
        Build a filtered complex from the given grayscale image

        Args:
            img: MxN numpy array of unsigned 8bit ints
        """
        M,N = img.shape
        self.img = img
        self.rows = M
        self.cols = N
        self.size = 4*N*M - 2*N - 2*M + 1
        if maxval is None:
            maxval = np.max(img)
        self.maxval = maxval
        self.build_complex()


    def build_complex(self):
        """
        Perform a linear pass through the image to build the filtered complex

        This method loops through all the cells in the cubical image complex, recording their degree, dimension, and boundary.  The image is iterated through left to right, top to bottom, first populating vertices, then edges, then 2-cells.  See :func:boundary for more details.
        """
        matrix = {} # boundary matrix
        dimension = np.zeros(self.size, np.uint8)
        degree = np.zeros(self.size)
        location = np.zeros((self.size,2), np.int)

        # Linear pass through all the simplicies
        i = 0
        # process the vertices
        dim=0
        for x in range(self.rows):
            for y in range(self.cols):
                matrix[i] = [] # vertex
                dimension[i] = dim
                degree[i] = self.degree(x,y,dim)
                location[i] = x,y
                #queue.append[(i,0)] # cell is index and degree
                i+=1

        # process edges
        # edges are processed so that at each pixel location we only consider
        # the edge running right and the edge running down from the current
        # location
        dim=1
        for x in range(self.rows):
            for y in range(self.cols):
                # right edge
                if y < self.cols-1:
                    matrix[i] = self.boundary(x,y,dim, 'right')
                    dimension[i] = dim
                    degree[i] = self.degree(x,y,dim, 'right')
                    location[i] = x,y
                    i+=1
                # down edge
                if x < self.rows-1:
                    matrix[i] = self.boundary(x,y,dim, 'down')
                    dimension[i] = dim
                    degree[i] = self.degree(x,y,dim, 'down')
                    location[i] = x,y
                    i+=1
        # process 2 cells
        # 2 cells are processed similarly to the edges, at each pixel location # we take the two cell to the bottom right of the vertex
        dim=2
        for x in range(self.rows-1):
            for y in range(self.cols-1):
                matrix[i] = self.boundary(x,y,dim)
                dimension[i] = dim
                degree[i] = self.degree(x,y,dim)
                location[i] = x,y
                i+=1

        #sanity check
        assert(i==self.size)

        # Need to sort results so that cells of lower degree appear first
        m,deg,dim, loc = self.convert(matrix,degree,dimension, location)
        self.matrix=m
        self.degree = deg
        self.dimension=dim
        self.location = loc


    def boundary(self, x,y,dim, direction=None):
        """
        Determine indices for the boundary of the cell at current location

        The image is processed left to right top to bottom. The first thing that happens is that all 0 cells are put into 1-1 correspondece with the image pixels.

        0 --------- 1 --------- 2 --------- 3 -------- ... -------- N-1
        |           |           |           |                        |
        N -------- N+1 ------- N+2 ------- N+3 ------- ... ------- 2N-1
        |           |           |           |                        |


        |           |           |           |                        |
        (M-1)N -- (M-1)N+1 -- (M-1)N+2 -- (M-1)N+3 --  ...   ----  MN-1

        Once the vertices are added the edges are added by running through each vertex in the natural order (0 to MN-1).  At vertex i, two edges get added: the edge running to the right of the vertex and the edge running down from the vertex, that is  RIGHT: i -- i+1, DOWN: i -- i+N.  Finally the 2 cells get added so in the natural linear order left to right top to bottom.

        Thus given an x,y location and the dimension of the current cell,
        determine the linear index of the boundary.  The linear index is the
        index at which the cell is added to the filtration according to the
        rules/ordering described above.

        Adding 2 cells is awkward.  First we need to offset by the number of vertices since the boundary of the 2 cells are edges.  Then we need to find the linear index in which edges are added, but they are added in an alternating fashion making it difficult.  Finally we have to correct for when x,y are at the edges of the complex.  The top and left edge are adjacent in indexing by the way edges are added.  Say i, i+1 are the indices for the top and left edge.  The right edge is i+3 if we are not in the last 2 cell column since the top edge of the next cell gets added first, otherwise its i+2 (since there is no next cell to the right).  Finally the bottom is i + row_jump where row_jump is the number of edges that get added in each row.  If we are in the very last row we need to account for this since in this case there were no down edges added so the bottom edges just increase by one, making row_jump too large.

        Args:
            x,y: pixel location
            dim: dimesion of current cell
            direction: direction of edge, either 'right' or 'left'
        """
        if dim == 0: return []
        linear_index = x*self.cols + y
        if dim == 1:
            if direction == 'right':
                bnd = set([linear_index, linear_index+1])
            if direction == 'down':
                bnd = set([linear_index, linear_index + self.cols])
        if dim == 2:
            offset = self.num_cells(dim=0) # number of vertices
            row_jump = 2*(self.cols-1) + 1
            top = offset + row_jump*x + 2*y
            left = top + 1
            right = top + 3 if y < self.cols -2 else top + 2
            b = top + row_jump
            bottom = b if x < self.rows-2 else b - y
            bnd = set([top,left, right, bottom])
        return bnd

    def degree(self, x,y,dim, direction=None):
        """
        Compute the degree (brightness) of the corresponding cell

        Since we perform a superlevel filtration (thresholding from highest to lowest values) the degree is actually the maximum value (e.g. 255) minus the actual brightness so that way bright pixels have lower degree (aka are added to the filtration first)
        """
        img = self.img
        if dim ==0: return self.maxval - img[x,y]
        if dim ==1:
            if direction=='right':
                deg = self.maxval - min(img[x,y], img[x,y+1])
            if direction=='down':
                deg = self.maxval - min(img[x,y], img[x+1,y])
        if dim==2:
            deg= self.maxval - min(img[x,y], img[x+1,y],
                                     img[x,y+1],img[x+1,y+1])
        return deg


    def num_cells(self,dim=None):
        """
        Return the number of cells in a given dimension
        """
        if dim is None:
            return self.size
        if dim==0:
            return self.rows*self.cols
        if dim==1:
            return (self.rows-1)*self.cols + self.rows * (self.cols-1)
        if dim==2:
            return (self.rows-1)*(self.cols-1)


    def convert(self,matrix, degree,dimension, location):
        """
        Convert indices from natural order to degree order

        The complex is built so that cells enter the filtration according to dimension and then left to right top to bottom 'natural' indexing.  These need to be converted to degree indexing so that cells of lower degree enter the filtration first.
        """

        index = np.argsort(degree,kind='mergesort') # need stable sorting
        inverse = {v:i for i,v in enumerate(index)}

        deg = degree[index]
        dim = dimension[index]
        loc = location[index]
        for k in matrix.keys():
            s = set([inverse[m] for m in matrix[k]])
            matrix[k] = s
        matrix = {i:matrix[index[i]] for i in range(len(matrix))}
        return matrix, deg, dim, loc


    def homology_image(self,pairs,sigma=5, width=50, mode='birth', kern_norm=False):
        """
        Convert homological information into an image

        Possibly use this image as another channel and feed to CNN
        """
        himg = np.zeros(self.img.shape)
        pers = self.persistence(pairs, norm=True)
        loc = self.pairs_to_location(pairs)
        if mode=='birth':
            centers = loc[:,1:3]
        else:
            centers = loc[:,3:5]

        for i,c in enumerate(centers):
            p = pers[i]
            kern = kernel(0,sigma=sigma*p   ,width=width, height=p, norm=kern_norm)
            x = np.arange(c[0]-width, c[0]+width+1, dtype=np.int)
            y = np.arange(c[1]-width, c[1]+width+1, dtype=np.int)
            x = np.clip(x,0,himg.shape[0]-1,x)
            y - np.clip(y,0,himg.shape[1]-1,y)
            X,Y = np.meshgrid(x,y)

            himg[X,Y] += kern
        return himg


    def pairs_to_location(self,pairs):
        """
        Convert pairs to birth-death location

        Given a list of index pairs convert each pair to the x,y location of its birth simplex and death simplex relative to the image.

        Args:
            pairs: (Nx3 array) dim, birth index, death index
        Returns:
            locs: (Nx5 array) dim, bx,by,dx,dy
        """
        locs = np.zeros((pairs.shape[0], 5))
        num_cells = self.num_cells()
        for i,pair in enumerate(pairs):
            birth = self.location[pair[1]]
            death = self.location[pair[2]] if pair[2]<num_cells else [-1,-1]
            locs[i,0] = pair[0]
            locs[i,1:3] = birth
            locs[i,3:5] = death
        return locs



    def scatter_birth(self,pairs,dim=0, **kwargs):
        """
        Scatter the birth points on an axis
        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            _,ax = plt.subplots()
        loc = self.pairs_to_location(pairs)
        loc = loc[loc[:,0] == dim]
        ax.scatter(loc[:,2], loc[:,1], **kwargs)

    def scatter_death(self,pairs,dim=1,**kwargs):
        """
        Scatter the birth points on an axis
        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            _,ax = plt.subplots()

        loc = self.pairs_to_location(pairs)
        loc = loc[loc[:,0] == dim]

        # one H_0 class lasts to infinity aka has no death point
        if dim == 0: loc = loc[loc[:,3] !=-1]
        ax.scatter(loc[:, 4], loc[:,3], **kwargs)


def kernel(mean,sigma, width, height, norm=True):
    """
    Rescaled guassian kernel
    """
    if norm:
        denom = np.sqrt(2*np.pi*sigma*sigma)
    else:
        denom = 1

    probs=[height*np.exp(-z*z/(2*sigma*sigma))/denom
            for z in range(-width,width+1)]
    return np.outer(probs,probs)

def twist_pdiagram(pairs,dim, **kwargs):
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

def _plotdiag(ax):
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims,lims, c='black')


if __name__=='__main__':
    matrix = {  0:[], 1:[],2:[], # vertices
                3:set([0,1]), 4:set([1,2]), 5:set([0,2]), # edges
                6:set([3,4,5]) # 2 cell
             }
    dim = np.array([0,0,0,1,1,1,2],dtype=np.uint) # dimension of each cell

    F = Filtration(matrix,dim)
    pairs = twist_persistence(F)
    print(pairs)
    twist_pdiagram(pairs,1)
    plt.show()


    img = cv2.imread('../demo/single_neuron.png',0)
    imF = ImageFiltration(img)
    pairs = twist_persistence(imF)
    twist_pdiagram(pairs,1)
    plt.show()
