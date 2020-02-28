import numpy as np
import matplotlib.pyplot as plt
import random
try:
    import cv2
    cv2_available = True
except:
    cv2_available = False


try:
    from CGAL.CGAL_Kernel import Point_2
    from CGAL.CGAL_Triangulation_2 import Delaunay_triangulation_2
    cgal_available = False
except:
    cgal_available = False

from eztda import Cell, FilteredComplex, homology

######################################################
# Image Specific Filtration
######################################################
class PixelCell(Cell):
    """
    Subclass of Cell specified for building a filtered complex using the pixels
    in an image. Here degree is the value of the pixel

    Args:
        x: x index associated to cell (relative to image)
        y: y index associated to cell
        celltype: 0,1,2,3 - 0 cell, horz 1 cell, vert 1 cell, 2 cell
        degree: corresponding pixel value
    """
    def __init__(self,x,y,celltype, degree=None):
        self.x = x
        self.y = y
        self.type = celltype
        if celltype not in [0,1,2,3]: raise ValueError('Improper Cell Type')
        dimension = 1
        if celltype == 0: dimension = 0
        if celltype == 3: dimension = 2

        Cell.__init__(self,dimension, degree)

    # helpers
    def _has_horizontal_extent(self):
        "True if horizontal edge or 2 cell"
        return self.type == 1 or self.type == 3

    def _has_vertical_extent(self):
        "True if vertical edge or 2 cell"
        return self.type ==2 or self.type == 3

    # builtins
    def __eq__(self,b):
        "For pixel cells equality is based on position and type "
        return (self.x,self.y,self.type) == (b.x, b.y, b.type)

    def __lt__(a,b):
        """
        Sort Pixel Cell by degree, x,y position and cell type.
        """
        if a.degree is None or b.degree is None:
            raise ValueError('Cant Sort cells without degree')
        if a.degree < b.degree: return True
        if a.degree > b.degree: return False
        if a.type < b.type: return True
        if a.type > b.type: return False
        if a.y < b.y: return True
        if a.y > b.y: return False
        if a.x < b.x: return True
        if a.x > b.x: return False
        return False

    def __hash__(self):
        return hash((self.x, self.y, self.type))

    def __str__(self):
        return 'PixelCell\nx:' + str(self.x) + ' y:' + str(self.y) + '\ntype:' + str(self.type) + '\nDegree:'+str(self.degree)





class ImageComplex(FilteredComplex):
    """
    Convert a grayscale image to a FilteredComplex object. The filtration is by
    superlevel sets with filtration parameter ranging from max(image) to 0.
    The filtration is performed by considering each pixel a vertex.  Edges and
    2-cells are constructed by giving them the value of the minimum of their
    coboundary.  Since the filtration is descending through pixel values this
    guarantees that at each step we have a closed subcomplex.

    Parameters:
        image: NxM numpy array (grayscale image)

    Methods:
        boundary(self,cell): return boundary cells of cell
        _isvalid(self,cell): check this cell is contained in image
    """

    def __init__(self,image, maxval=None):
        """
        Range through each pixel of image and create the corresponding cells.
        """
        image = np.array(image,dtype=float)
        if maxval is None:
            maxval = np.max(image)
        self.image = image
        N,M = np.shape(image)
        self.N = N-1 # superlevel filtration, size of complex in 2 cell span
        self.M = M-1
        # build complex
        cells = []
        ntype = 4
        for x in range(self.N+1):
            for y in range(self.M+1):
                for t in range(ntype):
                    # assign minimum value of coboundary
                    if t == 0: val = image[x,y]
                    if t == 1:
                        if y != self.M:
                            val = min(image[x,y], image[x,y+1])
                    if t == 2:
                        if x != self.N:
                            val = min(image[x,y], image[x+1,y])
                    if t == 3:
                        if y != self.M and x != self.N:
                            val = min(image[x,y], image[x+1,y], image[x,y+1], image[x+1,y+1] )

                    cell = PixelCell(x,y,t,maxval-val) # descending filtration
                    if self._isvalid(cell):
                        cell.boundary = self.boundary(cell)
                        cells.append(cell)
        FilteredComplex.__init__(self,cells)


        # Number of cells
        # verticies (N+1)*(M+1)
        # + edges   N*(M+1) + (N+1)*M
        # + 2cells  N*M
        # ---------------
        # 4NM + 2N + 2M + 1
        #self.size = 4*self.N*self.M + 2*self.N + 2*self.M +1
        #assert(self.size == len(self.cells))


    def boundary(self,pixcell):
        """
        Compute the boundary of a pixelcell in the ImageComplex
        """
        bnd = []
        newtype = 0
        x,y = pixcell.x, pixcell.y
        if pixcell.type == 0: return bnd

        if pixcell._has_horizontal_extent():
            if pixcell.type == 3: newtype = 2
            left_bnd = PixelCell(x,y, newtype)
            right_bnd = PixelCell(x,y+1, newtype)
            if self._isvalid(left_bnd): bnd.append(left_bnd)
            if self._isvalid(right_bnd): bnd.append(right_bnd)
        if pixcell._has_vertical_extent():
            if pixcell.type == 3: newtype = 1
            below_bd = PixelCell(x,y, newtype)
            above_bd = PixelCell(x+1,y,newtype)
            if self._isvalid(below_bd): bnd.append(below_bd)
            if self._isvalid(above_bd): bnd.append(above_bd)
        return bnd


    def _isvalid(self,cell):
        if cell.y > self.M or cell.x > self.N: return False
        if cell.y == self.M and cell._has_horizontal_extent(): return False
        if cell.x == self.N and cell._has_vertical_extent(): return False
        return True

    # Post process homology pairs
    def pairs_to_points(self,pairs):
        """
        Convert pairs as indices to pairs as points

        The persistent homology computation has the option of returning the persistent pairs as indices into the cells of the filtration.  We convert these indices back into the actual x,y coordinates of the cell from which they came allowing us to visualize the birth-death points in an image for a particular cycle.

        Args:
            pairs: results from persistence algorithm (with option degree=False)
        Returns:
            points: dictionary so that points[k] are the points of the k-dim
                    homology and points[k][i] = (b.x,b.y,d.x,d.y) is the ith pair with birth and death coordinates (b.x,b.y), and (d.x,d.y) respectively.
        """
        cells = self.cells
        points = np.zeros((pairs.shape[0],5))
        for i,pair in enumerate(pairs):
            points[i,0] = pair[0]
            points[i,1:] = (cells[pair[1]].x, cells[pair[1]].y,
                            cells[pair[2]].x, cells[pair[2]].y)
        return points

        # points = dict()
        # for k in pairs.keys():
        #     points[k] = np.array( [(cells[pair[0]].x, cells[pair[0]].y,
        #                             cells[pair[1]].x, cells[pair[1]].y)
        #                             for pair in pairs[k] ]
        #                         )
        # return points
    #####################################
    # Post-process the cycles(Get rid of the tails)
    #####################################
    
    def fill_mask(self,data, start_coords, fill_value):
        """
        Flood fill algorithm 
        Parameters
        ----------
        data : (M, N) ndarray of uint8 type
            Image with flood to be filled. Modified inplace.
        start_coords : tuple
            Length-2 tuple of ints defining (row, col) start coordinates.
        fill_value : int
            Value the flooded area will take after the fill.
        
        Returns
        -------
        None, ``data`` is modified inplace.
        """
        xsize, ysize = data.shape
        orig_value = data[start_coords[0], start_coords[1]]
        stack = set(((start_coords[0], start_coords[1]),))
        if fill_value == orig_value:
            raise ValueError("Filling region with same value "
                     "already present is unsupported. "
                     "Did you already fill this region?")

        while stack:
            x, y = stack.pop()

            if data[x, y] == orig_value:
                data[x, y] = fill_value
                if x > 0:
                    stack.add((x - 1, y))
                if x < (xsize - 1):
                    stack.add((x + 1, y))
                if y > 0:
                    stack.add((x, y - 1))
                if y < (ysize - 1):
                    stack.add((x, y + 1))
                
    def remove_non_boundary(self,good_cycles):
        #Helper function to remove tails from the raw cycles found by persistence algorithm
        #if plot=True, it allows to see individual cycles as a matrix
        #we use fill_mask to floodfill everywhere on the mask except the hole bounded by the cycle.
        #we start floodfilling from (0,0), so we need to use 2 pixels bigger image along left-right and up-down just in case there is a 
        #cycle whose coordinates go through (0,0)
        #"input:cycles with tails to be removed"
        #"Returns:coordinates of the clean cycles and the correponding matrix representation 1-pixel bigger than the original image"
        #"from all four directions"
        good_cycles_cleaned=[]
        masks=[]
        for k in range(len(good_cycles)):
            mask=self.overlay(good_cycles[[k]])
            self.fill_mask(mask[:,:,0],(0,0),0.5)
            for i in self.cycle2pixel(good_cycles[k]):
                if mask[i[0]+2,i[1]+1,0]==0:pass#break
                elif mask[i[0]+1,i[1]+2,0]==0:pass#break
                elif mask[i[0],i[1]+1,0]==0:pass#break
                elif mask[i[0]+1,i[1],0]==0:pass#break
                else: mask[i[0]+1,i[1]+1,0]=0.5
            if mask[:,:,0].all()==0.5: good_cycles_cleaned.append(good_cycles[k]);mask=self.overlay(good_cycles[[k]]);masks.append(mask)
            else: self.fill_mask(mask[:,:,0],(0,0),0); cycle=np.transpose(np.nonzero(mask[:,:,0])) ;  good_cycles_cleaned.append(cycle) ; masks.append(mask)
        pixels = np.vstack([cycle for cycle in good_cycles_cleaned])
        mask_good_clean = np.zeros((self.image.shape[0]+2, self.image.shape[1]+2, 4))
        mask_good_clean[pixels[:,0]+1, pixels[:,1]+1,0] = 1
        mask_good_clean[pixels[:,0]+1, pixels[:,1]+1,3] = 1
        return good_cycles_cleaned,mask_good_clean,masks
        
    #####################################
    # Filter cycles based on properties
    #####################################
    
    def to_contour(self,good_pairs,masks,circ=0, conv=0, A_high=None, A_low=0):
        """
        Convert cycles to cv2 contour, filter them in process
        """
        area = np.zeros(len(masks))
        circularity = np.zeros(len(masks))
        convexity = np.zeros(len(masks))
        contours = []
        for i,c in enumerate(masks):
            mask = masks[i].astype(np.uint8)
            ret, thresh = cv2.threshold(mask[:,:,0], 0, 255, 0)
            cnts, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            A = cv2.contourArea(cnts[0])
            area[i] = A
            s = cv2.arcLength(cnts[0], True)
            if s > 0:
                circularity[i] = 4*np.pi*A / s**2
            else:
                circularity[i] = 0
            hull = cv2.convexHull(cnts[0])
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                convexity[i] = area[i] / hull_area
            else:
                convexity[i] =0

            contours.append(cnts[0])

        if A_high is None:
            A_high = np.max(area)
        index = (circularity > circ) & (convexity > conv) & (area > A_low) & (area < A_high)
        contours = np.array(contours)
        return contours[index]

    def draw_contour(self,contours, img=None, index=None, thickness=1):
        """
        Draw contours on copy of image
        """
        if img is None:
            img = self.image.astype(np.uint8)
        imcopy = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        if index== None:
            for i in range(len(contours)):
                cv2.drawContours(imcopy, contours,i,[255,0,0], thickness) #(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        else:cv2.drawContours(imcopy, contours,index,[255,0,0], thickness)
        return imcopy
    
    def to_contour1(self,good_pairs,good_cycles,circ=0, conv=0, A_high=None, A_low=0):
        """
        Convert cycles to cv2 contour, filter them in process
        """
        area = np.zeros(len(good_cycles))
        circularity = np.zeros(len(good_cycles))
        convexity = np.zeros(len(good_cycles))
        contours = []
        for i,c in enumerate(good_cycles):
            mask = self.overlay([c]).astype(np.uint8)
            ret, thresh = cv2.threshold(mask[:,:,0], 0, 255, 0)
            cnts, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            A = cv2.contourArea(cnts[0])
            area[i] = A
            s = cv2.arcLength(cnts[0], True)
            if s > 0:
                circularity[i] = 4*np.pi*A / s**2
            else:
                circularity[i] = 0
            hull = cv2.convexHull(cnts[0])
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                convexity[i] = area[i] / hull_area
            else:
                convexity[i] =0

            contours.append(cnts[0])

        if A_high is None:
            A_high = np.max(area)
        index = (circularity > circ) & (convexity > conv) & (area > A_low) & (area < A_high)
        contours = np.array(contours)
        return contours[index]

    #####################################
    # Image Helpers
    #####################################


    def overlay(self,cycles):
        pixels = np.vstack([self.cycle2pixel(cycle) for cycle in cycles])
        mask = np.zeros((self.image.shape[0]+2, self.image.shape[1]+2, 4))
        mask[pixels[:,0]+1, pixels[:,1]+1,0] = 1
        mask[pixels[:,0]+1, pixels[:,1]+1,3] = 1
        return mask
    
    def normalize_bright(self,img,cycles,norm):
        cycle_coords=[]
        for i in range(len(cycles)):
            cycle_coords.append(np.array(self.cycle2pixel(cycles[i])))
        cycle_coords=np.array(cycle_coords)
        
        cycle_brightness=[]
        for i in range(len(cycles)):
            for j in range(len(cycles[i])):
                if img[cycle_coords[i][j][0],cycle_coords[i][j][1]] not in cycle_brightness:
                    cycle_brightness.append(img[cycle_coords[i][j][0],cycle_coords[i][j][1]])
        cycle_brightness=np.array(cycle_brightness)

        normalized=max(cycle_brightness)-min(cycle_brightness)
        cycle_brightness_normalized=(cycle_brightness-min(cycle_brightness))/normalized
        
        mask_tobe_filled = np.zeros((img.shape[0]+2, img.shape[1]+2, 4))
        for i in range(len(cycles)):
            for j in range(len(cycles[i])):
                if (img[cycle_coords[i][j][0],cycle_coords[i][j][1]]-min(cycle_brightness))/normalized>=norm:
                    mask_tobe_filled[cycle_coords[i][j][0]+1, cycle_coords[i][j][1]+1,0] = 1
                    mask_tobe_filled[cycle_coords[i][j][0]+1, cycle_coords[i][j][1]+1,3] = 1
        return(mask_tobe_filled)


    def plot_cycles(self,cycles, **kwargs):
        pixels = [self.cycle2pixel(cycle) for cycle in cycles]
        ax = plot_cycles(self.image, pixels, **kwargs)
        return ax




    def cycle2pixel(self, cycle):
        """
        convert a cylce of cells to a cycle of x,y positions

        Args:
            filtration: FilteredComplex object
            cycle: a list filled with cells (aka a cycle rep)

        Returns:
            pixels: a list of (x,y) tuples corresponding to the cells
        """
        return np.array([self.index2pixel(index) for index in cycle])


    def index2pixel(self, index):
        """
        Convert an index to x,y pixel data.

        Args:
            filtration: a Filtration object
            index: an index into filtration.cells

        Return:
            tuple: (x,y,degree)
        """
        cell = self.cells[index]
        return (cell.x, cell.y)
    
    def barcode(self,h1_pairs,persistence):
        y=np.arange(len(h1_pairs))
        deg_pairs = self.to_degree(h1_pairs)
        fig,ax=plt.subplots(1,1,figsize=(7,7))
        ax=plt.barh(y,deg_pairs[:,2]-deg_pairs[:,1], left=deg_pairs[:,1])
        for i in range(len(h1_pairs)):
            if deg_pairs[i,2]-deg_pairs[i,1]>persistence:
                ax[i].set_color('r')
        N=h1_pairs[deg_pairs[:,2]-deg_pairs[:,1]>persistence]
        return(len(N))
    
    
############################################
# Local Density Adjusted Delaunay Filtration
############################################
class LDA:
    """
    Compute the LDA-α-k triangulation of a given set of points.

    This class takes a list of points and computes the Delaunay triangluation.
    It then initializes rmin, which is the minimum radius of every maximal ball in a k-neighborhood of each vertex.  Finally it finds the minimal value of α for which each face and edge will be present.

    References:
    1. "Shape reconstruction from unorganized set of points" 2010, Maillot et al.
    2. "Automatic Recognition of 2D Shapes from a Set of Points", 2011, Presles et al.
    3. "Shape Reconstruction from an Unorganized Point Cloud with Outliers", 2012, Maillot et al.

    Parameters:
        points: (npoints, 2) array of points in the plane
        k: depth of neighborhood for algorithm (see reference 3)
    """

    def __init__(self,points = None, k=0):
        self.points = self._cgal_points(points)
        self.k = k
        self.tri = Delaunay_triangulation_2()
        self.tri.insert(self.points)
        self.radius = {f:self.circumradius(f) for f in self.tri.finite_faces()}
        self.rmin = self.compute_rmin()
        self.face_alpha,self.edge_alpha = self._tag_alpha()

    def compute_rmin(self, k=None):
        """
        Assign each vertex the value of the radius of the smallest maximal ball that passes through a point in the vertex's k-neighborhood.

        See reference 1 & 2.
        """
        if k is None: k = self.k
        rmin ={v:np.inf for v in self.tri.finite_vertices()}
        for f in self.tri.finite_faces():
            for i in range(3):
                rmin[f.vertex(i)] = min(self.radius[f], rmin[f.vertex(i)])

        # find minimum up to kth neighborhood
        i=0
        while(i<k):
            i += 1
            for vertex in self.tri.finite_vertices():
                neighbors = self.tri.incident_vertices(vertex)
                if neighbors.hasNext():
                    done = neighbors.next()
                while(1):
                    v = neighbors.next()
                    if not self.tri.is_infinite(v):
                        rmin[vertex] = min(rmin[v], rmin[vertex])
                    if v == done:
                        break
        return rmin


    def _tag_alpha(self):
        """
        Each face of the Delaunay triangulation comes with a natural value of α for which it first appears in the LDA-α-k filtration so we 'tag' each face with this value.  Each edge also comes with 2 values of α, the minimum and maximum α values of its incident faces.  The two values are useful for computing the LDA-α-k-shape and the minimum value is useful in computing the filtered complex.
        """
        face_alpha = {f:self._face_alpha(f) for f in self.tri.finite_faces()}
        edge_alpha = {e:self._edge_alpha(e, face_alpha) for e in self.tri.finite_edges()}
        return face_alpha, edge_alpha


    def _face_alpha(self,face):
        """
        Computes the maximal α such that face is α-empty or an α-eliminator
        ball (notation from ref 1 and 2 resp.)  This is the filtration value
        for which face first shows up in the filtered complex since after this
        α the face is present and before it the face is not present.
        """
        radius = self.radius[face]
        return min([1 - self.rmin[face.vertex(i)]/radius for i in range(3)])

    def _edge_alpha(self,edge, face_alpha):
        """
        Return the min and max of the face_alpha values for the indicent faces. The minimum value is used for the filtered complex.  For the α-shape the edge is present if and only if it is either incident to an infinite face or the value of α is between min and max.
        """
        f,h = edge[0], edge[0].neighbor(edge[1])
        if self.tri.is_infinite(f):
            alpha1 = face_alpha[h]
            alpha2 = np.inf
        elif self.tri.is_infinite(h):
            alpha1 = face_alpha[f]
            alpha2 = np.inf
        else:
            alpha1,alpha2 = face_alpha[f], face_alpha[h]
        return [min(alpha1,alpha2), max(alpha1,alpha2)]

    def circumradius(self, face):
        """
        Compute the circumradius of the triangle corresponding to face
        """
        C = self.tri.circumcenter(face)
        c = np.array([C.x(), C.y()])
        P = face.vertex(0).point()
        p = np.array([P.x(), P.y()])
        return np.linalg.norm(c-p)

    def equal_edge(self,edge):
        """
        Each edge has 2 representations so return the other rep given one
        """
        T = self.tri
        v1,v2 = edge[0].vertex(T.cw(edge[1])),edge[0].vertex(T.ccw(edge[1]))
        other_face = edge[0].neighbor(edge[1])
        for i in range(3):
            if (other_face.vertex(T.ccw(i)) == v1 and other_face.vertex(T.cw(i)) == v2):
                return (other_face,i)
        return (other_face,-1)



    def _cgal_points(self,points):
        """
        Convert a list of points to a list of CGAL Point_2 objects
        """
        if points is None or isinstance(points[0],Point_2):
            return points
        else:
            return [Point_2(point[0],point[1]) for  point in points]


    def plot_edges(self, alpha=1,flip=False,ax=None):
        if ax is None:
            fig,ax = plt.subplots()
        for e in self.tri.finite_edges():
            if self.edge_alpha[e][0] < alpha:
                s = self.tri.segment(e)
                pts = [ s.source(), s.target() ]
                xs = [ pts[0].x(), pts[1].x() ]
                ys = [ pts[0].y(), pts[1].y() ]
                if flip:
                    ax.plot(ys,xs,c="b")
                else:
                    ax.plot( xs, ys, c="b" )
        return ax


class LDACell(Cell):
    """
    Cell type for building the LDAComplex.
    """
    def __init__(self,handle,dimension,degree):
        self.handle = handle
        Cell.__init__(self,dimension,degree)

    def __hash__(self):
        return self.handle.__hash__()

    def __eq__(self,other):
        return self.handle == other.handle

class LDAComplex(FilteredComplex):
    """
    The filtered complex corresponding to the LDA-α-k filtration
    """
    def __init__(self, points, k=0):
        self.lda = LDA(points, k)
        cells = {}
        for v in self.lda.tri.finite_vertices():
            cell = LDACell(v,0,0)
            cell.boundary = self.boundary(cell)
            cells[v] = cell
        for e in self.lda.tri.finite_edges():
            alpha = self.lda.edge_alpha[e][0]
            cell = LDACell(e,1,alpha)
            cell.boundary = self.boundary(cell)

            cells[e]= cell
            #face,i = e[0],e[1]
            #vertices = []
            #vertices.append(face.vertex(self.lda.tri.cw(i)))
            #vertices.append(face.vertex(self.lda.tri.ccw(i)))
            #vertices are added at minimum degree of all its adjacent edges
            #for v in vertices:
            #    if v in cells:
            #        new_alpha = min(alpha,cells[v].degree)
            #        cells[v] = LDACell(v,0,new_alpha)
            #    else:
            #        cells[v] = LDACell(v,0,alpha)

        for f in self.lda.tri.finite_faces():
            alpha = self.lda.face_alpha[f]
            cell = LDACell(f,2,alpha)
            cell.boundary = self.boundary(cell)
            cells[f] = cell
        filt_cells = [cell for cell in cells.values()]
        FilteredComplex.__init__(self, filt_cells)


    def boundary(self,cell):
        bnd = []
        dim = cell.dimension
        if dim == 1:
            f,i = cell.handle[0],cell.handle[1]
            c1 = LDACell(f.vertex(self.lda.tri.cw(i)),0,0)
            c2 = LDACell(f.vertex(self.lda.tri.ccw(i)),0,0)
            bnd = [c1,c2]
        if dim == 2:
            for i in range(3):
                edge = (cell.handle,i)
                if edge in self.lda.edge_alpha:
                    alpha = self.lda.edge_alpha[edge]
                else:
                    edge = self.lda.equal_edge(edge)
                    alpha = self.lda.edge_alpha[edge]
                c = LDACell(edge,1,alpha)
                bnd.append(c)
        return bnd

    def plot_cycle(self,cycle, flip=True, ax=None):
        """
        Plot the cycle represent in 2D
        """
        if ax is None:
            fig,ax = plt.subplots()
        for i in cycle:
            cell = self.cells[i]
            s = self.lda.tri.segment(cell.handle)
            pts = [ s.source(), s.target() ]
            xs = [ pts[0].x(), pts[1].x() ]
            ys = [ pts[0].y(), pts[1].y() ]
            if flip:
                ax.plot( ys, xs, c="b" )
            else:
                ax.plot(xs,ys, c='b')


    def cycle2point(self,cycle, flip=True):
        """
        Convert cycle to list of x,y points
        """
        points = [self.index2source(index,flip=flip) for index in cycle]
        points.extend([self.index2target(index,flip=flip) for index in cycle])
        return np.array(points)

    def index2source(self,index, flip=True):
        cell = self.cells[index]
        s = self.lda.tri.segment(cell.handle)
        point = s.source()
        if flip:
            return [point.y(), point.x()]
        else:
            return [point.x(), point.y()]

    def index2target(self,index, flip=True):
        cell = self.cells[index]
        s = self.lda.tri.segment(cell.handle)
        point = s.target()
        if flip:
            return [point.y(),point.x()]
        else:
            return [point.x(), point.y()]

    def index2point(self, index, flip=True):
        """
        Convert an index to x,y pixel data.

        Args:
            self: a Filtration object
            index: an index into filtration.cells

        Return:
            tuple: (x,y) point of cell
        """
        cell = self.cells[index]
        s = self.lda.tri.segment(cell.handle)
        pts = [ s.source(), s.target() ]
        if flip:
            all_points = [pts[0].y(), pts[0].x(), pts[1].y(),pts[1].x()]
        else:
            all_points = [pts[0].x(), pts[0].y(), pts[1].x(),pts[1].y() ]

        return all_points




#####################################
# CGAL helpers
#####################################
def _face2point(f,i):
    """
    Return the [x,y] coordinates of the ith vertex of f
    """
    return [f.vertex(i).point().x(), f.vertex(i).point().y()]

def _vertex2point(v):
    """
    Return the [x,y] corrdinate of the vertex
    """
    return [v.point().x(), v.point().y()]


#####################################
# Image Helpers
#####################################
def cycle2pixel(filtration, cycle):
    """
    convert a cylce of cells to a cycle of x,y positions

    Args:
        filtration: FilteredComplex object
        cycle: a list filled with cells (aka a cycle rep)

    Returns:
        pixels: a list of (x,y) tuples corresponding to the cells
    """
    pixels = []
    for index in cycle:
        pixels.append(index2pixel(filtration, index))
    return pixels


def index2pixel(filtration, index):
    """
    Convert an index to x,y pixel data.

    Args:
        filtration: a Filtration object
        index: an index into filtration.cells

    Return:
        tuple: (x,y,degree)
    """
    cell = filtration.cells[index]
    return (cell.x, cell.y)



if __name__=='__main__':
    from eztda import pdiagram
    xx, yy = np.mgrid[:12, :12]
    circle = (xx -5 ) ** 2 + (yy - 5) ** 2
    mask = np.logical_and(circle <= 26, circle >= 18)
    img = np.zeros((3,3), dtype=np.uint8)
    img[:,0] = 255
    img[0,:] = 255
    img[:,2] = 255
    img[2,:] = 255
    plt.imshow(img)
    plt.show()
    ic = ImageComplex(img)
    for cell in ic.cells:
        print(cell.x,cell.y,cell.type,cell.degree)
    pairs,cycles = ic.persistence(cyclereps=True)
    pdiagram(pairs[1])
    print(pairs)
    print(cycles[1])

    plt.show()
