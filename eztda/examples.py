from eztda import Cell, FilteredComplex, pdiagram, LDAComplex, ImageComplex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def triangle():
    """
    Example Persistent Homology Computation on Filtered triangle

    The triangle fills in one vertex, edge, and face at a time.  The face closes the triangle forms at degree 5 and isn't filled in by the face until
    degree 10 leaving one persistent cycle.
    """

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
    print('Persistence pairs\n' + '-'*30 + '\n', pairs)
    print('\nCycle Representative\n' + '-'*30 + '\n',cycles)
    pdiagram(pairs, dim=1) #h1 diagram


def circle(npoints=100):
    """
    Example Persistent Homology Computation on Filtered Circle

    The circle is filtered by the Density Adaptive Complex.  The Density Adaptive complex is subcomplex of the Delaunay complex on the points.  It is parameterized by α∈[0,1] such that when α=0 the complex is the set of points and when α=1 the complex is the full Delaunay complex.
    """
    samples = np.random.normal(0,1,(npoints,2))
    circle = samples/np.reshape(np.linalg.norm(samples,axis=1), (npoints,1))
    noisy_circle = circle+np.random.normal(0,.1,(npoints,2))
    dacomp = LDAComplex(noisy_circle)
    pairs,cycles = dacomp.persistence(cyclereps=True)

    fig,ax = plt.subplots(1,3, figsize=(12,8))
    ax[0].scatter(*zip(*noisy_circle))
    pdiagram(pairs[1],ax=ax[1])
    dacomp.plot_cycle(cycles[1][0],ax=ax[2])
    for a in ax:
        a.set_aspect('equal')
    return fig, ax



def image(img):
    """
    Example Persistent Homology Computation on Filtered Image

    The image is a grayscale image with bright pixels corresponding to features and dark pixels corresponding to background.  The filtered complex is parameterized by a scalar, t, representing a threshold value.  A cell enters the complex if its pixel is brighter than max(img)-t so that dark pixels enter last and bright pixels enter first.
    """
    #img = mpimg.imread(fn)
    #img = np.dot(img[...,:3], [0.299, 0.587, 0.114]) #convert to gray
    imgcomp = ImageComplex(img)
    pairs,cycles = imgcomp.persistence(cyclereps=True)
    fig,ax = plt.subplots(1,2,figsize=(12,6))
    ax[0].imshow(img)
    imgcomp.plot_cycles([cycles[1][0]],ax=ax[1])
