import numpy as np

class Grid:
    def __init__(self, Nx, Ny):
        # Set properties of the class Grid
        self.xlow = 0
        self.ylow = 0
        self.xhigh = 31
        self.yhigh = 31
        self.Nx = Nx
        self.Ny = Ny

        self.points,self.cells = self.__createRectGrid()

    def __createRectGrid(self):
        # vectors for division
        x = np.linspace(self.xlow,self.xhigh,self.Nx)
        y = np.linspace(self.ylow,self.yhigh,self.Ny)

        # number of cells in each dir is Nx-1 or Ny-1
        nc_x = self.Nx-1
        nc_y = self.Ny-1
    
        # meshgrid -> #nodes x 2
        xx,yy = np.meshgrid(x,y)
        points = np.stack((xx,yy), axis=2)
        points = points.reshape(self.Nx*self.Ny,2)

        # rectangles -> first one explicitely, then column and row shift
        cells = np.array([0, 1, self.Nx, self.Nx +1])  
        cells = np.tile(cells, (nc_x*nc_y,1))
        
        # column shift
        shift = np.arange(0,nc_x*nc_y).reshape(nc_x*nc_y,1)
        shift = np.repeat(shift,4,axis=1)
        cells = cells + shift
        
        # row shift
        shift = np.arange(0,nc_y).reshape(nc_y,1)
        shift = np.repeat(np.repeat(shift,nc_x,axis=0),4,axis=1)
        cells = cells + shift

        return points,cells