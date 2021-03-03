import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib import animation

class Cell(object):
    '''
    Single cell that can be part of a larger grid
    '''

    def __init__(self, C, candidate, body, id):
        '''
        Args:
            C:         float in [0, 1], concentration of cell
            candidate: bool, whether or not cell is a candidate to become part of the object
            body:      bool, whether or not cell is part of the body
            id:        float in [0, 1) random variable for uniqueness of each cell
        '''
        self.C = C
        self.candidate = candidate
        self.body = body
        self.id = id
        self.P_growth = 0

class Grid(object):
    '''
    Grid with cells that grow through Diffusion Limited Aggregation

    Args:
    shape: 2-tuple, number of rows and columns of the grid
    w: float, omega parameter for SOR
    eta: float, parameter for candidate selection
    seed: 2-tuple, coordinates of initial object
    steps: int, amount of times of new cell is attached to the object
    e: float, convergence meausure (when the difference between two SOR iterations is considered 0)
    '''

    def __init__(self, shape, w, eta, seed, steps, epsilon):
        self.Nrows = shape[0]
        self.Ncols = shape[1]
        self.seed = seed
        self.w = w
        self.eta = eta
        self.steps = steps
        self.epsilon = epsilon
        self.cells = np.array([[Cell(0, False, False, np.random.random()) for j in np.arange(self.Ncols)] for i in np.arange(self.Nrows)])
        self.iterations = []
        self.body = [self.cells[seed[0], seed[1]]]
        # self.candidates = [self.cells[seed[0]-1, seed[1]], self.cells[seed[0]+1, seed[1]], self.cells[seed[0], seed[1]-1], self.cells[seed[0], seed[1]+1]]
        self.candidates = [self.cells[seed[0]-1, seed[1]], self.cells[seed[0], seed[1]-1], self.cells[seed[0], seed[1]+1]]

        # top row as source
        self.cells[0, :] = np.array([Cell(1, False, False, np.random.random()) for j in np.arange(self.Ncols)])
        
        # initial body
        self.cells[seed[0], seed[1]] = Cell(0, False, True, np.random.random())
        
        # initial candidates
        self.cells[seed[0]-1, seed[1]] = Cell(0, True, False, np.random.random())
        # self.cells[seed[0]+1, seed[1]] = Cell(0, True, False, np.random.random())
        self.cells[seed[0], seed[1]-1] = Cell(0, True, False, np.random.random())
        self.cells[seed[0], seed[1]+1] = Cell(0, True, False, np.random.random())


    def cells_C(self):
        ''''
        returns the current grid of concentrations
        '''
        return np.array([[self.cells[i, j].C for j in np.arange(self.Ncols)] for i in np.arange(self.Nrows)])


    def true_solution(self):
        '''
        True solution of the diffusion equation as time approaches infinity
        '''
        final_C = np.linspace(self.cells[0, 0].C, self.cells[-1, 0].C, self.Nrows)
        for j in np.arange(self.Ncols):
            for i in np.arange(self.Nrows):
                self.cells[i, j].C = final_C[i]

        self.cells[self.seed[0], self.seed[1]].C = 0

        
    def SOR(self):
        '''
        updates the concentration of all cells in the grid using Successive Over Relaxation except those part of the body 
        '''
        n_iter = 0        
        old_C = np.ones((self.Nrows, self.Ncols))

        while np.amax(np.abs(self.cells_C() - old_C)) > self.epsilon:

            n_iter += 1
            old_C = copy.deepcopy(self.cells_C())

            full_copy = copy.deepcopy(self.cells)

            for i in np.arange(1, self.Nrows-1):
                for j in np.arange(self.Ncols):

                    if self.cells[i, j].body:
                        continue
                    
                    # # GS method
                    # self.cells[i, j].C = 0.25 * (old_C[i+1, j] + self.cells[i-1, j].C + old_C[i, (j+1)%self.Ncols] + self.cells[i, j-1].C)
                    
                    neighbours = [self.cells[i-1, j], self.cells[i, j-1], full_copy[i+1, j], full_copy[i, (j+1)%self.Ncols]]
                    n_neighbours = len([neighbour for neighbour in neighbours if not neighbour.body])
                    
                    # SOR method
                    if n_neighbours > 0:
                        self.cells[i, j].C = self.w/n_neighbours * (old_C[i+1, j] + old_C[i, (j+1)%self.Ncols] + self.cells[i, j-1].C + self.cells[i-1, j].C) + (1-self.w) * old_C[i, j]
                    
                    # keep concentrations bounded
                    if self.cells[i, j].C < 0: self.cells[i, j].C = 0
                    elif self.cells[i, j].C > 1: self.cells[i, j].C = 1

        self.iterations += [n_iter]
        # print(n_iter)


    def get_neighbours(self, i, j):
        '''
        i, j: ints row and col index of cell whose neighbours should be returned
        returns all neighbouring cells excluding diagonals
        '''
        return [self.cells[(i+1)%self.Nrows, j], self.cells[i-1, j], self.cells[i, (j+1)%self.Ncols], self.cells[i, j-1]]


    def update_candidates(self):
        '''
        returns a list of all candidates of in the grid
        '''
        for i in np.arange(1, self.Nrows-1):
            for j in np.arange(self.Ncols):
                if self.cells[i, j].body:

                    # source and sinks can't become part of the body
                    if i != 1:
                        self.cells[i-1, j].candidate = True
                    if i != self.Nrows - 2:
                        self.cells[i+1, j].candidate = True
                    self.cells[i, j-1].candidate = True
                    self.cells[i, (j+1) % self.Ncols].candidate = True


    def set_P_growth(self):
        '''
        determines growth probability for all candidates
        '''

        # concentration all candidates
        total = np.sum([self.cells[i, j].C ** self.eta for i in np.arange(self.Nrows) for j in np.arange(self.Ncols) if self.cells[i, j].candidate])

        for i in np.arange(self.Nrows):
            for j in np.arange(self.Ncols):
                if self.cells[i, j].candidate:
                        self.cells[i, j].P_growth = self.cells[i, j].C ** self.eta / total
                else:
                    self.cells[i, j].P_growth = 0 
                           

    def DLA(self):
        '''
        Diffusion Limted Aggregattion algorithm
        '''
        time_grid = [[[self.cells[i, j].C for j in np.arange(self.Ncols)] for i in np.arange(self.Nrows)]]

        self.true_solution()

        for k in range(self.steps):

            # calculate the diffusion over the next time step
            self.SOR()
        
            self.update_candidates()

            # determine growth probabilities for all candidates
            self.set_P_growth()

            # pick a candidate with the growth probability as weight
            p = np.array([self.cells[i, j].P_growth for i in range(self.Nrows) for j in range(self.Ncols)])

            new_object = np.random.choice(self.cells.ravel(), p=p)

            # coordinates new object
            obj_coords = np.where(self.cells == new_object)

            # add choosen candidate to the body and remove from candidates
            added_cell = Cell(0, False, True, np.random.random())
            self.cells[obj_coords[0], obj_coords[1]] = added_cell
            self.body += [added_cell]

            time_grid += [[[self.cells[i, j].C for j in np.arange(self.Ncols)] for i in np.arange(self.Nrows)]]

            if k%10==0:
                print('iteration ', k)
        
        return np.array(time_grid), self.iterations

if __name__ == '__main__':

    steps = 100
    size = (100, 100)
    # seed = (-3, 3)
    seed = (size[1]-1, int(size[1]/2))
    w = 1.85
    eta = 1.5
    epsilon = 1e-3

    model = Grid(size, w, eta, seed, steps, epsilon)
    grid, iters = model.DLA()

    fig, ax = plt.subplots()
    
    im = ax.imshow(grid[0])

    def animate(i):
        i -= 1
        im.set_array(grid[i])  # update the data
        return [im]


    # Init only required for blitting to give a clean slate.
    def init():
        im.set_data(grid[0])
        return [im]

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                interval=25, blit=True)


    # writergif = animation.PillowWriter(fps=60, bitrate=-1) 
    # ani.save('diffeq.gif', writer=writergif)
    plt.show()