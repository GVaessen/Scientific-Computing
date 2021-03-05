import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib import animation
import pandas as pd
from scipy.sparse import diags

class Cell(object):
    '''
    Single cell that can be part of a larger grid
    '''

    def __init__(self, C, candidate, in_body, id, loc):
        '''
        Args:
            C:         float in [0, 1], concentration of cell
            candidate: bool, whether or not cell is a candidate to become part of the object
            body:      bool, whether or not cell is part of the body
            id:        float in [0, 1) random variable for uniqueness of each cell
        '''
        self.C = C
        self.candidate = candidate
        self.in_body = in_body
        self.id = id
        self.P_growth = 0
        self.loc = loc

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
    method: str, 'SOR' or 'shifts' determines which numerical method is used for the diffusion equation 
    '''

    def __init__(self, shape, w, eta, seed, steps, epsilon, method):
        self.Nrows = shape[0]
        self.Ncols = shape[1]
        self.seed = seed
        self.w = w
        self.eta = eta
        self.steps = steps
        self.epsilon = epsilon
        self.method = method
        self.cells = np.array([[Cell(0, False, False, np.random.random(), (i,j)) for j in np.arange(self.Ncols)] for i in np.arange(self.Nrows)])
        self.iterations = []
        self.body = [self.cells[seed[0], seed[1]]]

        # self.candidates = [self.cells[seed[0]-1, seed[1]], self.cells[seed[0]+1, seed[1]], self.cells[seed[0], seed[1]-1], self.cells[seed[0], seed[1]+1]]
        self.candidates = [self.cells[seed[0]-1, seed[1]], self.cells[seed[0], seed[1]-1], self.cells[seed[0], seed[1]+1]]

        # top row as source
        self.cells[0, :] = np.array([Cell(1, False, False, np.random.random(), (0,j)) for j in np.arange(self.Ncols)])
        
        # initial body
        # self.cells[seed[0], seed[1]] = Cell(0, False, True, np.random.random(), (seed[0], seed[1]))
        
        # initial candidates
        self.cells[seed[0]-1, seed[1]] = Cell(0, True, False, np.random.random(), (seed[0]-1, seed[1]))
        # self.cells[seed[0]+1, seed[1]] = Cell(0, True, False, np.random.random())
        self.cells[seed[0], seed[1]-1] = Cell(0, True, False, np.random.random(), (seed[0], seed[1]-1))
        self.cells[seed[0], seed[1]+1] = Cell(0, True, False, np.random.random(), (seed[0], seed[1]+1))

        self.concentrations = self.cells_C()


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

        # self.cells[self.seed[0], self.seed[1]].C = 0
        self.concentrations = self.cells_C()

    def shift_solve(self):
        '''
        solve the diffusion equation directly instead of iteratively using matrix shifts, matrix must be diagonal
        '''

        old_C = np.ones((self.Nrows, self.Ncols))

        while np.amax(np.abs(self.concentrations - old_C)) > self.epsilon:

            old_C = self.concentrations.copy()

            # construct L and U with 1's next to diagonals
            L = diags(np.ones(self.Nrows-1), offsets=-1).toarray()
            U = diags(np.ones(self.Nrows-1), offsets=1).toarray()
 
            new_C = 0.25 * (L @ self.concentrations + self.concentrations @ L + U @ self.concentrations + self.concentrations @ U)
            
            # implement boundary conditions
            new_C[0] = np.ones(self.Ncols)
            new_C[-1] = np.zeros(self.Ncols)
            
            self.concentrations = new_C

        
    def SOR(self):
        '''
        updates the concentration of all cells in the grid using Successive Over Relaxation except those part of the body 
        '''
        n_iter = 0        
        old_C = np.ones((self.Nrows, self.Ncols))

        while np.amax(np.abs(self.concentrations - old_C)) > self.epsilon:

            n_iter += 1

            old_C = self.concentrations[:]

            full_copy = copy.deepcopy(self.cells)

            new_C = np.ones((self.Nrows, self.Ncols))

            for i in np.arange(1, self.Nrows-1):
                for j in np.arange(self.Ncols):

                    if self.cells[i, j].in_body:
                        continue
                    
                    neighbours = [self.cells[i-1, j], self.cells[i, j-1], full_copy[i+1, j], full_copy[i, (j+1)%self.Ncols]]
                    n_neighbours = len([neighbour for neighbour in neighbours if not neighbour.in_body])
                    
                    # SOR method
                    if n_neighbours > 0:
                        self.cells[i, j].C = self.w/n_neighbours * (self.cells[i+1, j].C + self.cells[i, (j+1)%self.Ncols].C 
                                                                    + self.cells[i, j-1].C + self.cells[i-1, j].C) + (1-self.w) * self.cells[i, j].C

                    # keep concentrations bounded
                    if self.cells[i, j].C < 0: self.cells[i, j].C = 0
                    elif self.cells[i, j].C > 1: self.cells[i, j].C = 1

                    new_C[i, j] = self.cells[i, j].C

            self.concentrations = new_C

        self.iterations += [n_iter]



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

        for bodycell in self.body:

            i = bodycell.loc[0]
            j = bodycell.loc[1]

            # source and sinks can't become part of the body
            if i != 1:
                if self.cells[i-1, j] not in self.candidates + self.body:
                    self.cells[i-1, j].candidate = True
                    self.candidates += [self.cells[i-1, j]]
                
            if i != self.Nrows-1:
                if self.cells[i+1, j] not in self.candidates + self.body:
                    self.cells[i+1, j].candidate = True
                    self.candidates += [self.cells[i+1, j]]

            if self.cells[i, j-1] not in self.candidates + self.body:
                self.cells[i, j-1].candidate = True
                self.candidates += [self.cells[i, j-1]]

            if self.cells[i, (j+1) % self.Ncols] not in self.candidates + self.body:
                self.cells[i, (j+1) % self.Ncols].candidate = True
                self.candidates += [self.cells[i, (j+1) % self.Ncols]]

            # print('after ', len(self.candidates))
            


    def set_P_growth(self):
        '''
        determines growth probability for all candidates
        '''

        # concentration all candidates
        total = np.sum([candidate.C**self.eta for candidate in self.candidates])

        for candidate in self.candidates:
            i = candidate.loc[0]
            j = candidate.loc[1]

            self.cells[i, j].P_growth = (self.cells[i, j].C ** self.eta) / total
                           

    def DLA(self):
        '''
        Diffusion Limted Aggregattion algorithm
        '''

        self.true_solution()

        time_grid = [[[self.cells[i, j].C for j in np.arange(self.Ncols)] for i in np.arange(self.Nrows)]]

        for k in range(self.steps):

            # calculate the diffusion over the next time step
            if self.method == 'SOR':
                self.SOR()
            elif self.method == 'shifts':
                self.shift_solve()
            else:
                raise ValueError(f'mehtod can be SOR or shifts not {self.method}')
        

            self.update_candidates()

            # determine growth probabilities for all candidates
            self.set_P_growth()

            # pick a candidate with the growth probability as weight
            p = np.array([candidate.P_growth for candidate in self.candidates])

            # new_object = np.random.choice(self.cells.ravel(), p=p)
            new_object = np.random.choice(self.candidates, p=p)

            # coordinates new object
            obj_coords = np.where(self.cells == new_object)

            # add choosen candidate to the body and remove from candidates
            self.cells[obj_coords[0], obj_coords[1]][0].in_body = True
            self.cells[obj_coords[0], obj_coords[1]][0].candidate = False
            self.cells[obj_coords[0], obj_coords[1]][0].C = 0
            self.concentrations[obj_coords[0], obj_coords[1]] = 0

            # print(self.body)
            self.body += [self.cells[obj_coords[0], obj_coords[1]][0]]
            self.candidates.remove(self.cells[obj_coords[0], obj_coords[1]][0])

            for bodycell in self.body:
                self.concentrations[bodycell.loc[0], bodycell.loc[1]] = 0
            self.concentrations[-1,:] = 0
            time_grid += [self.concentrations]

            if k%100==0:
                print('eta ',self.eta,', iteration ', k)
        
        return np.array(time_grid), self.iterations

def plot_DLA():

    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots( 2,3, figsize=(9,6))

    df = pd.read_csv('DLA_3.csv').to_numpy()

    data = []
    for i in range(1,100):
            data += [df[i][1:]]

    ax1.imshow(np.array(data))
    ax1.set_title('$\eta=3$')
    ax1.axis('off')

    df = pd.read_csv('DLA_2.csv').to_numpy()

    data = []
    for i in range(1,100):
            data += [df[i][1:]]

    ax2.imshow(np.array(data))
    ax2.set_title('$\eta=2$')
    ax2.axis('off')
    
    
    df = pd.read_csv('DLA_1.5.csv').to_numpy()

    data = []
    for i in range(1,100):
            data += [df[i][1:]]

    ax3.imshow(np.array(data))
    ax3.set_title('$\eta=1.5$')
    ax3.axis('off')

    df = pd.read_csv('DLA_1.csv').to_numpy()

    data = []
    for i in range(1,100):
        data += [df[i][1:]]

    ax4.imshow(np.array(data))
    ax4.set_title('$\eta=1$')
    ax4.axis('off')

    df = pd.read_csv('DLA_0.5.csv').to_numpy()

    data = []
    for i in range(1,100):
        data += [df[i][1:]]

    ax5.imshow(np.array(data))
    ax5.set_title('$\eta=0.5$')
    ax5.axis('off')

    df = pd.read_csv('DLA_0.1.csv').to_numpy()

    data = []
    for i in range(1,100):
            data += [df[i][1:]]

    ax6.imshow(np.array(data))
    ax6.set_title('$\eta=0.1$')
    ax6.axis('off')

    plt.tight_layout()
    plt.savefig('DLA_eta_comp_6.pdf' , bbox_inches='tight')
    plt.show()

if __name__ == '__main__':

    steps = 100
    size = (100, 100)
    seed = (size[1]-1, int(size[1]/2))
    w = 1.85
    eta = 1.5
    epsilon = 1e-3
    method = 'shifts'

    model = Grid(size, w, eta, seed, steps, epsilon, method)
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

    # plot_DLA()

