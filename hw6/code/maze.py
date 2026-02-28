import random as random

class Maze:
    def __init__(self):
        '''
        Initialize the maze object
        '''
       
        self.xdim = 10
        self.ydim = 10
        
        self.start = (0, 0)

        self.snakes = [(8,0), (4,1),(9,3), (1,3), (6,3), (8,3), (7,9)]
        self.escapes = [(6, 0), (8,1), (0,4), (5,5), (3, 6)]
        
        self.state = self.start

        self.p_stable = 0.8

        # Actions: 0=N, 1=S, 2=E, 3=W
        self.moves = [(0,1), (0, -1), (1,0), (-1, 0)]

       
    def reset(self):
        '''
        Use this to respawn at the starting room of the maze
        '''
        self.state = self.start
        return self.state
    
    def step(self, direction):
        '''
        Use this function to move through the maze.
        The direction argument is how you indicate which direction you wish to move in the maze. 
        0 to move North, 1 to move South, 2 to move East, 3 to move West
        
        Returns 
        1) tuple containing the actual location after your attempted move
        2) -1, 0 or 1 to indicate if you have found yourself in a snake room, a plain room, or an exit room
        '''

        x_change = self.moves[direction][0]
        y_change = self.moves[direction][1]
        
        # See if maze shifts
        rand = random.random()
        
        if rand > self.p_stable:
            #Maze has shifted
            rand2 = random.randint(0,3)
            x_change += self.moves[rand2][0]
            y_change += self.moves[rand2][1]

        # Update location in maze
        x, y = self.state
        self.state = ((x+x_change) % self.xdim, (y+y_change) % self.ydim)
        location = ((x+x_change) % 10, (y+y_change) % 10)
        
        # Get Room Type
        if location in self.escapes:
            return self.state, 1
        elif location in self.snakes:
            return self.state, -1
        else:
            return self.state, 0
        
