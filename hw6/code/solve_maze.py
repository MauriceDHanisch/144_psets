import sys
import random
import matplotlib.pyplot as plt
import numpy as np
from maze import Maze

def get_next_state_intended(state, direction):
    # Action map from maze: 0=N, 1=S, 2=E, 3=W 
    # but wait, the maze has moves explicitly defined:
    moves = [(0,1), (0, -1), (1,0), (-1, 0)]
    x_change = moves[direction][0]
    y_change = moves[direction][1]
    
    x, y = state
    return ((x+x_change) % 10, (y+y_change) % 10)

def train_agent():
    env = Maze()
    
    # Parameters
    alpha = 0.5
    epsilon_start = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    epochs = 998
    
    # Rewards / Valuations
    R_unexplored = 0.0 # Standard plain room
    R_snake = -100.0
    R_exit = 100.0
    
    # Initialize V table
    V = {}
    for x in range(10):
        for y in range(10):
            V[(x, y)] = R_unexplored
            
    # Hardcode terminal states (so they never decay from their terminal value)
    for snake in env.snakes:
        V[snake] = R_snake
    for exit in env.escapes:
        V[exit] = R_exit
            
    epsilon = epsilon_start
    
    training_steps = []
    
    for epoch in range(epochs):
        state = env.reset()
        t = 0
        while True:
            # Epsilon greedy policy based on strictly adjacent rooms
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                best_val = -float('inf')
                best_actions = []
                for a in range(4):
                    next_s = get_next_state_intended(state, a)
                    val = V[next_s]
                    if val > best_val:
                        best_val = val
                        best_actions = [a]
                    elif val == best_val:
                        best_actions.append(a)
                action = random.choice(best_actions)
                
            next_state, reward_type = env.step(action)
            
            # The environment step determines our actual next state.
            # Update rule from the problem: V(i) = (1-a)V(i) + a * V(i')
            if state not in env.snakes and state not in env.escapes:
                V[state] = (1 - alpha) * V[state] + alpha * V[next_state]
            
            state = next_state
            t += 1
            
            if reward_type == 1 or reward_type == -1:
                break
                
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        training_steps.append(t)
        
    return V, training_steps, env

def test_agent(V, env, episodes=1000):
    success_count = 0
    test_steps = []
    for _ in range(episodes):
        state = env.reset()
        t = 0
        success = False
        while True:
            # Greedy
            best_val = -float('inf')
            best_actions = []
            for a in range(4):
                next_s = get_next_state_intended(state, a)
                val = V[next_s]
                if val > best_val:
                    best_val = val
                    best_actions = [a]
                elif val == best_val:
                    best_actions.append(a)
            action = random.choice(best_actions)
            
            next_state, reward_type = env.step(action)
            state = next_state
            t += 1
            
            if reward_type == 1:
                success_count += 1
                success = True
                break
            elif reward_type == -1:
                break
            
            # Prevent infinite loops in testing
            if t > 500:
                break
        test_steps.append(t)
        
    return success_count, test_steps

def plot_valuations(V, env):
    grid = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            grid[y, x] = V[(x, y)] # plotting y on rows, x on cols matches typical Cartesian grid if inverted, but let's just use standard matrix
            
    plt.figure(figsize=(10, 8))
    # By passing origin='upper', row 0 is at Top. 
    # Col x, Row y matches array index grid[y, x]
    im = plt.imshow(grid, cmap="RdYlGn")
    plt.colorbar(im)
    plt.title("Learned Room Valuations V(s)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    # Annotate numbers
    for x in range(10):
        for y in range(10):
            val = grid[y, x]
            # Print integer part so "99.9" becomes "99" instead of rounding to "100.0"
            plt.text(x, y, f"{int(val)}", ha='center', va='center', color='black' if -50 < val < 50 else 'white', fontsize=10)

    # Annotate snakes and escapes
    for (x, y) in env.snakes:
        plt.text(x, y, 'S', ha='center', va='center', color='black', fontsize=16, fontweight='bold')
    for (x, y) in env.escapes:
        plt.text(x, y, 'E', ha='center', va='center', color='blue', fontsize=16, fontweight='bold')
        
    plt.text(0, 0, 'START', ha='center', va='center', color='purple', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('../latex/figs/maze_learned_values.png')
    plt.close()

if __name__ == "__main__":
    V, t_steps, env = train_agent()
    print("Training finished. Running test episodes...")
    successes, test_steps = test_agent(V, env, 1000)
    print(f"Successes: {successes}/1000")
    if successes == 1000:
        print("Agent successfully escapes 1000/1000 times!")
        
    plot_valuations(V, env)
    print("Saved valuation grid to latex/figs/maze_learned_values.png")
