import numpy as np
import matplotlib.pyplot as plt

class BraessSimulator:
    def __init__(self, num_drivers=4000):
        self.num_drivers = num_drivers
        
    def get_costs_part_a(self, n_top, n_bottom):
        """
        Path 1: S -> A -> E: T(x) = x/100 + 45
        Path 2: S -> B -> E: T(x) = 45 + x/100
        n_top: drivers on Path 1
        n_bottom: drivers on Path 2
        """
        cost_top = n_top / 100 + 45
        cost_bottom = 45 + n_bottom / 100
        return cost_top, cost_bottom

    def get_costs_part_b(self, n_top, n_bottom, n_middle):
        """
        Path 1: S -> A -> E: T(x) = x_SA/100 + T_AE(x_AE)
        Path 2: S -> B -> E: T(x) = T_SB(x_SB) + x_BE/100
        Path 3: S -> A -> B -> E: T(x) = x_SA/100 + 0 + x_BE/100
        
        Flows:
        x_SA = n_top + n_middle
        x_AE = n_top
        x_SB = n_bottom
        x_BE = n_bottom + n_middle
        
        Costs:
        T_SA = x_SA / 100
        T_AE = 45
        T_SB = 45
        T_BE = x_BE / 100
        T_AB = 0
        """
        x_SA = n_top + n_middle
        x_BE = n_bottom + n_middle
        
        cost_top = x_SA / 100 + 45
        cost_bottom = 45 + x_BE / 100
        cost_middle = x_SA / 100 + 0 + x_BE / 100
        
        return cost_top, cost_bottom, cost_middle

    def run_simulation(self, part='a', iterations=1000, initial_dist=None):
        if part == 'a':
            if initial_dist is None:
                n_top, n_bottom = self.num_drivers, 0
            else:
                n_top, n_bottom = initial_dist
            
            history = {'top': [], 'bottom': [], 'average': []}
            
            for _ in range(iterations):
                c_top, c_bottom = self.get_costs_part_a(n_top, n_bottom)
                history['top'].append(c_top)
                history['bottom'].append(c_bottom)
                avg_time = (n_top * c_top + n_bottom * c_bottom) / self.num_drivers
                history['average'].append(avg_time)
                
                # Move 1% from slowest to fastest
                move_count = int(0.01 * (n_top if c_top > c_bottom else n_bottom))
                if c_top > c_bottom:
                    n_top -= move_count
                    n_bottom += move_count
                elif c_bottom > c_top:
                    n_bottom -= move_count
                    n_top += move_count
            
            return (n_top, n_bottom), history

        elif part == 'b':
            # initial_dist from part a: (n_top, n_bottom)
            n_top, n_bottom = initial_dist
            n_middle = 0
            
            history = {'top': [], 'bottom': [], 'middle': [], 'average': []}
            
            for _ in range(iterations):
                c_top, c_bottom, c_middle = self.get_costs_part_b(n_top, n_bottom, n_middle)
                costs = [c_top, c_bottom, c_middle]
                history['top'].append(c_top)
                history['bottom'].append(c_bottom)
                history['middle'].append(c_middle)
                avg_time = (n_top * c_top + n_bottom * c_bottom + n_middle * c_middle) / self.num_drivers
                history['average'].append(avg_time)
                
                # Move 1% from slowest path to fastest path
                # Identify slowest and fastest
                slowest_idx = np.argmax(costs)
                fastest_idx = np.argmin(costs)
                
                if slowest_idx == fastest_idx:
                    continue
                
                drivers = [n_top, n_bottom, n_middle]
                move_count = int(0.01 * drivers[slowest_idx])
                
                drivers[slowest_idx] -= move_count
                drivers[fastest_idx] += move_count
                
                n_top, n_bottom, n_middle = drivers
                
            return (n_top, n_bottom, n_middle), history
