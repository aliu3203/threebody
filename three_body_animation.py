"""
Three Body Problem Animation
Simulates and animates the gravitational interaction of three bodies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp

class ThreeBodyProblem:
    """Class to simulate the three body problem."""
    
    def __init__(self, masses, initial_positions, initial_velocities, G=1.0, softening=1e-6):
        """
        Initialize the three body problem.
        
        Parameters:
        -----------
        masses : array-like
            Masses of the three bodies [m1, m2, m3]
        initial_positions : array-like
            Initial positions [[x1, y1], [x2, y2], [x3, y3]]
        initial_velocities : array-like
            Initial velocities [[vx1, vy1], [vx2, vy2], [vx3, vy3]]
        G : float
            Gravitational constant (default: 1.0)
        softening : float
            Softening parameter to prevent infinite forces at close encounters (default: 1e-6)
            Should be very small to only prevent numerical overflow, not alter physics
        """
        self.masses = np.array(masses)
        self.G = G
        self.softening = softening
        self.n_bodies = 3
        
        # Calculate length scale from initial positions
        initial_pos = np.array(initial_positions)
        # Find maximum distance between any two bodies
        max_dist = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                dist = np.linalg.norm(initial_pos[i] - initial_pos[j])
                max_dist = max(max_dist, dist)
        
        # Also consider typical separation (mean distance)
        mean_dist = 0.0
        count = 0
        for i in range(3):
            for j in range(i+1, 3):
                dist = np.linalg.norm(initial_pos[i] - initial_pos[j])
                mean_dist += dist
                count += 1
        mean_dist /= count if count > 0 else 1
        
        # Use the larger of max_dist or mean_dist as length scale
        self.length_scale = max(max_dist, mean_dist) if max_dist > 0 else 1.0
        
        # Calculate maximum acceleration based on length scale
        # Use a reasonable fraction of G*M/L^2 where M is typical mass and L is length scale
        typical_mass = np.mean(self.masses)
        # Maximum acceleration: cap at ~100 * typical gravitational acceleration at length scale
        # This prevents extreme accelerations while allowing normal behavior
        self.max_acceleration = 400.0 * self.G * typical_mass / (self.length_scale ** 2)
        
        # Flatten initial conditions for ODE solver
        # Format: [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
        self.initial_state = np.zeros(12)
        self.initial_state[0:2] = initial_positions[0]
        self.initial_state[2:4] = initial_positions[1]
        self.initial_state[4:6] = initial_positions[2]
        self.initial_state[6:8] = initial_velocities[0]
        self.initial_state[8:10] = initial_velocities[1]
        self.initial_state[10:12] = initial_velocities[2]
        
        # Store trajectory
        self.trajectory = None
        self.time_points = None
        
    def equations_of_motion(self, t, state):
        """
        Compute the derivatives for the three body problem.
        
        Parameters:
        -----------
        t : float
            Current time
        state : array
            Current state [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
            
        Returns:
        --------
        derivatives : array
            Derivatives [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]
        """
        # Extract positions and velocities
        r1 = state[0:2]
        r2 = state[2:4]
        r3 = state[4:6]
        v1 = state[6:8]
        v2 = state[8:10]
        v3 = state[10:12]
        
        # Compute accelerations due to gravitational forces with softening
        # Softening prevents infinite forces when bodies get very close
        # Formula: a = G * m * r / (r^2 + epsilon^2)^(3/2)
        # Only apply softening when distance is very small to preserve physics
        
        def compute_acceleration(r_vec, mass_other, softening_param):
            """Compute gravitational acceleration with adaptive softening and capping."""
            r_sq = np.dot(r_vec, r_vec)
            r = np.sqrt(r_sq)
            
            # Compute acceleration
            if r < 1e-4:
                # Use softening for very close encounters
                r_soft = (r_sq + softening_param**2) ** 1.5
                a_vec = self.G * mass_other * r_vec / r_soft
            else:
                # Use true inverse square law for normal distances
                a_vec = self.G * mass_other * r_vec / (r_sq * r)
            
            # Cap acceleration magnitude to prevent extreme values
            a_mag = np.linalg.norm(a_vec)
            if a_mag > self.max_acceleration:
                # Scale down to maximum while preserving direction
                a_vec = a_vec * (self.max_acceleration / a_mag)
            
            return a_vec
        
        # Force on body 1
        r12 = r2 - r1
        r13 = r3 - r1
        
        a1_from_2 = compute_acceleration(r12, self.masses[1], self.softening)
        a1_from_3 = compute_acceleration(r13, self.masses[2], self.softening)
        
        a1 = a1_from_2 + a1_from_3
        
        # Force on body 2
        r21 = r1 - r2
        r23 = r3 - r2
        
        a2_from_1 = compute_acceleration(r21, self.masses[0], self.softening)
        a2_from_3 = compute_acceleration(r23, self.masses[2], self.softening)
        
        a2 = a2_from_1 + a2_from_3
        
        # Force on body 3
        r31 = r1 - r3
        r32 = r2 - r3
        
        a3_from_1 = compute_acceleration(r31, self.masses[0], self.softening)
        a3_from_2 = compute_acceleration(r32, self.masses[1], self.softening)
        
        a3 = a3_from_1 + a3_from_2
        
        # Return derivatives
        derivatives = np.zeros(12)
        derivatives[0:2] = v1  # dx1/dt, dy1/dt
        derivatives[2:4] = v2  # dx2/dt, dy2/dt
        derivatives[4:6] = v3  # dx3/dt, dy3/dt
        derivatives[6:8] = a1  # dvx1/dt, dvy1/dt
        derivatives[8:10] = a2  # dvx2/dt, dvy2/dt
        derivatives[10:12] = a3  # dvx3/dt, dvy3/dt
        
        return derivatives
    
    def solve(self, t_span, t_eval=None, max_step=0.01):
        """
        Solve the three body problem.
        
        Parameters:
        -----------
        t_span : tuple
            Time span (t_start, t_end)
        t_eval : array-like, optional
            Time points at which to evaluate the solution
        max_step : float
            Maximum step size for the solver
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) / max_step))
        
        solution = solve_ivp(
            self.equations_of_motion,
            t_span,
            self.initial_state,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
            dense_output=False
        )
        
        self.time_points = solution.t
        self.trajectory = solution.y.T  # Shape: (n_time_points, 12)
        
        return solution
    
    def get_positions(self, t_index):
        """
        Get positions of all bodies at a given time index.
        
        Parameters:
        -----------
        t_index : int
            Time index
            
        Returns:
        --------
        positions : array
            Positions [[x1, y1], [x2, y2], [x3, y3]]
        """
        if self.trajectory is None:
            raise ValueError("Must solve the problem first!")
        
        positions = np.zeros((3, 2))
        positions[0] = self.trajectory[t_index, 0:2]
        positions[1] = self.trajectory[t_index, 2:4]
        positions[2] = self.trajectory[t_index, 4:6]
        
        return positions


def create_animation(three_body, trail_length=100, interval=50):
    """
    Create an animated visualization of the three body problem.
    
    Parameters:
    -----------
    three_body : ThreeBodyProblem
        Solved three body problem instance
    trail_length : int
        Number of previous positions to show as trails
    interval : int
        Animation interval in milliseconds
    """
    if three_body.trajectory is None:
        raise ValueError("Must solve the problem first!")
    
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')  # Remove all axes and labels
    
    # Black and white colors - use different shades of gray/white for bodies
    body_colors = ['white', '0.7', '0.4']  # White, light gray, darker gray
    
    # Initialize plot elements
    bodies = []
    trail_collections = []
    
    for i in range(3):
        # Bodies (white/gray points) - smaller for point mass representation
        body, = ax.plot([], [], 'o', color=body_colors[i], markersize=6, 
                       markeredgecolor='white', markeredgewidth=0.5, 
                       zorder=5)
        bodies.append(body)
        
        # Trail collections for fading effect (initially empty)
        trail_collection = LineCollection([], colors='white', linewidths=1.5, zorder=1)
        ax.add_collection(trail_collection)
        trail_collections.append(trail_collection)
    
    # Center of mass point (yellow) - smaller to match point masses
    com_point, = ax.plot([], [], 'o', color='yellow', markersize=5,
                        markeredgecolor='yellow', markeredgewidth=0.5,
                        zorder=6)
    
    # Set axis limits based on trajectory
    all_positions = three_body.trajectory[:, [0, 2, 4, 1, 3, 5]]  # All x and y coordinates
    x_min, x_max = all_positions[:, [0, 1, 2]].min(), all_positions[:, [0, 1, 2]].max()
    y_min, y_max = all_positions[:, [3, 4, 5]].min(), all_positions[:, [3, 4, 5]].max()
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.1
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    
    def animate(frame):
        """Update function for animation."""
        # Get positions at current frame
        positions = three_body.get_positions(frame)
        
        # Calculate center of mass
        total_mass = np.sum(three_body.masses)
        com = np.zeros(2)
        for i in range(3):
            com += three_body.masses[i] * positions[i]
        com /= total_mass
        
        # Update center of mass position
        com_point.set_data([com[0]], [com[1]])
        
        # Update body positions
        for i in range(3):
            bodies[i].set_data([positions[i, 0]], [positions[i, 1]])
            
            # Update trails with fading effect
            start_idx = max(0, frame - trail_length)
            trail_positions = three_body.trajectory[start_idx:frame+1, 2*i:2*i+2]
            
            if len(trail_positions) > 1:
                # Create line segments for fading effect
                segments = []
                alphas = []
                
                for j in range(len(trail_positions) - 1):
                    segment = [trail_positions[j], trail_positions[j + 1]]
                    segments.append(segment)
                    # Fade from 1.0 (most recent) to 0.0 (oldest)
                    alpha = (j + 1) / len(trail_positions)
                    alphas.append(alpha)
                
                # Create LineCollection with fading colors
                colors_list = [(1.0, 1.0, 1.0, alpha) for alpha in alphas]  # White with varying alpha
                trail_collections[i].set_segments(segments)
                trail_collections[i].set_colors(colors_list)
            else:
                trail_collections[i].set_segments([])
        
        return bodies + trail_collections + [com_point]
    
    # Create animation
    n_frames = len(three_body.time_points)
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=interval,
        blit=False, repeat=True  # blit=False because LineCollection doesn't support blitting well
    )
    
    plt.tight_layout()
    return fig, anim


def main():
    """Main function to run the three body problem animation."""
    # Example 1: Figure-8 solution (equal masses, specific initial conditions)
    print("Setting up three body problem...")
    
    # Masses (equal for figure-8 solution)
    masses = [1.0, 1.0, 1.0]
    """
    # Initial positions for figure-8 solution
    # This is a periodic solution where the three bodies trace a figure-8 pattern
    initial_positions = [
        [-0.97000436, 0.24308753],  # Body 1
        [0.97000436, -0.24308753],  # Body 2
        [0.0, 0.0]                   # Body 3 (at origin)
    ]
    
    # Initial velocities for figure-8 solution
    initial_velocities = [
        [0.466203685, 0.43236573],   # Body 1
        [0.466203685, 0.43236573],   # Body 2
        [-0.93240737, -0.86473146]   # Body 3
    ]
    """
    # Alternative: Simple three body system (uncomment to use)
    initial_positions = [
        [-0.5, 0.0],   # Body 1
        [1.0, 0.7],    # Body 2
        [0.0, 1.0]     # Body 3
    ]
    initial_velocities = [
        [0.0, 0.0],   # Body 1
        [0.0, 0.0],    # Body 2
        [0.0, 0.0]     # Body 3
    ]
    
    # Create and solve the problem
    three_body = ThreeBodyProblem(masses, initial_positions, initial_velocities, G=1.0)
    
    # Solve for a time period
    t_span = (0, 20)  # Time span
    print("Solving equations of motion...")
    three_body.solve(t_span, max_step=0.02)
    
    print("Creating animation...")
    fig, anim = create_animation(three_body, trail_length=50, interval=40)
    
    print("Animation ready! Close the window to exit.")
    plt.show()
    
    # Optionally save the animation
    # print("Saving animation...")
    # anim.save('three_body_animation.gif', writer='pillow', fps=30)
    # print("Animation saved as 'three_body_animation.gif'")


if __name__ == "__main__":
    main()

