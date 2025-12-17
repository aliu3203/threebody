"""
Three Body Problem Animation
Simulates and animates the gravitational interaction of three bodies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

class ThreeBodyProblem:
    """Class to simulate the three body problem."""
    
    def __init__(self, masses, initial_positions, initial_velocities, G=1.0):
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
        """
        self.masses = np.array(masses)
        self.G = G
        self.n_bodies = 3
        
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
        
        # Compute accelerations due to gravitational forces
        # Force on body 1
        r12 = r2 - r1
        r13 = r3 - r1
        r12_norm = np.linalg.norm(r12)
        r13_norm = np.linalg.norm(r13)
        
        # Avoid division by zero
        if r12_norm < 1e-10:
            a1_from_2 = np.zeros(2)
        else:
            a1_from_2 = self.G * self.masses[1] * r12 / (r12_norm ** 3)
            
        if r13_norm < 1e-10:
            a1_from_3 = np.zeros(2)
        else:
            a1_from_3 = self.G * self.masses[2] * r13 / (r13_norm ** 3)
        
        a1 = a1_from_2 + a1_from_3
        
        # Force on body 2
        r21 = r1 - r2
        r23 = r3 - r2
        r21_norm = np.linalg.norm(r21)
        r23_norm = np.linalg.norm(r23)
        
        if r21_norm < 1e-10:
            a2_from_1 = np.zeros(2)
        else:
            a2_from_1 = self.G * self.masses[0] * r21 / (r21_norm ** 3)
            
        if r23_norm < 1e-10:
            a2_from_3 = np.zeros(2)
        else:
            a2_from_3 = self.G * self.masses[2] * r23 / (r23_norm ** 3)
        
        a2 = a2_from_1 + a2_from_3
        
        # Force on body 3
        r31 = r1 - r3
        r32 = r2 - r3
        r31_norm = np.linalg.norm(r31)
        r32_norm = np.linalg.norm(r32)
        
        if r31_norm < 1e-10:
            a3_from_1 = np.zeros(2)
        else:
            a3_from_1 = self.G * self.masses[0] * r31 / (r31_norm ** 3)
            
        if r32_norm < 1e-10:
            a3_from_2 = np.zeros(2)
        else:
            a3_from_2 = self.G * self.masses[1] * r32 / (r32_norm ** 3)
        
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
            rtol=1e-8,
            atol=1e-10
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
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Three Body Problem Animation', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Colors for the three bodies
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']
    body_names = ['Body 1', 'Body 2', 'Body 3']
    
    # Initialize plot elements
    bodies = []
    trails = []
    labels = []
    
    for i in range(3):
        # Bodies (larger points)
        body, = ax.plot([], [], 'o', color=colors[i], markersize=15, 
                       markeredgecolor='black', markeredgewidth=1.5, 
                       label=body_names[i], zorder=5)
        bodies.append(body)
        
        # Trails (lines showing path)
        trail, = ax.plot([], [], '-', color=colors[i], alpha=0.4, linewidth=1.5, zorder=1)
        trails.append(trail)
        
        # Labels
        label = ax.text(0, 0, body_names[i], fontsize=10, color=colors[i], 
                       fontweight='bold', ha='center', va='center', zorder=6)
        labels.append(label)
    
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
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    def animate(frame):
        """Update function for animation."""
        # Get positions at current frame
        positions = three_body.get_positions(frame)
        
        # Update body positions
        for i in range(3):
            bodies[i].set_data([positions[i, 0]], [positions[i, 1]])
            
            # Update trails (show last trail_length positions)
            start_idx = max(0, frame - trail_length)
            trail_positions = three_body.trajectory[start_idx:frame+1, 2*i:2*i+2]
            if len(trail_positions) > 0:
                trails[i].set_data(trail_positions[:, 0], trail_positions[:, 1])
            
            # Update labels (positioned slightly offset from body)
            offset = 0.15
            labels[i].set_position((positions[i, 0], positions[i, 1] + offset))
        
        # Update time text
        current_time = three_body.time_points[frame]
        time_text.set_text(f'Time: {current_time:.2f}')
        
        return bodies + trails + labels + [time_text]
    
    # Create animation
    n_frames = len(three_body.time_points)
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=interval,
        blit=True, repeat=True
    )
    
    plt.tight_layout()
    return fig, anim


def main():
    """Main function to run the three body problem animation."""
    # Example 1: Figure-8 solution (equal masses, specific initial conditions)
    print("Setting up three body problem...")
    
    # Masses (equal for figure-8 solution)
    masses = [1.0, 1.0, 1.0]
    
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
    
    # Alternative: Simple three body system (uncomment to use)
    # initial_positions = [
    #     [-1.0, 0.0],   # Body 1
    #     [1.0, 0.0],    # Body 2
    #     [0.0, 1.5]     # Body 3
    # ]
    # initial_velocities = [
    #     [0.0, -0.3],   # Body 1
    #     [0.0, 0.3],    # Body 2
    #     [0.0, 0.0]     # Body 3
    # ]
    
    # Create and solve the problem
    three_body = ThreeBodyProblem(masses, initial_positions, initial_velocities, G=1.0)
    
    # Solve for a time period
    t_span = (0, 20)  # Time span
    print("Solving equations of motion...")
    three_body.solve(t_span, max_step=0.02)
    
    print("Creating animation...")
    fig, anim = create_animation(three_body, trail_length=200, interval=30)
    
    print("Animation ready! Close the window to exit.")
    plt.show()
    
    # Optionally save the animation
    # print("Saving animation...")
    # anim.save('three_body_animation.gif', writer='pillow', fps=30)
    # print("Animation saved as 'three_body_animation.gif'")


if __name__ == "__main__":
    main()

