"""
Three Body Problem 3D Animation
Simulates and animates the gravitational interaction of three bodies in 3D space.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.integrate import solve_ivp

class ThreeBodyProblem3D:
    """Class to simulate the three body problem in 3D."""
    
    def __init__(self, masses, initial_positions, initial_velocities, G=1.0, softening=1e-6):
        """
        Initialize the three body problem in 3D.
        
        Parameters:
        -----------
        masses : array-like
            Masses of the three bodies [m1, m2, m3]
        initial_positions : array-like
            Initial positions [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]
        initial_velocities : array-like
            Initial velocities [[vx1, vy1, vz1], [vx2, vy2, vz2], [vx3, vy3, vz3]]
        G : float
            Gravitational constant (default: 1.0)
        softening : float
            Softening parameter to prevent infinite forces at close encounters (default: 1e-6)
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
        typical_mass = np.mean(self.masses)
        self.max_acceleration = 400.0 * self.G * typical_mass / (self.length_scale ** 2)
        
        # Flatten initial conditions for ODE solver
        # Format: [x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3]
        self.initial_state = np.zeros(18)
        self.initial_state[0:3] = initial_positions[0]
        self.initial_state[3:6] = initial_positions[1]
        self.initial_state[6:9] = initial_positions[2]
        self.initial_state[9:12] = initial_velocities[0]
        self.initial_state[12:15] = initial_velocities[1]
        self.initial_state[15:18] = initial_velocities[2]
        
        # Store trajectory
        self.trajectory = None
        self.time_points = None
        
    def equations_of_motion(self, t, state):
        """
        Compute the derivatives for the three body problem in 3D.
        
        Parameters:
        -----------
        t : float
            Current time
        state : array
            Current state [x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3]
            
        Returns:
        --------
        derivatives : array
            Derivatives [vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3]
        """
        # Extract positions and velocities
        r1 = state[0:3]
        r2 = state[3:6]
        r3 = state[6:9]
        v1 = state[9:12]
        v2 = state[12:15]
        v3 = state[15:18]
        
        # Compute accelerations due to gravitational forces with softening
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
        derivatives = np.zeros(18)
        derivatives[0:3] = v1  # dx1/dt, dy1/dt, dz1/dt
        derivatives[3:6] = v2  # dx2/dt, dy2/dt, dz2/dt
        derivatives[6:9] = v3  # dx3/dt, dy3/dt, dz3/dt
        derivatives[9:12] = a1  # dvx1/dt, dvy1/dt, dvz1/dt
        derivatives[12:15] = a2  # dvx2/dt, dvy2/dt, dvz2/dt
        derivatives[15:18] = a3  # dvx3/dt, dvy3/dt, dvz3/dt
        
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
        self.trajectory = solution.y.T  # Shape: (n_time_points, 18)
        
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
            Positions [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]
        """
        if self.trajectory is None:
            raise ValueError("Must solve the problem first!")
        
        positions = np.zeros((3, 3))
        positions[0] = self.trajectory[t_index, 0:3]
        positions[1] = self.trajectory[t_index, 3:6]
        positions[2] = self.trajectory[t_index, 6:9]
        
        return positions


def create_3d_animation(three_body, trail_length=100, interval=50):
    """
    Create an animated 3D visualization of the three body problem.
    
    Parameters:
    -----------
    three_body : ThreeBodyProblem3D
        Solved three body problem instance
    trail_length : int
        Number of previous positions to show as trails
    interval : int
        Animation interval in milliseconds
    """
    if three_body.trajectory is None:
        raise ValueError("Must solve the problem first!")
    
    fig = plt.figure(figsize=(12, 12), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    
    # Black and white colors - use different shades of gray/white for bodies
    body_colors = ['white', '0.7', '0.4']  # White, light gray, darker gray
    
    # Initialize plot elements
    bodies = []
    trail_lines = []
    
    for i in range(3):
        # Bodies (white/gray points) - smaller for point mass representation
        body, = ax.plot([], [], [], 'o', color=body_colors[i], markersize=6,
                       markeredgecolor='white', markeredgewidth=0.5, zorder=5)
        bodies.append(body)
        
        # Trail lines (will be updated with fading effect)
        trail_line, = ax.plot([], [], [], '-', color=body_colors[i], alpha=0.4, 
                             linewidth=1.5, zorder=1)
        trail_lines.append(trail_line)
    
    # Center of mass point (yellow) - smaller to match point masses
    com_point, = ax.plot([], [], [], 'o', color='yellow', markersize=5,
                        markeredgecolor='yellow', markeredgewidth=0.5, zorder=6)
    
    # Set axis limits based on trajectory
    all_x = np.concatenate([three_body.trajectory[:, 0], 
                           three_body.trajectory[:, 3], 
                           three_body.trajectory[:, 6]])
    all_y = np.concatenate([three_body.trajectory[:, 1], 
                           three_body.trajectory[:, 4], 
                           three_body.trajectory[:, 7]])
    all_z = np.concatenate([three_body.trajectory[:, 2], 
                           three_body.trajectory[:, 5], 
                           three_body.trajectory[:, 8]])
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    z_min, z_max = all_z.min(), all_z.max()
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    padding = 0.1
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
    
    # Add very faint axes for reference
    faint_color = '0.15'  # Very dark gray, almost black
    
    # Set faint tick colors
    ax.tick_params(axis='x', colors=faint_color, which='both')
    ax.tick_params(axis='y', colors=faint_color, which='both')
    ax.tick_params(axis='z', colors=faint_color, which='both')
    
    # Make axis lines faint
    ax.xaxis.line.set_color(faint_color)
    ax.yaxis.line.set_color(faint_color)
    ax.zaxis.line.set_color(faint_color)
    
    # Make panes very faint/transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(faint_color)
    ax.yaxis.pane.set_edgecolor(faint_color)
    ax.zaxis.pane.set_edgecolor(faint_color)
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Very faint grid
    ax.grid(True, color=faint_color, alpha=0.2, linestyle='--', linewidth=0.5)
    
    def animate(frame):
        """Update function for animation."""
        # Get positions at current frame
        positions = three_body.get_positions(frame)
        
        # Calculate center of mass
        total_mass = np.sum(three_body.masses)
        com = np.zeros(3)
        for i in range(3):
            com += three_body.masses[i] * positions[i]
        com /= total_mass
        
        # Update center of mass position
        com_point.set_data([com[0]], [com[1]])
        com_point.set_3d_properties([com[2]])
        
        # Update body positions and trails
        for i in range(3):
            bodies[i].set_data([positions[i, 0]], [positions[i, 1]])
            bodies[i].set_3d_properties([positions[i, 2]])
            
            # Update trails with fading effect
            start_idx = max(0, frame - trail_length)
            trail_positions = three_body.trajectory[start_idx:frame+1, 3*i:3*i+3]
            
            if len(trail_positions) > 1:
                # Create fading effect by plotting segments with varying alpha
                # For simplicity, we'll use a single line with overall alpha
                # More sophisticated fading would require multiple line segments
                trail_lines[i].set_data(trail_positions[:, 0], trail_positions[:, 1])
                trail_lines[i].set_3d_properties(trail_positions[:, 2])
                
                # Adjust alpha based on trail length (fade older parts)
                if len(trail_positions) > 10:
                    # Gradually fade the trail
                    alpha = 0.4 * (1.0 - 0.5 * (frame - start_idx) / trail_length)
                    trail_lines[i].set_alpha(max(0.1, alpha))
            else:
                trail_lines[i].set_data([], [])
                trail_lines[i].set_3d_properties([])
        
        return bodies + trail_lines + [com_point]
    
    # Create animation
    n_frames = len(three_body.time_points)
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=interval,
        blit=False, repeat=True
    )
    
    plt.tight_layout()
    return fig, anim


def main():
    """Main function to run the three body problem 3D animation."""
    print("Setting up three body problem in 3D...")
    
    # Masses (equal masses)
    masses = [1.0, 1.0, 1.0]
    
    # Initial positions in 3D (example: bodies starting in a triangle)
    initial_positions = [
        [-1.0, 0.0, 0.0],   # Body 1
        [1.0, 0.0, 0.0],    # Body 2
        [0.0, 1.5, 0.5]     # Body 3
    ]
    
    # Initial velocities in 3D
    initial_velocities = [
        [0.0, 0.2, 0.0],    # Body 1
        [0.0, -0.2, 0.0],   # Body 2
        [0.0, 0.0, 0.0]     # Body 3
    ]
    
    # Alternative: Figure-8 like solution extended to 3D
    # initial_positions = [
    #     [-0.97000436, 0.24308753, 0.1],  # Body 1
    #     [0.97000436, -0.24308753, -0.1],  # Body 2
    #     [0.0, 0.0, 0.0]                   # Body 3
    # ]
    # initial_velocities = [
    #     [0.466203685, 0.43236573, 0.0],   # Body 1
    #     [0.466203685, 0.43236573, 0.0],   # Body 2
    #     [-0.93240737, -0.86473146, 0.0]  # Body 3
    # ]
    
    # Create and solve the problem
    three_body = ThreeBodyProblem3D(masses, initial_positions, initial_velocities, G=1.0)
    
    # Solve for a time period
    t_span = (0, 20)  # Time span
    print("Solving equations of motion...")
    three_body.solve(t_span, max_step=0.02)
    
    print("Creating 3D animation...")
    fig, anim = create_3d_animation(three_body, trail_length=200, interval=30)
    
    print("Animation ready! Close the window to exit.")
    plt.show()
    
    # Optionally save the animation
    # print("Saving animation...")
    # anim.save('three_body_3d_animation.gif', writer='pillow', fps=30)
    # print("Animation saved as 'three_body_3d_animation.gif'")


if __name__ == "__main__":
    main()

