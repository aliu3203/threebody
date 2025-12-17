# Three Body Problem Animation

A Python implementation of the three body problem with animated visualization.

## Description

This project simulates the gravitational interaction of three bodies and creates a beautiful animated visualization showing their trajectories. The simulation uses numerical integration (Runge-Kutta method) to solve the equations of motion.

## Features

- Numerical solution of the three body problem using scipy's ODE solver
- Real-time animated visualization with trails showing the path of each body
- Configurable initial conditions, masses, and gravitational constant
- Includes the famous figure-8 periodic solution as a default example

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the animation:

```bash
python three_body_animation.py
```

To run the 3D animation:

```bash
python three_body_3d_animation.py
```

If using python3, the prefix should be pip3 and python3 respectively.

The script will:
1. Set up initial conditions (default: figure-8 solution)
2. Solve the equations of motion numerically
3. Display an animated visualization

## Customization

You can modify the initial conditions in the `main()` function:

- **Masses**: Change the `masses` array
- **Initial Positions**: Modify `initial_positions` 
- **Initial Velocities**: Adjust `initial_velocities`
- **Time Span**: Change `t_span` to simulate for longer/shorter periods
- **Animation Speed**: Adjust `interval` parameter in `create_animation()`

## Key Notes

- "Collisions" with other masses are permitted. To prevent repelling phenomena, acceleration was capped relative to the length scales of the initial conditions, multiplied by a factor of 400

## Example Configurations

### Figure-8 Solution (Default)
A periodic solution where three equal-mass bodies trace a figure-8 pattern.

### Simple Three Body System
Uncomment the alternative initial conditions in `main()` for a different configuration.

## Requirements

- Python 3.7+
- numpy
- matplotlib
- scipy

## License

This project is open source and available for educational purposes.

# threebody
