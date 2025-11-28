import matplotlib.pyplot as plt
import numpy as np
import math

def find_solution_equation():
    """
    Find integer solutions for 5x + 3y = 4 using linear Diophantine equation
    and simulate the pouring process based on the solution.
    """
    
    print("Solving the equation: 5x + 3y = 4")
    print("Where x = operations with 5L container, y = operations with 3L container")
    print("=" * 60)
    
    # Find integer solutions
    solutions = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            if 5*x + 3*y == 4:
                solutions.append((x, y))
    
    print(f"Found {len(solutions)} integer solutions:")
    for i, (x, y) in enumerate(solutions, 1):
        print(f"Solution {i}: x = {x}, y = {y} -> 5({x}) + 3({y}) = {5*x + 3*y}")
    
    return solutions

def simulate_from_equation(x, y):
    """
    Simulate the pouring process based on the equation solution 5x + 3y = 4
    """
    print(f"\nSimulating for solution: x = {x}, y = {y}")
    print("=" * 50)
    
    container_5L = 0
    container_3L = 0
    steps = []
    
    def record_step(action):
        steps.append(f"{action} | 5L: {container_5L}L, 3L: {container_3L}L")
    
    # Interpret the solution
    if x == 2 and y == -2:
        # This corresponds to our manual solution
        record_step("Start")
        
        # Fill 5L (positive x operation)
        container_5L = 5
        record_step("Fill 5L container")
        
        # Pour to 3L (this uses 3L container positively)
        pour = min(container_5L, 3 - container_3L)
        container_5L -= pour
        container_3L += pour
        record_step(f"Pour {pour}L to 3L")
        
        # Empty 3L (negative y operation)
        container_3L = 0
        record_step("Empty 3L container")
        
        # Pour remaining to 3L
        pour = min(container_5L, 3 - container_3L)
        container_5L -= pour
        container_3L += pour
        record_step(f"Pour {pour}L to 3L")
        
        # Fill 5L again (second positive x operation)
        container_5L = 5
        record_step("Fill 5L container again")
        
        # Pour to 3L until full (using 3L container)
        pour = min(container_5L, 3 - container_3L)
        container_5L -= pour
        container_3L += pour
        record_step(f"Pour {pour}L to 3L")
        
    # Display steps
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step}")
    
    print(f"Final: 5L container has {container_5L}L")
    return container_5L, steps

def plot_solutions(solutions):
    """
    Create a plot showing all integer solutions of 5x + 3y = 4
    """
    plt.figure(figsize=(12, 8))
    
    # Extract x and y coordinates from solutions
    x_vals = [sol[0] for sol in solutions]
    y_vals = [sol[1] for sol in solutions]
    
    # Plot the line 5x + 3y = 4
    x_line = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y_line = (4 - 5 * x_line) / 3
    
    # Plot the line
    plt.plot(x_line, y_line, 'b-', linewidth=2, label='5x + 3y = 4', alpha=0.7)
    
    # Plot integer solutions
    colors = ['red' if (x == 2 and y == -2) else 'green' for x, y in solutions]
    sizes = [100 if (x == 2 and y == -2) else 60 for x, y in solutions]
    
    plt.scatter(x_vals, y_vals, c=colors, s=sizes, zorder=5)
    
    # Annotate the practical solution
    for i, (x, y) in enumerate(solutions):
        if x == 2 and y == -2:
            plt.annotate(f'Practical Solution\n(x={x}, y={y})', 
                        xy=(x, y), xytext=(x+1, y-2),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontweight='bold')
        else:
            plt.annotate(f'({x}, {y})', xy=(x, y), xytext=(x+0.1, y+0.1))
    
    plt.xlabel('x (5L Container Operations)', fontsize=12)
    plt.ylabel('y (3L Container Operations)', fontsize=12)
    plt.title('Integer Solutions of 5x + 3y = 4\n(Water Measurement Problem)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.legend()
    
    # Add explanation text
    plt.figtext(0.02, 0.02, 
                "Interpretation:\n"
                "x = net operations with 5L container\n"
                "y = net operations with 3L container\n"
                "Positive: fill, Negative: empty", 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    return plt

def plot_pouring_process(steps):
    """
    Create a visualization of the pouring process
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data from steps
    steps_text = [step.split('|')[0].strip() for step in steps]
    container_5L = [int(step.split('5L: ')[1].split('L')[0]) for step in steps]
    container_3L = [int(step.split('3L: ')[1].split('L')[0]) for step in steps]
    
    step_numbers = list(range(len(steps)))
    
    # Plot 1: Water levels over time
    width = 0.35
    x = np.arange(len(steps))
    
    bars1 = ax1.bar(x - width/2, container_5L, width, label='5L Container', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, container_3L, width, label='3L Container', color='green', alpha=0.7)
    
    ax1.set_xlabel('Step Number')
    ax1.set_ylabel('Water Amount (Liters)')
    ax1.set_title('Water Levels in Containers During Process')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Step {i+1}' for i in range(len(steps))], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}L', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}L', ha='center', va='bottom')
    
    # Plot 2: Container visualization at final step
    containers = ['5L Container', '3L Container']
    final_levels = [container_5L[-1], container_3L[-1]]
    capacities = [5, 3]
    colors = ['blue', 'green']
    
    bars = ax2.bar(containers, final_levels, color=colors, alpha=0.7, label='Current Water')
    ax2.bar(containers, [cap - level for cap, level in zip(capacities, final_levels)], 
            bottom=final_levels, color=colors, alpha=0.3, label='Remaining Capacity')
    
    ax2.set_ylabel('Liters')
    ax2.set_title('Final State: 4 Liters in 5L Container')
    ax2.set_ylim(0, max(capacities) + 0.5)
    
    # Add value labels and capacity lines
    for i, (bar, capacity) in enumerate(zip(bars, capacities)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{height}L', ha='center', va='center', fontweight='bold', color='white')
        ax2.axhline(y=capacity, color='red', linestyle='--', alpha=0.5)
        ax2.text(len(containers)/2, capacity + 0.1, f'Capacity: {capacity}L', 
                ha='center', va='bottom', color='red')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

def plot_3d_solution_space():
    """
    Create a 3D plot showing the solution space
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh grid
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    Z = 5*X + 3*Y
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, linewidth=0, antialiased=True)
    
    # Plot the solutions where Z = 4
    solutions = []
    for x_val in range(-10, 11):
        for y_val in range(-10, 11):
            if 5*x_val + 3*y_val == 4:
                solutions.append((x_val, y_val, 4))
    
    if solutions:
        x_sol = [sol[0] for sol in solutions]
        y_sol = [sol[1] for sol in solutions]
        z_sol = [sol[2] for sol in solutions]
        
        # Color the practical solution differently
        colors = ['red' if (x == 2 and y == -2) else 'yellow' for x, y, z in solutions]
        sizes = [100 if (x == 2 and y == -2) else 50 for x, y, z in solutions]
        
        ax.scatter(x_sol, y_sol, z_sol, c=colors, s=sizes, marker='o', depthshade=False)
    
    ax.set_xlabel('x (5L Operations)')
    ax.set_ylabel('y (3L Operations)')
    ax.set_zlabel('5x + 3y')
    ax.set_title('3D Solution Space: 5x + 3y = 4 Plane')
    
    # Add a plane at z=4
    xx, yy = np.meshgrid(range(-10, 11), range(-10, 11))
    zz = np.full(xx.shape, 4)
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='red')
    
    return plt

# Run the complete analysis
if __name__ == "__main__":
    # Find all mathematical solutions
    solutions = find_solution_equation()
    
    # Simulate the practical solution
    final_amount = 0
    steps = []
    if solutions:
        # Use the solution that matches our manual approach: 5(2) + 3(-2) = 10 - 6 = 4
        target_solution = None
        for x, y in solutions:
            if x == 2 and y == -2:
                target_solution = (x, y)
                break
        
        if target_solution:
            final_amount, steps = simulate_from_equation(*target_solution)
            if final_amount == 4:
                print("✓ Successfully measured 4 liters!")
            else:
                print("✗ Failed to measure 4 liters")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Plot 1: Integer solutions
    plt1 = plot_solutions(solutions)
    plt1.savefig('water_problem_solutions.png', dpi=300, bbox_inches='tight')
    plt1.show()
    
    # Plot 2: Pouring process
    if steps:
        plt2 = plot_pouring_process(steps)
        plt2.savefig('water_pouring_process.png', dpi=300, bbox_inches='tight')
        plt2.show()
    
    # Plot 3: 3D solution space
    plt3 = plot_3d_solution_space()
    plt3.savefig('3d_solution_space.png', dpi=300, bbox_inches='tight')
    plt3.show()
    
    # Mathematical interpretation
    print("\n" + "=" * 60)
    print("MATHEMATICAL INTERPRETATION")
    print("=" * 60)
    
    print("The problem can be modeled as: 5x + 3y = 4")
    print("Where:")
    print("  x = net operations with 5L container")
    print("  y = net operations with 3L container")
    print("  Positive values: filling the container")
    print("  Negative values: emptying the container")
    
    # GCD analysis
    a, b, c = 5, 3, 4
    gcd_ab = math.gcd(a, b)
    
    print(f"\nGCD({a}, {b}) = {gcd_ab}")
    if c % gcd_ab == 0:
        print(f"Since {gcd_ab} divides {c}, integer solutions exist!")
        print(f"Number of solutions in range [-10,10]: {len(solutions)}")
    
    print("\nThe practical solution (x=2, y=-2) means:")
    print("  - Use the 5L container positively 2 times (fill it twice)")
    print("  - Use the 3L container negatively 2 times (empty it twice)")
    print("  - Net result: 5×2 + 3×(-2) = 10 - 6 = 4 liters")
    