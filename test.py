import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
import sympy

# Helper function to plot vectors in 2D
def plot_vector(ax, origin, vector, color='k', label=None, linestyle='-', head_width=0.1, head_length=0.15):
    ax.quiver(origin[0], origin[1], vector[0], vector[1],
              angles='xy', scale_units='xy', scale=1,
              color=color, label=label, linestyle=linestyle,
              headwidth=head_width / ax.get_xlim_size() * 10, # Adjust head size relative to plot
              headlength=head_length / ax.get_ylim_size() * 10)

# Helper function to plot vectors in 3D
def plot_vector_3d(ax, origin, vector, color='k', label=None, linestyle='-'):
    ax.quiver(origin[0], origin[1], origin[2],
              vector[0], vector[1], vector[2],
              color=color, label=label, linestyle=linestyle, length=np.linalg.norm(vector), normalize=False)


def analyze_and_visualize_matrix(A_np, matrix_name="Matrix A"):
    print(f"\n--- Analyzing {matrix_name} ---")
    print(f"A = \n{A_np}\n")

    n = A_np.shape[0]
    A_sym = sympy.Matrix(A_np)
    lam = sympy.symbols('lambda')
    I_sym = sympy.eye(n)

    # 1. Characteristic Polynomial
    char_poly_matrix = A_sym - lam * I_sym
    char_poly = sympy.det(char_poly_matrix)
    print(f"Characteristic Polynomial P(λ) = det(A - λI) = {sympy.simplify(char_poly)} = 0")

    # 2. Eigenvalues (from numpy for numerical stability, sympy for exact if possible)
    eigenvalues_np, eigenvectors_np = np.linalg.eig(A_np)
    # Sort them for consistent output, largest magnitude first for numpy
    idx = np.argsort(np.abs(eigenvalues_np))[::-1]
    eigenvalues_np = eigenvalues_np[idx]
    eigenvectors_np = eigenvectors_np[:, idx]


    print(f"Numerical Eigenvalues (from numpy): {np.round(eigenvalues_np, 5)}")
    # For printing, try to get symbolic roots if polynomial is simple enough
    try:
        symbolic_roots = sympy.roots(char_poly, lam)
        print(f"Symbolic Eigenvalues (from sympy roots): {symbolic_roots}")
    except NotImplementedError:
        print("Symbolic roots for this polynomial are complex to find.")


    # 3. Algebraic Multiplicity
    # Use rounded numpy eigenvalues for determining uniqueness and counts
    unique_eigenvalues, counts = np.unique(np.round(eigenvalues_np, 5), return_counts=True)
    algebraic_multiplicities = dict(zip(unique_eigenvalues, counts))
    print("\nAlgebraic Multiplicities (m_a):")
    for val, count in algebraic_multiplicities.items():
        print(f"  λ = {val:.{5}f}: m_a = {count}")

    # 4. Geometric Multiplicity & Eigenvectors
    print("\nEigenvectors and Geometric Multiplicities (m_g):")
    geometric_multiplicities = {}
    all_eigenvectors_for_plotting = [] # Store (eigenvalue, eigenvector) pairs

    for i, val_rounded in enumerate(unique_eigenvalues):
        # Find the original complex eigenvalue that corresponds to this rounded unique one
        # This is important if eigenvalues are complex or very close
        original_val_indices = np.where(np.isclose(eigenvalues_np, val_rounded))[0]
        original_val = eigenvalues_np[original_val_indices[0]] # Take the first match

        print(f"  For λ = {original_val:.{5}f}:")

        # Geometric multiplicity: dim(Null(A - λI)) = n - rank(A - λI)
        # Use original_val for higher precision in (A - λI)
        null_space_matrix = A_np - original_val * np.eye(n)
        # For rank calculation, it's often better to use a small tolerance or convert to float if it's not already
        rank = np.linalg.matrix_rank(null_space_matrix.astype(complex if np.iscomplexobj(A_np) or np.iscomplexobj(original_val) else float))
        geom_mult = n - rank
        geometric_multiplicities[val_rounded] = geom_mult
        print(f"    Geometric Multiplicity (m_g) = {n} - rank(A - ({original_val:.{5}f})I) = {n} - {rank} = {geom_mult}")

        # Eigenvectors from numpy for this unique eigenvalue
        # These are the columns in eigenvectors_np corresponding to original_val
        print(f"    Basis for Eigenspace E_({original_val:.{5}f}) (Eigenvectors from numpy):")
        current_eigenvalue_eigenvectors = []
        for k in original_val_indices:
            # Only take up to geom_mult linearly independent ones if numpy gives more due to numerical issues
            # However, numpy.linalg.eig usually gives a full set of L.I. eigenvectors if diagonalizable
            # If not, the concept of "eigenvectors for this eigenvalue" is tricky.
            # We trust numpy.linalg.eig gives L.I. vectors.
            # The check for diagonalizability will use m_a vs m_g.
            vec = eigenvectors_np[:, k]
            print(f"      v{k+1} = {np.round(vec, 5)}")
            all_eigenvectors_for_plotting.append((original_val, vec))
            current_eigenvalue_eigenvectors.append(vec)
        
        # If m_g > number of vecs from original_val_indices (e.g. repeated real eigenvalue)
        # This part is tricky as numpy.linalg.eig might return a full set of eigenvectors
        # even if m_g < m_a for some eigenvalues, potentially spanning a larger space
        # than the true eigenspace if the matrix is defective.
        # The rank calculation is the most reliable for m_g.

    # 5. Diagonalizability
    is_diagonalizable = True
    print("\nDiagonalizability Check:")
    for val_rounded in unique_eigenvalues:
        ma = algebraic_multiplicities[val_rounded]
        mg = geometric_multiplicities[val_rounded]
        print(f"  For λ = {val_rounded:.{5}f}: m_a = {ma}, m_g = {mg}. Condition m_g = m_a is {mg == ma}.")
        if mg != ma:
            is_diagonalizable = False

    if is_diagonalizable:
        print("Result: The matrix IS diagonalizable (all m_g == m_a).")
        # Check if sum of geometric multiplicities is n (equivalent condition)
        sum_mg = sum(geometric_multiplicities.values())
        if sum_mg == n:
            print(f"Sum of geometric multiplicities = {sum_mg} == n ({n}). Confirmed.")
        else: # Should not happen if individual m_g == m_a and sum of m_a == n
            print(f"Error: Sum of geometric multiplicities = {sum_mg} != n ({n}), but individual checks passed.")

    else:
        print("Result: The matrix IS NOT diagonalizable (at least one m_g < m_a).")

    # --- Visualization ---
    # Only visualize real eigenvectors for simplicity in this script
    real_eigenvectors_for_plotting = [(val, vec) for val, vec in all_eigenvectors_for_plotting if np.all(np.isreal(vec)) and np.isreal(val)]
    if not real_eigenvectors_for_plotting:
        print("\nNo real eigenvalues/eigenvectors to visualize.")
        return

    if n == 2 and real_eigenvectors_for_plotting:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Visualization for {matrix_name}")
        ax.axhline(0, color='grey', lw=0.5)
        ax.axvline(0, color='grey', lw=0.5)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.grid(True, linestyle=':', alpha=0.7)

        # Plot unit circle and its transformation
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = np.array([np.cos(theta), np.sin(theta)])
        ax.plot(circle[0, :], circle[1, :], color='lightgray', linestyle='--', label='Unit Circle')

        transformed_circle = A_np @ circle
        ax.plot(transformed_circle[0, :], transformed_circle[1, :], color='cyan', label='Transformed Circle (A @ Unit Circle)')

        colors = ['red', 'green', 'purple', 'orange'] # Cycle through colors
        plotted_labels = set()

        # Store min/max for plot limits
        all_x_coords = [0] + list(transformed_circle[0,:])
        all_y_coords = [0] + list(transformed_circle[1,:])

        for i, (val, vec) in enumerate(real_eigenvectors_for_plotting):
            val = np.real(val) # Ensure val is real for plotting
            vec = np.real(vec) # Ensure vec is real

            # Normalize eigenvector for consistent plotting length (optional, but good for viz)
            # vec_norm = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 1e-6 else vec
            vec_norm = vec # Plot original eigenvector length

            color = colors[i % len(colors)]

            # Eigenvector v
            label_v = f"v (λ={val:.2f})" if f"v (λ={val:.2f})" not in plotted_labels else None
            plot_vector(ax, [0,0], vec_norm, color=color, linestyle='-', label=label_v)
            if label_v: plotted_labels.add(label_v)
            all_x_coords.extend([0, vec_norm[0]])
            all_y_coords.extend([0, vec_norm[1]])


            # Transformed Eigenvector Av
            Av = A_np @ vec_norm
            label_Av = f"Av (scaled by λ)" if f"Av (scaled by λ)" not in plotted_labels and i==0 else None # only one label for this
            plot_vector(ax, [0,0], Av, color=color, linestyle=':', label=label_Av, head_width=0.12, head_length=0.18)
            if label_Av: plotted_labels.add(label_Av)
            all_x_coords.extend([0, Av[0]])
            all_y_coords.extend([0, Av[1]])


        # Set plot limits
        padding = 0.5
        min_x, max_x = min(all_x_coords), max(all_x_coords)
        min_y, max_y = min(all_y_coords), max(all_y_coords)
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)

        ax.legend(fontsize='small')
        ax.set_aspect('equal', adjustable='box') # Make x and y scales equal

    elif n == 3 and real_eigenvectors_for_plotting:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Eigenvectors for {matrix_name}")
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")

        # Origin lines
        ax.plot([0,0],[0,0],[-1,1], color='grey', lw=0.5) # Z
        ax.plot([0,0],[-1,1],[0,0], color='grey', lw=0.5) # Y
        ax.plot([-1,1],[0,0],[0,0], color='grey', lw=0.5) # X


        colors = ['red', 'green', 'blue', 'purple', 'orange']
        plotted_labels = set()

        all_coords = [0]
        for i, (val, vec) in enumerate(real_eigenvectors_for_plotting):
            val = np.real(val)
            vec = np.real(vec)
            color = colors[i % len(colors)]

            # Eigenvector v
            label_v = f"v (λ={val:.2f})" if f"v (λ={val:.2f})" not in plotted_labels else None
            plot_vector_3d(ax, [0,0,0], vec, color=color, linestyle='-', label=label_v)
            if label_v: plotted_labels.add(label_v)
            all_coords.extend(list(vec))

            # Transformed Eigenvector Av
            Av = A_np @ vec
            label_Av = f"Av (scaled by λ)" if f"Av (scaled by λ)" not in plotted_labels and i==0 else None
            plot_vector_3d(ax, [0,0,0], Av, color=color, linestyle=':', label=label_Av)
            if label_Av: plotted_labels.add(label_Av)
            all_coords.extend(list(Av))

        limit_val = max(np.abs(all_coords)) * 1.1 if all_coords else 1
        ax.set_xlim([-limit_val, limit_val])
        ax.set_ylim([-limit_val, limit_val])
        ax.set_zlim([-limit_val, limit_val])
        ax.legend(fontsize='small')
    plt.tight_layout()


# --- Define Example Matrices ---

# 1. Simple Diagonalizable 2x2 (Symmetric, distinct real eigenvalues)
A1 = np.array([[3, 1],
               [1, 2]])

# 2. Diagonalizable 2x2 (Non-symmetric, distinct real eigenvalues)
A2 = np.array([[4, -1],
               [2,  1]]) # Eigenvalues: 3, 2

# 3. Non-Diagonalizable 2x2 (Shear matrix, repeated eigenvalue, m_g < m_a)
A3 = np.array([[1, 1],
               [0, 1]]) # Eigenvalue: 1 (m_a=2), but m_g=1

# 4. Diagonalizable 2x2 (Identity * scalar - scaling matrix, repeated eigenvalue, m_g = m_a)
A4 = np.array([[2, 0],
               [0, 2]]) # Eigenvalue: 2 (m_a=2), m_g=2

# 5. Diagonalizable 3x3 (Your example)
A5 = np.array([[1, 1, 0],
               [1, 1, 0],
               [0, 0, 2]]) # Eigenvalues: 0, 2, 2 (m_a(2)=2, m_g(2)=2)

# 6. Non-Diagonalizable 3x3 (Jordan block form)
A6 = np.array([[2, 1, 0],
               [0, 2, 1],
               [0, 0, 2]]) # Eigenvalue: 2 (m_a=3), m_g=1

# 7. Matrix with complex eigenvalues (Rotation + Scaling)
A7 = np.array([[1, -1],
               [1,  1]]) # Eigenvalues: 1+i, 1-i

# --- Run Analysis ---
analyze_and_visualize_matrix(A1, "Matrix A1 (2x2 Symmetric Diagonalizable)")
analyze_and_visualize_matrix(A2, "Matrix A2 (2x2 Non-Symmetric Diagonalizable)")
analyze_and_visualize_matrix(A3, "Matrix A3 (2x2 Shear, Non-Diagonalizable)")
analyze_and_visualize_matrix(A4, "Matrix A4 (2x2 Scaling, Diagonalizable)")
analyze_and_visualize_matrix(A5, "Matrix A5 (Your 3x3 Example, Diagonalizable)")
analyze_and_visualize_matrix(A6, "Matrix A6 (3x3 Jordan Block, Non-Diagonalizable)")
analyze_and_visualize_matrix(A7, "Matrix A7 (2x2 Complex Eigenvalues)")


plt.show()