###################################
# PHASE PORTRAIT ALT              #
###################################
from scipy.optimize import fsolve
#from scipy.integrate import odeint
import autograd.numpy as np
from autograd import jacobian
import matplotlib.pyplot as plt
#######################################
##### OSDR known model parameters #####
#######################################

# After the description of the neighbourhood dynamics ODE in Methods, we have
# dX/dt = X(p^+(X) - p^-(X)), with constant division rate parameter p^+(X) = 1 / (1+np.exp(-(intercept+a*F))), and death rate parameter p^-(X) = b.

# We expect this equation to have X decrease above its 16 SS, and increase below. This value is chosen to best correspond to the neighbourhood dynamics drawn in S2H.

# Define the system of ODEs
def ODE_system(state):
    F, M = state
    a, intercept = -0.120, -2.456
    b = 1 / (1+np.exp(-(intercept+a*16)))
    dF_dt = F * (1 / (1+np.exp(-(intercept+a*F))) - b)
    dM_dt = M * (1 / (1+np.exp(-(intercept+a*M))) - b)
    return [dF_dt, dM_dt]

def ODE_system_np(state):
    F, M = state
    a, intercept = -0.120, -2.456
    b = 1 / (1+np.exp(-(intercept+a*16)))
    dF_dt = F * (1 / (1+np.exp(-(intercept+a*F))) - b)
    dM_dt = M * (1 / (1+np.exp(-(intercept+a*M))) - b)
    return np.array([dF_dt, dM_dt])


# Generate rates for streamlines adapted to the log2 scale
def streamlines(exp_F, exp_M):
    F = 2**exp_F
    M = 2**exp_M
    dF_dt, dM_dt = ODE_system([F, M])
    return dF_dt, dM_dt


# Find nullclines using fsolve
def nullclines():
    F_range = np.logspace(0, 8, 400, base=2)
    M_range = np.logspace(0, 8, 400, base=2)

    F_nullcline = [fsolve(lambda F: ODE_system([F, M])[0], 16) for M in M_range]
    M_nullcline = [fsolve(lambda M: ODE_system([F, M])[1], 16) for F in F_range]
    
    return F_range, M_range, F_nullcline, M_nullcline

# Find fixed points using fsolve
def find_fixed_points():
    expected = [[0, 0], [16, 0], [0, 16], [16, 16]]
    fixed_points = [fsolve(ODE_system, expectation) for expectation in expected]
    return fixed_points

# Calculate stability using the Jacobian
def is_stable(fp):
    jac = jacobian(ODE_system_np)
    J = jac(fp)
    eigenvalues = np.linalg.eigvals(J)
    return all(e.real < 0 for e in eigenvalues)

def is_unstable(fp):
    jac = jacobian(ODE_system_np)
    J = jac(fp)
    eigenvalues = np.linalg.eigvals(J)
    return any(e.real > 0 for e in eigenvalues)

# Plot the phase portrait
def plot_phase_portrait():
    # Parameters for meshgrid
    exp_F_mesh = np.linspace(0, 8, 30)
    exp_M_mesh = np.linspace(0, 8, 30)
    exp_F, exp_M = np.meshgrid(exp_F_mesh, exp_M_mesh)

    # Calculate the growth rates for the streamlines
    F_rate, M_rate = streamlines(exp_F, exp_M)
    F_rate_scaled = F_rate / (2**exp_F)
    M_rate_scaled = M_rate / (2**exp_M)
    
    plt.figure()

    # Streamplot
    plt.streamplot(exp_F, exp_M, F_rate_scaled, M_rate_scaled,
                   color="black")

    # Nullclines
    F_range, M_range, F_nullcline, M_nullcline = nullclines()
    plt.plot(np.log2(F_nullcline), np.log2(M_range), 'b-', label='F nullcline')
    plt.plot(np.log2(F_range), np.log2(M_nullcline), 'r-', label='M nullcline')
    

    # Fixed points
    fixed_points = find_fixed_points()
    print(fixed_points)
    # with stability analysis
    label_added = {'Stable': False, 'Unstable': False, 'Semi-stable': False} #point label tracker to avoid redundancy
    for fp in fixed_points:
        x=(np.log2(fp[0])) if fp[0]!=0 else fp[0]
        y=(np.log2(fp[1])) if fp[1]!=0 else fp[1]
        stability=''
        if is_stable(fp):
            stability = 'Stable'
        elif is_unstable(fp):
            stability = 'Unstable'
        else:
            stability = 'Semi-stable'
        print(stability)
        fcolor = 'black' if stability == 'Stable' or stability == 'Semi-stable' else 'white'
        ecolor = 'black' if stability == 'Stable' or stability =='Unstable' else 'red'
        if label_added[stability]==False:
            plt.scatter(x, y, s=100, edgecolors=ecolor, facecolors=fcolor, label=f'{stability} Fixed Point', zorder=2)
            label_added[stability]=True
        else:
            plt.scatter(x, y, s=100, edgecolors=ecolor, facecolors=fcolor, zorder=2)

    # Labels and legend
    plt.xlabel('log2(F)')
    plt.ylabel('log2(M)')
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    plt.legend()
    plt.title('Phase Portrait')
    plt.grid(True)
    plt.show()


