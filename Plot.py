# Import Libraries
import numpy as np  # For scientific computation
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt  # For plotting capabilities
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from termcolor import colored
from scipy.constants import *  # For Scientific Constants
import colorcet as cc  # For colormaps
import seaborn as sns # For color schemes
from mpmath import mpf, mpc, mp # For Precision Arithmetic

My_Color_Map = ""

# Initialize Variables

V = []
V_Old = []
V_Top = []
V_Bottom = []
V_Left = []
V_Right = []
Delta_V = []

E = []
Ex = []
Ey = []
E_Calculated = []

x_Total_Points = 0
x_Min = 0
x_Max = 0
x_Mid = 0
x = []

y_Total_Points = 0
y_Min = 0
y_Max = 0
y_Mid = 0
y = []

x_Meshgrid = []
y_Meshgrid = []

sigma = []
delta = 0

id = []
'''
ID's: 
0 >> Boundary Condition
1 >> Undetermined
2 >> Set
'''

thickness = 0
radius = 0
surface_Charge_Density = 0
factor = 0


def plot_Potential_3D(x, y, V):

    # Create Figure
    fig = plt.figure()
    # Create Axes for Figure
    ax = fig.add_subplot(111, projection='3d')
    # Define Surface
    surf = ax.plot_surface(x, y, V, rstride=1, cstride=1, cmap=My_Color_Map,
                           linewidth=0.1, antialiased=True)
    # Define View
    ax.view_init(35, 35)

    # Set Title and Names of Axes
    ax.set_xlabel('x (PU)', fontsize=12)
    ax.set_ylabel('y (PU)', fontsize=12)
    ax.set_zlabel('Potential (PU)', fontsize=12)

    # Set Colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    return None


def plot_electric_Field_2D(x, y, E):
    global My_Color_Map
    global Ex
    global Ey

    # Create Figure
    fig, ax = plt.subplots()

    Ex = E[1]
    Ey = E[0]


    # Create Stream Plot
    strm = ax.streamplot(x, y, E[1], E[0], color=E[0], linewidth=1.5, cmap=My_Color_Map,
                         density=2.0, arrowstyle='->', arrowsize=1.0)
    #E[1] >> Velocity of x
    #E[0] >> Velocity of y

    # Set Title and Names of Axes
    plt.xlabel('x (PU)', fontsize=12)
    plt.ylabel('y (PU)', fontsize=12)

    # Create Quiver Plot
    norm = matplotlib.colors.Normalize()
    norm.autoscale(np.linalg.norm(E))
    sm = matplotlib.cm.ScalarMappable(cmap=My_Color_Map, norm=norm)
    sm.set_array([])
    plt.quiver(x, y, E[1], E[0], color = cc.gray)
    plt.colorbar(sm)
    plt.show()

    return None


def laplace(V):
    global x_Total_Points
    global y_Total_Points

    # Solve laplace's Equation Numerically
    for i in range(1, x_Total_Points - 1):
        for j in range(1, y_Total_Points - 1):
            V[i, j] = 0.25 * (V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1])
    return V


def poisson(V):
    global x_Total_Points
    global y_Total_Points
    global sigma  # Surface Charge Density
    global delta  # Stepsize

    # Solve poisson's Equation Numerically
    for r in range(1, y_Total_Points - 1):
        for c in range(1, x_Total_Points - 1):
            V[r, c] = 0.25 * (
                    V[r + 1, c] + V[r - 1, c] + V[r, c + 1] + V[r, c - 1] + (sigma[r][c] / epsilon_0 * delta ** 2))
    return V


def electric_Field(V):
    global E

    # Calculate the Electric Field by Taking the Divergence
    E = np.gradient(-V)

    return E


def create_2D_Space():
    global x_Total_Points
    global x_Min
    global x_Max
    global x_Mid
    global x

    global y_Total_Points
    global y_Min
    global y_Max
    global y_Mid
    global y

    global x_Meshgrid
    global y_Meshgrid

    global delta

    global id

    global factor

    # Define the x-coordinates
    x_Total_Points = 50
    x_Min = -5
    x_Max = 5
    x = np.linspace(x_Min, x_Max, x_Total_Points)
    x_Mid = int(x_Total_Points / 2)

    # Define the y-coordinates
    y_Total_Points = x_Total_Points
    y_Min = x_Min
    y_Max = x_Max
    y = np.linspace(y_Min, y_Max, y_Total_Points)
    y_Mid = int(y_Total_Points / 2)

    delta = (x_Max - x_Min) / x_Total_Points
    #Delta is the same for both the x- and y-axis.

    # Create a Meshgrid based on the Coordinates
    x_Meshgrid, y_Meshgrid = np.meshgrid(x, y)

    # Create ID Matrix
    id = np.ones((x_Total_Points, y_Total_Points))

    factor = x_Max

    return None


def initial_Fields():
    global V
    global V_Old
    global id

    global E
    global Ex
    global Ey

    global x_Total_Points
    global y_Total_Points

    V = np.zeros((x_Total_Points, y_Total_Points))
    V_Old = np.zeros_like(V)
    Ex = np.zeros_like(V)
    Ey = np.zeros_like(V)

    return None


def dirichlet():
    global V
    global V_Top
    global V_Bottom
    global V_Left
    global V_Right
    global id

    global x_Total_Points
    global y_Total_Points

    V_Top = 0.0
    V_Bottom = 0.0
    V_Left = 0.0
    V_Right = 0.0

    V[:, x_Total_Points - 1] = V_Right
    V[:, 0] = V_Left
    V[y_Total_Points - 1, :] = V_Bottom
    V[0, :] = V_Top

    id[:, x_Total_Points - 1] = 0
    id[:, 0] = 0
    id[y_Total_Points - 1, :] = 0
    id[0, :] = 0

    return None


def potential_Parallel_Capacitor():
    global V
    global id
    global sigma
    global thickness
    global delta

    thickness = 0.5

    i = int(thickness / delta)

    V[3:y_Total_Points - 3, 5:5+i] = 10
    V[3:y_Total_Points - 3, x_Total_Points - 4-i:x_Total_Points - 4] = -10

    id[3:y_Total_Points - 3, 5:5 + i] = 2
    id[3:y_Total_Points - 3, x_Total_Points - 4-i:x_Total_Points - 4] = 2


    return None


def potential_Point():
    global V
    global id
    global sigma
    global thickness
    global delta
    global surface_Charge_Density

    surface_Charge_Density = 10e-12
    sigma = np.zeros_like(V)

    V[y_Mid][x_Mid] = 0
    id[y_Mid][x_Mid] = 2
    sigma[y_Mid][x_Mid] = surface_Charge_Density

    return None

def potential_Line_Charge():
    global V
    global id
    global sigma
    global thickness
    global delta
    global surface_Charge_Density

    thickness = 0.25
    i = int(thickness / delta)

    surface_Charge_Density = 5e-12

    sigma = np.zeros_like(V)

    V[y_Mid-i:y_Mid+i, 5:x_Total_Points-5] = 10
    sigma[y_Mid - i:y_Mid + i, 5:x_Total_Points - 5] = surface_Charge_Density
    id[y_Mid - i:y_Mid + i, 5:x_Total_Points - 5] = 2



    return None


def laplace_Parallel_Capacitor():
    global V
    global V_Old
    global Delta_V

    global E_Calculated

    # Set Iterations
    iterations = 2000

    # Use laplace's Equation to Calculate V
    for i in range(0, iterations):
        V = laplace(V)
        V_Old = V

        dirichlet()
        potential_Parallel_Capacitor()

    # Calculate the Electric Field
    E_Calculated = electric_Field(V)

    return None


def poisson_Point_Charge():
    global V
    global V_Old
    global Delta_V

    global E_Calculated

    # Set Iterations
    iterations = 2000

    # Use poisson's Equation to Calculate V

    for i in range(0, iterations):
        V = poisson(V)
        V_Old = V

        dirichlet()

        # Calculate the Change in Potential
        Delta_V = V - V_Old

    # Calculate the Electric Field
    E_Calculated = electric_Field(V)
    

    return None

def poisson_Line_Charge():
    global V
    global V_Old
    global Delta_V

    global E_Calculated

    # Set Iterations
    iterations = 2000

    # Use poisson's Equation to Calculate V

    for i in range(0, iterations):
        V = poisson(V)
        V_Old = V

        dirichlet()

        # Calculate the Change in Potential
        Delta_V = V - V_Old

    # Calculate the Electric Field
    E_Calculated = electric_Field(V)
    

    return None


def plot_Graphs():
    plot_Potential_3D(x_Meshgrid, y_Meshgrid, V)
    plot_electric_Field_2D(x_Meshgrid, y_Meshgrid, E_Calculated)
    plt.show()
    return None


def parallel_Capacitor():
    global My_Color_Map
    My_Color_Map = cm.inferno

    create_2D_Space()
    initial_Fields()
    dirichlet()
    potential_Parallel_Capacitor()
    laplace_Parallel_Capacitor()
    plot_Graphs()


def point_Charge():
    global My_Color_Map
    global x
    global V
    My_Color_Map = cm.inferno

    create_2D_Space()
    initial_Fields()
    dirichlet()
    potential_Point()
    poisson_Point_Charge()
    plot_Graphs()
    return None

def line_Charge():
    global My_Color_Map
    global x
    global V
    My_Color_Map = cm.inferno

    create_2D_Space()
    initial_Fields()
    dirichlet()
    potential_Line_Charge()
    poisson_Line_Charge()
    plot_Graphs()
    return None

print('Generating graphs for Point Charge...')
point_Charge()
print('Generating graphs for Line Charge...')
line_Charge()
print('Generating graphs for Parallel Capacitor...')
parallel_Capacitor()
