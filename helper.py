# Helper functions for "Bernoulli Line and Bloch Sphere" laboratory
# 
# These functions are implement to prevent Pythonic distractions 
# from the key tasks of the laboratory. If learners have sufficent
# background the code below is written to be instructional and provides
# links to other sources.
#
#
# James Whitfield (2022)
# Dartmouth College 
# Amazon Visiting Academic
#
# AJ Cressman (2022)
# Dartmouth College

import matplotlib.pyplot as plt
import numpy as np
from braket.aws import AwsDevice
from braket.circuits import Circuit, Noise
from braket.devices import LocalSimulator

def run_circuit_task1(circuit,n_shots=1000, device=None, verbose=False, plot=True):
    """run_circuit_task1(circuit,n_shots=1000, device=None, verbose=False, plot=True)
    Runs a given single qubit experiment.

        Args:
            circuit

        Optional args:
            n_shots     Integer number of times to run the circuit. Default: 1000.
            device      The default device is the LocalSimulator
            verbose     Boolean variable for printing extra output. Default: False.
            plot        Boolean to create plot. Default: True.

        Returns:
            counts:     A dictionary with keys "0" and "1" containing the counts.

        Examples:
            >>> circuit = Circuit().h(0)
            >>> run_circuit_task1(circuit)
            >>> run_circuit_task1(circuit,n_shots=100)
            >>> run_circuit_task1(circuit,n_shots=300,verbose=True,plot=False)
    """

    if circuit.qubit_count > 1:
        print("Only single qubit circuits please. Returning zero.")
        return 0

    # We will be using the LocalSimulator (rather than an actual quantum device, or one of AWS's managed simulators).
    # The default device is the "Local State Vector Simulator".
    # For more documentation, see: https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html
    if device is None:
        device = LocalSimulator()

    # execute task on simulator
    # Note: this is a very common workflow -- set up circuit, run, get result as measurement counts, plot
    counts = device.run(circuit, shots=n_shots).result().measurement_counts

    # counts is a python dictionary
    # (technically, a Counter: https://docs.python.org/3/library/collections.html#collections.Counter):
    if verbose:
        print(counts)

    # plot measurement counts as percentages
    # plt.bar(x-values, y-values, ...options)
    # More documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
    
    if plot:
        plt.bar(("0", "1"), (100 * counts["0"] / n_shots, 100 * counts["1"] / n_shots))
        plt.ylim([0, 100])
        plt.xlabel("Outcome")
        plt.ylabel("Percent")

    if verbose:
        print("Zero counts:", counts["0"])
        print("One  counts:", counts["1"])

    return counts

def run_bernoulli_task(p,n_shots=None,verbose=False,plot=True):
    """run_bernoulli_task(p,n_shots=None,verbose=False,plot=True)
    
    Runs a given Bernoulli experiment.

        Args:
            p           Probability of Outcome 0

        Optional args:
            n_shots     Integer number of times to run the circuit. Default: 1000.
            verbose     Boolean to print extra. Default: False.
            plot        Boolean to create plot. Default: True.

        Returns:
            counts:     A dictionary with keys "0" and "1" containing the counts.

        
        Examples:
            >>> p=0.8
            >>> n=1000            
            >>> run_bernoulli_task(p,n)
            >>> run_bernoulli_task(p,n,verbose=True,plot=False)
    """
    if n_shots is None:
        n_shots = 1000

    circuit = Circuit()

    # Define a bit flip noise channel
    # Since the channel is considered noise, it can only cause error with prob 0 to 0.5.
    # Otherwise, it is performing a faulty X gate
    if p > 0.5:
        # apply Z gate to first qubit (counting starts at 0)
        # Note: If we left this out, we would just have an empty circuit, and our simulation would fail!
        circuit.z(0)
        noise = Noise.BitFlip(probability = 1 - p)
    else:
        # apply X gate to first qubit (counting starts at 0)
        circuit.x(0)
        noise = Noise.BitFlip(probability = p)

    circuit.apply_gate_noise(noise)

    # Our usual workflow, with a different device
    # Here we are using the "Local Density Matrix Simulator", because we are running noise instructions.
    # Students: change this back to the default (LocalSimulator()) and see what error message you get.
    device = LocalSimulator("braket_dm")
    counts = device.run(circuit, shots=n_shots).result().measurement_counts
    
    if(plot):
        plt.bar(("0", "1"), (100 * counts["0"] / n_shots, 100 * counts["1"] / n_shots))
        plt.ylim([0, 100])
        plt.xlabel("outcome")
        plt.ylabel("percent")

    if(verbose):
        print("Zero counts:", counts["0"])
        print("One  counts:", counts["1"])

    return counts



def get_bernoulli_val_to_estimate(seedval=None):
    """get_bernoulli_val_to_estimate(seedval=None)
    This function returns a random Bernoulli parameter p 
    
    **Do not print the value if you are performing an estimation procedure**

    Optional args:
        seedval         A seed value for the random number generator.
                        This ensures that the same random number is generated 
                        each time the function is run.
    Return:
        p_unknown       Bernoulli parameter to be estimated
    """

    # Don't dig into this too much...
    # Try to figure it out without printing p!
    
    if(seedval):
        np.random.seed(seedval)
        
    p_unknown = np.random.rand()
    
    return p_unknown

# This function will generate a circuit for a random state.
# Students: write some code to estimate the state of the qubit.
# Be as quantitative as possible.  And a plot would be nice!
def get_target_bias_parameters(seedval=1):
    """get_target_bias_parameters(seedval=1)
    This function returns random vector of unit length
    
    **Do not print the values if you are performing an estimation procedure**

    Optional args:
        seedval         A seed value for the random number generator.
                        This ensures that the same random number is generated 
                        each time the function is run.
    Return:
        x,y,z           Parameters to be estimated
    """
    # Don't dig into this too much...
    # Try to figure it out without printing the circuit parameters!
    np.random.seed(seedval)

    # initial
    xi = np.random.randn()
    yi = np.random.randn()
    zi = np.random.randn()

    R = np.sqrt(xi * xi + yi * yi + zi * zi)

    x = xi / R
    y = yi / R
    z = zi / R

    return x, y, z



def get_circuit_of_quantum_state_to_estimate(seedval=None):
    """get_circuit_of_quantum_state_to_estimate(seedval=None)
    
    This function returns a circuit to create a random qubit state
    
    **Do not print the circuit if you are performing an estimation procedure**

    Optional args:
        seedval         A seed value for the random number generator.
                        This ensures that the same random numbers are generated 
                        each time the function is run.
    Return:
        cir             A state preparation circuit
    """
    
    if(seedval):
        np.random.seed(seedval)
    
    state_prep_circuit = Circuit()
    state_prep_circuit.rx(0, np.random.rand() * 2 * np.pi)
    state_prep_circuit.ry(0, np.random.rand() * 2 * np.pi)

    return state_prep_circuit



def plot_qubit_state(qubit_biases,computed_uncertainties=None):
    """plot_qubit_state(qubit_biases,computed_uncertainties=None)
    
    Plots a pure state qubit along with uncertainties

    Args:
        qubit_biases                Dictionaries with keys "x", "y", "z". 
                                    Values should be normalized to unit vector length.
        computed_uncertainties      Dictionaries with keys "x", "y", "z" giving uncertainies.

    Examples:
        >>> qubit_biases           = {"x": 0.429, "y": -0.01, "z": 0.248}
        >>> computed_uncertainties = {"x": 0.5,   "y": 0.25,  "z": 0.25 }
        >>> plot_qubit_state(qubit_biases)
        >>> plot_qubit_state(qubit_biases,computed_uncertainties)

    """


    # More fun with matplotlib: https://matplotlib.org/3.5.0/gallery/
    fig = plt.figure(dpi=256)
    ax = fig.add_subplot(projection="3d")
    # here we create a 3D mesh of a sphere
    # Students: check to see that this makes sense
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="red", linewidth=0.1)
    ax.set_title("Vectors+Sphere")
    ax.view_init(45, 35)  # the viewing angle

    # Plot axes
    # We use the `quiver` function to plot arrows
    # More documentation: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.axes.Axes.quiver.html
    ax.quiver(
        # starting x-position of each arrow
        [0, 0, 0],
        # starting y-position of each arrow
        [0, 0, 0],
        # starting z-position of each arrow
        [0, 0, 0],
        # length in x-direction of each arrow
        [1, 0, 0],
        # length in y-direction of each arrow
        [0, 1, 0],
        # length in z-direction of each arrow
        [0, 0, 1],
        linewidth=1,
    )

    # Plot state
    ax.quiver(
        # starting x-position of each arrow
        [0],
        # starting y-position of each arrow
        [0],
        # starting z-position of each arrow
        [0],
        # length in x-direction of each arrow
        [qubit_biases["x"]],
        # length in y-direction of each arrow
        [qubit_biases["y"]],
        # final z-pos
        [qubit_biases["z"]],
        linewidth=3,
        color="green",
    )

    # Plot uncertainties
    # Note: separated for readability, but can be recombined into one big plot easily
    # start at center of bias and add 0.5 * uncertainty
    ax.quiver(
        # starting x-position of each arrow
        [qubit_biases["x"], 0, 0],
        # starting y-position of each arrow
        [0, qubit_biases["y"], 0],
        # starting z-position of each arrow
        [0, 0, qubit_biases["z"]],
        # length in x-direction of each arrow
        [0.5 * computed_uncertainties["x"], 0, 0],
        # length in y-direction of each arrow
        [0, 0.5 * computed_uncertainties["y"], 0],
        # length in z-direction of each arrow
        [0, 0, 0.5 * computed_uncertainties["z"]],
        color="black",
        linewidth=2,
    )
    # start at center of bias and subtract 0.5 * uncertainty
    ax.quiver(
        # starting x-position of each arrow
        [qubit_biases["x"], 0, 0],
        # starting y-position of each arrow
        [0, qubit_biases["y"], 0],
        # starting z-position of each arrow
        [0, 0, qubit_biases["z"]],
        # length in x-direction of each arrow
        [-0.5 * computed_uncertainties["x"], 0, 0],
        # length in y-direction of each arrow
        [0, -0.5 * computed_uncertainties["y"], 0],
        # length in z-direction of each arrow
        [0, 0, -0.5 * computed_uncertainties["z"]],
        color="black",
        linewidth=2,
    )
