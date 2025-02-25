from qiskit.circuit import Gate  # Import the Gate class for custom gates
import numpy as np  # Import NumPy for numerical operations

class VirtualZGate(Gate):
    """
    Custom Virtual-Z Gate for Qiskit.

    This gate represents a virtual Z rotation and is parameterized by an angle `theta`.

    Mathematically, it performs the following unitary operation:

        U = [[exp(-i * π * theta),  0]
             [0,  exp(i * π * theta)]]

    The `theta` parameter determines the phase shift applied to the qubit.

    Attributes:
        - theta (float): The phase rotation angle.
    """

    def __init__(self, theta):
        """
        Initialize the Virtual-Z Gate.

        Args:
            theta (float): Rotation angle for the Z gate.
        """
        super().__init__("virt_z", 1, [theta])  # Define a 1-qubit gate with a parameter

    def to_matrix(self):
        """
        Compute the unitary matrix representation of the Virtual-Z Gate.

        Returns:
            np.ndarray: The 2x2 unitary matrix corresponding to the Virtual-Z Gate.
        """
        theta = self.params[0]  # Extract the parameter (rotation angle)
        return np.array([
            [np.exp(-1j * np.pi * theta), 0],  # Phase shift for |0⟩ state
            [0, np.exp(1j * np.pi * theta)]   # Phase shift for |1⟩ state
        ])


class GPIGate(Gate):
    """
    Custom GPI(ϕ) Gate for Qiskit.

    This gate applies a phase-dependent transformation on a single qubit.

    Mathematically, it performs the following unitary operation:

        U = [[0, e^(-2πiϕ)]
             [e^(2πiϕ), 0]]

    The parameter `ϕ` defines the phase applied to the qubit.

    Attributes:
        - ϕ (float): Phase parameter.
    """

    def __init__(self, phi):
        """
        Initialize the GPI Gate.

        Args:
            phi (float): Phase shift angle.
        """
        super().__init__("gpi", 1, [phi])  # Define a 1-qubit gate with one parameter

    def to_matrix(self):
        """
        Compute the unitary matrix representation of the GPI Gate.

        Returns:
            np.ndarray: The 2x2 unitary matrix corresponding to the GPI Gate.
        """
        phi = self.params[0]  # Extract the phase parameter
        return np.array([
            [0, np.exp(-2j * np.pi * phi)],
            [np.exp(2j * np.pi * phi), 0]
        ])


class GPI2Gate(Gate):
    """
    Custom GPI2(ϕ) Gate for Qiskit.

    This gate applies a generalized phase transformation on a single qubit.

    Mathematically, it performs the following unitary operation:

        U = (1/√2) * [[ 1, -i e^(-2πiϕ)]
                       [-i e^(2πiϕ), 1]]

    The parameter `ϕ` defines the phase shift applied to the qubit.

    Attributes:
        - ϕ (float): Phase parameter.
    """

    def __init__(self, phi):
        """
        Initialize the GPI2 Gate.

        Args:
            phi (float): Phase shift angle.
        """
        super().__init__("gpi2", 1, [phi])  # Define a 1-qubit gate with one parameter

    def to_matrix(self):
        """
        Compute the unitary matrix representation of the GPI2 Gate.

        Returns:
            np.ndarray: The 2x2 unitary matrix corresponding to the GPI2 Gate.
        """
        phi = self.params[0]  # Extract the phase parameter
        factor = 1 / np.sqrt(2)  # Normalization factor
        exp_pos = -1j * np.exp(2j * np.pi * phi)
        exp_neg = -1j * np.exp(-2j * np.pi * phi)

        return factor * np.array([
            [1, exp_neg],
            [exp_pos, 1]
        ])
