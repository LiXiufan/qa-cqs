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
            [np.exp(-1j *  theta), 0],  # Phase shift for |0⟩ state
            [0, np.exp(1j *  theta)]   # Phase shift for |1⟩ state
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
            [0, np.exp(-1j * phi)],
            [np.exp(1j * phi), 0]
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
        exp_pos = -1j * np.exp(1j * phi)
        exp_neg = -1j * np.exp(-1j * phi)

        return factor * np.array([
            [1, exp_neg],
            [exp_pos, 1]
        ])

class FullMSGate(Gate):
    """
    Custom Full Mølmer-Sørensen (MS) Gate for Qiskit.

    This gate applies a fully entangling operation between two qubits with two phase parameters.

    Mathematically, it performs the following unitary operation:

        U = (1/√2) * [[1,  0,  0,  -i e^(-2πi (ϕ₀ + ϕ₁)) ]
                      [0,  1,  -i e^(-2πi (ϕ₀ - ϕ₁)),  0 ]
                      [0,  -i e^(2πi (ϕ₀ - ϕ₁)),  1,  0 ]
                      [-i e^(2πi (ϕ₀ + ϕ₁)),  0,  0,  1 ]]

    The parameters `ϕ₀` and `ϕ₁` control the phase shifts applied to the qubits.

    Attributes:
        - ϕ₀ (float): Phase parameter 1
        - ϕ₁ (float): Phase parameter 2
    """

    def __init__(self, phi0, phi1):
        """
        Initialize the FullMS Gate.

        Args:
            phi0 (float): First phase shift angle.
            phi1 (float): Second phase shift angle.
        """
        super().__init__("fullMS", 2, [phi0, phi1])  # Define a 2-qubit gate with 2 parameters

    def to_matrix(self):
        """
        Compute the unitary matrix representation of the FullMS Gate.

        Returns:
            np.ndarray: The 4x4 unitary matrix corresponding to the FullMS Gate.
        """
        phi0, phi1 = self.params  # Extract the phase parameters
        factor = 1 / np.sqrt(2)  # Normalization factor

        exp_14 = -1j * np.exp(-1j * (phi0 + phi1))
        exp_23 = -1j * np.exp(-1j * (phi0 - phi1))
        exp_32 = -1j * np.exp(1j * (phi0 - phi1))
        exp_41 = -1j * np.exp(1j * (phi0 + phi1))

        return factor * np.array([
            [1, 0, 0, exp_14],
            [0, 1, exp_23, 0],
            [0, exp_32, 1, 0],
            [exp_41, 0, 0, 1]
        ])


class PartialMSGate(Gate):
    """
    Custom Partial Mølmer-Sørensen (MS) Gate for Qiskit.

    This gate applies an entangling operation with two phase parameters and an additional
    rotation parameter `θ` controlling the entanglement strength.

    Mathematically, it performs the following unitary operation:

        U = [[ cos(πθ),  0,  0,  -i e^(-2πi (ϕ₀ + ϕ₁)) sin(πθ) ]
             [ 0,  cos(πθ),  -i e^(-2πi (ϕ₀ - ϕ₁)) sin(πθ),  0 ]
             [ 0,  -i e^(2πi (ϕ₀ - ϕ₁)) sin(πθ),  cos(πθ),  0 ]
             [ -i e^(2πi (ϕ₀ + ϕ₁)) sin(πθ),  0,  0,  cos(πθ) ]]

    The parameters `ϕ₀`, `ϕ₁`, and `θ` control the entangling operation.

    Attributes:
        - ϕ₀ (float): Phase parameter 1
        - ϕ₁ (float): Phase parameter 2
        - θ (float): Rotation parameter (controls entanglement strength)
    """

    def __init__(self, phi0, phi1, theta):
        """
        Initialize the PartialMS Gate.

        Args:
            phi0 (float): First phase shift angle.
            phi1 (float): Second phase shift angle.
            theta (float): Rotation parameter.
        """
        super().__init__("partialMS", 2, [phi0, phi1, theta])  # Define a 2-qubit gate with 3 parameters

    def to_matrix(self):
        """
        Compute the unitary matrix representation of the PartialMS Gate.

        Returns:
            np.ndarray: The 4x4 unitary matrix corresponding to the PartialMS Gate.
        """
        phi0, phi1, theta = self.params  # Extract parameters
        cos_theta = np.cos(theta/2)
        sin_theta = np.sin(theta/2)

        e_pos = -1j * np.exp(1j * (phi0 + phi1))
        e_neg = -1j * np.exp(-1j * (phi0 + phi1))
        e_diff_pos = -1j * np.exp(1j * (phi0 - phi1))
        e_diff_neg = -1j * np.exp(-1j * (phi0 - phi1))

        return np.array([
            [cos_theta, 0, 0, e_neg * sin_theta],
            [0, cos_theta, e_diff_neg * sin_theta, 0],
            [0, e_diff_pos * sin_theta, cos_theta, 0],
            [e_pos * sin_theta, 0, 0, cos_theta]
        ])
