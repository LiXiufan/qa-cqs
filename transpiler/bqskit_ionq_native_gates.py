# Re-import necessary libraries after execution state reset

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


# we want theta from 0 to pi
def normalize_theta(param):
    new_param = param
    while new_param > np.pi / 2:
        new_param -= np.pi / 2
    while new_param < 0:
        new_param += np.pi / 2
    return new_param


class VirtualZGate(QubitGate, DifferentiableUnitary, CachedClass):
    r"""
    A gate representing an arbitrary rotation around the Z axis.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\exp({-i\theta}) & 0 \\\\
        0 & \\exp({i\theta}) \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'virt_z'

    def get_unitary(self, params=None) -> UnitaryMatrix:
        r"""Return the unitary for this gate, see :class:`Unitary` for more."""
        if params is None:
            params = []
        self.check_parameters(params)

        exp1 = np.exp(-1j * params[0])
        exp2 = np.exp(1j * params[0])

        return UnitaryMatrix(
            [
                [exp1, 0],
                [0, exp2],
            ],
        )

    def get_grad(self, params=None) -> npt.NDArray[np.complex128]:
        r"""
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if params is None:
            params = []
        self.check_parameters(params)

        d1exp = -1j * np.exp(-1j * params[0])
        d2exp = 1j * np.exp(1j * params[0])

        return np.array(
            [
                [
                    [d1exp, 0],
                    [0, d2exp],
                ],
            ], dtype=np.complex128,
        )


class GPIGate(QubitGate, DifferentiableUnitary, CachedClass):
    r"""
    A gate representing the GPI(param) transformation.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        0 & e^{-2\pi i \phi} \\\\
        e^{2\pi i \phi} & 0
        \\end{pmatrix}

    This gate applies a phase-dependent transformation on the qubit.

    Attributes:
        - params (float): The phase parameter.
    """

    _num_qudits = 1  # This gate operates on a single qubit
    _num_params = 1  # It has one parameter: param
    _qasm_name = 'GPI'  # Custom gate name for QASM compatibility

    def get_unitary(self, params=None) -> UnitaryMatrix:
        r"""
        Return the unitary matrix representation of the GPI gate.

        Args:
            params (list[float]): List containing the phase parameter params.

        Returns:
            UnitaryMatrix: A 2x2 unitary matrix implementing the GPI gate.
        """
        if params is None:
            params = []
        self.check_parameters(params)

        phi = params[0]  # Extract the phase parameter
        return UnitaryMatrix(
            [
                [0, np.exp(-1j * phi)],  # Upper diagonal element
                [np.exp(1j * phi), 0]  # Lower diagonal element
            ]
        )

    def get_grad(self, params=None) -> npt.NDArray[np.complex128]:
        r"""
        Compute the gradient of the GPI gate with respect to param.

        Args:
            params (list[float]): List containing the phase parameter params.

        Returns:
            np.ndarray: A (1, 2, 2) gradient matrix.
        """
        if params is None:
            params = []
        self.check_parameters(params)

        phi = params[0]
        d_exp_pos = 1j * np.exp(1j * phi)  # Derivative w.r.t. param
        d_exp_neg = -1j * np.exp(-1j * phi)

        return np.array(
            [
                [
                    [0, d_exp_neg],  # Gradient at (0,1)
                    [d_exp_pos, 0]  # Gradient at (1,0)
                ]
            ],
            dtype=np.complex128
        )


class GPI2Gate(QubitGate, DifferentiableUnitary, CachedClass):
    r"""
    A gate representing the GPI2(param) transformation.

    This gate is defined by the following unitary matrix:

    .. math::

        \\frac{1}{\\sqrt{2}} \\begin{pmatrix}
        1 & -i e^{-2\pi i \phi} \\\\
        -i e^{2\pi i \phi} & 1
        \\end{pmatrix}

    This is a generalization of the GPI gate with additional phase properties.

    Attributes:
        - params (float): The phase parameter.
    """

    _num_qudits = 1  # This gate operates on a single qubit.
    _num_params = 1  # It takes one parameter: param (phi).
    _qasm_name = 'GPI2'  # The name used in QASM representation.

    def get_unitary(self, params=None) -> UnitaryMatrix:
        r"""
        Computes and returns the unitary matrix for the GPI2 gate.

        Args:
            params (list[float]): A single parameter params.

        Returns:
            UnitaryMatrix: A 2x2 unitary matrix representing the gate.
        """
        if params is None:
            params = []
        self.check_parameters(params)  # Ensure the parameter list is valid.

        phi = params[0]  # Extract the phase parameter param.
        factor = 1 / np.sqrt(2)  # Normalization factor.
        exp_pos = -1j * np.exp(1j * phi)  # e^(2πiparam) * -i
        exp_neg = -1j * np.exp(-1j * phi)  # e^(-2πiparam) * -i

        return UnitaryMatrix(
            factor * np.array([
                [1, exp_neg],  # First row
                [exp_pos, 1]  # Second row
            ])
        )

    def get_grad(self, params=None) -> npt.NDArray[np.complex128]:
        r"""
        Computes the gradient of the GPI2 gate with respect to param.

        This is useful for gradient-based optimization techniques in
        variational quantum algorithms.

        Args:
            params (list[float]): A single parameter param.

        Returns:
            np.ndarray: A (1, 2, 2) gradient matrix.
        """
        if params is None:
            params = []
        self.check_parameters(params)  # Validate input parameters.

        phi = params[0]  # Extract the phase parameter.
        factor = 1 / np.sqrt(2)  # Normalization factor.

        # Compute the derivatives of the exponential terms.
        d_exp_pos = np.exp(1j * phi)
        d_exp_neg = -np.exp(-1j * phi)

        return np.array(
            [
                factor * np.array([
                    [0, d_exp_neg],  # Gradient (0,1)
                    [d_exp_pos, 0]  # Gradient (1,0)
                ])
            ],
            dtype=np.complex128
        )


class FullMSGate(QubitGate, DifferentiableUnitary, CachedClass):
    r"""
    The FullMS (Mølmer-Sørensen) gate with two phase parameters.

    This gate applies an entangling operation between two qubits and
    is defined by the following unitary matrix:

    .. math::

        \\frac{1}{\\sqrt{2}} \\begin{bmatrix}
        1 & 0 & 0 & -i e^{-2\pi i (\phi_0 + \phi_1)} \\\\
        0 & 1 & -i e^{-2\pi i (\phi_0 - \phi_1)} & 0 \\\\
        0 & -i e^{2\pi i (\phi_0 - \phi_1)} & 1 & 0 \\\\
        -i e^{2\pi i (\phi_0 + \phi_1)} & 0 & 0 & 1
        \\end{bmatrix}

    Attributes:
        - param₀ (float): Phase parameter 1
        - param₁ (float): Phase parameter 2
    """

    _num_qudits = 2  # This gate operates on two qubits
    _num_params = 2  # It has two parameters: param₀ and param₁
    _qasm_name = 'fullMS'  # Custom gate name for QASM compatibility

    def get_unitary(self, params=None) -> UnitaryMatrix:
        r"""
        Return the unitary matrix representation of the FullMS gate.

        Args:
            params (list[float]): List containing the phase parameters param₀, param₁.

        Returns:
            UnitaryMatrix: A 4x4 unitary matrix implementing the FullMS gate.
        """
        if params is None:
            params = []
        self.check_parameters(params)

        phi0, phi1 = params  # Extract the phase parameters
        factor = 1 / np.sqrt(2)  # Normalization factor

        exp_14 = -1j * np.exp(-1j * (phi0 + phi1))
        exp_23 = -1j * np.exp(-1j * (phi0 - phi1))
        exp_32 = -1j * np.exp(1j * (phi0 - phi1))
        exp_41 = -1j * np.exp(1j * (phi0 + phi1))

        return UnitaryMatrix(
            factor * np.array([
                [1, 0, 0, exp_14],
                [0, 1, exp_23, 0],
                [0, exp_32, 1, 0],
                [exp_41, 0, 0, 1]
            ])
        )

    def get_grad(self, params=None) -> npt.NDArray[np.complex128]:
        r"""
        Compute the gradient of the FullMS gate with respect to param₀ and param₁.

        Args:
            params (list[float]): List containing the phase parameters param₀, param₁.

        Returns:
            np.ndarray: A (2, 4, 4) gradient matrix.
        """
        if params is None:
            params = []
        self.check_parameters(params)

        phi0, phi1 = params
        factor = 1 / np.sqrt(2)

        # Compute derivatives
        d_14 = -np.exp(-1j * (phi0 + phi1))
        d_21 = -np.exp(-1j * (phi0 - phi1))
        d_32 = np.exp(1j * (phi0 - phi1))
        d_41 = np.exp(1j * (phi0 + phi1))

        return np.array([
            factor * np.array([
                [0, 0, 0, d_14],
                [0, 0, d_21, 0],
                [0, d_32, 0, 0],
                [d_41, 0, 0, 0]
            ]),
            factor * np.array([
                [0, 0, 0, d_14],
                [0, 0, -d_21, 0],
                [0, -d_32, 0, 0],
                [d_41, 0, 0, 0]
            ])
        ], dtype=np.complex128)


class PartialMSGate(QubitGate, DifferentiableUnitary, CachedClass):
    r"""
    The Partial Mølmer-Sørensen (MS) gate with two phase parameters and an angle θ.

    This gate is defined by the following unitary matrix:

    .. math::

        \\begin{bmatrix}
        \\cos(\\pi \\theta) & 0 & 0 & -i e^{-2\pi i (\phi_0 + \phi_1)} \\sin(\\pi \\theta) \\\\
        0 & \\cos(\\pi \\theta) & -i e^{-2\pi i (\phi_0 - \phi_1)} \\sin(\\pi \\theta) & 0 \\\\
        0 & -i e^{2\pi i (\phi_0 - \phi_1)} \\sin(\\pi \\theta) & \\cos(\\pi \\theta) & 0 \\\\
        -i e^{2\pi i (\phi_0 + \phi_1)} \\sin(\\pi \\theta) & 0 & 0 & \\cos(\\pi \\theta)
        \\end{bmatrix}

    This gate allows for partial entanglement based on the parameter θ.

    Attributes:
        - param₀ (float): Phase parameter 1
        - param₁ (float): Phase parameter 2
        - θ (float): Rotation angle
    """

    _num_qudits = 2  # This gate operates on two qubits
    _num_params = 3  # It has three parameters: param₀, param₁, and θ
    _qasm_name = 'partialMS'  # Custom gate name for QASM compatibility

    def get_unitary(self, params=None) -> UnitaryMatrix:
        r"""
        Return the unitary matrix representation of the PartialMS gate.

        Args:
            params (list[float]): List containing param₀, param₁, and θ.

        Returns:
            UnitaryMatrix: A 4x4 unitary matrix implementing the PartialMS gate.
        """
        if params is None:
            params = []
        self.check_parameters(params)

        phi0, phi1, theta = params  # Extract the phase parameters and rotation angle
        cos = np.cos(normalize_theta(theta) / 2)
        sin = np.sin(normalize_theta(theta) / 2)
        e_pos = -1j * np.exp(1j * (phi0 + phi1))
        e_neg = -1j * np.exp(-1j * (phi0 + phi1))
        e_diff_pos = -1j * np.exp(1j * (phi0 - phi1))
        e_diff_neg = -1j * np.exp(-1j * (phi0 - phi1))

        return UnitaryMatrix(
            np.array([
                [cos, 0, 0, e_neg * sin],
                [0, cos, e_diff_neg * sin, 0],
                [0, e_diff_pos * sin, cos, 0],
                [e_pos * sin, 0, 0, cos]
            ])
        )

    def get_grad(self, params=None) -> npt.NDArray[np.complex128]:
        r"""
        Compute the gradient of the PartialMS gate with respect to param₀, param₁, and θ.

        Args:
            params (list[float]): List containing param₀, param₁, and θ.

        Returns:
            np.ndarray: A (3, 4, 4) gradient matrix.
        """
        if params is None:
            params = []
        self.check_parameters(params)

        phi0, phi1, theta = params
        cos = np.cos(normalize_theta(theta) / 2)
        sin = np.sin(normalize_theta(theta) / 2)
        e_pos = -1j * np.exp(1j * (phi0 + phi1))
        e_neg = -1j * np.exp(-1j * (phi0 + phi1))
        e_diff_pos = -1j * np.exp(1j * (phi0 - phi1))
        e_diff_neg = -1j * np.exp(-1j * (phi0 - phi1))

        # Compute derivatives
        d_cos = -sin / 2
        d_sin = cos / 2
        d_e_pos = np.exp(1j * (phi0 + phi1))
        d_e_neg = -np.exp(-1j * (phi0 + phi1))
        d_e_diff_pos = np.exp(1j * (phi0 - phi1))
        d_e_diff_neg = -np.exp(-1j * (phi0 - phi1))

        return np.array([
            np.array([
                [0, 0, 0, d_e_neg * sin],
                [0, 0, d_e_diff_neg * sin, 0],
                [0, d_e_diff_pos * sin, 0, 0],
                [d_e_pos * sin, 0, 0, 0]
            ], dtype=np.complex128),
            np.array([
                [0, 0, 0, d_e_neg * sin],
                [0, 0, -d_e_diff_neg * sin, 0],
                [0, -d_e_diff_pos * sin, 0, 0],
                [d_e_pos * sin, 0, 0, 0]
            ], dtype=np.complex128),
            np.array([
                [d_cos, 0, 0, d_sin * e_neg],
                [0, d_cos, d_sin * e_diff_neg, 0],
                [0, d_sin * e_diff_pos, d_cos, 0],
                [d_sin * e_pos, 0, 0, d_cos]
            ], dtype=np.complex128)
        ])


class ZZGate(QubitGate, DifferentiableUnitary, CachedClass):
    r"""
    The ZZ gate, a two-qubit gate that applies a phase shift depending on the
    computational basis states of two qubits.

    It is given by the following parameterized unitary:

    .. math::

        R_{ZZ}(\theta) = \exp(-i \frac{\theta}{2} Z \otimes Z) =
        \begin{bmatrix}
        e^{-i \theta / 2} & 0 & 0 & 0 \\
        0 & e^{i \theta / 2} & 0 & 0 \\
        0 & 0 & e^{i \theta / 2} & 0 \\
        0 & 0 & 0 & e^{-i \theta / 2}
        \end{bmatrix}

    Attributes:
        - θ (float): Rotation angle.
    """

    _num_qudits = 2  # This gate operates on two qubits.
    _num_params = 1  # It has one parameter: θ (rotation angle).
    _qasm_name = 'ZZ'  # Custom gate name for QASM compatibility.

    def get_unitary(self, params=None) -> UnitaryMatrix:
        """
        Return the unitary matrix representation of the ZZ gate.

        Args:
            params (list[float]): List containing θ.

        Returns:
            UnitaryMatrix: A 4x4 unitary matrix implementing the ZZ gate.
        """
        if params is None:
            params = []
        self.check_parameters(params)

        theta = normalize_theta(params[0])  # Normalize theta
        exp_pos = np.exp(1j * theta / 2)
        exp_neg = np.exp(-1j * theta / 2)

        return UnitaryMatrix(
            np.array([
                [exp_neg, 0, 0, 0],
                [0, exp_pos, 0, 0],
                [0, 0, exp_pos, 0],
                [0, 0, 0, exp_neg]
            ])
        )

    def get_grad(self, params=None) -> npt.NDArray[np.complex128]:
        """
        Compute the gradient of the ZZ gate with respect to θ.

        Args:
            params (list[float]): List containing θ.

        Returns:
            np.ndarray: A (1, 4, 4) gradient matrix.
        """
        if params is None:
            params = []
        self.check_parameters(params)

        theta = normalize_theta(params[0])  # Normalize theta
        d_exp_pos = 1j / 2 * np.exp(1j * theta / 2)
        d_exp_neg = -1j / 2 * np.exp(-1j * theta / 2)

        return np.array([
            np.array([
                [d_exp_neg, 0, 0, 0],
                [0, d_exp_pos, 0, 0],
                [0, 0, d_exp_pos, 0],
                [0, 0, 0, d_exp_neg]
            ], dtype=np.complex128)
        ])
