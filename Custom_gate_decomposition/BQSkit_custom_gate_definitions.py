# Re-import necessary libraries after execution state reset
from __future__ import annotations
from bqskit import Circuit, compile
from bqskit.compiler.machine import MachineModel
from bqskit.ir.gates import RXXGate, RXGate, RZGate,RYGate,RYYGate,HGate,CNOTGate
from bqskit.ext import bqskit_to_qiskit
import numpy as np
from bqskit.ir.gate import Gate


import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class VirtualZGate(QubitGate, DifferentiableUnitary, CachedClass):
    """
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

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        exp1 = np.exp(-1j * params[0])
        exp2 = np.exp(1j * params[0])

        return UnitaryMatrix(
            [
                [exp1, 0],
                [0, exp2],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
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
    """
    A gate representing the GPI(ϕ) transformation.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        0 & e^{-2\pi i \phi} \\\\
        e^{2\pi i \phi} & 0
        \\end{pmatrix}

    This gate applies a phase-dependent transformation on the qubit.

    Attributes:
        - ϕ (float): The phase parameter.
    """

    _num_qudits = 1  # This gate operates on a single qubit
    _num_params = 1  # It has one parameter: ϕ
    _qasm_name = 'GPI'  # Custom gate name for QASM compatibility

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """
        Return the unitary matrix representation of the GPI gate.

        Args:
            params (list[float]): List containing the phase parameter ϕ.

        Returns:
            UnitaryMatrix: A 2x2 unitary matrix implementing the GPI gate.
        """
        self.check_parameters(params)

        phi = params[0]  # Extract the phase parameter
        return UnitaryMatrix(
            [
                [0, np.exp(-2j * np.pi * phi)],  # Upper diagonal element
                [np.exp(2j * np.pi * phi), 0]   # Lower diagonal element
            ]
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Compute the gradient of the GPI gate with respect to ϕ.

        Args:
            params (list[float]): List containing the phase parameter ϕ.

        Returns:
            np.ndarray: A (1, 2, 2) gradient matrix.
        """
        self.check_parameters(params)

        phi = params[0]
        d_exp_pos = 2j * np.pi * np.exp(2j * np.pi * phi)  # Derivative w.r.t. ϕ
        d_exp_neg = -2j * np.pi * np.exp(-2j * np.pi * phi)

        return np.array(
            [
                [
                    [0, d_exp_neg],  # Gradient at (0,1)
                    [d_exp_pos, 0]   # Gradient at (1,0)
                ]
            ],
            dtype=np.complex128
        )


class GPI2Gate(QubitGate, DifferentiableUnitary, CachedClass):
    """
    A gate representing the GPI2(ϕ) transformation.

    This gate is defined by the following unitary matrix:

    .. math::

        \\frac{1}{\\sqrt{2}} \\begin{pmatrix}
        1 & -i e^{-2\pi i \phi} \\\\
        -i e^{2\pi i \phi} & 1
        \\end{pmatrix}

    This is a generalization of the GPI gate with additional phase properties.

    Attributes:
        - ϕ (float): The phase parameter.
    """

    _num_qudits = 1  # This gate operates on a single qubit.
    _num_params = 1  # It takes one parameter: ϕ (phi).
    _qasm_name = 'GPI2'  # The name used in QASM representation.

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """
        Computes and returns the unitary matrix for the GPI2 gate.

        Args:
            params (list[float]): A single parameter ϕ.

        Returns:
            UnitaryMatrix: A 2x2 unitary matrix representing the gate.
        """
        self.check_parameters(params)  # Ensure the parameter list is valid.

        phi = params[0]  # Extract the phase parameter ϕ.
        factor = 1 / np.sqrt(2)  # Normalization factor.
        exp_pos = -1j * np.exp(2j * np.pi * phi)  # e^(2πiϕ) * -i
        exp_neg = -1j * np.exp(-2j * np.pi * phi)  # e^(-2πiϕ) * -i

        return UnitaryMatrix(
            factor * np.array([
                [1, exp_neg],  # First row
                [exp_pos, 1]  # Second row
            ])
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Computes the gradient of the GPI2 gate with respect to ϕ.

        This is useful for gradient-based optimization techniques in
        variational quantum algorithms.

        Args:
            params (list[float]): A single parameter ϕ.

        Returns:
            np.ndarray: A (1, 2, 2) gradient matrix.
        """
        self.check_parameters(params)  # Validate input parameters.

        phi = params[0]  # Extract the phase parameter.
        factor = 1 / np.sqrt(2)  # Normalization factor.

        # Compute the derivatives of the exponential terms.
        d_exp_pos = 2 * np.pi * np.exp(2j * np.pi * phi)
        d_exp_neg = -2 * np.pi * np.exp(-2j * np.pi * phi)

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
    """
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
        - ϕ₀ (float): Phase parameter 1
        - ϕ₁ (float): Phase parameter 2
    """

    _num_qudits = 2  # This gate operates on two qubits
    _num_params = 2  # It has two parameters: ϕ₀ and ϕ₁
    _qasm_name = 'fullMS'  # Custom gate name for QASM compatibility

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """
        Return the unitary matrix representation of the FullMS gate.

        Args:
            params (list[float]): List containing the phase parameters ϕ₀, ϕ₁.

        Returns:
            UnitaryMatrix: A 4x4 unitary matrix implementing the FullMS gate.
        """
        self.check_parameters(params)

        phi0, phi1 = params  # Extract the phase parameters
        factor = 1 / np.sqrt(2)  # Normalization factor

        exp_14 = -1j * np.exp(-2j * np.pi * (phi0 + phi1))
        exp_23 = -1j * np.exp(-2j * np.pi * (phi0 - phi1))
        exp_32 = -1j * np.exp(2j * np.pi * (phi0 - phi1))
        exp_41 = -1j * np.exp(2j * np.pi * (phi0 + phi1))

        return UnitaryMatrix(
            factor * np.array([
                [1, 0, 0, exp_14],
                [0, 1, exp_23, 0],
                [0, exp_32, 1, 0],
                [exp_41, 0, 0, 1]
            ])
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Compute the gradient of the FullMS gate with respect to ϕ₀ and ϕ₁.

        Args:
            params (list[float]): List containing the phase parameters ϕ₀, ϕ₁.

        Returns:
            np.ndarray: A (2, 4, 4) gradient matrix.
        """
        self.check_parameters(params)

        phi0, phi1 = params
        factor = 1 / np.sqrt(2)

        # Compute derivatives
        d_14 = -2 * np.pi * np.exp(-2j * np.pi * (phi0 + phi1))
        d_21 = -2 * np.pi * np.exp(-2j * np.pi * (phi0 - phi1))
        d_32 = 2 * np.pi * np.exp(2j * np.pi * (phi0 - phi1))
        d_41 = 2 * np.pi * np.exp(2j * np.pi * (phi0 + phi1))

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
    """
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
        - ϕ₀ (float): Phase parameter 1
        - ϕ₁ (float): Phase parameter 2
        - θ (float): Rotation angle
    """

    _num_qudits = 2  # This gate operates on two qubits
    _num_params = 3  # It has three parameters: ϕ₀, ϕ₁, and θ
    _qasm_name = 'partialMS'  # Custom gate name for QASM compatibility

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """
        Return the unitary matrix representation of the PartialMS gate.

        Args:
            params (list[float]): List containing ϕ₀, ϕ₁, and θ.

        Returns:
            UnitaryMatrix: A 4x4 unitary matrix implementing the PartialMS gate.
        """
        self.check_parameters(params)

        phi0, phi1, theta = params  # Extract the phase parameters and rotation angle
        cos = np.cos(np.pi * theta)
        sin = np.sin(np.pi * theta)
        e_pos = -1j * np.exp(2j * np.pi * (phi0 + phi1))
        e_neg = -1j * np.exp(-2j * np.pi * (phi0 + phi1))
        e_diff_pos = -1j * np.exp(2j * np.pi * (phi0 - phi1))
        e_diff_neg = -1j * np.exp(-2j * np.pi * (phi0 - phi1))

        return UnitaryMatrix(
            np.array([
                [cos, 0, 0, e_neg * sin],
                [0, cos, e_diff_neg * sin, 0],
                [0, e_diff_pos * sin, cos, 0],
                [e_pos * sin, 0, 0, cos]
            ])
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Compute the gradient of the PartialMS gate with respect to ϕ₀, ϕ₁, and θ.

        Args:
            params (list[float]): List containing ϕ₀, ϕ₁, and θ.

        Returns:
            np.ndarray: A (3, 4, 4) gradient matrix.
        """
        self.check_parameters(params)

        phi0, phi1, theta = params
        cos = np.cos(np.pi * theta)
        sin = np.sin(np.pi * theta)
        e_pos = -1j * np.exp(2j * np.pi * (phi0 + phi1))
        e_neg = -1j * np.exp(-2j * np.pi * (phi0 + phi1))
        e_diff_pos = -1j * np.exp(2j * np.pi * (phi0 - phi1))
        e_diff_neg = -1j * np.exp(-2j * np.pi * (phi0 - phi1))

        # Compute derivatives
        d_cos = -np.pi * sin
        d_sin = np.pi * cos
        d_e_pos = 2 * np.pi * np.exp(2j * np.pi * (phi0 + phi1))
        d_e_neg = -2 * np.pi * np.exp(-2j * np.pi * (phi0 + phi1))
        d_e_diff_pos = 2 * np.pi * np.exp(2j * np.pi * (phi0 - phi1))
        d_e_diff_neg = -2 * np.pi * np.exp(-2j * np.pi * (phi0 - phi1))

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




