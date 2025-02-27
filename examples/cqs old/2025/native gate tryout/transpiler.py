from typing import List, Union, Optional, Callable
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import exp, sqrt, array, pi
from qiskit.circuit.library import UnitaryGate, HGate, ZGate, XGate, RXGate, RXXGate, YGate, RZGate, RGate, MSGate, RXXGate, RZZGate, CRZGate
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit import Gate
from qiskit.circuit.parameterexpression import ParameterValueType, ParameterExpression
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.library import PhaseGate, SXGate, UGate, CXGate, IGate
from networkx import Graph
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import XGate, SXGate, RZGate, CZGate, ECRGate
from qiskit.circuit import Measure, Delay, Parameter, Reset
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

class IonQGPIGate(Gate):
    def __init__(self, phi: ParameterValueType, label: Optional[str] = None, *, duration=None, unit="dt"):
        super().__init__("ionq-gpi", 1, [phi], label=label, duration=duration, unit=unit)

    def _define(self):
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        rules = [(RGate(pi, self.params[0]), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc.append(instr, qargs, cargs)
        self.definition = qc

    # def control(
    #     self,
    #     num_ctrl_qubits: int = 1,
    #     label: str | None = None,
    #     ctrl_state: str | int | None = None,
    #     annotated: bool | None = None,
    # ):
    #     """Return a (multi-)controlled-GPI gate.
    #
    #     Args:
    #         num_ctrl_qubits: number of control qubits.
    #         label: An optional label for the gate [Default: ``None``]
    #         ctrl_state: control state expressed as integer,
    #             string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
    #         annotated: indicates whether the controlled gate should be implemented
    #             as an annotated gate. If ``None``, this is set to ``True`` if
    #             the gate contains free parameters, in which case it cannot
    #             yet be synthesized.
    #
    #     Returns:
    #         ControlledGate: controlled version of this gate.
    #     """
    #     if annotated is None:
    #         annotated = any(isinstance(p, ParameterExpression) for p in self.params)
    #
    #     gate = super().control(
    #         num_ctrl_qubits=num_ctrl_qubits,
    #         label=label,
    #         ctrl_state=ctrl_state,
    #         annotated=annotated,
    #     )
    #     return gate

    def inverse(self, annotated: bool = False):
        """Return inverse GPI gate.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RXXGate` with an inverted parameter value.

        Returns:
            IonQGPIGate: inverse gate.
        """
        return IonQGPIGate(self.params[0])  # self-inverse

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the GPI gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return array([[0, exp(-1j*self.params[0])], [exp(1j*self.params[0]), 0]], dtype=dtype)
    # TODO: finish the array for the rest of gates and try to see
    #  1. Finish the controlled native gate;
    #  2. Try to do some optimization to the compilation;
    #  3. Explore the error model.

class IonQGPI2Gate(Gate):
    def __init__(self, phi: ParameterValueType, label: Optional[str] = None, *, duration=None, unit="dt"):
        super().__init__("ionq-gpi2", 1, [phi], label=label, duration=duration, unit=unit)

    def _define(self):
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        rules = [(RGate(pi/2, self.params[0]), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc.append(instr, qargs, cargs)
        self.definition = qc

    # def control(
    #     self,
    #     num_ctrl_qubits: int = 1,
    #     label: str | None = None,
    #     ctrl_state: str | int | None = None,
    #     annotated: bool | None = None,
    # ):
    #     """Return a (multi-)controlled-GPI2 gate.
    #
    #     Args:
    #         num_ctrl_qubits: number of control qubits.
    #         label: An optional label for the gate [Default: ``None``]
    #         ctrl_state: control state expressed as integer,
    #             string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
    #         annotated: indicates whether the controlled gate should be implemented
    #             as an annotated gate. If ``None``, this is set to ``True`` if
    #             the gate contains free parameters, in which case it cannot
    #             yet be synthesized.
    #
    #     Returns:
    #         ControlledGate: controlled version of this gate.
    #     """
    #     if annotated is None:
    #         annotated = any(isinstance(p, ParameterExpression) for p in self.params)
    #
    #     gate = super().control(
    #         num_ctrl_qubits=num_ctrl_qubits,
    #         label=label,
    #         ctrl_state=ctrl_state,
    #         annotated=annotated,
    #     )
    #     return gate

    def inverse(self, annotated: bool = False):
        """Return inverse GPI2 gate.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RXXGate` with an inverted parameter value.

        Returns:
            IonQGPIGate: inverse gate.
        """
        return IonQGPI2Gate(self.params[0]+pi)  # self-inverse


class IonQVirtualZGate(Gate):
    def __init__(self,
                 theta: ParameterValueType,
                 label: Optional[str] = None, *, duration=None, unit="dt"):
        super().__init__("ionq-virtualz", 1, [theta], label=label, duration=duration, unit=unit)

    def _define(self):
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        rules = [(RZGate(self.params[0]), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc.append(instr, qargs, cargs)
        self.definition = qc

    # def control(
    #         self,
    #         num_ctrl_qubits: int = 1,
    #         label: str | None = None,
    #         ctrl_state: str | int | None = None,
    #         annotated: bool | None = None,
    # ):
    #     """Return a (multi-)controlled-RZ gate.
    #
    #     Args:
    #         num_ctrl_qubits: number of control qubits.
    #         label: An optional label for the gate [Default: ``None``]
    #         ctrl_state: control state expressed as integer,
    #             string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
    #         annotated: indicates whether the controlled gate should be implemented
    #             as an annotated gate. If ``None``, this is set to ``True`` if
    #             the gate contains free parameters and more than one control qubit, in which
    #             case it cannot yet be synthesized. Otherwise it is set to ``False``.
    #
    #     Returns:
    #         ControlledGate: controlled version of this gate.
    #     """
    #     # deliberately capture annotated in [None, False] here
    #     if not annotated and num_ctrl_qubits == 1:
    #         gate = CRZGate(self.params[0], label=label, ctrl_state=ctrl_state)
    #         gate.base_gate.label = self.label
    #     else:
    #         # If the gate parameters contain free parameters, we cannot eagerly synthesize
    #         # the controlled gate decomposition. In this case, we annotate the gate per default.
    #         if annotated is None:
    #             annotated = any(isinstance(p, ParameterExpression) for p in self.params)
    #
    #         gate = super().control(
    #             num_ctrl_qubits=num_ctrl_qubits,
    #             label=label,
    #             ctrl_state=ctrl_state,
    #             annotated=annotated,
    #         )
    #     return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted Virtual Z gate

        :math:`RZ(\lambda)^{\dagger} = RZ(-\lambda)`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RZGate` with an inverted parameter value.

        Returns:
            IonQVirtualZGate: inverse gate.
        """
        return IonQVirtualZGate(-self.params[0])


# we note that when implementing the fully entangled MS gate and partially entangled MS gate, we assume that
# there is no phase term (with respect to \phi).

class IonQFullMSGate(Gate):
    def __init__(self, label: Optional[str] = None, ctrl_state: Optional[Union[str, int]] = None, *, duration=None, unit="dt", _base_label=None):
        super().__init__("ionq-full-ms", 2, [], duration=duration, unit=unit)

    def _define(self):
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q)
        rules = [(RXXGate(pi/2), [q[0], q[1]], [])]
        for instr, qargs, cargs in rules:
            qc.append(instr, qargs, cargs)
        self.definition = qc

    # def control(
    #     self,
    #     num_ctrl_qubits: int = 1,
    #     label: str | None = None,
    #     ctrl_state: str | int | None = None,
    #     annotated: bool | None = None,
    # ):
    #     """Return a (multi-)controlled-RXX gate.
    #
    #     Args:
    #         num_ctrl_qubits: number of control qubits.
    #         label: An optional label for the gate [Default: ``None``]
    #         ctrl_state: control state expressed as integer,
    #             string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
    #         annotated: indicates whether the controlled gate should be implemented
    #             as an annotated gate. If ``None``, this is set to ``True`` if
    #             the gate contains free parameters, in which case it cannot
    #             yet be synthesized.
    #
    #     Returns:
    #         ControlledGate: controlled version of this gate.
    #     """
    #     if annotated is None:
    #         annotated = any(isinstance(p, ParameterExpression) for p in self.params)
    #
    #     gate = super().control(
    #         num_ctrl_qubits=num_ctrl_qubits,
    #         label=label,
    #         ctrl_state=ctrl_state,
    #         annotated=annotated,
    #     )
    #     return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted fully entangled MS gate

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RZGate` with an inverted parameter value.

        Returns:
            IonQFullMSGate: inverse gate.
        """
        # RXX(-pi/2)
        return IonQPartialMSGate(-pi/2)
class IonQPartialMSGate(Gate):
    def __init__(self,
                 theta: ParameterValueType,
                 label: Optional[str] = None, ctrl_state: Optional[Union[str, int]] = None, *, duration=None, unit="dt"):
        super().__init__("ionq-partial-ms", 2, [theta], label=label, duration=duration, unit=unit)

    def _define(self):
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q)
        rules = [(RXXGate(self.params[0]), [q[0], q[1]], [])]
        for instr, qargs, cargs in rules:
            qc.append(instr, qargs, cargs)
        self.definition = qc

    # def control(
    #     self,
    #     num_ctrl_qubits: int = 1,
    #     label: str | None = None,
    #     ctrl_state: str | int | None = None,
    #     annotated: bool | None = None,
    # ):
    #     """Return a (multi-)controlled-RXX gate.
    #
    #     Args:
    #         num_ctrl_qubits: number of control qubits.
    #         label: An optional label for the gate [Default: ``None``]
    #         ctrl_state: control state expressed as integer,
    #             string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
    #         annotated: indicates whether the controlled gate should be implemented
    #             as an annotated gate. If ``None``, this is set to ``True`` if
    #             the gate contains free parameters, in which case it cannot
    #             yet be synthesized.
    #
    #     Returns:
    #         ControlledGate: controlled version of this gate.
    #     """
    #     if annotated is None:
    #         annotated = any(isinstance(p, ParameterExpression) for p in self.params)
    #
    #     gate = super().control(
    #         num_ctrl_qubits=num_ctrl_qubits,
    #         label=label,
    #         ctrl_state=ctrl_state,
    #         annotated=annotated,
    #     )
    #     return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted partially entangled MS gate

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RZGate` with an inverted parameter value.

        Returns:
            IonQPartialMSGate: inverse gate.
        """
        return IonQPartialMSGate(-self.params[0])

class IonQZZGate(Gate):
    def __init__(self,
                 theta: ParameterValueType,
                 label: Optional[str] = None, ctrl_state: Optional[Union[str, int]] = None, *, duration=None, unit="dt"):
        super().__init__("ionq-zz", 2, [theta], label=label, duration=duration, unit=unit)

    def _define(self):
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q)
        rules = [(RZZGate(self.params[0]), [q[0], q[1]], [])]
        for instr, qargs, cargs in rules:
            qc.append(instr, qargs, cargs)
        self.definition = qc

    # def control(
    #     self,
    #     num_ctrl_qubits: int = 1,
    #     label: str | None = None,
    #     ctrl_state: str | int | None = None,
    #     annotated: bool | None = None,
    # ):
    #     """Return a (multi-)controlled-RXX gate.
    #
    #     Args:
    #         num_ctrl_qubits: number of control qubits.
    #         label: An optional label for the gate [Default: ``None``]
    #         ctrl_state: control state expressed as integer,
    #             string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
    #         annotated: indicates whether the controlled gate should be implemented
    #             as an annotated gate. If ``None``, this is set to ``True`` if
    #             the gate contains free parameters, in which case it cannot
    #             yet be synthesized.
    #
    #     Returns:
    #         ControlledGate: controlled version of this gate.
    #     """
    #     if annotated is None:
    #         annotated = any(isinstance(p, ParameterExpression) for p in self.params)
    #
    #     gate = super().control(
    #         num_ctrl_qubits=num_ctrl_qubits,
    #         label=label,
    #         ctrl_state=ctrl_state,
    #         annotated=annotated,
    #     )
    #     return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted ZZ Gate gate

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RZGate` with an inverted parameter value.

        Returns:
            IonQVirtualZGate: inverse gate.
        """
        return IonQZZGate(-self.params[0])

class FakeIonQAriaBackend(BackendV2):
    """Fake IonQ Aria backend."""

    def __init__(self, n=25):
        """Instantiate a new fake ionq Aria backend.

        Args:
            n (int): the qubit number
        """
        super().__init__(name="Fake IonQ Aria backend")

        V = [i for i in range(n)]
        E = [(j, i) for i in range(n) for j in range(i)]
        self._graph = Graph()
        self._graph.add_nodes_from(V)
        self._graph.add_edges_from(E)
        self._target = Target(
            "Fake IonQ Aria backend", num_qubits=n
        )
        props_1 = {(qubit,): None for qubit in V}
        props_2 = {(edge[0], edge[1]): None for edge in E}

        # circuit parameters
        theta = [Parameter("theta"+str(i)) for i in range(5)]

        self._target.add_instruction(IonQGPIGate(theta[0]), props_1)
        self._target.add_instruction(IonQGPI2Gate(theta[1]), props_1)
        self._target.add_instruction(IonQVirtualZGate(theta[2]), props_1)
        self._target.add_instruction(IonQFullMSGate(), props_2)
        self._target.add_instruction(IonQPartialMSGate(theta[3]), props_2)
        self._target.add_instruction(IonQZZGate(theta[4]), props_2)
        self._target.add_instruction(Measure(), props_1)

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return None

    @property
    def graph(self):
        return self._graph

    @classmethod
    def _default_options(cls):
        return Options(shots=1024)

    def run(self, circuit, **kwargs):
        raise NotImplementedError(
            "This backend does not contain a run method"
        )


def add_equivalence():
    # circuit parameters
    theta = [Parameter("theta" + str(i)) for i in range(5)]

    # add equivalence I
    I_q = QuantumRegister(1, "I_q")
    I_qc = QuantumCircuit(I_q)
    I_qc.append(IonQVirtualZGate(0), [I_q[0]], [])
    SessionEquivalenceLibrary.add_equivalence(IGate(), I_qc)

    # add equivalence H
    H_q = QuantumRegister(1, "H_q")
    H_qc = QuantumCircuit(H_q)
    H_qc.append(IonQVirtualZGate(pi), [H_q[0]], [])
    H_qc.append(IonQGPI2Gate(pi/2), [H_q[0]], [])
    SessionEquivalenceLibrary.add_equivalence(HGate(), H_qc)

    # add equivalence X
    X_q = QuantumRegister(1, "X_q")
    X_qc = QuantumCircuit(X_q)
    X_qc.append(IonQGPIGate(0), [X_q[0]], [])
    SessionEquivalenceLibrary.add_equivalence(XGate(), X_qc)

    # add equivalence Y
    Y_q = QuantumRegister(1, "Y_q")
    Y_qc = QuantumCircuit(Y_q)
    Y_qc.append(IonQGPIGate(pi/2), [Y_q[0]], [])
    SessionEquivalenceLibrary.add_equivalence(YGate(), Y_qc)

    # add equivalence Z
    Z_q = QuantumRegister(1, "Z_q")
    Z_qc = QuantumCircuit(Z_q)
    Z_qc.append(IonQVirtualZGate(pi), [Z_q[0]], [])
    SessionEquivalenceLibrary.add_equivalence(ZGate(), Z_qc)

    # add equivalence Rz
    Rz_q = QuantumRegister(1, "Rz_q")
    Rz_qc = QuantumCircuit(Rz_q)
    Rz_qc.append(IonQVirtualZGate(theta[0]), [Rz_q[0]], [])
    SessionEquivalenceLibrary.add_equivalence(RZGate(theta[0]), Rz_qc)

    # add equivalence CX
    CX_q = QuantumRegister(2, "CX_q")
    CX_qc = QuantumCircuit(CX_q)
    CX_qc.append(IonQGPI2Gate(pi/2), [CX_q[0]], [])
    CX_qc.append(IonQFullMSGate(), [CX_q[0], CX_q[1]], [])
    CX_qc.append(IonQGPI2Gate(pi), [CX_q[0]], [])
    CX_qc.append(IonQGPIGate(0), [CX_q[0]], [])
    CX_qc.append(IonQGPI2Gate(pi/2), [CX_q[0]], [])
    CX_qc.append(IonQGPIGate(pi/2), [CX_q[0]], [])
    CX_qc.append(IonQGPI2Gate(pi/2), [CX_q[1]], [])
    CX_qc.append(IonQGPIGate(pi/2), [CX_q[1]], [])
    SessionEquivalenceLibrary.add_equivalence(CXGate(), CX_qc)

    # add equivalence RXX
    RXX_q = QuantumRegister(2, "RXX_q")
    RXX_qc = QuantumCircuit(RXX_q)
    RXX_qc.append(IonQPartialMSGate(theta[1]), [RXX_q[0], RXX_q[1]], [])
    SessionEquivalenceLibrary.add_equivalence(RXXGate(theta[1]), RXX_qc)

    # add equivalence RZZ
    RZZ_q = QuantumRegister(2, "RZZ_q")
    RZZ_qc = QuantumCircuit(RZZ_q)
    RZZ_qc.append(IonQZZGate(theta[2]), [RZZ_q[0], RZZ_q[1]], [])
    SessionEquivalenceLibrary.add_equivalence(RZZGate(theta[2]), RZZ_qc)

def transpile_to_ionq_native_gates(qc):
    add_equivalence()
    backend = FakeIonQAriaBackend(n=qc.num_qubits)
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    return pm.run(qc)





# q = QuantumRegister(1, "q")
# def_sy_h = QuantumCircuit(q)
# def_sy_h.append(GPIGate(Parameter('theta')), [q[0]], [])
# def_sy_h.append(GPI2Gate(Parameter('theta')), [q[0]], [])
# def_sy_h.append(RZGate(Parameter('theta')))
# SessionEquivalenceLibrary.add_equivalence(
#     HGate(), def_sy_h)
