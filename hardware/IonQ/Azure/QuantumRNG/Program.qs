//  Two qubit random number generator, this program is written in Q#.

namespace QuantumRNG {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Canon;

    @EntryPoint()
    operation GenerateRandomBits() : Result[] {
        use qubits = Qubit[2];
        ApplyToEach(H, qubits);
        return MultiM(qubits);
    }
}














