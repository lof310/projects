# Symbolic Language for Small AI Reasoning

This is a new symbolic language I'm designing specifically for small AI models (<150M parameters) to perform logical reasoning.

## Design Goals

- **Vocabulary:** <600 tokens total (vs. Lean's thousands)
- **No natural language:** Purely symbolic, easier for small models to parse. But it is also easier to translate directly to Natural Language.
- **Coverage:** Supports proofs, reasoning, CoT, manual computations and logical deduction
- **Motivation:** Lean is more of a programming language it is too complex for small models to learn effectively.

## Example: Euclid's Proof of infinitude of Primes.

```lean
hyp p : Nat -> Prop.
hyp q : forall n : Nat , exists m : Nat , ( p m /\ m > n ).
{
    assume h :  ̃ q.
    let S := { m : Nat | p m }.
    assume k : exists c : Nat , ( | S | = c ).
    let M := ( prod r : S , r ) + 1.
    step ( M mod r <> 0 ) ⇝ (  ̃ ( r | M ) ) for r in S.
    step ( M > 1 ) ⇝ ( exists t : Nat , p t /\ t | M ).
    by deduce.
    show ( t in S /\ t notin S ).
    by contra.
} qed.