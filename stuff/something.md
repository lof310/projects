Euclid's Proof of infinitude of Primes.

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
