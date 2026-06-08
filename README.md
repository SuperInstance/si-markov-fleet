# si-markov-fleet

> **Proof of Concept:** Markov chain analysis for fleet state transitions — stationary distributions reveal long-term budget equilibrium, mixing time tells how fast the fleet converges.

## The Insight

Fleet budget transitions can be modeled as a Markov chain:
- **States** = possible budget configurations (conserving, spending, recovering...)
- **Transitions** = probability of moving between configurations
- **Stationary distribution** π = long-term probability of each state
- **Mixing time** = how many rounds until the fleet forgets its starting state

Key theorem: For an ergodic chain, π is unique and the chain converges regardless of starting state.

## What This Proves

1. **Power iteration converges**: π = πP finds the stationary distribution
2. **Uniform chains mix instantly**: All states equally likely after 1 step
3. **Asymmetric chains have biased equilibria**: π₁/π₂ = P₂₁/P₁₂
4. **Absorbing states capture mass**: Once entered, never left
5. **Entropy rate measures randomness**: Uniform chains have maximum entropy
6. **Communicating classes partition states**: Separate sub-fleets

## Usage

```rust
use si_markov_fleet::*;

// Define transition matrix
let data = vec![vec![0.9, 0.1], vec![0.2, 0.8]];
let p = TransitionMatrix::from_vec(data).unwrap();

// Stationary distribution
let pi = p.stationary(1000, 1e-10);
println!("π = {:?}", pi); // [0.667, 0.333]

// Mixing time
let t = p.mixing_time(0.01, 10000);
println!("Mixing time: {}", t);

// Hitting time (expected steps to reach state j from i)
let h = p.hitting_time(1, 1000);
println!("Hitting time to state 1: {:?}", h);

// Entropy rate
let entropy = p.entropy_rate(1000);

// Fleet model
let fleet = FleetMarkov::new(p, vec!["conserving".into(), "spending".into()]);
println!("Ergodic: {}", fleet.is_ergodic());
```

## Modules

- `TransitionMatrix` — NxN stochastic matrix with validation
- `stationary()` — π via power iteration (left eigenvector of P)
- `mixing_time()` — steps until ||P^n - π|| < ε
- `hitting_time()` — expected steps to reach target state
- `communicating_classes()` — partition into communicating classes
- `is_aperiodic()` / `is_ergodic()` — chain classification
- `entropy_rate()` — Shannon entropy rate H = -Σ πᵢPᵢⱼ log(Pᵢⱼ)
- `FleetMarkov` — high-level fleet model with labels

## Connection to Conservation Law

The conservation constraint γ + η = C restricts which transitions are valid:
- **Conserving transitions**: stay on the manifold (high probability)
- **Off-manifold transitions**: violate conservation (low probability)
- **Stationary distribution** concentrated on conserving states = healthy fleet
- **Mixing time** = recovery speed after perturbation

If the stationary distribution has mass on "non-conserving" states, the fleet is structurally prone to conservation violations.

## Tests: 22

Covers: matrix validation, row sums, construction errors, stationary uniform/doubly-stochastic/asymmetric/eigenvector, mixing time finite/fast, hitting time self/positive, communicating classes connected/disconnected, aperiodicity, ergodicity, entropy rate positive/max, fleet model, absorbing chain, identity chain.

## License

MIT
