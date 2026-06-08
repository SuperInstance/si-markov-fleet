//! Markov chain analysis for fleet state transitions.
//!
//! Model fleet budget transitions as a Markov chain:
//! - States = possible budget configurations
//! - Transitions = budget reallocations between agents
//! - Stationary distribution = long-term budget equilibrium
//! - Mixing time = how fast the fleet reaches equilibrium

/// Transition matrix P where P[i][j] = Pr(state j | state i).
#[derive(Debug, Clone)]
pub struct TransitionMatrix {
    pub n: usize,
    pub data: Vec<Vec<f64>>,
}

impl TransitionMatrix {
    pub fn new(n: usize) -> Self {
        // Default: uniform transitions
        let p = 1.0 / n as f64;
        Self { n, data: vec![vec![p; n]; n] }
    }

    pub fn from_vec(data: Vec<Vec<f64>>) -> Result<Self, String> {
        let n = data.len();
        for (i, row) in data.iter().enumerate() {
            if row.len() != n { return Err(format!("Row {} has wrong length", i)); }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 0.01 {
                return Err(format!("Row {} sums to {} (expected 1.0)", i, sum));
            }
        }
        Ok(Self { n, data })
    }

    pub fn identity(n: usize) -> Self {
        let mut data = vec![vec![0.0; n]; n];
        for i in 0..n { data[i][i] = 1.0; }
        Self { n, data }
    }

    pub fn uniform(n: usize) -> Self { Self::new(n) }

    /// Absorbing chain: states 0..k are transient, k..n are absorbing.
    pub fn absorbing(n: usize, k_absorb: usize) -> Self {
        let mut data = vec![vec![0.0; n]; n];
        for i in 0..k_absorb.min(n) { data[i][i] = 1.0; } // absorbing states
        for i in k_absorb.min(n)..n {
            // Transient: distribute probability to neighbors
            let count = (n - k_absorb.min(n)).max(1);
            let p = 1.0 / count as f64;
            for j in k_absorb.min(n)..n { data[i][j] = p; }
        }
        Self { n, data }
    }

    /// Check if all rows sum to 1.
    pub fn is_valid(&self) -> bool {
        for row in &self.data {
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 0.01 { return false; }
        }
        true
    }

    /// Multiply by vector: result[i] = Σ_j P[i][j] * v[j]
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        (0..self.n).map(|i| {
            self.data[i].iter().zip(v.iter()).map(|(pij, vj)| pij * vj).sum()
        }).collect()
    }

    /// Stationary distribution via power iteration.
    pub fn stationary(&self, max_iter: usize, tol: f64) -> Vec<f64> {
        let mut pi = vec![1.0 / self.n as f64; self.n];
        for _ in 0..max_iter {
            // π_new[j] = Σ_i π[i] * P[i][j]  (left eigenvector)
            let mut next = vec![0.0; self.n];
            for i in 0..self.n {
                for j in 0..self.n {
                    next[j] += pi[i] * self.data[i][j];
                }
            }
            let diff: f64 = next.iter().zip(pi.iter()).map(|(a, b)| (a - b).abs()).sum();
            pi = next;
            if diff < tol { break; }
        }
        pi
    }

    /// Mixing time: min n such that ||P^n[i] - π||₁ < ε for all i.
    pub fn mixing_time(&self, epsilon: f64, max_iter: usize) -> usize {
        let pi = self.stationary(max_iter, 1e-12);
        let mut current = vec![vec![0.0; self.n]; self.n];
        for i in 0..self.n { current[i][i] = 1.0; } // start from each state

        for step in 1..=max_iter {
            // Multiply each row by P
            for i in 0..self.n {
                current[i] = self.mul_vec(&current[i]);
            }
            // Check convergence
            let mut max_dist = 0.0_f64;
            for i in 0..self.n {
                let dist: f64 = current[i].iter().zip(pi.iter()).map(|(a, b)| (a - b).abs()).sum();
                max_dist = max_dist.max(dist);
            }
            if max_dist < epsilon { return step; }
        }
        max_iter
    }

    /// Hitting time: expected steps to reach j from i.
    pub fn hitting_time(&self, target: usize, max_iter: usize) -> Vec<f64> {
        let mut h = vec![0.0; self.n];
        h[target] = 0.0;
        for _ in 0..max_iter {
            let mut new_h = h.clone();
            for i in 0..self.n {
                if i == target { continue; }
                let mut expected = 1.0;
                for j in 0..self.n {
                    expected += self.data[i][j] * h[j];
                }
                new_h[i] = expected;
            }
            let diff: f64 = new_h.iter().zip(h.iter()).map(|(a, b)| (a - b).abs()).sum();
            h = new_h;
            if diff < 1e-10 { break; }
        }
        h
    }

    /// Communicating classes (states that can reach each other).
    pub fn communicating_classes(&self) -> Vec<Vec<usize>> {
        let n = self.n;
        let mut visited = vec![false; n];
        let mut classes = vec![];

        for start in 0..n {
            if visited[start] { continue; }
            // BFS to find all states reachable from start
            let mut reachable = vec![false; n];
            let mut queue = vec![start];
            reachable[start] = true;
            while let Some(s) = queue.pop() {
                for j in 0..n {
                    if self.data[s][j] > 1e-12 && !reachable[j] {
                        reachable[j] = true;
                        queue.push(j);
                    }
                }
            }
            // Check if all reachable can reach back to start
            let mut class = vec![];
            for i in 0..n {
                if !reachable[i] { continue; }
                // Check if i can reach start
                let mut can_reach = vec![false; n];
                let mut q = vec![i];
                can_reach[i] = true;
                while let Some(s) = q.pop() {
                    for j in 0..n {
                        if self.data[s][j] > 1e-12 && !can_reach[j] {
                            can_reach[j] = true;
                            q.push(j);
                        }
                    }
                }
                if can_reach[start] {
                    class.push(i);
                    visited[i] = true;
                }
            }
            if !class.is_empty() { classes.push(class); }
        }
        classes
    }

    /// Is the chain aperiodic? (all states have period 1)
    pub fn is_aperiodic(&self) -> bool {
        // Check if any state has self-loop (sufficient for aperiodicity)
        for i in 0..self.n {
            if self.data[i][i] > 1e-12 { return true; }
        }
        false
    }

    /// Is the chain ergodic? (irreducible + aperiodic)
    pub fn is_ergodic(&self) -> bool {
        let classes = self.communicating_classes();
        classes.len() == 1 && self.is_aperiodic()
    }

    /// Entropy rate: H = -Σ_ij π_i P_ij log(P_ij)
    pub fn entropy_rate(&self, max_iter: usize) -> f64 {
        let pi = self.stationary(max_iter, 1e-12);
        let mut h = 0.0;
        for i in 0..self.n {
            for j in 0..self.n {
                if self.data[i][j] > 1e-12 && pi[i] > 1e-12 {
                    h -= pi[i] * self.data[i][j] * self.data[i][j].ln();
                }
            }
        }
        h
    }
}

/// Fleet Markov model.
#[derive(Debug, Clone)]
pub struct FleetMarkov {
    pub chain: TransitionMatrix,
    pub state_labels: Vec<String>,
}

impl FleetMarkov {
    pub fn new(chain: TransitionMatrix, labels: Vec<String>) -> Self {
        Self { chain: chain, state_labels: labels }
    }

    pub fn stationary(&self) -> Vec<f64> { self.chain.stationary(10000, 1e-12) }
    pub fn mixing_time(&self, epsilon: f64) -> usize { self.chain.mixing_time(epsilon, 10000) }
    pub fn entropy_rate(&self) -> f64 { self.chain.entropy_rate(10000) }
    pub fn is_ergodic(&self) -> bool { self.chain.is_ergodic() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_valid() {
        let p = TransitionMatrix::uniform(3);
        assert!(p.is_valid());
    }

    #[test]
    fn test_rows_sum_to_one() {
        let p = TransitionMatrix::uniform(4);
        for row in &p.data {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_from_vec_valid() {
        let data = vec![vec![0.5, 0.5], vec![0.3, 0.7]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        assert!(p.is_valid());
    }

    #[test]
    fn test_from_vec_invalid() {
        let data = vec![vec![0.5, 0.3], vec![0.2, 0.7]]; // row 0 sums to 0.8
        assert!(TransitionMatrix::from_vec(data).is_err());
    }

    #[test]
    fn test_stationary_uniform() {
        let p = TransitionMatrix::uniform(3);
        let pi = p.stationary(1000, 1e-10);
        for v in &pi {
            assert!((v - 1.0/3.0).abs() < 1e-6, "Should be uniform, got {}", v);
        }
    }

    #[test]
    fn test_stationary_doubly_stochastic() {
        let data = vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0], vec![1.0, 0.0, 0.0]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        let pi = p.stationary(1000, 1e-10);
        for v in &pi { assert!((v - 1.0/3.0).abs() < 0.01); }
    }

    #[test]
    fn test_stationary_asymmetric() {
        let data = vec![vec![0.9, 0.1], vec![0.2, 0.8]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        let pi = p.stationary(1000, 1e-10);
        // π₁/π₂ = 0.2/0.1 = 2, so π₁ = 2/3, π₂ = 1/3
        assert!((pi[0] - 2.0/3.0).abs() < 0.15, "π₁ = {}", pi[0]);
        assert!((pi[1] - 1.0/3.0).abs() < 0.15, "π₂ = {}", pi[1]);
    }

    #[test]
    fn test_stationary_is_eigenvector() {
        let data = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        let pi = p.stationary(1000, 1e-10);
        // π·P = π  →  π_new[j] = Σ_i π[i] P[i][j]
        let mut next = vec![0.0; 2];
        for i in 0..2 { for j in 0..2 { next[j] += pi[i] * p.data[i][j]; } }
        for i in 0..2 { assert!((next[i] - pi[i]).abs() < 1e-6, "π[{}] = {} but πP = {}", i, pi[i], next[i]); }
    }

    #[test]
    fn test_mixing_time_finite() {
        let data = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        let t = p.mixing_time(0.01, 10000);
        assert!(t < 100, "Mixing time should be reasonable: {}", t);
    }

    #[test]
    fn test_mixing_time_uniform_fast() {
        let p = TransitionMatrix::uniform(3);
        let t = p.mixing_time(0.01, 1000);
        assert!(t <= 2, "Uniform chain mixes instantly: {}", t);
    }

    #[test]
    fn test_hitting_time_self() {
        let p = TransitionMatrix::uniform(3);
        let h = p.hitting_time(0, 1000);
        assert!((h[0] - 0.0).abs() < 1e-10, "Hitting time to self = 0");
    }

    #[test]
    fn test_hitting_time_positive() {
        let data = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        let h = p.hitting_time(1, 1000);
        assert!(h[0] > 0.0, "Hitting time from 0 to 1 > 0");
    }

    #[test]
    fn test_communicating_classes_connected() {
        let data = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        let classes = p.communicating_classes();
        assert_eq!(classes.len(), 1, "Connected chain has 1 class");
    }

    #[test]
    fn test_communicating_classes_disconnected() {
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        let classes = p.communicating_classes();
        assert_eq!(classes.len(), 2, "Disconnected chain has 2 classes");
    }

    #[test]
    fn test_aperiodic() {
        let data = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        assert!(p.is_aperiodic());
    }

    #[test]
    fn test_periodic() {
        let data = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        assert!(!p.is_aperiodic());
    }

    #[test]
    fn test_ergodic() {
        let data = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        assert!(p.is_ergodic());
    }

    #[test]
    fn test_entropy_rate_positive() {
        let data = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        let h = p.entropy_rate(1000);
        assert!(h > 0.0, "Entropy rate should be positive");
    }

    #[test]
    fn test_entropy_rate_uniform_max() {
        let p = TransitionMatrix::uniform(4);
        let h = p.entropy_rate(1000);
        // Maximum entropy for 4 states: log(4) ≈ 1.386
        assert!(h > 1.0, "Uniform entropy should be near max: {}", h);
    }

    #[test]
    fn test_fleet_markov() {
        let data = vec![vec![0.8, 0.2], vec![0.3, 0.7]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        let fm = FleetMarkov::new(p, vec!["conserving".into(), "spending".into()]);
        let pi = fm.stationary();
        assert_eq!(pi.len(), 2);
        assert!(fm.is_ergodic());
    }

    #[test]
    fn test_absorbing_chain() {
        let data = vec![vec![1.0, 0.0, 0.0], vec![0.5, 0.5, 0.0], vec![0.3, 0.0, 0.7]];
        let p = TransitionMatrix::from_vec(data).unwrap();
        let pi = p.stationary(1000, 1e-10);
        // State 0 is absorbing, 1 and 2 can reach it
        assert!(pi[0] > 0.3, "Absorbing state should have mass: π₀={}", pi[0]);
    }

    #[test]
    fn test_identity_chain() {
        let p = TransitionMatrix::identity(3);
        let pi = p.stationary(100, 1e-10);
        // Identity: any distribution is stationary, power iteration keeps initial
        for v in &pi { assert!(v.is_finite()); }
    }
}
