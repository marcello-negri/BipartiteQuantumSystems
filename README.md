# BipartiteQuantumSystems
Partial permutational symmetry in quantum bipartite systems. Code for a Semester Research Project within the Trapped Ion Quantum Information group (TIQI) at the ETH 2020.

**Abstract**: Several physical systems of interest consist of an ensemble of two level systems coupled to a bosonic mode in an open system setting. Therefore, it is fundamental to be able to efficiently simulate such systems in order to test hypothesis and make predictions. When all two level systems are assumed to be identical the **permutational symmetry** of the ensemble allows to **exponentially reduce the computational resources** needed. This powerful result has been implemented in the open source library **Permutational Invariant Quantum Solver** (PIQS) and has been successfully applied to explore a wide variety of physical phenomena such as superradiant light emission, spin squeezing, phase transitions, and the ultrastrong coupling regime. As soon as a one two level system can be distinguished from the others though, the permutational symmetry is broken and the exponential complexity is recovered. In this project we explore how to extend the applicability of PIQS to mixed-species ensembles characterised by **partial permutational symmetry** and if this can be done with the same exponential reduction of computational resources. In particular, we focus on the case of a **bipartite system** composed of two level systems belonging to two different species. This example allows to show that the speedup is **still exponential** and gives a way to estimate the complexity for a more **general mixed-species scenario**, which is of great scientific interest.
