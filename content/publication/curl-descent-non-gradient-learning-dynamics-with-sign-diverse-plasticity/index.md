---
title: "Curl Descent: Non-Gradient Learning Dynamics with Sign-Diverse Plasticity"
publication_types:
  - "1"
authors:
  - Hugo Ninou
  - Jonathan Kadmon
  - N. Alex Cayco-Gajic
doi: https://doi.org/10.48550/arXiv.2510.02765
publication: NeurIPS 2025 Spotlight
abstract: "Learning in large neural networks requires solving the credit
  assignment problem: how can individual synapses be updated to improve global
  performance? Normative theories assert that synaptic weights should be
  adjusted proportionally to a local gradient estimate to descend a global
  objective function. Recent years have seen various proposals for biologically
  plausible learning rules that approximate gradient descent in neural networks.
  However, this view is difficult to reconcile with the  diversity of plasticity
  rules observed experimentally. This diversity means that weight updates may
  not always align with a global gradient, introducing fundamentally
  non-gradient components into the learning dynamics. This problem is often
  overlooked when training artificial neural network models, raising our central
  question: Can networks still learn effectively when their dynamics include
  non-gradient terms?"
draft: false
featured: true
image:
  filename: screenshot-2025-10-15-at-17.57.18.png
  focal_point: Smart
  preview_only: false
date: 2025-10-15T15:55:16.790Z
---
Neural networks using unsupervised learning rules, such as Hebbian plasticity, can be shown to minimize objective functions like those for feature extraction, but only when paired with specific architectures \[Pehlevan et al. 2015, Neural Computation]. For instance, the learning dynamics of a recurrent neural network composed solely of excitatory neurons equipped with Hebbian plasticity can be expressed as the gradient of an objective function. However, this property is fragile; introducing inhibitory neurons with plastic synapses governed by the same Hebbian rule renders the learning dynamics provably non-gradient (Fig. A). This occurs because the first-order recurrent terms in the neural activity, y=(I+WD)f (where f is an external input and D is a diagonal matrix with elements ±1 for neuron types), introduce non-symmetric terms into the weight update equations that cannot be derived from any scalar potential. This illustrates a more general principle: while certain architectures can be framed as performing gradient descent, many biologically plausible scenarios featuring diverse populations naturally give rise to learning dynamics that combine a gradient term with a non-gradient component.


To investigate the impact of these non-gradient terms, we adopt an analytically tractable teacher-student framework \[Saxe et al. 2014]. In this supervised setting, a student network learns to match a teacher’s output by minimizing its mean-squared error (Fig. B). We modify the standard gradient descent update in a way that mirrors the effect of inhibitory neurons in the recurrent unsupervised setting, by flipping the sign of the synaptic weight updates for specific synapses originating from a rule-flipped population of neurons (Fig. C). This results in deterministic non-gradient learning dynamics we term ‘curl descent’. 


Under curl descent, the fixed points of the learning dynamics coincide with those of gradient descent. However, their stability may change depending on network architectureWe analyze this stability by examining the Jacobian matrix of the learning dynamics at solution points (C=0, Fig. B). A solution is stable if all eigenvalues of its Jacobian have negative real parts. Using random matrix theory, we characterize a dynamical phase transition \[Fruchart et al. 2021, Nature] where the solution manifold becomes unstable. We show that this transition is determined by two factors: the compression ratio of the network and the fraction of rule-flipped synapses introduced either in the hidden or the readout layer. 


Our analysis reveals that expansive networks (with more hidden units than inputs) are more robust to the introduction of these adversarial synapses than contractive ones (Fig. DE). Introducing rule-flipped synapses in the hidden layer leads to poor performance (Fig. F) and chaotic synaptic weight dynamics in the unstable phase (Fig. H). Surprisingly however, in the unstable phase corresponding to the readout weights, the learning dynamics still managed to find low-error regions (Fig. G), and could in some situations improve convergence speed. These results generalize to nonlinear networks (Fig. I) and over a wide range of hyperparameters, including training set size, the student weights’ initialization, task complexity and fraction of rule-flipped synapses. 
Taken together, our theoretical and numerical results reveal that networks featuring non-gradient learning dynamics—arising from diverse synaptic populations—can still learn a global objective function, depending on their architecture and the fraction of rule-flipped synapses. Compressive networks are more prone to destabilization of their solution points when adversarial synapses are introduced, however this instability is not always detrimental. While adding non-gradient terms in the hidden weights can lead to catastrophic chaotic dynamics, introducing them in the readout weights enables the network to find alternative low-error solutions, and escape saddle points that typically trap gradient descent, leading to faster convergence.


Our results suggest that strict adherence to a global gradient may not always be the most efficient learning strategy. The diversity of plasticity rules observed experimentally might allow biological circuits to leverage non-gradient dynamics for more rapid and effective optimization. Future work will extend this framework to more structured, non-i.i.d. inputs and more complex architectures (multidimensional outputs, recurrent networks, deep networks). These extensions will allow us to examine whether non-gradient learning dynamics can explain phenomena such as representational drift—where ongoing synaptic reorganization occurs without impairing performance—ultimately bringing theory closer to the conditions under which biological and artificial networks learn in the real world.

![](cosyne2026_figure.png)

Caption. A Excitatory-inhibitory recurrent neural network equipped with a Hebbian learning rule. yi: neural activity; Wij: synaptic weights; f: external input; Dii: diagonal matrix with elements ±1 for neuron types; : time constants. The recurrent neural dynamics (top equation) are evaluated at steady state up to first order. The learning dynamics (bottom equation) cannot be expressed as the gradient of a scalar function. B Teacher-student framework : both networks share the same architecture and receive inputs drawn from an i.i.d. normal distribution. The teacher’s weights are fixed while the student weights change through learning to minimize the mean-squared error C. C Schematic of the modified weight update. D,E Analytical phase diagrams. Stability of the solution manifold as a function of the compression ratio c and the fraction of rule-flipped synapses in each layer h(hidden) and r(readout). F,G Simulation results. Test-error as a function of c and h(hidden) and r(readout). The black curve shows the analytically derived stability boundary. H Example chaotic weight dynamics in the unstable regime. I Test-error for gradient descent and curl descent with a single rule-flipped readout weight.