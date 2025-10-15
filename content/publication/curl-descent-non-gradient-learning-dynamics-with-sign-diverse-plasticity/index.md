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
abstract: 'Gradient-based algorithms are a cornerstone of artificial neural
  network training, yet it remains unclear whether biological neural networks
  use similar gradient-based strategies during learning. Experiments often
  discover a diversity of synaptic plasticity rules, but whether these amount to
  an approximation to gradient descent is unclear. Here we investigate a
  previously overlooked possibility: that learning dynamics may include
  fundamentally non-gradient "curl"-like components while still being able to
  effectively optimize a loss function. Curl terms naturally emerge in networks
  with inhibitory-excitatory connectivity or Hebbian/anti-Hebbian plasticity,
  resulting in learning dynamics that cannot be framed as gradient descent on
  any objective. To investigate the impact of these curl terms, we analyze
  feedforward networks within an analytically tractable student-teacher
  framework, systematically introducing non-gradient dynamics through neurons
  exhibiting rule-flipped plasticity. Small curl terms preserve the stability of
  the original solution manifold, resulting in learning dynamics similar to
  gradient descent. Beyond a critical value, strong curl terms destabilize the
  solution manifold. Depending on the network architecture, this loss of
  stability can lead to chaotic learning dynamics that destroy performance. In
  other cases, the curl terms can counterintuitively speed learning compared to
  gradient descent by allowing the weight dynamics to escape saddles by
  temporarily ascending the loss. Our results identify specific architectures
  capable of supporting robust learning via diverse learning rules, providing an
  important counterpoint to normative theories of gradient-based learning in
  neural networks.'
draft: false
featured: true
image:
  filename: screenshot-2025-10-15-at-17.57.18.png
  focal_point: Smart
  preview_only: false
date: 2025-10-15T15:55:16.790Z
---
