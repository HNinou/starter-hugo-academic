---
title: "Paper deep dive : Linking connectivity, dynamics and computations in
  low-rank recurrent neural networks, Mastrogiuseppe & Ostojic"
date: 2024-09-08T15:36:22.897Z
draft: false
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
## Table of Contents
1. [Introduction](#introduction)
2. [Theoretical Framework](#theoretical-framework)
    1. [Building a firing rate model](#building-a-firing-rate-model)
        1. [The firing rate](#the-firing-rate)
        2. [The total synaptic current](#the-total-synaptic-current)
    2. [Networks with low-rank connectivity matrices](#networks-with-low-rank-connectivity-matrices)
        1. [Networks with unit-rank structure](#networks-with-unit-rank-structure)
        2. [Dynamical Mean-Field Theory](#dynamical-mean-field-theory)
        3. [Dynamical Mean-Field Theory extension to the {{< math >}}$\tau_r \gg \tau_s${{< /math >}} case](#dynamical-mean-field-theory-extension-to-the-\(\tau_r-\gg-\tau_s\)-case)
3. [Spontaneous Activity](#spontaneous-activity)
    1. [Reproduction of the paper's phase diagram](#reproduction-of-the-papers-phase-diagram)
    2. [Comparison with the phase diagram in the {{< math >}}$\tau_r \gg \tau_s${{< /math >}} case](#comparison-with-the-phase-diagram-in-the-\(\tau_r-\gg-\tau_s\)-case)
4. [Response to an external input](#response-to-an-external-input)
    1. [Reproduction of figure 2.D of the article](#reproduction-of-figure-2D-of-the-article)
5. [Conclusion](#conclusion)

## Introduction

Cortical networks, consisting of highly interconnected neurons with recurrent synapses, are believed to be the fundamental units of mammalian brains. These networks' connectivity lies somewhere between fully structured and fully random. Since the 1980s, several approaches to cortical network connectivity design have been proposed, but they lack a unifying conceptual framework. 

This article aims to link the dynamics of recurrent neural networks to their connectivity matrix and show how one can design low-rank connectivity structures to implement specific computations, illustrated through four tasks.

## Theoretical Framework

The model used to describe cortical neural networks is a firing rate model. Each node (neuron) in the network is represented by its firing rate {{< math >}}$\phi(x_i)${{< /math >}} where {{< math >}}$\phi(x) = \tanh(x)${{< /math >}}, and {{< math >}}$i \in [1, \dots, N]${{< /math >}}, with {{< math >}}$N${{< /math >}} being the number of neurons. The evolution of the neuron's firing rate is governed by:

{{< math >}}$$
\dot{x_i}(t) = -x_i(t) + \sum_{j=1}^{N} J_{ij} \phi(x_j(t)) + I_i 
$$ {{< /math >}}  
\label{fr_model}

where {{< math >}}$J_{ij}${{< /math >}} is the connectivity matrix representing the synaptic connections and {{< math >}}$I_i${{< /math >}} is the external input to neuron {{< math >}}$i${{< /math >}}. This firing rate model simplifies the simulation of cortical networks while focusing on certain approximations.

### Building a firing rate model

#### The firing rate

The firing rate, {{< math >}}$r(t)${{< /math >}}, is modeled as a function of the total synaptic current, {{< math >}}$x(t)${{< /math >}}. With membrane capacitance and resistance in mind, the firing rate can be expressed as:

{{< math >}}$$
\tau_r \frac{dr}{dt} = -r + \phi(x(t)) 
$$ {{< /math >}}  
\label{v(I)}

Here, {{< math >}}$\tau_r${{< /math >}} is the characteristic time, typically 20 ms.

#### The total synaptic current

The total synaptic current {{< math >}}$x_i(t)${{< /math >}} is written as:

{{< math >}}$$
x_i(t) = \sum_{j=1}^{N} J_{ij} \int_{-\infty}^{t} d\tau K(t-\tau) r_j(\tau) 
$$ {{< /math >}}  
\label{I(t)withK}

Taking {{< math >}}$K(t) = \frac{\exp(-t/\tau_s)}{\tau_s}${{< /math >}}, where {{< math >}}$\tau_s${{< /math >}} is the synaptic conductance time constant, results in:

{{< math >}}$$
\tau_s \frac{dx_i(t)}{dt} = -x_i(t) + \left( \sum_{j=1}^{N} J_{ij} r_j(t) + I_i \right) 
$$ {{< /math >}}  
\label{I(u)}

This equation simplifies to two cases: {{< math >}}$\tau_r \gg \tau_s${{< /math >}} or {{< math >}}$\tau_r \ll \tau_s${{< /math >}}, determining how firing rate models approximate input-output dynamics in neurons.

### Networks with low-rank connectivity matrices

The connectivity matrix {{< math >}}$J_{ij}${{< /math >}} is modeled as the sum of a random matrix {{< math >}}$\chi${{< /math >}} and a low-rank structured matrix {{< math >}}$P${{< /math >}}:

{{< math >}}$$
J_{ij} = g \chi_{ij} + P_{ij} 
$$ {{< /math >}}  
\label{eqJ(chi,p)}

Where {{< math >}}$g${{< /math >}} controls the random strength, and {{< math >}}$P_{ij}${{< /math >}} is of order {{< math >}}$1/N${{< /math >}}. A key assumption is that the variance of {{< math >}}$\chi_{ij}${{< /math >}} scales as {{< math >}}$1/N${{< /math >}}, which allows networks of different sizes to be compared theoretically.

#### Networks with unit-rank structure

For simplicity, assume {{< math >}}$P_{ij} = \frac{m_i n_j}{N}${{< /math >}}, where {{< math >}}$m = \{m_i\}${{< /math >}} and {{< math >}}$n = \{n_j\}${{< /math >}} are {{< math >}}$N${{< /math >}}-dimensional vectors. The dynamics of the system are governed by the random strength {{< math >}}$g${{< /math >}} and the structure strength {{< math >}}$m^T n / N${{< /math >}}. This model has parallels to Hopfield networks but does not require symmetry or Dale's law.

#### Dynamical Mean-Field Theory

In large networks with low-dimensional connectivity (scaling as {{< math >}}$1/N${{< /math >}}), the activity of each neuron is denoted as {{< math >}}$\mu_i = \kappa m_i${{< /math >}}, where {{< math >}}$\kappa = \langle n_i [\phi_i] \rangle_i${{< /math >}}. Non-zero {{< math >}}$\kappa${{< /math >}} implies network activity projected along {{< math >}}$m${{< /math >}}.

### Dynamical Mean-Field Theory extension to the {{< math >}}$\tau_r \gg \tau_s${{< /math >}} case

Starting from equation {{< math >}}$\dot{r_i} = -r_i + \phi(\eta_i)${{< /math >}}, with {{< math >}}$\eta_i${{< /math >}} representing the total input, we derive the following results:

{{< math >}}$$
\mu_i = [x_i] = m_i \kappa 
$$ {{< /math >}}  
{{< math >}}$$
\Delta^I_0 = [x_i^2] - [x_i]^2 = g^2 \langle [\phi_i^2] \rangle 
$$ {{< /math >}}

This approach captures the stationary behavior of both the original and extended models.

## Spontaneous Activity

### Reproduction of the paper's phase diagram

Simulations reproduced the phase diagram showing chaotic and structured activity for different parameter values, as seen in figure {{< math >}}$1${{< /math >}}.

### Comparison with the phase diagram in the {{< math >}}$\tau_r \gg \tau_s${{< /math >}} case

The extended model with {{< math >}}$\tau_r \gg \tau_s${{< /math >}} exhibits similar stationary and chaotic activity to the original model.

## Response to an external input

### Reproduction of figure 2.D of the article

The network response along the {{< math >}}$m${{< /math >}} vector for different input intensities was consistent with the paper's results, confirming that the alternative model retains key properties of the original model.

## Conclusion

Through critical analysis, we identified an unexplored model in which {{< math >}}$\tau_r${{< /math >}} is not negligible compared to {{< math >}}$\tau_s${{< /math >}}. This alternative model demonstrates similar behavior in both spontaneous activity and external stimuli responses, suggesting that it is a viable extension for modeling low-rank recurrent neural networks in a biologically plausible framework.
