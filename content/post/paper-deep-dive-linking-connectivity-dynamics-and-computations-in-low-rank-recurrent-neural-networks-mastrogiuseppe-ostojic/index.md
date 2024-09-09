---
title: "Paper deep dive : Linking connectivity, dynamics and computations in
  low-rank recurrent neural networks, Mastrogiuseppe & Ostojic, 2018"
date: 2024-09-08T15:36:22.897Z
summary: "Cortical networks, which consist in highly interconnected neurons with
  recurrent synapses are believed to make for the fundamental units of mammalian
  brains. Observations show that cortical connectivity lies somewhere between
  fully structured and fully random. Several functional approaches have been
  made for connectivity design of cortical networks since the 80's but they lack
  a unifying conceptual picture. To address this matter, authors point out that
  all these approaches share something in common: the fact that the resulting
  connectivity matrices are low rank. This article [1] aims at linking the
  recurrent neural networks' dynamics to their connectivity matrix and showing
  how one can design the low-rank connectivity structure of such networks to
  implement specific computations. The latter point is illustrated on four
  specific tasks."
draft: false
featured: false
image:
  filename: capture-d’écran-2024-09-09-à-15.18.02.png
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

Cortical networks, which consist in highly interconnected neurons with recurrent synapses are believed to make for the fundamental units of mammalian brains. Observations show that cortical connectivity lies somewhere between fully structured and fully random. Several functional approaches have been made for connectivity design of cortical networks since the 80's but they lack a unifying conceptual picture. To address this matter, authors point out that all these approaches share something in common: the fact that the resulting connectivity matrices are low rank. This article [1] aims at linking the recurrent neural networks' dynamics to their connectivity matrix and showing how one can design the low-rank connectivity structure of such networks to implement specific computations. The latter point is illustrated on four specific tasks.

## Theoretical Framework

The model used here to describe cortical neural networks is a firing rate model. This means that each node (i.e. neuron) in the network is represented by its firing rate {{< math >}}$\phi(x_i)${{< /math >}} with {{< math >}}$\phi(x)=\textrm{tanh}(x)${{< /math >}} being the current-to-rate transfer function, and {{< math >}}$i\in [1\dots N]${{< /math >}}, with {{< math >}}$N${{< /math >}} the number of neurons. The evolution of a neuron's firing rate is governed by equation (1).

{{< math >}}$$
    \dot{x_i}(t) = -x_i(t) + \sum_{j=1}^N J_{ij} \phi(x_j(t)) + I_i \quad (1)
$$ {{< /math >}}

where {{< math >}}$J_{ij}${{< /math >}} is the connectivity matrix representing the synaptic connections of the network and {{< math >}}$x_i${{< /math >}} is the external current input to neuron {{< math >}}$i${{< /math >}}. Note here that {{< math >}}$\phi(x_i)${{< /math >}} representing the firing rate can have negative values. This can be dealt with by replacing the {{< math >}}$\textrm{tanh}${{< /math >}} function, which makes the calculations easier, by a sigmoid without causing major changes to the theoretical results.

As all the results of this article come from analysis and simulations of this model, it is crucial to understand its limitations as for the description of cortical neural networks. With this goal in mind, we explore some developments in order to understand the underlying hypotheses and limitations of this model by building it again from scratch [2].

### Building a firing rate model

The behaviour of a neuron can be described by the neuronal response function {{< math >}}$y(t)${{< /math >}} that encodes the exact time at which it fires spikes. A model involving {{< math >}}$y(t)${{< /math >}} is called a spiking model. 
{{< math >}}$$
        y(t) = \sum_{i=1}^n \delta(t-t_i),    \quad  r(t) = \int_t^{t+\Delta t}  \underbrace{\frac{1}{\Delta t}\langle y(\tau) \rangle  d\tau}_{\textrm{average over the trials}} \quad (2)
$$ {{< /math >}}

Firing rate models focus on the quantity {{< math >}}$r(t)${{< /math >}} in eq. (2) which is an approximation of the exact spike sequence {{< math >}}$y(t)${{< /math >}}. They have the advantage of being easier to simulate on computers as they do not take into account the short time scale dynamics of the spikes. As we want to model the total input for the neurons, we can look at {{< math >}}$r(t)${{< /math >}} instead of {{< math >}}$y(t)${{< /math >}} if there is not too much variability between two trials. Indeed, upon summing over different synapses, one has low variability (Central Limit Theorem) if the entries are numerous and uncorrelated.

Firing rate models are relevant when

1. The firing of neurons in a network is uncorrelated (there is little synchronous firing)
2. The precise patterns of spike timing are unimportant. Indeed, the information regarding those precise patterns is lost when averaging over the trials.

#### Figure 1: Sketch of a modeled neuron with presynaptic inputs {{< math >}}$r_i${{< /math >}} and postsynaptic output {{< math >}}$r(t)${{< /math >}}. {{< math >}}$x(t)${{< /math >}} is the total synaptic current or input.
![Neuron drawing](Neuron_drawing.png)


In order to fully describe a firing rate model, we have to specify the dependence of the postsynaptic firing rate on the total synaptic input {{< math >}}$r(x)${{< /math >}} and the dependence of the total synaptic input on the presynaptic inputs {{< math >}}$x(r_1,r_2,r_3,r_4)${{< /math >}}.

#### The firing rate

Let's first describe the firing rate {{< math >}}$r(t)${{< /math >}} as a function of the total synaptic current {{< math >}}$x(t)${{< /math >}}. We could simply write {{< math >}}$r(t) = \phi(x(t))${{< /math >}}, {{< math >}}$\phi${{< /math >}} being the current-to-rate function, but due to the membrane capacitance and resistance, we should rather express the firing rate {{< math >}}$r(t)${{< /math >}} as a low-pass filtered version of its steady state with characteristic time {{< math >}}$\tau_r${{< /math >}}, usually of the order of 20 ms.

{{< math >}}$$
    \tau_r \frac{dr}{dt} = -r + \phi(x(t))
$$ {{< /math >}}


Note that in reality, it is the membrane potential, not the firing rate, that is a low pass of the input current and that the dynamics of the two are not the same. 

#### The total synaptic current

We now want to write the total synaptic current of neuron {{< math >}}$i${{< /math >}}, {{< math >}}$x_i(t)${{< /math >}} as a function of the {{< math >}}$N${{< /math >}} presynaptic firing rates {{< math >}}$r_j(t)${{< /math >}} for {{< math >}}$j \in [1 \dots N]${{< /math >}} and their associated weights {{< math >}}$J_{ij}${{< /math >}}. Note that {{< math >}}$J_{ij}>0${{< /math >}} corresponds to an excitatory synapse while {{< math >}}$J_{ij}<0${{< /math >}} corresponds to an inhibitory one. We introduce the synaptic kernel response function {{< math >}}$K(t)${{< /math >}} that is simply the response current induced at time {{< math >}}$t${{< /math >}} by a spike at time {{< math >}}$t=0${{< /math >}}. Assuming that the effects of a spike sum linearly, we can then write the total synaptic current as 

{{< math >}}$$
    x_i(t) = \sum_{j=1}^{N} J_{ij} \int_{-\infty}^{t} d\tau K(t-\tau) \underbrace{y_j(\tau)}_{\sum_{k=1}^{N} \delta(\tau-t_k)}
$$ {{< /math >}}

which in the firing rate model approximation writes

{{< math >}}$$
    x_i(t) = \sum_{j=1}^{N} J_{ij} \int_{-\infty}^{t} d\tau K(t-\tau) r_j(\tau) \quad (3)
$$ {{< /math >}}

By taking {{< math >}}$K(t) = \exp(-t/\tau_s)/\tau_s${{< /math >}} (with {{< math >}}$\tau_s${{< /math >}} the time constant that describes the decay of the synaptic conductance, usually of the order of a few milliseconds), then (3) can be written as a differential equation

{{< math >}}$$
    \tau_s \frac{dx_i(t)}{dt} = -x_i(t) + \Big( \sum_{j=1}^{N} J_{ij} r_j(t) + I_i\Big) \quad (4)
$$ {{< /math >}}

This is actually equation (1) of the paper in which {{< math >}}$\tau_s${{< /math >}} was chosen equal to 1 for simplicity. Equations (4) and (2) give us the two parts needed to describe a firing rate model which can be simplified in two extreme cases:

- {{< math >}}$\tau_r \gg \tau_s${{< /math >}}: Then {{< math >}}$x_i(t) = \sum_{j=1}^{N} J_{ij} r_j(t) + I_i${{< /math >}} and {{< math >}}$r${{< /math >}} is a low-pass of {{< math >}}$x(t)${{< /math >}}.
- {{< math >}}$\tau_r \ll \tau_s${{< /math >}}: Then {{< math >}}$r(t) = \phi(x(t))${{< /math >}}, {{< math >}}$r(t)${{< /math >}} follows {{< math >}}$x(t)${{< /math >}} instantaneously.

Authors implicitly consider that we're in the situation where {{< math >}}$\tau_r \ll \tau_s${{< /math >}} which is not obvious at all as both characteristic times seem to be of the same order of magnitude. This observation led me to explore what would have been the paper's results if they had instead considered the situation where {{< math >}}$\tau_r \gg \tau_s${{< /math >}}. This latter case yields the following equation for the system that has to be compared to equation (1):

{{< math >}}$$
    \dot{r_i} = -r_i + \phi(\sum_{j=1}^N J_{ij}r_j+I_i) \quad (5)
$$ {{< /math >}}

In the following, we will reproduce some of both theoretical and simulatory results of the paper and try to extend them to a system governed by equation (5).

### Networks with low-rank connectivity matrices

We start by placing ourselves in the same context as in the article i.e. with a negligible membrane relaxation characteristic time.

The connectivity matrix {{< math >}}$J_{ij}${{< /math >}} is the sum of an uncontrolled random matrix {{< math >}}$\chi${{< /math >}} and of a structured low ranked known matrix {{< math >}}$P${{< /math >}}. {{< math >}}$J_{ij}${{< /math >}} is thus defined by

{{< math >}}$$
    J_{ij} = \underbrace{g \chi_{ij}}_{\textrm{mean $0$, variance $g^2/N$}} + \underbrace{P_{ij}}_{\textrm{of order $1/N$}}
$$ {{< /math >}}


Note that there is no biological reason for which {{< math >}}$\chi_{ij}${{< /math >}} should have a variance scaling as {{< math >}}$1/N${{< /math >}}. This constraint makes possible the comparison of networks of different sizes from a theoretical perspective (especially in the case {{< math >}}$N \to \infty${{< /math >}}) and can be dealt with by adjusting the random strength {{< math >}}$g${{< /math >}} at will.

#### Networks with unit-rank structure

We start with {{< math >}}$P_{ij} = \frac{m_i n_j}{N}${{< /math >}} with {{< math >}}$m=\{m_i\}${{< /math >}} and {{< math >}}$n=\{n_j\}${{< /math >}} two N-dimensional vectors. Authors define two important parameters, {{< math >}}$g${{< /math >}} the random strength, and {{< math >}}$m^Tn/N${{< /math >}} the structure strength, that govern the type of dynamics of the system. The type of networks studied here is related to the Hopfield networks studied in class. However the unit-rank terms here do not require to be symmetric and can be correlated to each other. Regarding the biological plausibility of this proposed connectivity, it can be noticed that Dale's law is not imposed here. 

#### Dynamical Mean-Field Theory

Under the assumption of a large network with a weak low-dimensional connectivity matrix (scaling as {{< math >}}$1/N${{< /math >}}) one can derive the activity of each neuron thanks to dynamical mean-field theory by considering the mean and variance of the input it receives. Authors find that the average equilibrium input to unit {{< math >}}$i${{< /math >}} is denoted {{< math >}}$\mu_i = \kappa m_i${{< /math >}} with {{< math >}}$\kappa = \langle n_i[\phi_i] \rangle_i${{< /math >}}, that is that the activity of the network is one dimensional, along the vector {{< math >}}$m${{< /math >}} as long as {{< math >}}$\kappa>0${{< /math >}}. As {{< math >}}$\kappa${{< /math >}} represents the activity projected on vector {{< math >}}$n${{< /math >}}, non-vanishing values of {{< math >}}$\kappa${{< /math >}} require a non-vanishing overlap between {{< math >}}$m${{< /math >}} and {{< math >}}$n${{< /math >}}.

#### Dynamical Mean-Field Theory extension to the {{< math >}}$\tau_r \gg \tau_s${{< /math >}} case

Starting from equation (5), we derive a Dynamical Mean-Field approach in order to express both {{< math >}}$\mu_i \equiv [x_i]${{< /math >}} and {{< math >}}$\Delta_0^I \equiv [x_i^2] - [x_i]^2${{< /math >}}. Similarly to the derivation proposed in the supplementary information of the paper, one can consider the case where {{< math >}}$I_i=0${{< /math >}} {{< math >}}\forall i{{< /math >}}. By denoting

{{< math >}}$$
    \eta_i(t) = \sum_{j=1}^N J_{ij} r_j = g \sum_{j=1}^N \chi_{ij} r_j + \frac{m_i}{N} \sum_{j=1}^N n_j r_j
$$ {{< /math >}}


equation (5) can be rewritten as

{{< math >}}$$
    \dot{r_i} = -r_i + \phi(\eta_i)
$$ {{< /math >}}


In the stationary scenario, we would then have

{{< math >}}$$
    r_i = \phi(\eta_i)
$$ {{< /math >}}

By applying {{< math >}}$\phi^{-1}${{< /math >}} to both sides of this equation, we fall back on equation 28 of the paper that gives us an expression for {{< math >}}$\mu_i${{< /math >}} and {{< math >}}$\Delta_0^I${{< /math >}}.

{{< math >}}$$
    \mu_i = [x_i] = m_i \kappa
$$ {{< /math >}}

{{< math >}}$$
    \Delta^I_0 = [x_i^2] - [x_i]^2 = g^2 \langle [\phi_i^2] \rangle
$$ {{< /math >}}

Conducting a DMF analysis in the chaotic scenario is however trickier, and we do not develop it in this project although it is an interesting lead. 

## Spontaneous Activity

### Reproduction of the paper's phase diagram

#### Figure 2: Phase diagram obtained from personal simulations (50×50 resolution). Left: measures the chaotic nature of the system. Right: measures the structured nature of the system.
![Phase diagram obtained from personal simulations](phase_diag_stat_chao_randomness_fixed.png)

#### Figure 3: Left: Theoretical phase diagram from the article. Right: Phase diagram obtained from personal simulations (50×50 resolution).
![Phase diagram](phase_diag_art.PNG)


In figure 1 of the article, a phase diagram is proposed. The result presented shows the phase diagram obtained from theoretical results. We chose to reproduce this phase diagram thanks to simulations (fig.3). To do so, finding the good statistics to describe the chaoticity and structuration of the system is crucial. Authors hint us in using the temporal variance of {{< math >}}$x_i${{< /math >}}, averaged over the number of neurons, to characterize chaoticity {{< math >}}$\langle \textrm{std}_t(x_i) \rangle_i${{< /math >}}. We use the activity along {{< math >}}$m${{< /math >}}, given by {{< math >}}$\kappa = \langle n_i[\phi_i] \rangle_i${{< /math >}} to characterize structure. We superimposed the phase diagrams for chaoticity and structure shown in figure 2 to obtain the one shown on the right panel of figure 3.

Each of the 2500 pixels on the simulated phase diagram represents a simulation. The statistics were measured after a transient phase was over for 100 seconds with parameter {{< math >}}$\Delta t = 0.5 \textrm{s}${{< /math >}}. Considering the ergodicity theorem, this allows us not to have to make several simulations for each pair of parameters, which would have multiplied by the same amount the time spent in simulating the transient phase.

The biologically plausible phase is the one where there is both structured and chaotic activity.

### Comparison with the phase diagram in the {{< math >}}$\tau_r \gg \tau_s${{< /math >}} case

We investigate the changes that using equation (5) instead of (1) would bring to the phase diagram describing the system's behavior. Interestingly, the phase diagram obtained for equation (5) is very similar to that of (1) (see figure 4). This was expected for the stationary part of the diagram but not necessarily for the chaotic part. This result hints us into thinking that the DMF theoretical results one would derive for our alternative model might be the same as the ones found in the paper.

#### Figure 4: Left: Phase diagram for {{< math >}}$\tau_r \ll \tau_s${{< /math >}}. Right: Phase diagram for {{< math >}}$\tau_r \gg \tau_s${{< /math >}} (resolution 50 × 50).
![Phase diagram](phase_diag_compare.png)


## Response to an external input

For further analysis, we aim at comparing the behavior of our alternative model to that of the paper in response to an external stimulus {{< math >}}$I${{< /math >}}. As in the article, we look at the response of the system along the {{< math >}}$m${{< /math >}} vector. Figure 5 shows the transient dynamics for both models when the system is subject to an external input with the same connectivity matrix and the same initial conditions. One can see that, as expected by the theory, the stationary states are the same while the transient phases slightly differ. Indeed, the system seems to evolve more slowly for the alternative model (figure 5 right).

#### Figure 5: Transient dynamics in the {{< math >}}$\tau_r \ll \tau_s${{< /math >}} (Left) and {{< math >}}$\tau_r \gg \tau_s${{< /math >}} (Right) scenarios in response to a step input.
![Transient dynamics](transient_dyn.png)

### Reproduction of figure 2.D of the article

Figure 6, corresponding to figure 2.D of the article, plays a major role in the four task implementations designed in the following of the paper. Reproducing this result for our alternative model basically ensures that it will be able to perform the proposed tasks as well. The reproduction of this result is presented in figure 7. The activity along vector {{< math >}}$m${{< /math >}} is recorded for 15 different intensities of the input in three case scenarios. First, in the case where {{< math >}}$m${{< /math >}}, {{< math >}}$n${{< /math >}}, and {{< math >}}$I${{< /math >}} are all orthogonal in respect to each other. The system hence shows no response along {{< math >}}$m${{< /math >}} as {{< math >}}$\kappa = \langle n_i[\phi_i] \rangle_i = 0${{< /math >}} (fig.7 left). Second, in the case where {{< math >}}$m${{< /math >}} and {{< math >}}$n${{< /math >}} are orthogonal and {{< math >}}$I${{< /math >}} has a component along {{< math >}}$n${{< /math >}}. The network then shows some activity along {{< math >}}$m${{< /math >}} (fig.7 center). Finally, in the case where {{< math >}}$m${{< /math >}} and {{< math >}}$n${{< /math >}} have an overlap and {{< math >}}$I${{< /math >}} is colinear to the orthogonal part of {{< math >}}$n${{< /math >}} with respect to {{< math >}}$m${{< /math >}}, the system shows a bistable activity along {{< math >}}$m${{< /math >}} (fig.7 right). This bistable activity vanishes when the input becomes too strong.

The same results were observed for our alternative model, and are presented in figure 8.

#### Figure 6
![Figure 2D of the article](2D_art.PNG)

#### Figure 7: Activity along {{< math >}}$m${{< /math >}} as a function of input strength in three different scenarios in the {{< math >}}$\tau_r \ll \tau_s${{< /math >}} scenario. Left: {{< math >}}$m${{< /math >}}, {{< math >}}$n${{< /math >}}, and {{< math >}}$I${{< /math >}} are orthogonal. Center: {{< math >}}$m${{< /math >}} and {{< math >}}$n${{< /math >}} are orthogonal, and {{< math >}}$I${{< /math >}} has a component along {{< math >}}$n${{< /math >}}. Right: {{< math >}}$m${{< /math >}} and {{< math >}}$n${{< /math >}} have an overlap.
![Activity along m](2Dorth.png)

#### Figure 8: Same as figure 7 but in the {{< math >}}$\tau_r \gg \tau_s${{< /math >}} scenario.
![Same as figure 2D but for the alternative model](2Dorth_alt.png)


## Conclusion

Thanks to a carefully conducted critical reading, we noticed that when building their firing rate model, authors implicitly did not explore an alternative model in which {{< math >}}$\tau_r${{< /math >}} (the membrane characteristic time) is not negligible with respect to {{< math >}}$\tau_s${{< /math >}} (the synaptic characteristic time). This alternative model is an interesting extension as it covers a theoretical framework ignored by the authors. We showed through a short theoretical analysis that both models were equivalent in the stationary case. Finally, we showed experimentally that the alternative model we proposed in equation (5) has very similar behavior to that of the paper's, both for spontaneous activity and in response to an external stimulus. Because this behavior analysis makes for the building block of the four task implementations, it is legitimate to think that the proposed alternative model will also be able to perform these tasks. The mini-project conducted here therefore shows that the low-rank recurrent neural networks studied in this paper are even more biologically plausible than they claim to be.

- - -

### References

* [1] Francesca Mastrogiuseppe and Srdjan Ostojic. Linking connectivity, dy-namics and computations in low-rank recurrent neural networks. Neuron, 99(3):609–623.e29, August 2018. arXiv: 1711.09672.
* [2] Peter Dayan and L. F. Abbott. Theoretical neuroscience: computational and mathematical modeling of neural systems. Computational neuro-science. Massachusetts Institute of Technology Press, Cambridge, Mass, 2001.
