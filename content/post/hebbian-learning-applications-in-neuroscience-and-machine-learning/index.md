---
title: "Hebbian learning : Applications in neuroscience and machine learning"
date: 2024-09-09T15:12:31.360Z
draft: true
featured: false
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
## Table of Contents

1. [Introduction](#introduction)
2. [What Defines an Intelligent System?](#1-what-defines-an-intelligent-system)
3. [Modeling Approaches in Neuroscience](#2-modeling-approaches-in-neuroscience)
   - [Mechanistic Models: The "How" Question](#21-mechanistic-models-the-how-question)
   - [Normative Models: The "Why" Question](#22-normative-models-the-why-question)
4. [Learning Paradigms in Machine Learning](#3-learning-paradigms-in-machine-learning)
5. [Experimental Evidence of Hebbian Learning](#4-experimental-evidence-of-hebbian-learning)
   - [Long-Term Potentiation (LTP) and Long-Term Depression (LTD)](#41-long-term-potentiation-ltp-and-long-term-depression-ltd)
   - [Inhibitory Plasticity](#42-inhibitory-plasticity)
6. [Hebbian Learning in Machine Learning](#5-hebbian-learning-in-machine-learning)
   - [Hebbian Learning for Principal Component Analysis (PCA)](#51-hebbian-learning-for-principal-component-analysis-pca)
   - [Hebbian Learning for Memory Retrieval](#6-hebbian-learning-for-memory-retrieval)
     - [Hopfield Networks](#61-hopfield-networks)
     - [Modern Hopfield Networks](#62-modern-hopfield-networks)
     - [Hopfield Networks with Inhibitory Neurons](#63-hopfield-networks-with-inhibitory-neurons)
7. [Normative Approaches to Memory Retrieval](#7-normative-approaches-to-memory-retrieval)
8. [Mathematical Extensions of Hebbian Learning](#8-mathematical-extensions-of-hebbian-learning)
9. [Conclusion](#9-conclusion)


**Introduction**

Hebbian learning is one of the most fundamental theories in neuroscience and machine learning, originating from Donald Hebb's work in 1949. It forms the basis for understanding synaptic plasticity and how networks of neurons can adapt and learn through repeated activation. This model not only explains neural mechanisms in the brain but also inspires computational methods for pattern recognition, memory retrieval, and self-organization in machine learning.

In this post, we will explore the full extent of Hebbian learning by reviewing its mathematical foundations, examining its role in neuroscientific models, and discussing modern applications in machine learning. By analyzing experimental evidence, computational models, and the most recent developments, we will highlight how Hebbian learning is key to understanding both biological and artificial intelligence systems.

---

## 1. **What Defines an Intelligent System?**

Before delving into the specifics of Hebbian learning, it's crucial to establish what constitutes an intelligent system. According to the presentation, intelligence in both biological and artificial systems can be characterized by three main components:

- **Neural Structure**: The physical arrangement of neurons and their connections.
- **Learning Rule**: The mechanism through which the system adapts and learns, with Hebbian learning being a prime example.
- **Data**: The information and experiences processed by the system.

Together, these factors define the architecture and behavior of any system that can be classified as intelligent, be it a brain or a machine learning model.

---

## 2. **Modeling Approaches in Neuroscience**

In neuroscience, two types of modeling approaches are widely used:

### 2.1 **Mechanistic Models: The "How" Question**
Mechanistic models aim to explain **how** neural phenomena occur. They describe the processes and components that lead to observable phenomena. For instance, Hebb's 1949 model is a qualitative mechanistic model that explains **pattern completion**, a cognitive process where incomplete information triggers the retrieval of a full memory. However, while mechanistic models tell us how the system behaves, they often leave the purpose or goal of the process unanswered.

### 2.2 **Normative Models: The "Why" Question**
In contrast, normative models address the **why** question. These models assume that neural processes have evolved to optimize certain objectives, often modeled as an optimization problem. For example, normative models assume there is a goal, such as maximizing memory retention, and that neural processes, including Hebbian learning, have evolved to achieve this.

---

## 3. **Learning Paradigms in Machine Learning**

Machine learning has developed several learning paradigms, many of which are inspired by the brain’s mechanisms of adaptation. The primary paradigms include:

- **Supervised Learning**: Involves a large amount of labeled data and typically uses error backpropagation to optimize weights in neural networks.
  
- **Unsupervised Learning**: Does not require labeled data, making it more biologically plausible. Hebbian learning and contrastive divergence are examples of unsupervised learning methods.

- **Reinforcement Learning**: Provides feedback only at the end of a task or sequence, with models receiving either positive or negative reinforcement based on their performance.

- **Self-Supervised Learning**: Uses the input data itself to generate supervisory signals, making it more efficient in data use.

- **Weakly Supervised Learning**: Involves limited feedback, usually only at the conclusion of a sequence of actions.

---

## 4. **Experimental Evidence of Hebbian Learning**

### 4.1 Long-Term Potentiation (LTP) and Long-Term Depression (LTD)
The experimental foundation for Hebbian learning lies in **long-term potentiation (LTP)** and **long-term depression (LTD)**, processes that strengthen or weaken synaptic connections over time based on activity patterns. LTP occurs when two neurons are stimulated simultaneously, enhancing their connection.

The typical experimental protocol for observing Hebbian learning involves the following steps:
1. **Measure initial EPSP** (Excitatory Post-Synaptic Potential) between two neurons.
2. **Simultaneously stimulate the neurons**.
3. **Measure the final EPSP** to observe any changes in synaptic strength.

In this context, Hebbian learning is observed between pyramidal neurons, and several parameters affect LTP/LTD induction:
- **Heterosynaptic processes**
- **Calcium levels**
- **Spike timing**
- **Cell types**

### 4.2 Inhibitory Plasticity
The investigation of **inhibitory plasticity**—the changes in synapses involving inhibitory neurons (which make up 20% of the brain)—is more complex due to the wide diversity of **GABAergic neurons**. However, recent studies have shown that **both Hebbian and Anti-Hebbian learning** occur at inhibitory synapses, contributing to:
- Maintaining **excitatory-inhibitory (EI) balance**.
- Preventing **winner-takes-all** dynamics, where only one neuron dominates.
- Implementing **gating mechanisms** that regulate signal flow.

Mechanistic models have demonstrated that inhibitory plasticity is crucial for various neural processes, especially memory retrieval.

---

## 5. **Hebbian Learning in Machine Learning**

### 5.1 Hebbian Learning for Principal Component Analysis (PCA)
Pehlevan and Chklovskii (2015) showed how Hebbian learning can be applied to **Principal Component Analysis (PCA)**. By formulating the learning process as a similarity-matching problem, the objective is to minimize the Frobenius norm:

{{< math >}}$$
\min \| X^\top X - Y^\top Y \|_F
$${{< /math >}}

This leads to the following learning rules:

{{< math >}}$$
\Delta M_{ij} = \frac{y_{T,i}(y_{T,j} - M_{T,ij} y_{T,i})}{\sum_T y_{t,i}^2}
$${{< /math >}}

{{< math >}}$$
\Delta W_{ij} = \frac{y_{T,i}(x_{T,j} - W_{T,ij} y_{T,i})}{\sum_T y_{t,i}^2}
$${{< /math >}}

These equations govern how synaptic weights are updated over time based on the activations of both neurons, highlighting the Hebbian nature of the learning process.

---

## 6. **Hebbian Learning for Memory Retrieval**

### 6.1 Hopfield Networks
In 1982, **Hopfield networks** became a pivotal application of Hebbian learning in computational neuroscience. Hopfield networks use binary neurons, and the activation rule is defined as:

{{< math >}}$$
y_i = \begin{cases} 
1 & \text{if } \sum_j W_{ij} y_j > \theta_i \\
-1 & \text{otherwise}
\end{cases}
$${{< /math >}}

These networks operate as **associative memory systems**, where patterns are stored and retrieved through synaptic interactions. The synaptic weight matrix **W** is symmetric, ensuring that the network’s dynamics will eventually converge to an attractor state that corresponds to a stored pattern. The learning rule for the synaptic weights is:

{{< math >}}$$
W_{ij} = \sum_{\mu \in \text{patterns}} F_i^\mu F_j^\mu
$${{< /math >}}

However, Hopfield networks have a limited capacity, storing about **0.14N** patterns, where **N** is the number of neurons. Improvements have been made to address these limitations.

### 6.2 Modern Hopfield Networks
Modern versions of Hopfield networks have continuous states and are continuous in time, providing greater flexibility but often at the cost of **biological plausibility**. Some modern networks, like those proposed by Krotov and Hopfield (2021), incorporate higher-order interaction terms, which allow for more sophisticated representations but reduce resemblance to actual neural circuits.

One exciting application of modern Hopfield networks is in **transformer architectures**, particularly in understanding how attention heads function. These networks, with multiple hidden layers, have provided insights into both biological and artificial attention mechanisms.

### 6.3 Hopfield Networks with Inhibitory Neurons
Recent advances have introduced **inhibitory neurons** into Hopfield networks, allowing for **Hebbian learning at excitatory synapses** and **Anti-Hebbian learning at inhibitory synapses**. This addition, explored by Mongillo et al. (2018), has increased memory capacity, although the precise role of inhibitory neurons in improving memory storage is not yet fully understood. Recent research by Gong et al. (2024) has employed Hebbian/Anti-Hebbian learning along with hyperparameter optimization to explore this dynamic further.

---

## 7. **Normative Approaches to Memory Retrieval**

A key challenge in neuroscience is to find normative models that explain why memory retrieval works the way it does. Consider a **recurrent neural network (RNN)** where the dynamics are defined as:

{{< math >}}$$
\tau \cdot y = W D y - y
$${{< /math >}}

Here, **W** is the weight matrix, and **D** is a diagonal matrix, with **Dii = 1** for excitatory neurons and **Dii = -1** for inhibitory neurons. The goal is to make **F**, the memory pattern, an attractor state. By minimizing the following distance function:

{{< math >}}$$
E = \frac{1}{2} \| W D F - F \|^2
$${{< /math >}}

We derive the following learning rule for updating **W**:

{{< math >}}$$
\Delta W_{ij} = FF^\top D - W FF^\top
$${{< /math >}}

This Hebbian/Anti-Hebbian learning rule demonstrates that memory retrieval can be framed as an optimization problem, providing a normative explanation for how memory works in neural networks.

---

## 8. **Mathematical Extensions of Hebbian Learning**

An extended form of the Hebbian/Anti-Hebbian learning rule is:

{{< math >}}$$
\Delta W = y_{post} y_{pre}^\top D
$${{< /math >}}

Substituting the respective pre-synaptic and post-synaptic terms gives:

{{< math >}}$$
\Delta W = F^\top (F + W D F) D = F F^\top D - W F F^\top + W F F^\top + W D F F^\top D
$${{< /math >}}

This extension can be linked to a minimization problem, suggesting that **recurrent activity** might be responsible for additional learning effects beyond simple Hebbian processes.

---

## 9. **Conclusion**

Hebbian learning remains a cornerstone of both neuroscience and machine learning. Experimental evidence and mechanistic models have consistently demonstrated its effectiveness in explaining neural phenomena like memory retrieval and pattern completion. Modern research is now focused on increasing biological plausibility by incorporating inhibitory neurons, although challenges remain in explaining these results through normative models.

Moving forward, Hebbian learning may continue to bridge the gap between brain-inspired computing and machine learning, offering new ways to understand intelligence, whether biological or artificial.

---

**References:**
- Pereira-Obilinovic et al., 2023
- Krotov and Hopfield, 2021
- Pehlevan and Chklovskii, 2015
- Mongillo et al., 2018
- Gong et al., 2024
- Hennequin et al., 2017
