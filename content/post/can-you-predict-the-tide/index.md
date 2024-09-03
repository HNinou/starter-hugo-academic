---
title: Can you predict the tide ?
date: 2023-08-31T20:41:40.089Z
draft: false
featured: false
image:
  filename: featured.png
  focal_point: Smart
  preview_only: false
---
# Can you predict the tide?

This project is part of the Information and Complexity course, and aims to answer the 'Can you predict the tide?' challenge from the FLUMINANCE team at Inria. The goal is to predict the excess of sea level using past pressure field measurements. In this report, we outline our approach which consists of two parts: 1) Analyzing the data set and 2) Implementing a relevant 'Encoder-Decoder' model for the prediction task

{{< math >}}
$$
\gamma*{n} = \frac{ \left | \left (\mathbf x*{n} - \mathbf x*{n-1} \right )^T \left [\nabla F (\mathbf x*{n}) - \nabla F (\mathbf x*{n-1}) \right ] \right |}{\left |\nabla F(\mathbf{x}*{n}) - \nabla F(\mathbf{x}_{n-1}) \right |^2}
$$
{{< /math >}}

## Analysis of data set

### Heatmaps

To choose the most appropriate algorithms to solve the problem, we first analyzed the dataset. We did a linear regression of the sea level at a given time based on each pressure field value at that time. We then centralized and reduced the pressure field values before performing the regression. The heatmap in Figure 1 shows the correlation between the pressure field and sea level for city 1 (left) and city 2 (right). We can already see that city 1 is located in the South-East quadrant of the heatmap, while city 2 is in the North-West.

![](heatmap1_p.png)

<img src="heatmap1_p.png" width = 0.49\textwidth"> <img src="heatmap2_p.png" width = 0.49\textwidth">

<div>Figure 1: Heatmap of correlation between the pressure field and sea level for city 1 (left) and city 2 (right)</div>

We can also look at the spatial derivatives of the pressure field, or horizontal wind (derivative along x-axis) and vertical wind (derivative along y-axis). The results are presented in Figures 2 and 3. We observe that the horizontal wind seems to contain more information than the vertical wind about sea level rise (the heatmaps for the vertical wind have many low coefficients).

<img src="heatmap1_wh.png" width = 0.49\textwidth"> <img src="heatmap2_wh.png" width = 0.49\textwidth">

<div>Figure 2: Heatmap of correlation between the horizontal wind and sea level for city 1 (left) and city 2 (right)</div>

<img src="heatmap1_wv.png" width = 0.49\textwidth"> <img src="heatmap2_wv.png" width = 0.49\textwidth">

<div>Figure 3: Heatmap of correlation between the vertical wind and sea level for city 1 (left) and city 2 (right)</div>

### Autocorrelation in time

We also analyzed the autocorrelation of the sea level over time. The heatmaps in Figure 4 show the autocorrelation for cities 1 and 2. One limitation is that the time between two consecutive points is not regular, with about two-thirds being evaluated at the same moment as the previous point. However, we can still see that city 2 has a much higher autocorrelation than city 1.

<img src="autocorr1.png" width = 0.49\textwidth"> <img src="autocorr2.png" width = 0.49\textwidth">

<div>Figure 4: Autocorrelation of the sea level over time for city 1 (left) and city 2 (right)</div>

## Implementation of an 'Encoder-Decoder' model

We implemented a 'Encoder-Decoder' model with recurrent cells that take in a vector from a time series and output an encoded version of the system state. The 'Encoder' then feeds this encoded state to the 'Decoder' which takes in the surplus and outputs the surplus at t+1. We implemented this in Python using the Pytorch library.

### Choice of recurrent cells

For the recurrent cells, we can choose different options. The basic cell is a simple matrix multiplication followed by a non-linear function, but this often leads to problems with learning (explosion or extinction of gradient). To address this, Long Short-Term Memory (LSTM) cells have been developed, and Gated Recurrent Units (GRU) cells are an alternative with fewer parameters and similar performance.

### Regularization using the Adam method

We used the Adam method for regularization with a L2 penalty to perform a gradient descent.

## Segmentation of data set in training-set and testing-set

We randomly split the data set into a training set (90%) and a testing set (10%). However, we noticed that the scores on the testing set were very close, sometimes even identical to those on the training set. This was because some points in the time series were very close or even overlapping, leading to overfitting of the training set onto the testing set. To address this, we split the data into blocks of correlated points in time (see figure 1).

## Naive implementation without using pressure fields

We first tried implementing the network as presented in Figure 5, where only the time series of the surplus was used and the pressure fields were ignored. We optimized several parameters such as number of layers per cell, size of 'hidden vector', or dropout rate to minimize the score. However, we found that with two layers, a 'hidden vector' size of 25, and a dropout rate of 20%, we could reach a minimum score of 0.63.

<img src="losses_no_slp.png" width = 0.6\linewidth">

<div>Figure 5: Scores for the training set and test set as a function of the number of epochs in the case of a naive implementation without using pressure fields. The minimum is reached after 27 epochs</div>

## Naive implementation with pressure fields and wind

We also tried implementing the network as presented in Figure 6, where a large vector of the surplus was added to the pressure field and wind at the time point closest in time (see figure 7). This resulted in a very high dimensional input. We found that with four layers, a 'hidden vector' size of 200, and a dropout rate of 50%, we could reach a minimum score of 0.58, which is an improvement of 5%. However, this comes at the cost of multiplying the number of parameters by $25000$.

<img src="encoder_decoder_plus.png" width = 1.3\textwidth">

<div>Figure 6: Implementation where a large vector containing the surplus as well as the pressure field and wind at the time point closest in time is used as input. The resulting vector has 5045 parameters</div>

<img src="losses_whole.png" width = 0.6\linewidth">

<div>Figure 7: Scores for the training set and test set as a function of the number of epochs in the case of a naive implementation with pressure fields and wind. The minimum is reached after 27 epochs</div>

## Implementation with dimensionality reduction of pressure field and wind

We propose a less naive implementation of this network, which takes advantage of the data analysis we have done. Instead of giving the entire pressure field and wind as input to the network, we perform a scalar product between these fields and their heatmaps. This reduces the size of the input vector from 5045 to just 2 (one per city). The size of the input vector is therefore only 8 rather than 5045. We also found that adding a dropout did not improve the performance of the network. With four layers and a 'hidden vector' size of 25, we obtain a score of 0.5, which is an improvement of 8% compared to the naive method including pressure fields and wind in their raw form. The proposed method is more performant and saves on resources as it requires 5000 times fewer parameters.

We also note that we only use one-fifth of the available pressure fields. A possible extension would be to consider not just the most recent field, but the four preceding ones as well. This might bring additional information about the correlation between the surplus and these fields at previous time points. However, this would require recalculating heatmaps that give the correlation between the surplus at t and the pressure field at times t-1, t-2, t-3, and t-4.