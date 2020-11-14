# TensorFlow implementation of Total Deep Variation

This repo provides an unofficial implementation of the Total Deep Variation paper by Kolber et al. CVPR 2020.
If you use this network, please cite their work appropriately.

The goal of this implementation in TensorFlow is to be easy to read and to adapt:

- all the code is in one file
- defaults are those from the paper
- there is no other imports than from TensorFlow

## Remarks

For now I only implemented the regularizer and its proximity operator.
I definitely want to implement the unrolled Forward-Backward, the training, and its subsequent experiments (I particularly love the eigenvectors one).
However, since I want to keep this repo in the spirit of a standalone implementation, I will do so in a dedicated experiments directory with Jupyter Notebooks (even if they are not the most flexible format).
