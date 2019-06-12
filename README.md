![](logo.png)

# Yellow submarine - solving Max-Cut problem using Strawberry Fields

This is the code for the algorithm for solving Max-Cut Problem using the Strawberry Fields and QMLT libraries. (TODO: links)


## Introduction

For full description please refer to our paper (TODO) describing the project and associated research (TODO).

The main functionality of the code is to solve the Max-Cut problem given a matrix representing a graph. It consists of two parts:
a) photonic quantum circuit representing graph
b) parametrized variational circuit

The parameters of the part b) are optimized during the training process to provide solution to the problem.


## Getting started

Please install the requirements by running: `pip install -r requirements.txt`. The example of usage for a fully connected graph is shown in the `example.py` file.

