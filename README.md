# DronesRL

# The assignment
Drones deliever packages to clients using Reinforcement Learning methods (deterministic and probabilistic)

# Part 1 - Deterministic Problem

## Introduction

In this exercise, we assume the role of a head of a drone delivery agency. Our main goal is to
deliver the packages to your clients in the shortest time possible. To achieve this most efficiently,
we make use of the search algorithms, with the first task being modeling
the problem precisely. The goal is to deliver to each customer every package on his list in the shortest amount of
steps (or turns) possible.

## Environment

The environment is a rectangular grid - given as a list of lists (exact representation can be found
in the “Input” section). Each point on a grid represents an area. An area can be either passable
or impassable for the delivery drones. Moreover, there are packages lying in different locations
around the grid. The packages can be picked up by drones and delivered to clients.
Clients can move on a pre-determined and known path, and each client has a list of required
packages. In general, the client can request non-existing packages (then the problem is
unsolvable). Moreover, there could be packages that no one needs. One drone can carry up to
two packages at once.
The task is done in timestamps (or turns), so the environment changes only after you apply an
action.

## Input 

![image](https://user-images.githubusercontent.com/80041689/178791974-8b1592d0-6c24-44e7-b4a6-22035fe66587.png)

# Part 2 - Probabilistic Problem

## Introduction

In this part, we continue to lead the drone delivery company, but this time the clients
exhibit a non-deterministic behavior.

## Environment
The environment is similar to the one in Part 1: the same grid, with the same packages and
clients.

Key differences include:

● Clients don’t have a predetermined path, but rather choose their move probabilistically.

● The agent is not required to output a whole plan but rather given a state output an action

● The execution runs for a limited, pre-determined number of turns

● The goal of this part is to collect as many points as possible (more on that in the
“Points” section)

## Clients’ behavior

For each client, you are given two parameters governing its behavior: the starting location and
the probability to move in each direction (up, down, left, right, or stay in place). The clients can
move only to adjacent tiles. If a client is located near the edge of the grid, the probability of
movement beyond the edge is distributed between other directions proportionally. 

For example, if a client is on the upper edge of the grid (thus can’t move up), and has probabilities [0.5, 0.2,
0.05, 0.1, 0.15] – it will move according to probabilities [0, 0.4, 0.1, 0.2, 0.3].

# Part 3 

## Introduction

The exercise continues with the drone delivery company, but this time the drone must navigate
an unknown environment under RL settings. There is no client, only a stationary place to drop
the packages off.

## Environment

Each turn, the environment outputs an observation (obs0) which is passed to the drone agent
who in turn plots an action. The action is passed back to the environment to process and output
a result: new observation and a reward.

In this section we also recieved a ready Trainer class. It’s a simple class that combines both the
environment and the agent.
