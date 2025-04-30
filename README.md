# IMPLEMENTING HIERARCHICAL REFINEMENT FOR OPTIMAL TRANSPORT
This project aims to implement the hierarchical refinement process using the JAX library as part of the Optimal Transport course at ENSAE for the 2024-2025 academic year.  
The proposed implementation is based on the following research paper : *Hierarchical Refinement: Optimal Transport to Infinity and Beyond*, available at the following link : [https://arxiv.org/pdf/2503.03025](https://arxiv.org/pdf/2503.03025).  
The first implementation of the algorithm is available at the following [repository](https://github.com/raphael-group/HiRef/tree/main)

## Requirements
Nothing more than [jax](https://docs.jax.dev/en/latest/quickstart.html) and [numpy](https://numpy.org/) basically to run the code.

```sh
pip install -r requirements.txt
```

## Documentation
The documentation of the project has been made with [pdoc](https://pdoc.dev/) . You can access it with the following command
```sh
pdoc xxx
```

## Abstract of the problem
We implemented a hierachical refinement algorithm base on the article in the introduction.
This implementation is quite simple in terms of optimization but its goal is to explain the structure of such algorithm.

You may find a notebook where we explain the advantages of this method and some of its subutilities.
