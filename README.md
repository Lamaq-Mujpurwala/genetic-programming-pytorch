# This project was the most intresting and challenging rpoject I have worked on so far. From developing a basic GA algorithm in C to implementing it on PyTorch has given me a perfect hands-on experience to learn and interpret complex tasks. 

# 🧬 Evolving Neural Network Crossover Operators using Genetic Programming

> ⚠️ **NOTE: This repository is a Work In Progress.**  
> The codebase is currently under **refactoring**, since the goal of the project was to learn hands on and the implementation has been achieved, work on refactoring is a task I intend to do at some point to make the code more accessible. There is no timeline decided yet,  
> Initial experiments have been conducted **only on the Wine dataset**. Support for additional datasets (e.g., CIFAR-10, Higgs, etc.) will be added in upcoming commits.

---

This project is a **from-scratch implementation** of a research paper on *GP-evolved crossover operators for Genetic Algorithms optimizing Neural Network weights*. It explores how **Genetic Programming (GP)** can evolve **custom crossover operators** to enhance **Genetic Algorithms (GA)** for training neural networks — combining deep learning and evolutionary computation.

> ⚠️ This repository is not a reproduction, but an **independent re-implementation and analysis** of the methodology from the referenced research.

---

## 🚀 Features

- ✅ Complete Genetic Algorithm (GA) to evolve Neural Network weights
- ✅ Fully functional Genetic Programming (GP) engine to evolve crossover operators
- ✅ Support for both **reusable** and **disposable** crossover strategies
- ✅ Implemented with the powerful [`DEAP`](https://github.com/DEAP/deap) library
- ✅ Neural networks built and accelerated using **PyTorch + CUDA**
- ✅ Manual **bloat control**, crossover validation, and evolutionary metrics tracking
- ✅ Experiment 2 replicated: **GA w/o crossover vs GA w/ evolved crossover (on Wine dataset)**

---

## 🧠 What This Project Covers

### 📚 Research Understanding
- Deep dive into the paper’s methodology
- Mathematical insight into crossover fitness and GP evolution logic

### 🧬 Evolutionary Computing
- Mutation, tournament selection, elitism
- GP primitives and terminals customized for NN weight vectors
- Safe and stable crossover operator evaluation

### 🧠 Deep Learning
- PyTorch-based neural network models
- Custom weight manipulation and evaluation
- Training-free evaluation using pre-initialized networks

---

## 🛠️ Technologies Used

| Tool        | Purpose                            |
|-------------|-------------------------------------|
| **Python**  | Core language                       |
| **DEAP**    | Genetic Algorithms & GP framework   |
| **PyTorch** | Neural network evaluation & GPU use |
| **WSL2**    | Linux-based setup for reproducibility |
| **CUDA**    | GPU acceleration                    |

---

## 📊 Experiments Replicated

- ✅ **Experiment 2** from the paper:
  - GA without crossover
  - GA with GP-evolved reusable crossover
  - Results compared on **Wine dataset** *(more datasets coming soon)*

---


## 📁 Repository Structure

```bash
.
├── Reusable_GECO/         # ✅ Final implementation of Experiment 2 (GA w/o crossover vs GA w/ GP crossover)
│   ├── GA_without_crossover.py   # GA implementation
│   ├── GP_for_RGECO.py           # GP implementation evolving crossover operators
│   └── GA_with_GP_crossover.py   # End-to-end execution script
│
├── Datasets/              # 🧰 Dataset utilities (Wine, others in progress)
│
├── DEAP/                  # 🧪 Initial exploratory work and learning experiments with the DEAP library
│
├── Refactored/            # 🔨 In-progress refactor of the entire codebase into cleaner modules
│   └── (To be finalized)
│
├── README.md              # 📘 You're here
└── Research Paper.pdf     # 📄 Reference research paper
```
