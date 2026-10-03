# Capstone Project – Black-Box Optimisation (BBO)

## Section 1: Project Overview

This is a Capstone project part of Imperial College Professional Certificate in Machine Learning and Artificial Intelligence course. The project focuses on black-box optimisation (BBO), where the objective is to identify the input that maximises each of eight unknown functions. Unlike conventional optimisation problems, the mathematical form and gradients of these functions are unavailable; the only information accessible is the scalar output returned after querying a chosen input.

The aim is for each function (ranging from 2 to 8 input dimensions, with all variables constrained to the range [0,1]), to identify the input that produces the highest output within a defined number of evaluation opportunities.

**Relevance**

Many real-world machine learning problems share this structure. Examples include hyperparameter optimisation, experimental design and drug discovery, where objective functions are expensive to evaluate and cannot be expressed analytically. As a result, every evaluation must be used efficiently.

**Overall Approach**

The project employs Bayesian Optimisation with a Gaussian Process (GP) surrogate model. After each evaluation, the GP is updated to provide both a predicted objective value and an estimate of uncertainty across the search space. An Upper Confidence Bound (UCB) acquisition function then selects the next evaluation point by balancing:

- **Exploitation** – sampling regions predicted to have high objective values.
- **Exploration** – sampling uncertain regions that may contain better solutions.

This strategy aims to maximise information gained from each expensive query.

---

## Section 2: Inputs and Outputs

**Inputs**

Each function accepts a real-valued input vector `x`, with every component constrained to the interval [0,1]. The dimensionality differs across functions:

| Function | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Dimensions | 2 | 2 | 3 | 4 | 4 | 5 | 6 | 8 |

**Outputs**

Each query returns a single scalar value `y`. Output magnitudes vary substantially between functions, ranging from values close to zero (Function 1) to several thousand (Function 5).

**Example**

```
Function 1:
Input:  x = [0.3, 0.8]
Output: y = 1.3 × 10⁻⁷⁹
```

For each function, the initial observations are stored as paired NumPy files:
- `initial_inputs.npy`
- `initial_outputs.npy`

---

## Section 3: Challenge Objectives

The objective is to maximise each unknown function while operating under several constraints:

- **Limited number of queries** – Only one new query per function is permitted on each round.
- **Unknown objective** – No analytical expression or gradient information is available; the function must be inferred entirely from the data.
- **Bounded search space** – Every input dimension is restricted to the interval [0,1].
- **Increasing dimensionality** – Functions range from 2 to 8 dimensions, limiting the practicality of exhaustive search methods.
- **Unknown noise characteristics** – Since observation noise is not known in advance, fitting reliable surrogate model hyperparameters requires careful validation.

---

## Section 4: Technical Approach / Development Record

### Week 1

Implemented a standard Bayesian Optimisation pipeline using a Gaussian Process with an RBF kernel and a UCB acquisition function.

Candidate points were generated using:
- Grid search for two-dimensional functions.
- Random sampling for higher-dimensional functions.

Function 1 proved to be an exception because its extremely small output values caused unstable GP behaviour. Instead of relying on unreliable model predictions, the next query was selected manually from an unexplored region of the search space.

### Week 2

- Constrained the allowable range of the length-scale parameter.
- Fixed random seeds to improve reproducibility.
- Ensured each newly observed data point was incorporated into the training set before refitting the Gaussian Process.

### Week 3

Introduced a structured exploration to compare different Gaussian Process configurations.

The evaluation involved combinations of:
- Kernel type
- Length-scale bounds
- UCB exploration parameter (β)

Configurations were ranked using log marginal likelihood, allowing kernel selection to be guided by empirical evidence rather than defaults.
