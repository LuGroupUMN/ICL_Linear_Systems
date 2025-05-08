# ICL_Linear_Systems
This project provides code for [In-Context Learning of Linear Systems: 
Generalization theory and Application to Operator Learning]


# Example 1: Linear system with random matrix
Consider the linear $Ax=y$, where $A \in \mathbb{R}^{d \times d}$ is a $d$
 by $d$ matrix, that is defined by $A = QDQ^T$. Here $Q$ is a random random orthonormal matrix obtained from the QR decomposition of a random matrix. 
 
We denote by $U_{d} \left[a,b\right]$ the distribution of $d \times d$ diagonal matrices with whose entries are independently sampled from the uniform distribution $U[a,b]$, that is $D \sim U_{d}\left[a,b\right]$ as $D= \diag(\lambda_1,\dots,\lambda_{d}), ~
\lambda_i \overset{\mathrm{iid}}{\sim} U\left[a,b\right]$. The vector $y$ is sampled from a multivariate normal distribution $\mathcal{N}(0, \Sigma_d(\rho))$, where $\Sigma_d(\rho)$ denotes an equal-correlated covariance matrix, which is defined as $\Sigma_d(\rho) = (1-\rho)I_d + \rho F_d$, with $F_d\in \mathbb{R}^{d\times d}$ being a matrix of all ones.

## 1.1 Test the in-domain generalization

### 1.1.1 Test error vs. Number of Tasks (N)

#### Step 1: Prepare the Data
Navigate to the `RM/test_N` directory. Configuration settings can be adjusted in `config.py`.

```
python prepare_data.py
```


#### Step 2: Train the Model

```
python train.py
```
#### Step 3: Test the Model

```
python test.py
```
