by 2154286 郑伟丞 

### Q1 : $inf\{sin(n):n>1\}=？$，write down the answer and prove it.

answer:  $inf\{sin(n):n>1\}=-1$

$inf\{sin(x):x \in \mathbb{R}\}=-1$

So the problem is equivalent to proving $\forall \epsilon >0 ,\exist k,n \in \mathbb{N}^*,|2k\pi-\frac{\pi}{2}-n|<\epsilon$

Let $\{x_t\}=-\frac{\pi}{2}+t,t \in \mathbb{N}$

$\exists p_t \in \mathbb{Z},\exists r_t\in[0,2\pi),-\frac{\pi}{2}+t=x_t=2p_t\pi+r_t$

Obviously, for each $i \neq j$, $2\pi=\frac{(i-j)-(r_i-r_j)}{p_i-p_j}$  and $r_i \neq r_j$

Split $[0,2\pi) $ into $\lfloor \frac{2\pi}{2\epsilon}\rfloor+1$ pieces: $[0,\delta),[\delta,2\delta),...,[θ-\delta,θ),\delta=\frac{2\pi}{\lfloor \frac{2\pi}{2\epsilon}\rfloor+1}$

Since $\{r_t\} $ is infinite, there exists $r_i,r_j$ in the same one of the above intervals,and thus

$0<r_i-r_j<\delta<2\epsilon$

Since $x_i-x_j=i-j=2(p_i-p_j)\pi+r_i-r_j$

Then $0<(i-j)+2(p_j-p_i)\pi=r_i-r_j<2\epsilon$

Obviously, $\forall \nu \in \mathbb{R},\exists \mu \in \mathbb{Z},|\mu(r_i-r_j)-\nu| \leq \frac{1}{2}(r_i-r_j)<\epsilon$

Then have $\nu = \frac{\pi}{2}+2w\pi,w\in\mathbb{Z} $ and $r_i-r_j=(i-j)+2(p_j-p_i)\pi$,

it's satisfied that $|\mu(i-j)+2[\mu(p_j-p_i)-w]\pi-\frac{\pi}{2}|<\epsilon$

Then have $k=\mu(p_j-p_i)-w,n=-\mu(i-j)$

when $i>j$,$\exists i,j \in \mathbb{N}^*,\exists w\in \mathbb{Z},\exists \mu \in \mathbb{Z}^-$ to satisfy $k,n \in \mathbb{N}^*$

when $i>j$,$\exists i,j \in \mathbb{N}^*,\exists w\in \mathbb{Z},\exists \mu \in \mathbb{N}^*$ to satisfy $k,n \in \mathbb{N}^*$

$\forall \epsilon >0,\exist k,n\in\mathbb{N}^*,|2k\pi-\frac{\pi}{2}-n|<\epsilon$

Above all, the original statement is proved.

### Q2 :Prove Cauchy-Schwartz inequality $\bm{x^Ty\le\|x\|_2\|y\|_2}$ in two ways.

Method1:

Assume that vector $y$ is a non-zero vector (if it is a zero vector, the equation holds)

Let $\bm{z=x-\frac {x*y}{\|y\|^2_2}*y}$

Then $\bm{z*z=\|x\|^2_2-\frac {2x*y}{\|y\|_2^2}x*y+\left(\frac {x*y}{\|y\|^2_2}\right)^2*\|y\|_2^2=\|z\|^2_2} \geq 0$

$\bm{\therefore \|x\|_2^2\|y\|_2^2-x(x*y)^2+(x*y)^2} \geq 0$

$\bm{\therefore \|x\|_2^2\|y\|_2^2\geq(x*y)^2}$

So $\bm{x^Ty\leq \|x\|^2_2\|y\|^2_2}$

Method2:

Let $\bm{z=\frac {x}{\|x\|_2^2}-\frac {y}{\|y\|_2^2}}$

Then $\bm{z*z=\frac {\|x\|_2^2}{\|x\|_2^2}-\frac {2x*y}{\|x\|_2^2\|y\|_2^2}+\frac {\|y\|_2^2}{\|y\|_2^2}}$

$\bm{\therefore 2-\frac {2x*y}{\|x\|_2^2\|y\|_2^2}}\geq0$

So $\bm{x^Ty\leq \|x\|_2^2\|y\|_2^2}$

### Q3 : Let $\bf{A}$ be a matrix of size $m\times n$. Denote the range space of $\bf{A}$ as $R(\bf{A})$ and the null space of $\bf{A}$ as $N(\bf{A})$,repectively. Prove $R(\bf{A})=\it{N}^\bot(\bf{A}^{\it{}T})$

$R(\bm{A})=\{\bm{y| y=Ax,x}\in \mathbb{R}^n\}$

$N(\bm{A}^T)=\{\bm{x}|\bm{A}^T=0,\bm{x}\in \mathbb{R}^m\}$

$N^\bot(\bm{A}^T)=\{\bm{y}|\langle \bm{x ,y}\rangle =0,\forall \bm{x}\in N(\bm{A}^T)\}$

Let $\bm{A}=(\bm{a_1,a_2,...a_n})$ , where $\bm{a_1,a_2,...,a_n} \in \mathbb{R}^m$

$\forall \bm{\mu} \in N(\bm{A}^T)$

$\bm{a_1^T\mu=a_2^T\mu=...=a^T_n\mu=0}$

So  $x_1\bm{a_1^T\mu}=x_2\bm{a_2^T\mu}=...=x_n\bm{a^T_n\mu}=0$ where $x_1,x_2,...,x_n \in \mathbb{R}$

So $(x_1\bm{a_1^T}+x_2\bm{a_2^T}+...+x_n\bm{a_n^T})\bm{\mu}=0$  where $x_1,x_2,...,x_n \in \mathbb{R}$

Let $\bm{x}=(x_1,x_2,...,x_n)^T$

So $x_1\bm{a_1^T}+x_2\bm{a_2^T}+...+x_n\bm{a_n^T}=\bm{x^TA^T}$

$\bm{x^TA^T\mu}=(\bm{Ax)^T\mu}=\langle\bm{Ax,\mu}\rangle=0$

So $N^\bot(\bm{A^T}) = \{\bm{y|y=Ax,x\in \mathbb{R}^n}\}=R(\bm{A})$

### Q4 : For any two matrices, prove $trace(\bm{AB})=trace(\bm{BA})$

$(\bm{AB})_{ij}=\sum_k \bm{A}_{ik}\bm{B}_{kj}$

$trace(\bm{AB})=\sum_n\sum_k\bm{A}_{nk}\bm{B}_{kn}$

$(\bm{BA})_{ij}=\sum_k \bm{B}_{ik}\bm{A}_{kj}$

$trace(\bm{BA})
$

$=\sum_n\sum_k \bm{B}*{nk}\bm{A}*{kn}$

$ =\sum_n\sum_k \bm{A}_{kn}\bm{B}_{nk}$

$=\sum_k\sum_n\bm{A}_{nk}\bm{B}_{kn}$  (from n is equivalent to k)

$=\sum_n\sum_k\bm{A}_{nk}\bm{B}_{kn}$

$=trace(\bm{BA})$

### Q5 : Prove $\bm{A} \succeq 0 \Longleftrightarrow \langle\bm{A,B} \rangle$ for all $\bm{B} \succeq 0$

1.prove $\bm{A} \succeq 0 \Longrightarrow \langle\bm{A,B}\rangle $ for all $\bm{B} \succeq 0$

$\because \bm{A} \succeq 0,\bm{B} \succeq0$

$\therefore \bm{A}=\bm{P^T\land P} , \bm{B}=\bm{Q^TDQ}$

where $\bm{P}$ is an orthogonal matrix, $\bm{\land}$ is a diagonal matrix composed of the eigenvalues of $\bm{A}$, and the same applies to $\bm{B}$

$\langle\bm{A,B} \rangle$

$=trace(\bm{AB^T})$

$=trace(\bm{AB})$

$=trace(\bm{P^T\land PQ^TDQ })$

$=trace(\bm{\land PQ^TDQP^T})$  (from $trace(\bm{AB})=trace(\bm{BA})$)

Let $\bm{C}=\bm{PQ^TDQP^T}$

$\bm{C}$ is a matrix with the same eigenvalues as $\bm{B}$, with non-negative diagonal elements, and $\bm{\land}$ is a diagonal matrix with non-negative diagonal elements.

$\therefore trace(\bm{\land C}) \geq 0$

So $\langle \bm{A,B} \rangle \geq 0$

2.prove $\langle\bm{A,B}\rangle$ for all $\bm{B} \succeq0 \Longrightarrow \bm{A}\succeq0$

Suppose $\bm{A}$ is not a positive semidefinite matrix

Then $\exists \bm{x}\in \mathbb{R} $,  $\bm{x^TAx}<0$

Let $\bm{B=xx^T}$

$\forall \bm{y}\in \mathbb{R}$, $\bm{y^TBy=y^Txx^Ty=(x^Ty)^Tx^Ty \geq0}$

$\therefore $  $\bm{B}$ must be a positive semidefinite matrix

$\langle\bm{A,B}\rangle$

$=trace(\bm{AB})=trace(\bm{Axx^T})$

Let $\bm{\mu=Ax}$

$\bm{x^T\mu}=\sum_{i=1}^nx_i\mu_i$

$trace(\bm{\mu x^T})=\sum_{i=1}^n\mu_ix_i$

$\therefore trace(\bm{\mu x^T})=\bm{x^T\mu}$

$\therefore trace(\bm{Axx^T})=\bm{x^TAx}<0$

$\therefore \langle\bm{A,B\rangle}<0$

Contradictory to the premise, the assumption is not true

So $\bm{A}\succeq0$

Above all, the original statement is proved.

### Q6 : Define $f(\bm{x})\triangleq\Vert\mathbf{A}\bm{x}-b\Vert_2^2. $  Compute $∇f(\bm{x})﻿and ∇^2f(\bm{x})$

$f(\bm{x})\triangleq (\bm{Ax-b})^2$

$\frac{\partial f(\bm{x})}{\partial x_k}$                                                            $(1)$

$=\frac {\partial\sum_i(\sum_j A_{ij}x_j-b_i)^2}{\partial x_k}$

$=2\sum_i(\sum_jA_{ij}x_j-b_i)A_{ik}$

Let $\bm{D=Ax-b}$

Then $D_i =\sum_jA_{ij}x_j-b_i$

So $(1)=2\sum_iD_iA_{ik}=2\sum_iA^T_{ki}D_i$   $(2)$

Let $\bm{B}=\bm{A^TD}$

Then $B_k=\sum_iA^T_{ki}D_i$

So $(2)=2B_k$

So $∇f(\bm{x})=2\bm{B}=x\bm{A^T(Ax-b)}$

$\frac {\partial^2 f(\bm{x})}{\partial x_k \partial x_l}=\frac{2\partial \sum_i(\sum_jA_{ij}x_j-b_i)A_{ik}}{\partial x_l}$

$=2\sum_iA_{ik}A_{il}$

$=2\sum_iA^T_{ki}A_{il}$

$=2(A^TA)_{kl}$

So $\nabla ^2 f(\bm{x})=2\bm{A^TA}$

### Q7 : Define $f(\bm{x}) \triangleq \|\bm{A-xx^T}\|_F^2$. Compute $\nabla f(\bm{x})$ and $\nabla ^2f(\bm{x}).$

$f(\bm{x}) \triangleq \sum_i\sum_j(A_{ij}-x_ix_j)^2$

$\frac {\partial f(\bm{x})}{\partial x_k}=2(A_{kk}-x_k^2)(-2x_k)-2\sum_{i\neq k}(A_{ik}-x_ix_k)x_i-2\sum_{j \neq k}(A_{ki}-x_kx_j)x_j$

$=-2(\sum_i(A_{ik}-x_ix_k)x_i+\sum_j(A_{kj}-x_kx_j)x_j)$     $(1)$

Let $\bm{D=A-xx^T}$

Then $D_{ij}=A_{ij}-x_ix_j$

So $(1)=-2[(D^Tx)_k+(Dx)_k]$

So $\nabla f(\bm{x})=-2(\bm{D^Tx+Dx})=-2[(\bm{A-xx^T})^T+(\bm{A-xx^T})]\bm{x}$

$$
\frac{\partial ^2f(\bm{x})}{\partial x_k \partial x_l}=\frac{-2\partial (\sum_i(A_{ik}-x_ix_k)x_i+\sum_j(A_{kj}-x_kx_j)x_j}{\partial x_l}=\begin{cases}  
4\sum_ix_i^2-4A_{kk}+8x_k^2 & k=l \\
8x_kx_l - 2A_{lk}-2A_{kl} & k \neq l \\
\end{cases}
$$

So $\nabla ^2f(\bm{x})=8xx^T+4x^TxI-2A-2A^T$

### Q8 : For the logistic regression example in lecture notes, compute $ \nabla E(\bm{x})$

$E(\bm{w})=-\sum_{n=1}^N\{t_nlny_n+(1-t_n)ln(1-y_n)\}$

$y_n=\sigma(\bm{w^T\phi_n})=\frac{1}{1+e^{-\bm{w}^T\phi_n}}$

$\nabla_{(\bm{w^T\phi_n})}E(\bm{w})=\nabla_{(\bm{w^T\phi_n})}y_n\nabla E(\bm{w})=y_n(1-y_n)(\frac {1-t_n}{1-y_n}-\frac{t_n}{y_n})=y_n-t_n $

$\bm{y}=(y_1,y_2,...,y_N)^T$

$\bm{t}=(t_1,t_2,...,t_N)^T$

$\bm{\phi}=(\phi_1^T,\phi_2^T,...,\phi_N^T)^T$

So $\nabla E(\bm{w})=\nabla_{\bm{w}}(\bm{\phi w})\nabla_{\bm{(\phi w)}}E(\bm{w})=\bm{\phi^T(y-t)}$

### 

### Q9 : Define $f(\bm{x}) \triangleq log\sum_{k=1}^{n}e^{x_k}.$ Prove $\nabla ^2f(\bm{x})\succeq 0$

$\frac{\partial f(\bm{x})}{\partial x_i}=\frac{e^{x_i}}{\sum_{k=1}^{n}e^{x_k}}$

$$
\frac{\partial ^2f(\bm{x})}{\partial x_i \partial x_j}=\frac{\partial (\frac{e^{x_i}}{\sum_{k=1}^ne^{x_k}})}{\partial x_j}=\begin{cases}  
\frac{e^{x_i}(\sum_{k=1}^ne^{x_k}-e^{x_i})}{(\sum_{k=1}^ne^{x_k})^2} & i=j \\
\frac {-e^{x_i+x_j}}{(\sum_{k=1}^ne^{x_k})^2}& i \neq j \\
\end{cases}
$$

$\forall \bm{x} \in \mathbb{R}$

$\bm{x^T}\nabla ^2f(\bm{x})\bm{x}=\sum_i\sum_jA_{ij}x_ix_j=\sum_i\frac{e^{x_i}(\sum_{k=1}^ne^{x_k}-e^{x_i})x_i^2}{(\sum_{k=1}^ne^{x_k})^2}-\sum_{i \neq j}\frac{e^{x_i+x_j}x_ix_j}{(\sum_{k=1}^ne^{x_k})^2}$

$=\frac{(\sum_{k=1}^ne^{x_k}x_k^2)(\sum_{k=1}^ne^{x_k})-(\sum_{k=1}^ne^{x_k}x_k)^2}{(\sum_{k=1}^ne^{x_k})^2}$

From Cauchy's inequality we get

$(\sum_{k=1}^ne^{x_k}x_k)^2 \leq (\sum_{k=1}^ne^{x_k}x_k^2)(\sum_{k=1}^ne^{x_k}) $

So $\bm{x^T}\nabla ^2f(\bm{x})\bm{x}  \geq 0$

So $\nabla ^2f(\bm{x})\succeq 0$

### Q10 : Find at least one example in either of the following two fields that can be formulated as an optimization problem and show how to formulate it

### 1.EDA software

### 2.cluster scheduling for data centers

##### 1.EDA software

Optimization problem statement:

A common optimization problem is layout optimization.
The goal of layout optimization is to arrange the location and connections of circuit components within a limited chip area to achieve optimal performance, power consumption, and reliability. Layout optimization can be expressed as a combinatorial optimization problem, that is, under a given set of components and area constraints, find a combination of component locations and connection methods that makes a certain objective function (such as delay, power consumption, performance, etc.) achieve Minimum or maximum value.
(Of course, given the performance or power consumption requirements, the same is true for minimizing the chip area. The optimization variable is still the layout of the original circuit components)

Suppose you want to design a digital circuit that contains several logic gates and flip-flops. You need to arrange the location and connection of these components within a rectangular chip area. The optimization goal is to achieve the best performance of the chip, that is, to minimize the delay, while also considering factors such as power consumption and area. The optimization variables are the coordinates and orientation of each component, and the path of each connection.

$min_{x,y,θ,r}  $   $f(\bm{x,y,θ,r})$

$s.t.$               

$g(\bm{x,y,θ,r}) \leq 0 $

$h(\bm{x,y,θ,r})=0$

Among them, $\bm{x,y}$are the coordinates of the component, $\bm{θ}$ is the direction of the component, and $\bm{r}$ is the path of the connection. $f(\bm{x,y,θ,r})$ is the objective function, which represents the performance indicators of the chip, such as delay, power consumption, etc. $g(\bm{x,y,θ,r})$ is an inequality constraint, which represents the area limit of the chip, the connection length limit, etc. $h(\bm{x,y,θ,r})$ is an equality constraint, which represents the logical relationship between components, the wiring topology, etc.

##### 2.cluster scheduling for data centers

Optimization problem statement:

There are multiple tasks that need to be performed in a data center cluster. Each task requires certain computing and storage resources and is executed on a specific server. A scheduling algorithm is designed to maximize data center resource utilization and minimize task completion time. This can be formulated as an optimization problem by defining the assignment and execution order of tasks, where the optimization variables are the assignment and execution order of tasks, and the goal is to maximize resource utilization and minimize the constraints of task completion time.

Consider a data center cluster with 10 servers, each with 8 CPU cores and 16 GB of memory. There are 20 tasks to be performed, each with different computing and storage resource requirements, as well as different server requirements.
(For example, task A requires 2 CPU cores and 4G memory and can only be executed on servers 1, 2, and 3; while task B requires 4 CPU cores and 8G memory and can be executed on any server... )

$min $   $\max_{i} y_i$

$s.t.$     

$\sum_{j=1}^m x_{ij}c_i \leq cy_i, i=1,...,n$

$\sum_{j=1}^mx_{ij}m_j \leq My_i ,i=1,...,n$

$\sum_{i=1}^nx_{ij}=1,j=1,...,m$

$x_{ij} \in \{0,1\}, i=1,...,n,j=1,...,m$

$y_i=[0,1],i=1,...,n$

Among them, $n$is the number of servers, $m$ is the number of tasks, and $x_{ij}$ is a binary variable indicating whether task $j$ is assigned to server $i$. $y_i$ is a continuous variable representing the resource utilization of server $i$. ​ $c_j$ and $m_j$ are the number of CPU cores and memory size required by task $j$ respectively. $c$ and $M$ are the number of CPU cores and memory size of each server respectively.

The objective function is to minimize the maximum resource utilization among all servers, that is, to balance the load of each server. Constraints include:

Each server's CPU and memory resources cannot exceed their total amount multiplied by their resource utilization.

Each task can only be assigned to one server and must meet the constraints of the servers it requires.

Resource utilization must be between 0 and 1.
