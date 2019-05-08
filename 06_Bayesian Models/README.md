
 # 6 ] Bayesian models
 
- ### Kalman Filters
  - Kalman Filters (Linear, Gaussian)
  - Extended Kalman Filter (Locally Linear, Gaussian)
  - Unscented Kalman Filter (Non Linear, Gaussian)
  - Particule Filter (Non linear, Non Gaussian)
- ### Bayes rule, Probability review :heavy_check_mark:

- ### Bayesian Regression/Classification
  - Linear
  - Logistic
  - Poisson
  - Linear Mixed Effect/pooled
  - Hierarchical Regression 



- ### Variationnal Inference and Score estimator
   - Score estimator : indicates how sensitive a likelihood function it to a its parameter theta : Explicitly, the score is the gradient of the log-   likelihood with respect to theta.
   - log derivative trick : http://wilkeraziz.github.io/slides/24-05-2018-uedin-dgm-discrete.pdf :heavy_check_mark:
   - Black-box variationnal inference : https://ermongroup.github.io/cs228-notes/extras/vae/ (applied to VAE) :heavy_check_mark:
   - EM algorithm :heavy_check_mark:
   - ELBO and KL divergence:heavy_check_mark:
   - Variance reduction techniques
   - Differentiation of the likelihood :heavy_check_mark:
   
- ### Gaussian Processes
   - Very simple idea : http://katbailey.github.io/post/gaussian-processes-for-dummies/ :heavy_check_mark:
   - The bible : http://www.gaussianprocess.org/gpml/ 
   - Kernel cookbook : https://www.cs.toronto.edu/~duvenaud/cookbook/ :heavy_check_mark:
   - Multitask Gaussian Process : https://homepages.inf.ed.ac.uk/ckiw/postscript/multitaskGP_v22.pdf :heavy_check_mark:
   
- ### Bayesian Optimisation for training :heavy_check_mark:
  - Very Basic Example : https://scikit-optimize.github.io/notebooks/bayesian-optimization.html
  - [A 3 pages course about acquisition_functions](./ressources/Bayesian_Optimization.pdf)
  - Good example : https://colab.research.google.com/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb#scrollTo=OcCIOa7KQD3w -> remember that the optimized function takes as input only the parameters we want to optimize
  
- ### Bayesian Nonparametric clustering

-### Topic models (e.g Latent Dirichlet allocation)

- ### Naive Bayes
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
  
- ### Linear dynamical systems (e.g., state space models, hidden Markov models)

- ### Bayesian Networks
  - Bayesian Neural Network 
  - Bayesian Convolutional Network
  - Bayesian Belief Network (BBN)
- ### Averaged One-Dependence Estimators (AODE)
- ### Bayesian Data Assimilation
- ### Probit Regression

 
- ## Deep exponential families eg  Deep latent Gaussian models


-collapsed Gibbs sampling 
- CRP clustering 
- Dirichlet process 
- evidence approximation 
- Gibbs sampling as a special case of Metropolis-Hastings 
- hierarchical Dirichlet process 
- IBP linear-Gaussian model 
- importance sampling 
- Jeffreys prior
- particle filter
- reversible jump MCMC

