2 周之后进行汇报

小组组成：2-3 for data intelligence, 1-2 for mobility platform

objective: engineer an intelligent mobile platform

problem solving: from scratch

project schedule
- baseline: 15th-16th July, "treasure hunt" robot
  - remote controlled vehicle with camera
  - ability to classify images
- treasure hunt task: 17th July, testing of baseline
- advanced model: 18th-26th July
- final demonstration: 27th July
- showcase: 29th July, show the design to everyone in SWS

system integration: something that AI isn't quite good at

---

**guideline**: get motivated to learn by oneself

teaching (human) vs instructing (computer)

supervised vs unsupervised learning: training data with/without labels(expected output)

reinforcement learning: training for maximizing the reward, still need some rules (physical, or sth specific) given

key components of a ML pipeline
- data for training
- model for transforming the data
- objective function for evaluating the performance of the model
- algorithm for optimizing the model's parameter

data assumption: the inputs are independent

regression evaluation: root mean squared error

classfication evaluation: false positive rate, false negative rate
- confusion matrix
- metrics: precision (positive prediction accuracy), recall (positive recognition rate), F1 score (harmonic mean of precision and recall), accuracy
- single measurement may lead to useless classifier
- multipule categories: **one-vs-rest**, with micro/macro averaging

linear model: $\hat{y}=f(w)$, every independent input **engaged linearly** in the production of $w$ ($w$ can be expressed in a matrix/vector form)
- however, $f$ doesn't need to be a linear function

tensor: collection of matrices

linear regression: the case when $f(w)=w$, $\hat{y}=X\theta$
- loss function: mean squared error
- get the optimal $\theta$ vector: solve directly and analytically, gradient descent

training requirements:
- test data ought to be representative of all aspects of the real world
- no data cleaning, just putting all the features as inputs **without engaging human expertise**
- ignore categorical features, using one-hot code to represent each category
- no normalization (0-minimum and 1-maximum) or standardization (0-mean and 1-deviation)

logistic regression: the case when $f(w)=\dfrac{1}{1+e^{-w}}$
- $\hat{y}$ can be treated as a probability, indicating the classification result
- loss function: cross-entropy loss

polynomial linear regression: introducing nonlinear relationship between $X$ and $y$, but it's still linear in $\theta$
- a proper polynomial degree required; the exceedingly high polynomial degree often lead to overfitting (lose the generalization capability)
- to avoid overfitting: introduce **$\theta$ complexity** to the loss function

validation set: for model selection and hyperparameter (network layer count, learning rate, etc.) tuning

---

naive bayes classifier: predict the probability that a document (statement) belongs to one class
- each document consists of several words $w_1, w_2, \cdots$, so the conditional probability eventually degrades to the word sequence level ($P(y|w_1, w_2, \cdots, w_n)$)
- using Bayes theory, simplifying with the assumption that the appearance of each word is independent (usually not the case actually)
- address the underflow issue caused by large-scale multiplying: using log probabilities
- address the out-of-vocabulary words & unrepresented classes issue: introducing **Add-k Smoothing**

decision tree: use different features sequentially (different features on different layers, probably depending on the upper layer result) to come up with an eventual classification
- can be used for both classification & regression task
- the diverging standard depends on the input data

building a decision tree:
- split the pieces of data into two based on an formerly unused feature on the route, until all the pieces of data come up with the same outcome
- find a better split: choosing the split that minimizing the impurity (**entropy**) of subtrees
- feature: the boundary appears parallel to the axis on the graph
- pros: the tree is easy to build, easy to interpret
- cons: sensitive to slight changes in data, doesn't guarantee the optimal tree

random forest

