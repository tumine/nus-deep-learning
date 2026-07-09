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

linear model: every independent input **engaged linearly** in the production of output (can be expressed in a matrix/vector form)
