RNN motivation:
- in CNN, each inference task is independent
- the input/output size is fixed
- however, **sequential data processing & generating** tasks such as sentiment extraction from a sentence need the ability to consider the whole sentence as a whole instead of simply processing single words
- for generative tasks, the previous output should act as the input for the next output, usually processed through the same network

properties expected for RNN
- capture the dependence between inputs
  - not able to simple send cumulative inputs to the network (violating the fact that the function input size is fixed)
  - the hidden states are feature maps of their respective inputs, so just passing the previous hidden state to the hidden state of the next network
  - in such means, $h_2$ can contain both features of $x_1$ and $x_2$
  - in a single network, achieve this by feed its own output back into itself for the next step
- account for various number of inputs
- execute the same function at different time step

the structure of RNN: $x_t\rightarrow h_t\rightarrow y_t$
- fully connnected layers always act as a predictor throughout all kinds of NNs
- in different kinds of NNs, the principle differences lie in the **feature extractor** (CNN-convolution layers, RNN-RNN layers)
- hidden state processing: $H_t=\phi(X_t W_{xh}+H_{t-1}W_{hh}+b_h)$
  - $W_{xh}, W_{hh}, b_h$ don't vary as time steps increase
  - $X_t W_{xh}+H_{t-1}W_{hh}$ can be **concatenated** as a single matrix multiplication

many-to-one RNN
- generate single output only when the input sequence ends
- the system **ignores or dropouts the output** until the loop finishes

one-to-many RNN: e.g. image captioning
- first using CNN to extract the features from the image

many-to-many RNN 1: position of state tagging

many-to-many RNN 2: encoder-decoder (many-to-one + one-to-many)

RNN training algorithm: backpropagation through time
- problem: mathematical instability (vanishing / exploding)
- solve exploding problem: gradient clipping (set a top value)
- solve vanishing problem: use LSTMs / GRUs

LSTMs: introducing memory cells
- structure of memory cells: input gates/forget gates/output gates
- each gate's output is generated through an FC layer
- memory cell (long-term memory) is invisible outside, while hidden state remains visible
- hidden state is determined according to the updated memory cell and output layer (indicating the important parts of the long-term memory)

autoencoders (AE): focus on optimizing the **feature extracting efficiency** rather than classification / regression efficiency

deep reinforcement learning (DRL): in tasks such as winning a match, the reward of a intermediate action remains opaque until the final results appear
