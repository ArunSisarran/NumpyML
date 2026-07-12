# Neural Network From Scratch: Cat / Not-Cat Classifier

**Goal:** Build a binary image classifier (cat vs. not-cat) using only Python and NumPy — no ML libraries (no scikit-learn, PyTorch, TensorFlow, Keras, etc.). The purpose is deep understanding, not a polished product. Every milestone below is something you should implement and verify yourself before moving to the next.

**Dataset:** Kaggle "Dogs vs. Cats" (or similar cats-vs-dogs dataset).

Since you noted the underlying math is mostly new to you, this plan front-loads a math-foundations phase before you touch backpropagation. Don't skip it — trying to implement backprop without this will turn into cargo-culting formulas you don't understand, which defeats the point of this project.

---

## Phase 0: Math Foundations (do this before writing any network code)

Work through these until you can explain each one in plain language AND do a small example by hand on paper.

- [ ] **Vectors & matrices** — what they represent, how matrix multiplication works mechanically (not just "numpy does it"), and why shapes have to align.
- [ ] **Dot products** — geometric and algebraic meaning; why a neuron's weighted sum is a dot product.
- [ ] **Functions & derivatives** — what a derivative represents (rate of change / slope), how to compute derivatives of simple functions (polynomials, exponentials).
- [ ] **The chain rule** — this is the single most important concept for backprop. Practice on composed functions like f(g(x)) by hand before worrying about neural nets at all.
- [ ] **Partial derivatives & gradients** — what it means to take a derivative with respect to one variable when there are many; what a gradient vector represents (direction of steepest increase).
- [ ] **Sigmoid function** — its formula, its shape, and critically, derive its derivative by hand (this shows up constantly in binary classification).

**Checkpoint:** You should be able to, without looking anything up, explain why gradient descent moves in the *negative* direction of the gradient, and derive the derivative of sigmoid on paper.

---

## Phase 1: Data Pipeline (no ML yet)

- [ ] Load raw cat/dog images from the Kaggle dataset using only basic image libraries (e.g. PIL/Pillow for decoding — NumPy for everything else). No pre-built data loaders from ML frameworks.
- [ ] Decide on and implement a fixed image resizing strategy (e.g. all images to 64x64). Understand *why* neural nets need fixed-size input.
- [ ] Convert images into flattened NumPy arrays. Understand exactly what shape your data is in and why (e.g. images → vectors).
- [ ] Normalize pixel values yourself (figure out why normalization matters for training stability).
- [ ] Relabel the dataset for your binary task: cat = 1, not-cat (dog) = 0.
- [ ] Split data into training and test sets manually (write your own split logic, don't reach for a library helper).

**Checkpoint:** You can load a batch of images and describe the exact shape and meaning of every axis in your resulting NumPy array.

---

## Phase 2: Single Neuron / Logistic Regression (the simplest possible "network")

Before building a multi-layer network, implement logistic regression as a neural network with zero hidden layers. This is the "hello world" of neural nets and contains almost all the core mechanics in miniature.

- [ ] Implement the forward pass: weighted sum of inputs + bias, passed through sigmoid.
- [ ] Implement a loss function (binary cross-entropy) and understand *why* this loss function is used for classification instead of something like mean squared error.
- [ ] Derive (by hand) the gradient of the loss with respect to the weights and bias.
- [ ] Implement the backward pass (computing those gradients in code) from your own derivation — not from a memorized formula.
- [ ] Implement parameter updates via gradient descent.
- [ ] Train this single neuron on your cat/not-cat data and get it running end to end, even if accuracy is mediocre.

**Checkpoint:** You have a working (if weak) classifier and can explain every line of the forward and backward pass in terms of the math from Phase 0.

---

## Phase 3: Generalize to a Multi-Layer Neural Network

- [ ] Design a network architecture on paper first: input layer size, number of hidden layers, number of neurons per hidden layer, output layer. Justify your choices.
- [ ] Implement forward propagation generalized across multiple layers (think about how to structure this with loops or a data structure of layers, not hardcoded steps).
- [ ] Implement additional activation functions (e.g. ReLU or tanh for hidden layers) and understand why hidden layers usually don't use sigmoid.
- [ ] Derive backpropagation for a multi-layer network by hand — this is where the chain rule from Phase 0 gets used repeatedly across layers. Do this derivation before coding it.
- [ ] Implement the generalized backward pass.
- [ ] Implement weight initialization strategies and understand why initializing all weights to zero breaks training.
- [ ] Train the full network and compare performance to your single-neuron baseline from Phase 2.

**Checkpoint:** You can explain, layer by layer, how an error signal at the output propagates backward to update weights in earlier layers.

---

## Phase 4: Making Training Actually Work Well

- [ ] Implement mini-batch gradient descent (vs. full-batch or single-sample) and understand the tradeoffs.
- [ ] Implement a way to track and plot training loss over time (basic NumPy/matplotlib is fine — this isn't an ML library).
- [ ] Diagnose and fix at least one real training problem you encounter (e.g. vanishing gradients, exploding loss, stuck accuracy) — don't just copy a fix, understand why it happened.
- [ ] Implement a learning rate and experiment with how changing it affects training.
- [ ] Add basic regularization (e.g. L2 regularization) and observe its effect on overfitting.

**Checkpoint:** You can look at a loss curve and diagnose what's likely going wrong in training.

---

## Phase 5: Evaluation and Honest Assessment

- [ ] Implement accuracy calculation on your held-out test set manually.
- [ ] Implement a confusion matrix by hand (true positive/negative, false positive/negative) — understand why accuracy alone can be misleading.
- [ ] Test the model on a handful of individual images and inspect predictions manually.
- [ ] Write up (for yourself) what your model's failure cases look like and hypothesize why.

**Checkpoint:** You can state your model's accuracy, its likely failure modes, and explain the confusion matrix without needing to look up the terms.

---

## Phase 6 (Stretch Goals, Optional)

Only attempt these once Phases 0-5 are solid and understood — this is where you start rebuilding "library" conveniences yourself.

- [ ] Implement a simple momentum-based optimizer (understand what problem momentum solves before implementing Adam).
- [ ] Implement dropout from scratch.
- [ ] Implement a basic convolutional layer from scratch (this is a significant jump in complexity — matrix flattening won't cut it for genuinely learning spatial features).
- [ ] Compare your from-scratch model's performance against a simple scikit-learn or PyTorch baseline, purely as a sanity check on your own implementation (not as your project's actual approach).

---

## Ground Rules

- No `sklearn`, `torch`, `tensorflow`, `keras`, or any autodiff/autograd library at any phase before the stretch goals' sanity-check comparison.
- NumPy is for array math (matrix multiplication, element-wise ops) — not for anything that does the "thinking" for you (no built-in neural net layers, optimizers, or loss functions from any library).
- Before implementing any formula, derive it or at least trace through why it's true. If you can't explain a piece of code you wrote, that's a signal to stop and understand it before moving on.
- It's fine to reference explanations of concepts (this plan doesn't forbid learning resources) — the constraint is on *code*, not on understanding.
