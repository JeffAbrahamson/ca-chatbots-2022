Campus Academy

# Chatbots

Note: the course is in French but the notes are in English, because
this is a domain where it's important to get used to finding resources
in English.

Plan to do all exercises in groups of 2-3 people.

We will mostly use Keras, although some examples will use pytorch.
We'll mostly use Google Colab.


## Overview

### Day 1

Morning: introductions, review.

Afternoon: Eliza codelab


### Day 2

Discussion:
* PCA, auto-encoders
  * Loss = L^2_2
  * Orthognonal, linear subspace
AE non-ortho, manifold

Example: eigenfaces

* In one example, 10 component deep AE = 50 component LSA (PCA)
* 2 components = visual

NLP introduction:
* BOW
* Stop words
* TF-IDF
* Word2vec

Compare documents via vectors.  Cosine distance.  Faster in fewer dimensions.

* Semantic hashing
  * "Supermarket search"
  * Reduce search space to a few buckets
  * Deep AE, logistic units in hidden units
  * Hope locality is related to similarity
  * Example: bisection of computer memory

Vectors in $\mathbb{R}^n$ vs $\mathbb{Z}^n$.

Shallow AE
* RBM with contrastive divergence
* not RBM with maximum likelihood
* Note: Hopfield nets

Codelab on document similarity.


### Day 3

Auto-encoders + codelab using images to understand auto-encoders visually.

LSTM auto-encoders + codelab


### Day 4

To be determined based on experience of first three days.

### Day 5

To be determined based on experience of first three days.


## Basics of ML

This part is hopefully review.

* Classification and regression
* Supervised and unsupervised (and reinforcement)
* NLP (as distinct from voice recognition, for example)
* Decomposition (e.g., text -> meaning, speech -> text/meaning)
* BOW, TF-IDF, word2vec, glove (note: we gain much, but we lose sequence information)

Some things to think about together and discuss:

* When training a chatbot, what are we learning?  What are we optimising?
* What is a possible loss function?
* How can we define training accuracy?
* How much data might we need?
* What domain knowledge is needed?


## Eliza

* [About (fr)](https://fr.wikipedia.org/wiki/ELIZA) et [(en)](https://en.wikipedia.org/wiki/ELIZA)
* [wadetb/elisa](https://github.com/wadetb/eliza)
* [dabraude/Pyliza](https://github.com/dabraude/Pyliza)

Discussion: principles, history (this dates from the mid-1960's)

Exercise: to understand how eliza works, make it work in French

Another interesting example from 2000:  [How may I help you?](papers/gorin-2000-how-may-i-help-you.pdf)


## Sequence-to-sequence modeling

Note that the devision of topics here is a bit artificial.  We can
talk about HMM, RNN, LSTM, seq2seq, attention, transformers, etc. as
distinct concepts, but in the real world there's mixing.

### Some articles

It's ok if you have trouble reading scientific papers.  It's hard and
it takes practice.  A couple suggestions can help a lot:

Read the abstract, the introduction, and the conclusion, in that
order.  Then skim the bibliography to get a feel for who is being
cited.  You'll be surprised how quickly you start to recognise authors
and papers, and this gives you a clue who people think is important in
the field.

Write a short summary of the paper, a note to a future you.  This is
really important.

If the paper is important to you, read the rest.


* [An intelligent Chatbot using deep learning with Bidirectional RNN and attention model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7283081/) ([local](papers/dhyani-2020-chatbot.pdf))
* [Seq2Seq AI Chatbot with Attention Mechanism](https://arxiv.org/abs/2006.02767) ([local](papers/sojasingarayar-2020-chatbot.pdf))
* [Conversational Chatbot using Deep Learning Algorithm and Attention Mechanism](https://www.ijser.org/researchpaper/Conversational-Chatbot-using-Deep-Learning-Algorithm-and-Attention-Mechanism.pdf) ([local](papers/prithvi-2020-chatbot.pdf))
* [Performance of Seq2Seq learning Chatbot with Attention layer in Encoder decoder model](https://www.researchgate.net/publication/351837227_Performance_of_Seq2Seq_learning_Chatbot_with_Attention_layer_in_Encoder_decoder_model) ([local](papers/raj-2021-chatbot.pdf))

Some classical foundation articles:

* [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) ([local](papers/NIPS-2017-attention-is-all-you-need-Paper.pdf))


### HMM

* [Chatbot with HMM](https://github.com/timothyouano/Chatbot-with-Hidden-Markov-Model-HMM)
* [Conversation/speech synthesis](https://www.chatbots.org/paper/synthesis_and_evaluation_of_conversational_characteristics_in_hmm-based_spe/)
* [Comparison of rhymer chatbots: HMM, RNN](https://www.derczynski.com/sheffield/papers/archive/innopolis/chernova.pdf) ([local](papers/chernova.pdf))

### LSTM

Some gentle introductory examples:

* [An annotated LSTM implementation](https://nn.labml.ai/lstm/index.html)
* [LSTM example 1](https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/time-series/Ch4-LSTM.html): predicting daily covid-19 cases in South Korea.
* [LSTM example 2](https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/time-series/Ch5-CNN-LSTM.html): an improved version
* [Generative chatbot using LSTM RNNs](https://hub.packtpub.com/build-generative-chatbot-using-recurrent-neural-networks-lstm-rnns/)
* [Gentle Introduction to LSTM Autoencoders](https://machinelearningmastery.com/lstm-autoencoders/)
* [LSTM with Python](https://machinelearningmastery.com/lstms-with-python/) (book, I haven't bought or read it)

And some chatbot examples:
* An [LSTM chatbot using LSTM](https://github.com/ShrishtiHore/Conversational_Chatbot_using_LSTM)
* An [LSTM Attention based generative chatbot](https://github.com/AdroitAnandAI/LSTM-Attention-based-Generative-Chat-bot)

### Attention

* (kaggle) [seq2seq chatbot with attention](https://www.kaggle.com/code/programminghut/seq2seq-chatbot-keras-with-attention/notebook)
* [Build a Chatbot by Seq2Seq and attention in Pytorch V1](https://chatbotslife.com/build-a-chatbot-by-seq2seq-and-attention-in-pytorch-v1-3cb296dd2a41)
* [ChatBot with Attention for beginners](https://www.kaggle.com/code/hijest/chatbot-with-attention-for-beginners-and-others/notebook)
* [LSTM Attention based Generative Chat bot](https://github.com/AdroitAnandAI/LSTM-Attention-based-Generative-Chat-bot)

### Transformers

* [Transformer Chatbot Tutorial with TensorFlow](https://medium.com/tensorflow/a-transformer-chatbot-tutorial-with-tensorflow-2-0-88bf59e66fe2)
