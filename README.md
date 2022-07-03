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


### Days 4 and 5

Google Cloud chatbot codelab.

    https://codelabs.developers.google.com/codelabs/chatbots-dialogflow-hangouts#0

which is much the same as this but with a (very slightly) different flavour:

    https://developers.google.com/learn/pathways/chatbots-dialogflow

Ideally, continue on with this one:

    https://developers.google.com/learn/pathways/custom-responsive-chatbots

In particular, this give you some experience with pre-packaged
services such as Google's.  (Amazon, Apple, Microsoft etc. all have
such things in one for or another.)

Ideally, go beyond what the codelab proposes (you have a day and a
half to reflect on what that means for you and your group) so that you
learn something specific that interests you.


## Auto-évaluation

To do no later than Tuesday evening:

https://forms.gle/ZGUYoxa1v5mfHBfB8

also

https://forms.gle/gCdoi9BB6VaLusGD8


## Presentations

After lunch on Tuesday, each group will do a 5-10 minute presentation
of their work.  Knowing that you've all worked on roughly the same
material, you'll want to think of how to present your work in a way
that doesn't look exactly like every one else's work.  For example,
you probably _don't_ want to summarise the Google presentation, as
everyone will have seen that.  Talk rather about your synthesis of
that information and what you've learned beyond what the nice lady in
the video says.

Remember, also, that what feels hard for you at the moment you are
presenting the material is not _necessarily_ the most important part
of what you have to present.

Remember to think through how you want to structure your presentation.
What should the listener take away at the end?

It might help to imagine that your boss asked you to learn enough
about chatbots to give a presentation to a technically competent
client, and that you want to show the client that you are competent to
work on whatever problems they have.

You should plan to do a live demo of your work, but that should be the
icing on the cake, not the cake.


## AMA

Question and answer time.


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
