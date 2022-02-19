# Topic Modelling on short-text data 

Extracting topics from a corpus of short-text data (e.g., survey answers, news titles, online chat records, blog comments) is considered as a difficult task for classical topic modelling approaches such as **LDA (Latent Dirichlet Allocation)**. This is because short-text is sparse data, and LDA's original MAP estimation algorithm works poorly on sparse data.

In recent 2020 paper, authors proposed a novel MAP estimation algorithm, ***called BOPE***, which uses Bernoulli randomness for Online Maximum a Posteriori Estimation:
- BOPE solves non-convex MAP problem via Bernoulli sampling and stochastic bounds. It is stochastic in nature and converges to a stationary point of the MAP problem at a rate of O(1/T) which is the state-of-the-art convergence rate, where T denotes the number of iterations.
- In particular, it is proven that Bernoulli randomness in BOPE plays the regularization role which reduces severe overfitting for probabilistic models in ill-posed cases such as short-text.

Taking these advantages into account, I used BOPE to solve MAP inference in LDA and assess its effectiveness on short-text data.

-------
The model is built from scratch on Python using libraries -- numpy, scipy, numba, pandas, matplotlib, NLTK.

Project directories:
```
├── LDA
│   ├── common
│   ├── evaluation
│   ├── input-data
│   ├── model
│   ├── output-data
├── preprocessing 
├── research_papers
```

------------------------------------------------------------------------
TABLE OF CONTENTS


A. LEARNING 

   1. SETTINGS FILE

   2. DATA FILE FORMAT

B. MEASURE

C. PRINTING TOPICS


------------------------------------------------------------------------
A. LEARNING 

The model class is implemented in `./LDA/model/LDA_BOPE.py` and it is run via `./LDA/model/run_model.py`. \
Before running the model, input data should be generated out of raw csv file via `./preprocessing/generate_input_data.py`. During preparation of the input data, each document inside raw csv file is succinctly represented as a sparse vector of word counts. The input data is a file where each line is of the form:

     [M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]

where [M] is the number of unique terms in the document, and the [count] associated with each term is how many times that term appeared in the document.  Note that [term_1] is an integer which indexes the term; it is not a string.

------------------------------------------------------------------------

B. MEASURE

Perplexity is a popular measure to see predictiveness and generalization of a topic model.

In order to compute perplexity of the model, testing data is needed. Each document in the testing data is randomly divided into two disjoint part w_obs and w_ho with the ratio 80:20
They are stored in [input-data folder] with corresponding file name is of the form:

data_test_part_1.txt\
data_test_part_2.txt

------------------------------------------------------------------------

C. PRINTING TOPICS

The Python script topics.py lets you print out the top N
words from each topic in a .topic file.  Usage is:

     python topics.py [beta file] [vocab file] [n words] [result file]














&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
&nbsp;

#### Performance Measures:
- Log Predictive Probability (LPP):

    Predictive Probability shows the predictiveness and generalization of a model M on new data. We follow the procedure in [this paper](https://arxiv.org/pdf/1206.6425.pdf) to compute this measurement. 

- Normalised Pointwise Mutual Information (NPMI): 

    NPMI measurements helps us to see the coherence or seman- tic quality of individual topics. According to [this paper](https://www.ams.org/journals/tran/1943-054-03/S0002-9947-1943-0009263-7/S0002-9947-1943-0009263-7.pdf), NPMI agrees well with human evaluation on interpretability of topic models.

