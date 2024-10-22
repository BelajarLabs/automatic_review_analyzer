## Introduction

The goal of this project is to design a classifier to use for sentiment analysis of product reviews. 
Our training set consists of reviews written by Amazon customers for various food products. 
The reviews, originally given on a 5 point scale, have been adjusted to a +1 or -1 scale, 
representing a positive or negative review, respectively.

Below are two example entries from our dataset. 
Each entry consists of the review and its label. 
The two reviews were written by different customers describing their experience with a sugar-free candy.

| Review                                                                                                                                                                                                                                            | label |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| *Nasty No flavor. The candy is just red, No flavor. Just plan and chewy. I would never buy them again*                                                                                                                                            | $-1$  |
| *YUMMY! You would never guess that they're sugar-free and it's so great that you can eat them pretty much guilt free! i was so impressed that i've ordered some for myself (w dark chocolate) to take to the office. These are just EXCELLENT!* | $1$   |

In order to automatically analyze reviews, you will need to complete the following tasks:

1. Implement and compare three types of linear classifiers: 
   the **perceptron** algorithm, the **average perceptron** algorithm, and the **Pegasos** algorithm.
2. Use your classifiers on the food review dataset, using some simple text features.
3. Experiment with additional features and explore their impact on classifier performance.

### The Data
The data consists of several reviews, each of which has been labeled with $-1$ or $+1$, 
corresponding to a negative or positive review, respectively. 
The original data has been split into four files:

- `reviews_train.tsv` (4000 examples)
- `reviews_validation.tsv` (500 examples)
- `reviews_test.tsv` (500 examples)

To get a feel for how the data looks, we suggest first opening the files with a text editor, spreadsheet program, 
or other scientific software package.

### Translating reviews to feature vectors
We will convert review texts into feature vectors using a **bag of words** approach.
We start by compiling all the words that appear in a training set of reviews into a **dictionary**, 
thereby producing a list of $d$ unique words.

We can then transform each of the reviews into a feature vector of length $d$ 
by setting the $i^{th}$ coordinate of the feature vector to $1$ 
if the $i^{th}$ word in the dictionary appears in the review, or $0$ otherwise. 
For instance, consider two simple documents "Mary loves apples" and "Red apples". 
In this case, the dictionary is the set $\{ \text {Mary}; \text {loves}; \text {apples}; \text {red}\}$, 
and the documents are represented as $(1; 1; 1; 0)$ and $(0; 0; 1; 1)$.

A bag of words models can be easily expanded to include phrases of length $m$. 
A **unigram** model is the case for which $m=1$. 
In the example, the unigram dictionary would be $(\text {Mary}; \text {loves}; \text {apples}; \text {red})$. 
In the **bigram** case, $m=2$, the dictionary is $(\text {Mary loves}; \text {loves apples}; \text {Red apples})$, and 
representations for each sample are $(1; 1; 0), (0; 0; 1)$. 
In this section, you will only use the unigram word features. 
These functions are implemented in the `bag_of_words` function.


## Hinge Loss
In this project we will be implementing linear classifiers beginning with the Perceptron algorithm. 
We will begin by writing loss function, a hinge-loss function. 
For this function we are given the parameters of the model $\theta$ and $\theta _0$. 
Additionally, we are given a feature matrix in which the rows are feature vectors and the columns are individual features,
and a vector of labels representing the actual sentiment of the corresponding feature vector.

### Hinge Loss on One Data Sample
First, implement the basic hinge loss calculation on a single data-point. 
Instead of the entire feature matrix, we are given one row, representing the feature vector of a single data sample, 
and its label of +1 or -1 representing the ground truth sentiment of the data sample.

### The Complete Hinge Loss
Now it's time to implement the complete hinge loss for a full set of data. 
The input will be a full feature matrix this time, and a vector of corresponding labels. 
The $k^{th}$ row of the feature matrix corresponds to the $k^{th}$ element of the labels vector. 
This function should return the appropriate loss of the classifier on the given dataset.

## Perceptron Algorithm

### Perceptron Single Step Update
Now we will implement the single step update for the perceptron algorithm (implemented with $0-1$ loss). 
You will be given the feature vector as an array of numbers, the current $\theta$ and $\theta_0$ parameters, 
and the correct label of the feature vector. 
The function should return a tuple in which the first element is the correctly updated value of $\theta$ 
and the second element is the correctly updated value of $\theta_0$.

> [!TIP]
> Because of numerical instabilities, it is preferable to identify $0$ with a small range $[-\varepsilon , \varepsilon ]$. 
> That is, when $x$ is a float, "$x=0$" should be checked with $|x| < \varepsilon$.

### Full Perceptron Algorithm
In this step you will implement the full perceptron algorithm. 
You will be given the same feature matrix and labels array as you were given in **The Complete Hinge Loss**. 
You will also be given $T$,
the maximum number of times that you should iterate through the feature matrix before terminating the algorithm. 
Initialize $\theta$ and $\theta_0$ to zero. 
This function should return a tuple in which the first element is the final value of $\theta$ and 
the second element is the value of $\theta_0$.

> [!TIP]
> Make sure you initialize `theta` to a 1D array of shape `(n,)` and **not** a 2D array of shape `(1, n)`.

> [!NOTE]
> Please call `get_order(feature_matrix.shape[0])`, and use the ordering to iterate the feature matrix in each iteration. 
> In practice, people typically just randomly shuffle indices to do stochastic optimization.

### Average Perceptron Algorithm 
The average perceptron will add a modification to the original perceptron algorithm: 
since the basic algorithm continues updating as the algorithm runs, 
nudging parameters in possibly conflicting directions, 
it is better to take an average of those parameters as the final answer. 
Every update of the algorithm is the same as before. 
The returned parameters $\theta$, however, are an average of the $\theta$s across the $nT$ steps:

$$\theta _{final} = \frac{1}{nT}(\theta ^{(1)} + \theta ^{(2)} + ... + \theta ^{(nT)})$$

You will now implement the average perceptron algorithm. 
This function should be constructed similarly to the Full Perceptron Algorithm above, 
except that it should return the average values of $\theta$ and $\theta_0$.

> [!TIP]
> Tracking a moving average through loops is difficult, but tracking a sum through loops is simple.

> [!NOTE]
> Please call `get_order(feature_matrix.shape[0])`, and use the ordering to iterate the feature matrix in each iteration. 
> In practice, people typically just randomly shuffle indices to do stochastic optimization.


## Pegasos Algorithm

Now you will implement the Pegasos algorithm. 
For more information, refer to the original paper at [original paper](https://www.notion.so/Automatic-Review-Analyzer-fa12e75898404964aeca1ad1f41db923?pvs=21).

The following pseudocode describes the Pegasos update rule.

$$
\begin{align*}
&\textmd{Pegasos update rule}\ \left(x^{(i)}, y^{(i)}, \lambda , \eta , \theta \right):\\
&\kern1.5em \textbf{if}\ y^{(i)}(\theta \cdot x^{(i)}) \leq 1 \ \textbf{then}\\
&\kern3em \textbf{update}\ \theta = (1 - \eta \lambda ) \theta + \eta y^{(i)}x^{(i)}\\
&\kern1.5em\textbf{else}:\\
&\kern3em \textbf{update}\ \theta = (1 - \eta \lambda ) \theta
\end{align*}
$$

The $\eta$ parameter is a decaying factor that will decrease over time. 
The $\lambda$ parameter is a regularizing parameter.

In this problem, you will need to adapt this update rule to add a bias term ($\theta_0$) to the hypothesis, 
but take care not to penalize the magnitude of $\theta_0$.

### Pegasos Single Step Update
This function is very similar to [**Perceptron Single Step Update**](#perceptron-single-step-update), 
except that it should utilize the Pegasos parameter update rules instead of those for perceptron. 
The function will also be passed a $\lambda$ and $\eta$ value to use for updates.

The Pegasos algorithm mixes together a few good ideas: 
regularization, hinge loss, sub-gradient updates, and decaying learning rate.

When using a bias, the bias update for a mistake becomes:

$$
\theta _0 = \theta _0 + \eta y^{(i)}
$$

### Full Pegasos Algorithm
The same feature matrix and labels array were given in [**Full Perceptron Algorithm**](#full-perceptron-algorithm). 
Also, the $T$, the maximum number of times to iterate through the feature matrix before terminating the algorithm. 
Initialize $\theta$ and $\theta _0$ to zero. 
For each update, set $\displaystyle \eta = \frac{1}{\sqrt{t}}$ where
$t$ is a counter for the number of updates performed so far (between $1$ and $nT$ inclusive).
This function should return a tuple in which the first element is the final value of $\theta$ and 
the second element is the value of $\theta _0$.