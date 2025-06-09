---
layout: appendix
title: Probabilities & Bayes rule (in memo)
description: "A quick and gentle introduction to probability and Bayes rule (in memo)"
---

### Appendix chapter 01: Probabilities & Bayes rule (in memo)

*Author: Michael Franke, adapted to memo by Robert D. Hawkins*

##### The 3-card problem: motivating Bayesian reasoning

Jones is not a magician or a trickster, so you do not have to fear. He just likes probability puzzles. He shows you his deck of three cards. One is blue on both sides. A second is blue on one side and red on the other. A third is red on both sides. Jones shuffles his deck, draws a random card (without looking), selects a random side of it (without looking) and shows it to you. What you see is a blue side. What do you think the probability is that the other side, which you presently do not see, is blue as well?

Many people believe that the chance that the other side of the card is blue is .5; that there is a 50/50 chance of either color on the back. After all, there are six sides in total, half of which are blue, and since you do not know which side is on the back, the odds are equal for blue and red.

This is faulty reasoning. It looks only at the base rate of sides. It fails to take into account the **observation-generating process**, i.e., the way in which Jones manipulates his cards and "generates an observation" for you. For the 3-cards problem, the observation-generating process can be visualized as in Fig. 1.

{% include figure.html 
file="../images/3-card-problem_process.png" 
caption="The observation-generating process for the 3-card problem. Jones selects a random card, then chooses a random side of it." 
number = "1"
width="500px" 
%}

The process tree in Fig. 1 also shows the probabilities with which events happen at each choice point during the observation-generating process. Each card is selected with equal probability $$\frac{1}{3}$$. The probability of showing a blue side is 1 for the blue-blue card, .5 for the blue-red card, and 0 for the red-red card. The leaves of the tree show the probabilities, obtained by multiplying all probabilities along the branches, of all 6 traversal of the tree (including the logically impossible ones, which naturally receive a probability of 0). 

If we combine our knowledge of the observation-generating process in Fig. 1 with the observation that the side shown was blue, we should eliminate the outcomes that are incompatible with it, as shown in Fig. 2. What remains are the probabilities assigned to branches that are compatible with our observation. But they do not sum to 1. If we therefore renormalize (here: division by .5), we end up believing that it is twice as likely for the side which we have not observed to be blue as well. The reason is because the blue-blue card is twice as likely to have generated what we observed than the blue-red card is.

{% include figure.html 
file="../images/3-card-problem_elimination.png" 
caption="The observation-generating process for the 3-card problem after eliminating outcomes incompatible with the observation 'blue'." 
number = "2"
width="500px" 
%}

The latter reasoning is actually an instance of Bayes rule. For our purposes, we can think of Bayes rule as a normatively correct way of forming (subjective) beliefs about which causes have likely generated an observed effect, i.e., a way of reasoning probabilistically and defeasibly about likely explanations for what has happened. In probabilistic pragmatics we will use Bayes rule to capture the listener's attempt of recovering what a speaker may have had in mind when she made a particular utterance. In other words, probabilistic pragmatics treats pragmatic interpretation as probabilistic 
inference to the best explanation of what worldly states of affairs, mental states and contextual factors would have caused the speaker to act in the manner observed.

You should now feel very uncomfortable. Did we not just say that most people fail at probabilistic reasoning tasks like the 3-card problem? (Other prominent examples would be the two-box problem or the [Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem).) Yes, we did. But there is also a marked contrast between probability puzzles and natural language understanding. One reason for why many people fail to show the correct probabilistic reasoning in the 3-card problem is because they neglect the precise nature of the observation-generating process. The process by which Jones selects his cards is not particularly familiar to us. (When is the last time you performed this card selection procedure.) In contrast, every speaker is intimately familiar with the production of utterances to express ideas. It is arguably a hallmark of human language that we experience ourselves in the role of the producer (compare [Hockett's design features of language](https://en.wikipedia.org/wiki/Hockett%27s_design_features), in particular **total feedback**). On this construal, pragmatic interpretation is (in part) a simulation process of what we might have said to express such-and-such in such-and-such contextual conditions.

##### Probability distributions & Bayes rule

If $$X$$ is a (finite) set of mutually discrete outcomes (results of an experiment, utterances, interpretations ... ), a (discrete) probability distribution over $$X$$ is a function $$P \colon X \rightarrow [0;1]$$ such that $$\sum_{x \in X} P(x) = 1$$. For any subset $$ Y \subseteq X$$ we further define $$P(Y) = \sum_{x \in Y} P(x)$$. 

Consider the example of a 2-dimensional probability table. It gives probabilities for meeting a person with a particular hair and eye color. 

|                    | brown            | blue         |    green     |
|:-------------------|:----------------:|:------------:|:------------:|
| **black**\t\t\t |\t   .4\t   |\t .02\t  |\t .01
| **blond**\t\t |\t  .1\t   |\t.3\t  |\t  .1
| **red**            |      .01       |     .01      |  .05

Formally, we have $$X = H \times E$$ with $$H = \{ \text{black}, \text{blond}, \text{red} \}$$ and $$E = \{ \text{brown}, \text{blue}, \text{green} \}$$. The probability of meeting a person with black hair and green eyes would accordingly be $$P(\langle \text{black}, \text{green} \rangle) = .01$$.

Denote with $$\text{"black"}$$ the proposition of meeting a person with black hair. This is a subset of the outcome space: $$\text{"black"} = \{ \langle h , e \rangle \in X \mid h = \text{black} \}$$. Similarly, for other hair and eye colors. The probability of meeting a person with black hair is obtained by **marginalization**: $$P(\text{"black"}) = .4 + .02 + .01 = .43$$.

If $$A, B \subseteq X$$ with $$B \neq \emptyset$$, the **conditional probability** of $$A$$ given $$B$$ is

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

For example, the conditional probability of a person to have black hair given that she has green eyes is:

$$P(\text{"black"} \mid \text{"green"}) = \frac{.01}{.01 + .1 + .05} = \frac{1}{16} = 0.0625$$

A direct consequence of this definition is **Bayes rule** which relates the conditional probability of $$A$$ given $$B$$ to the conditional probability of $$B$$ given $$A$$:

$$P(A \mid B) = \frac{P(B \mid A) \ P(A)}{P(B)}$$

Bayes rule is most useful for abductive reasoning to the best explanation of an observation (effect) based on some unobservable cause. Consider the 3-card problem again. We would like to infer which card Jones selected based on what we saw (i.e., only one side of it). In other words, we would like to know the conditional probability of, say the blue-blue card, given that we have observed one blue side: $$P(\text{blue-blue} \mid \text{obs. blue})$$. What we have is an observation-generating process that specifies the reverse, namely the conditional probability of observing blue or red, given a card. Formally, we would get:

$$P(\text{blue-blue} \mid \text{obs. blue}) = \frac{P(\text{obs. blue} \mid \text{blue-blue}) \ P(\text{blue-blue})}{P(\text{obs. blue})}$$

$$ = \frac{1 \cdot \frac{1}{3}}{\frac{1}{2}} = \frac{2}{3} $$

##### Bayes rule in memo

Let's implement this in memo and see the results:

```python
import jax
import jax.numpy as jnp
from enum import IntEnum

class Color(IntEnum):
    RED = 0
    BLUE = 1

cards = jnp.arange(3)
sides = jnp.arange(2)

@jax.jit
def check_color(card, side):
    return jnp.array([
        [0, 0],  # Card 0: both sides red
        [0, 1],  # Card 1: one red, one blue
        [1, 1]   # Card 2: both sides blue
    ])[card][side]

def color_prior(color):
    total_prob = 0
    for card in cards:
        for side in sides:
            if check_color(card, side) == color:
                total_prob += 1/6  # 1/3 for card * 1/2 for side
    return total_prob

def card_posterior(card, color):
    # Calculate P(color | card)
    p_color_given_card = 0
    for side in sides:
        if check_color(card, side) == color:
            p_color_given_card += 0.5  # 1/2 for each side
    
    # Calculate P(color)
    p_color = color_prior(color)
    
    # Calculate P(card)
    p_card = 1/3  # Equal probability for each card
    
    # Apply Bayes' rule
    return (p_color_given_card * p_card) / p_color

# Calculate and display prior probabilities
print("Prior probabilities of colors:")
print(f"P(RED) = {color_prior(Color.RED):.4f}")
print(f"P(BLUE) = {color_prior(Color.BLUE):.4f}")

# Display the full probability table
print("\nFull probability table:")
print("Card | Side 1 | Side 2")
print("----------------------")
for card in cards:
    side1 = "RED" if check_color(card, 0) == Color.RED else "BLUE"
    side2 = "RED" if check_color(card, 1) == Color.RED else "BLUE"
    print(f"{card+1:4d} | {side1:6s} | {side2:6s}")

# Calculate and display posterior probabilities
print("\nPosterior probabilities of cards given color:")
print("\nGiven RED:")
for card in cards:
    print(f"P(card {card+1} | RED) = {card_posterior(card, Color.RED):.4f}")
print("\nGiven BLUE:")
for card in cards:
    print(f"P(card {card+1} | BLUE) = {card_posterior(card, Color.BLUE):.4f}")
```
{: data-executable="true" data-thebe-executable="true"}

##### Working with memo: From simple enumeration to Bayesian inference

The `memo` library provides a powerful domain-specific language for expressing probabilistic computations. Let's work through several examples that demonstrate key concepts in probability and Bayesian reasoning.

**Example 1: Simple enumeration over discrete domains**

We start with the simplest case - enumerating over a discrete domain like coin flips:

```python
import jax
import jax.numpy as jnp
from enum import IntEnum
from memo import memo
from memo import domain as product

class Coin(IntEnum):
    TAILS = 0
    HEADS = 1

@memo
def f_enum[_c: Coin]():
    return _c

res = f_enum(print_table=True)
```
{: data-executable="true" data-thebe-executable="true"}

**Example 2: Computing conditional probabilities**

Next, we compute probabilities over the same domain using memo's probabilistic constructs:

```python
@memo
def g[_c: Coin]():
    observer: given(c in Coin, wpp=1)
    return Pr[observer.c == _c]

res = g(print_table=True)
```
{: data-executable="true" data-thebe-executable="true"}

**Example 3: Multiple coin flips**

Now let's consider flipping two coins and computing the probability that at least one comes up heads:

```python
SampleSpaceTwoFlips = product(
    f1=len(Coin),
    f2=len(Coin),
)

@jax.jit
def sumflips(s):
    return SampleSpaceTwoFlips.f1(s) + SampleSpaceTwoFlips.f2(s)

@memo
def flip_twice():
    student: given(s in SampleSpaceTwoFlips, wpp=1)
    return Pr[sumflips(student.s) >= 1]

flip_twice(print_table=True)
```
{: data-executable="true" data-thebe-executable="true"}

**Example 4: Multiple coin flips with constraints**

For a more complex example, let's flip 10 coins and compute the probability of getting between 4 and 6 heads (inclusive):

```python
nflips = 10
SampleSpace = product(**{f"f{i}": len(Coin) for i in range(1, nflips + 1)})

@jax.jit
def sumseq(s):
    return jnp.sum(jnp.array([SampleSpace._tuple(s)]))

@memo
def flip_n():
    student: given(s in SampleSpace, wpp=1)
    return Pr[sumseq(student.s) >= 4, sumseq(student.s) <= 6]

flip_n(print_table=True)
```
{: data-executable="true" data-thebe-executable="true"}

**Example 5: The 3-card problem in memo**

Now let's implement the 3-card problem using memo's probabilistic programming constructs. This demonstrates how memo can naturally express complex probabilistic reasoning:

```python
class Color(IntEnum):
    RED = 0
    BLUE = 1

cards = jnp.arange(3)
sides = jnp.arange(2)

@jax.jit
def check_color(card, side):
    return jnp.array([
        [0, 0],  # Card 0: both sides red
        [0, 1],  # Card 1: one red, one blue  
        [1, 1]   # Card 2: both sides blue
    ])[card][side]

# let's compute the prior probability of seeing each color
@memo
def color_prior[_c: Color]():
    agent: given(card in cards, wpp=1)
    agent: given(side in sides, wpp=1)
    return Pr[check_color(agent.card, agent.side) == _c]

color_prior(print_table=True)
```
{: data-executable="true" data-thebe-executable="true"}

**Example 6: Bayesian inference for the 3-card problem**

Finally, let's implement the full Bayesian inference for the 3-card problem. This shows how memo can express complex inference scenarios with observers, agents, and conditional reasoning:

```python
import xarray as xr

@memo
def card_posterior[_card: cards, _color: Color]():
    observer: knows(_card, _color)
    observer: thinks[
        friend: chooses(card in cards, wpp=1),
        friend: chooses(side in sides, wpp=1),
        friend: given(color in Color, 
                        wpp=color==check_color(card, side))
    ]
    observer: observes [friend.color] is _color
    return observer[Pr[friend.card == _card]]

res1 = card_posterior(print_table=True, return_aux=True, return_xarray=True)

# Extract and display the posterior probabilities
xa = res1.aux.xarray
pr = xa.loc[:, 'RED']
print(f"        P(card 1 | RED) = {pr.loc[0].sum():.4f}")
print(f"        P(card 2 | RED) = {pr.loc[1].sum():.4f}")
print(f"        P(card 3 | RED) = {pr.loc[2].sum():.4f}")
print("\n")
pr2 = xa.loc[:, 'BLUE']
print(f"        P(card 1 | BLUE) = {pr2.loc[0].sum():.4f}")
print(f"        P(card 2 | BLUE) = {pr2.loc[1].sum():.4f}")
print(f"        P(card 3 | BLUE) = {pr2.loc[2].sum():.4f}")
```
{: data-executable="true" data-thebe-executable="true"}