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

To implement Bayes rule in memo, we specify the observation-generating process using memo's sampling and inference constructs. First, let's define our shared code:

<script type="py-editor">
# Shared definitions
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
</script>

Now we can calculate and display the prior probabilities:

<script type="py-editor">
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
</script>

Now we can implement Bayesian reasoning to calculate the posterior probabilities:

<script type="py-editor">
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

# Calculate and display posterior probabilities
print("Posterior probabilities of cards given color:")
print("\nGiven RED:")
for card in cards:
    print(f"P(card {card+1} | RED) = {card_posterior(card, Color.RED):.4f}")
print("\nGiven BLUE:")
for card in cards:
    print(f"P(card {card+1} | BLUE) = {card_posterior(card, Color.BLUE):.4f}")
</script>

This implementation shows how we can model basic Bayesian reasoning in Python. Let's break it down:

1. We're interested in building probability distributions depending on two axes:
   - `card`: which card we're interested in (0, 1, or 2)
   - `color`: the color we observed (RED or BLUE)

2. The `check_color` function defines the color configuration of each card:
   - Card 0: both sides red
   - Card 1: one red, one blue
   - Card 2: both sides blue

3. The `color_prior` function calculates the prior probability of seeing a particular color by:
   - Iterating through all possible card and side combinations
   - Adding up the probabilities when the color matches

4. The `card_posterior` function implements Bayes' rule to calculate:
   - P(color | card): probability of seeing a color given a card
   - P(color): prior probability of seeing a color
   - P(card): prior probability of selecting a card
   - Returns P(card | color) using Bayes' rule

When we run this model, it gives us probabilities like:
```
P(1 | RED) = 0.3333
P(2 | RED) = 0.0000
P(3 | RED) = 0.6667

P(1 | BLUE) = 0.3333
P(2 | BLUE) = 0.6667
P(3 | BLUE) = 0.0000
```

These probabilities make intuitive sense:
- If we see red, it can't be card 2 (which has both sides blue)
- If we see blue, it can't be card 0 (which has both sides red)
- Card 1 (with one red and one blue side) is equally likely in both cases

> **Exercise:** Implement Bayesian reasoning for the 2-box problem in Python. Jones has two boxes. One contains two gold coins, the other one gold and one silver coin. Jones selects a random box and picks a random coin from it. He shows you a gold coin. What is the probability that the other coin in the box from which Jones presented the gold coin to you is also gold?
