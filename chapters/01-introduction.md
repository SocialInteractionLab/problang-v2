---
layout: chapter
title: Introducing the Rational Speech Act framework
description: "An introduction to language understanding as Bayesian inference"
---

### Chapter 1: Language understanding as Bayesian inference

Much work in formal, compositional semantics follows the tradition of positing systematic but inflexible theories of meaning. In practice, the meaning we derive from language depends innumerably contextual factors, both linguistic and situational. 
To formally explain these nuanced aspects of meaning and better understand the compositional mechanism that delivers them, recent work in formal pragmatics recognizes semantics not as one of the final steps in meaning calculation, but rather as one of the first. 
Within the Bayesian Rational Speech Act framework refp:frankgoodman2012, speakers and listeners reason about each other's reasoning about the literal interpretation of utterances. 
The resulting interpretation necessarily depends on the literal interpretation of an utterance, but is not wholly determined by it. 
This baisc move --- *reasoning* about likely interpretations --- provides ready explanations for complex phenomena ranging from metaphor refp:kaoetal2014metaphor and hyperbole refp:kaoetal2014 to the specification of thresholds in degree semantics refp:lassitergoodman2013.

The probabilistic pragmatics approach leverages the tools of structured probabilistic models formalized in a stochastic 𝞴-calculus to develop and refine a general theory of communication. The framework synthesizes the knowledge and approaches from diverse areas --- formal semantics, Bayesian models of inference, formal theories of measurement, philosophy of language, etc. --- into an articulated theory of language in practice. These new tools yield broader empirical coverage and richer explanations for linguistic phenomena through the recognition of language as a means of communication, not merely a vacuum-sealed formal system. By subjecting the heretofore off-limits land of pragmatics to articulated formal models, the rapidly growing body of research both informs pragmatic phenomena and enriches theories of semantics. In what follows, we consider the first foray into this framework.

### Introducing the Rational Speech Act framework

The Rational Speech Act (RSA) framework views communication as recursive reasoning between a speaker and a listener. The listener interprets the speaker's utterance by reasoning about a cooperative speaker trying to inform a naive listener about some state of affairs. Using Bayesian inference, the listener reasons about what the state of the world is likely to be given that a speaker produced some utterance, knowing that the speaker is reasoning about how a listener is most likely to interpret that utterance. Thus, we have (at least) three levels of inference. At the top, the sophisticated, **pragmatic listener**, $$L_{1}$$, reasons about the **pragmatic speaker**, $$S_{1}$$, and infers the state of the world given that the speaker chose to produce the utterance $$u$$. The speaker chooses $$u$$ by maximizing the probability that a naive, **literal listener**, $$L_{0}$$, would correctly infer the state of the world given the literal meaning of $$u$$.

To make this architecture more intelligible, let's consider a concrete example and a vanilla version of an RSA model. In its initial formulation, reft:frankgoodman2012 use the basic RSA framework to model referent choice in efficient communication. Let's suppose that there are only three objects that the speaker and listener want to talk about, as in Fig. 1.

{% include figure.html 
file="../images/rsa_scene.png" 
caption="Example referential communication scenario from Frank and Goodman. Speakers
choose a single word, <i>u</i>, to signal an object, <i>s</i>." 
number = "1"
width="400px" 
%}

In a **reference game**, a speaker wants to refer to one of the given objects. To simplify, we assume that the speaker may only choose one property (see below) with which to do so. In the example of Fig. 1, the **set of objects**

$$O = \{\text{blue-square}, \text{blue-circle}, \text{green-square}\}$$

contains the three objects given. The **set of utterances**

$$U = \{ \text{"square"}, \text{"circle"}, \text{"green"}, \text{"blue"}  \}$$

contains the four properties from which the speaker can choose.

A vanilla RSA model for this scenario consists of three recursively layered, conditional probability rules for speaker production and listener interpretation. These rules are summarized in Fig. 2 and will be examined one-by-one below. The overal idea is that a **pragmatic speaker** $$S_{1}$$ chooses a word $$u$$ to best signal an object $$o$$ to a **literal listener** $$L_{0}$$, who interprets $$u$$ as true and finds the objects that are compatible with the meaning of $$u$$. The **pragmatic listener** $$L_{1}$$ reasons about the speaker's reasoning and interprets $$u$$ accordingly, using Bayes' rule; $$L_1$$ also weighs in the prior probability of objects in the scenario (i.e., an object's salience, $$P(o)$$). By formalizing the contributions of salience and efficiency, the RSA framework provides an information-theoretic definition of informativeness in pragmatic inference.  

{% include figure.html 
file="../images/rsa_schema.png" 
caption="Schema of RSA reasoning hierarchy." 
width="400px" 
number = "2"
%}

### Literal listeners

At the base of this reasoning, the naive, literal listener $$L_{0}$$ interprets an utterance according to its meaning. That is, $$L_{0}$$ computes the probability of $$o$$ given $$u$$ according to the semantics of $$u$$ and the prior probability of $$o$$. A standard view of the semantic content of an utterance suffices: a mapping from states of the world to truth values. For example, the utterance $$\text{"blue"}$$ is true of states $$\text{blue-square}$$ and $$\text{blue-circle}$$ and false of state $$\text{green-square}$$. We write $$[\![u]\!] \colon O \mapsto \{0,1\}$$ for the denotation function of this standard, Boolean semantics of utterances in terms of states. The literal listener is then defined via a function $$P_{L_{0}} \colon U \mapsto \Delta^S$$ that maps each utterance to a probability distribution over world states, like so:

$$P_{L_{0}}(o\mid u) \propto [\![u]\!](o) \cdot P(o)$$

Here, $$P(o)$$ is an a priori belief regarding which state or object the speaker is likely to refer to in general. These prior beliefs can capture general world knowledge, perceptual salience, or other things. For the time being, we assume a flat prior belief according to which each object is equally likely. (As we move away from flat priors, we'll want to revise these assumptions so that $$L_0$$ (but not $$L_1$$!) uses a uniform prior over states. In fact, this is what reft:frankgoodman2012 assumed in their model. See [Appendix Chapter 4](app-04-BDA.html) for discussion.)

The literal listener rule can be written as follows:

```python
from memo import memo
import jax
import jax.numpy as np
from enum import IntEnum

# Define the objects and utterances using simple IntEnum
class Object(IntEnum):
    BLUE_SQUARE = 0
    BLUE_CIRCLE = 1
    GREEN_SQUARE = 2

class Utterance(IntEnum):
    BLUE = 0
    GREEN = 1
    SQUARE = 2
    CIRCLE = 3

# Denotation function: which of these utterances are true of which objects?
@jax.jit
def meaning(utterance, obj):
    return np.array([
        [True,  True,  False],  # BLUE: true for the blue objects
        [False, False, True ],  # GREEN: true for the green objects  
        [True,  False, True ],  # SQUARE: true for the square objects
        [False, True,  False]   # CIRCLE: true for the circle objects
    ])[utterance, obj]

@memo
def L0[_u: Utterance, _o: Object]():
    listener: knows(_u)
    listener: chooses(o in Object, wpp=meaning(_u, o))
    return Pr[listener.o == _o]
  
def test_literal_listener(utterance):
    print(f"Literal listener interpretation of '{utterance.name}':")
    outcomes = L0()
    for obj in Object:
        print(f"P({obj.name} | '{utterance.name}') = {outcomes[utterance][obj]:.3f}")
        
test_literal_listener(Utterance.BLUE)
```
{: data-executable="true" data-thebe-executable="true"}

> **Exercises:**
> 1. Try testing the literal listener's behavior with different utterances. What happens when you use `Utterance.SQUARE` or `Utterance.GREEN`?
> 2. Add a fourth object (e.g., a green circle) to the scenario. How does this change the literal listener's behavior?
> 3. In this example, we implicitly imposed a uniform prior over the three objects. Where in the model is this assumption being made? What happens when the listener's beliefs are not uniform over the possible objects of reference, e.g., the "green square" is very salient? (Hint: think about the `wpp` term in the `chooses` statement.)
> 4. The denotation function is manually specified as a Boolean matrix for this simple example. Can you make this a function of object properties to make the code more easily maintainable for different sets of utterances and objects? Note that we're under some strict constraints from JAX here: for the code to compile and run fast, we need to operate with arrays. 

Fantastic! We now have a way of specifying how a listener will interpret the truth functional meaning of a referential utterance in a given context of objects. Let's unpack some of the `memo` syntax. 

- The `knows` statement explicitly establishes that the listener agent has certain information available to them. In this case, we assume the listener heard an utterance `_u` so they can use it in their reasoning. What happens when we remove this line? 

- The `chooses` statement represents the listener making a choice among options. Here, the listener is choosing an object o from the set of possible Objects. The `wpp` parameter stands for "with probability proportional to" - it determines the probability weights for each choice. In this case, `wpp=meaning(_u, o)` means the probability of choosing object o is proportional to whether the (known) utterance _u is true of that object, as determined by the meaning function. This implements the literal semantics. 

- The return statement `Pr[listener.o == _o]` says that we want the probability that the listener's chosen object o equals the specific object _o that we're querying about. The `Pr[...]` syntax asks "what is the marginal probability that this condition is true?"

### Pragmatic speakers

One of Grice's core insights was that linguistic utterances are intentional actions. 
Thus, we model the speaker as a rational (Bayesian) actor. 
They choose an action (e.g., an utterance) according to its utility. 
Rationality is often defined as choosing an action that maximizes the agent's (expected) utility. 
Here, we consider a generalization in which speakers use a *softmax* function to approximate the (classical) rational choice.
(For more on the properties of the softmax funciton, see [this nice tutorial](https://alpslab.stanford.edu/papers/FrankeDegen_submitted.pdf).)

#### Bayesian decision-making

In the code box below, you'll see a generic *approximately rational* agent model. 
In memo, we use `factor` statements to increment log-probabilities, which effectively implements softmax decision-making. Each `factor` statement adds to the log-score of the current execution path, and memo normalizes these to compute probabilities. In effect, the function `agent` computes the distribution:

$$P(a_i) = \frac{\exp(\alpha \cdot \text{Util}(a_i))}{\sum_{j} \exp(\alpha \cdot \text{Util}(a_j))}$$

```python
# Define possible actions
class Action(IntEnum):
    A1 = 0
    A2 = 1
    A3 = 2

# Define utilities for the actions
@jax.jit
def utility(action):
    return np.array([0, 1, 2])[action]

# Define a rational agent who chooses actions according 
# to their expected utility
@memo
def agent[_a: Action](alpha):
    speaker: chooses(a in Action, wpp=exp(alpha * utility(a)))    
    return Pr[speaker.a == _a]

# Test the agent
print("Probability that an agent will take various actions:")
agent(1, print_table=True)
```
{: data-executable="true" data-thebe-executable="true"}

> **Exercises:**
> 1. Explore what happens when you change the actor's optimality parameter `alpha`. What does `alpha = 0` represent? What happens as `alpha` approaches infinity?
> 2. Explore what happens when you change the utilities in the `utility()` function to [4,5,6]? (Hint: think about what matters for the softmax -- absolute values or relative differences?). What if all utilities are equal? 
> 3. Add a fourth action `A4` with utility 10. How does the probability distribution change? 

#### A rational speech actor

In language understanding, the utility of an utterance is how well it communicates the state of the world $$s$$ to a listener. So, the speaker $$S_{1}$$ chooses utterances $$u$$ to communicate the state $$s$$ to the hypothesized literal listener $$L_{0}$$. Another way to think about this: $$S_{1}$$ wants to minimize the effort $$L_{0}$$ would need to arrive at $$s$$ from $$u$$, all while being efficient at communicating. $$S_{1}$$ thus seeks to minimize the surprisal of $$s$$ given $$u$$ for the literal listener $$L_{0}$$, while bearing in mind the utterance cost, $$C(u)$$. (This trade-off between efficacy and efficiency is not trivial: speakers could always use minimal ambiguity, but unambiguous utterances tend toward the unwieldy, and, very often, unnecessary. We will see this tension play out later in the book.)

Speakers act in accordance with the speaker's utility function $$U_{S_{1}}$$: utterances are more useful at communicating about some state as surprisal and utterance cost decrease. (See the [Appendix Chapter 2](app-02-utilities.html) for more on speaker utilities.)


$$U_{S_{1}}(u; s) = \log L_{0}(s\mid u) - C(u)$$

(In memo, we can access $$\log L_{0}(s\mid u)$$ through the literal listener's output probabilities.)

> **Exercise:** 
> Return to the first code box and find $$\log L_{0}(s\mid u)$$ for the utterance "blue" and each of the three possible reference objects.

With this utility function in mind, $$S_{1}$$ computes the probability of an utterance $$u$$ given some state $$s$$ in proportion to the speaker's utility function $$U_{S_{1}}$$. The term $$\alpha > 0$$ controls the speaker's optimality, that is, the speaker's rationality in choosing utterances. We define:

$$P_{S_{1}}(u\mid s) \propto \exp(\alpha U_{S_{1}}(u; s))\,,$$

which expands to:

$$P_{S_1}(u \mid s) \propto \exp(\alpha (\log L_{0}(s\mid u) - C(u)))\,.$$

The following code implements this model of the speaker.
In the following code, we assume that all utterances are equally costly (i.e., $$C(u) = C(u')$$ for all $$u, u'$$) (see [Appendix Chapter 3](app-03-costs.html) for more on message costs and how to implement them).

```python
@memo
def S1[_u: Utterance, _o: Object](alpha):
  speaker: knows(_o)
  speaker: chooses(u in Utterance, wpp=exp(alpha * log(L0[u, _o]())))
  return Pr[speaker.u == _u]

def test_pragmatic_speaker(alpha, obj):
    print(f"Literal listener interpretation of '{obj.name}':")
    outcomes = S1(alpha)
    for utterance in Utterance:
        print(f"P('{utterance.name}' | {obj.name}) = {outcomes[utterance][obj]}")

test_pragmatic_speaker(1, Object.GREEN_SQUARE)
```
{: data-executable="true" data-thebe-executable="true"}

> **Exercise:** Check the speaker's behavior for different objects. What happens when you test with `Object.GREEN_SQUARE` or `Object.BLUE_CIRCLE`? How does changing the `alpha` parameter affect the speaker's behavior?

We now have a model of the utterance generation process. With this in hand, we can imagine a listener who thinks about this kind of speaker.

### Pragmatic listeners

The pragmatic listener $$L_{1}$$ computes the probability of a state $$s$$ given some utterance $$u$$. By reasoning about the speaker $$S_{1}$$, this probability is proportional to the probability that $$S_{1}$$ would choose to utter $$u$$ to communicate about the state $$s$$, together with the prior probability of $$s$$ itself. In other words, to interpret an utterance, the pragmatic listener considers the process that *generated* the utterance in the first place. (Note that the listener model uses `observe`, which functions like `factor` with $$\alpha$$ set to $$1$$.)

<!-- <center>The pragmatic listener: P<sub>L<sub>1</sub></sub>(s|u) ∝ P<sub>S<sub>1</sub></sub>(u|s) · P(s)</center> -->

$$P_{L_{1}}(s\mid u) \propto P_{S_{1}}(u\mid s) \cdot P(s)$$

```python
@memo
def L1[_u: Utterance, _o: Object](alpha):
  listener: knows(_u)
  listener: chooses(o in Object, wpp = S1[_u, o](alpha))
  return Pr[listener.o == _o]

def test_pragmatic_listener(utterance, alpha=1.0):
    print(f"Pragmatic listener interpretation of '{utterance.name}' (alpha={alpha}):")
    outcomes = L1(alpha)
    for obj in Object:
        print(f"P({obj.name} | '{utterance.name}') = {outcomes[utterance][obj]:.3f}")

test_pragmatic_listener(Utterance.BLUE)        
```
{: data-executable="true" data-thebe-executable="true"}

### Putting it all together

So far, we've implemented the RSA framework by defining separate functions for each level of reasoning: `L0` for literal listeners, `S1` for pragmatic speakers, and `L1` for pragmatic listeners. While this approach works and gives us the correct results, it doesn't leverage the full power of memo, which makes the recursive reasoning structure more transparent. 
We can combine all these pieces into a single listener function that takes the level of recursion as an argument. This unified approach allows us to easily explore different depths of reasoning by simply changing a parameter. 

```python
@memo
@memo
def L[_u: Utterance, _o: Object](alpha, k):
  
    # we start by setting up the listener's mental model of the speaker
    listener: thinks[
      
        # the speaker is assumed to have some object in mind 
        speaker: given(o in Object, wpp=1),

        # and choose an utterance depending on the recursion level
        speaker: chooses(u in Utterance, wpp=
            meaning(u, o) 
            if k == 0 
            else exp(alpha * log(L[u, o](alpha, k - 1))))
    ]
    # now the listener actually hears the speaker produce an utterance
    listener: observes [speaker.u] is _u
    
    # and updates their beliefs about the object o accordingly
    listener: chooses(o in Object, wpp=Pr[speaker.o == o])
    
    # finally, we return a marginal distribution over the objects
    return Pr[listener.o == _o]

def test_pragmatic_listener(utterance, alpha=1.0, level = 1):
    print(f"Listener interpretation of '{utterance.name}' (alpha={alpha}, level = {level}):")
    
    outcomes = L(alpha, level)
    for obj in Object:
        print(f"P({obj.name} | '{utterance.name}') = {outcomes[utterance][obj]:.3f}")

test_pragmatic_listener(Utterance.BLUE, level = 0) 
test_pragmatic_listener(Utterance.BLUE, level = 1) 
```
{: data-executable="true" data-thebe-executable="true"}


> **Exercises:**
> 1. Explore what happens if you increase the `alpha` softmax temperature parameter. What if you increase the recursion level? Does the model converge? 
> 2. What happens when you set alpha=0 in the unified model? What about very large values of alpha? How does this parameter interact with the recursion level?
> 3. Try adding a new multi-word utterance (e.g., "blue square"). What should its meaning be? 
> 4. Add a simple cost function where longer utterances (like "blue square") are more costly than shorter ones (like "blue"). How does this affect the speaker's choices and the listener's interpretations across recursion levels?


In the [next chapter](02-pragmatics.html), we'll see how RSA models have been developed to model more complex aspects of pragmatic reasoning and language understanding.
