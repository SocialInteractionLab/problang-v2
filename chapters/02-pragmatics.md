---
layout: chapter
title: Modeling pragmatic inference
description: "Enriching the literal interpretations"
---

### Chapter 2: Enriching the literal interpretations

#### Application 1: Scalar implicature

Scalar implicature stands as the poster child of pragmatic inference. Utterances are strengthened---via implicature---from a relatively weak literal interpretation to a pragmatic interpretation that goes beyond the literal semantics: "Some of the apples are red," an utterance compatible with all of the apples being red, gets strengthened to "Some but not all of the apples are red."  The mechanisms underlying this process have been discussed at length. reft:goodmanstuhlmuller2013 apply an RSA treatment to the phenomenon and formally articulate the model by which scalar implicatures get calculated.

Assume a world with three apples; zero, one, two, or three of those apples may be red.
Further assume that speakers may describe the current state of the world in one of three ways: "none", "some," or "all" of the apples are red.
Putting priors and literal semantics together, we can implement the behavior of a literal listener as in the previous chapter. The literal listener simply interprets messages by assigning probability 0 to each state where the observed message is false, and a uniform probability to each true state.


```python
from memo import memo
import jax
import jax.numpy as np
from enum import IntEnum

# possible states of the world
State = np.arange(4)

# possible utterances
class Utterance(IntEnum):
  ALL = 0
  SOME = 1
  NONE = 2

@jax.jit
def meaning(utterance, state):
    return np.array([
        state == 3,  # ALL: true if 3/3 apples are red
        state >= 1,  # SOME: true if at least one is red
        state == 0,  # NONE: true if 0/3 apples are red
    ])[utterance]

@memo
def L0[_u: Utterance, _s: State]():
    listener: knows(_u)
    listener: chooses(s in State, wpp=meaning(_u, s))
    return Pr[listener.s == _s]

L0()
```
{: data-executable="true" data-thebe-executable="true"}

> **Exercises:**
> 1. Try examining the prior over states.
> 2. Try adding an utterance for "MOST".

Next, let's have a look at the speaker's behavior. Intuitively put, in the vanilla RSA model, the speaker will never choose a false message; the speaker chooses among the true utterances by prioritizing those with the strongest meaning (see [Appendix Chapter 2](app-02-utilities.html)). You can verify this behavior with the following code.

```python
@memo
def S1[_u: Utterance, _s: State](alpha):
    speaker: knows(_s)
    speaker: chooses(u in Utterance, wpp=imagine[
      listener: knows(u),
      listener: chooses(s in State, wpp=meaning(u, s)),
      exp(alpha * log(Pr[listener.s == _s]))
    ])
    return Pr[speaker.u == _u]

S1(1)
```
{: data-executable="true" data-thebe-executable="true"}

With this knowledge about the communication scenario---crucially, the availability of the "all" alternative utterance---a pragmatic listener is able to infer from the "some" utterance that a state in which the speaker would not have used the "all" utterance is more likely than one in which she would. The following code---a complete vanilla RSA model for scalar implicatures---implements the pragmatic listener. 

```python
@memo
def L1[_u: Utterance, _s: State](alpha):
    listener: thinks[
        speaker: chooses(s in State, wpp=1),
        speaker: chooses(u in Utterance, wpp=imagine[
            listener: knows(u),
            listener: chooses(s in State, wpp=meaning(u, s)),
            exp(alpha * log(Pr[listener.s == s]))
        ])
    ]
    listener: observes [speaker.u] is _u
    listener: chooses(s in State, wpp=Pr[speaker.s == s])
    return Pr[listener.s == _s]

L1(1)
```    
{: data-executable="true" data-thebe-executable="true"}

> **Exercises:**
> 1. Explore what happens if you make the speaker *less* optimal.
> 2. Subtract one of the utterances. What changed?
> 3. Add a new utterance. What changed?
> 4. Check what would happen if 'some' literally meant some-but-not-all 
> 5. Change the relative prior probabilities of the various states and see what happens to model predictions.

#### Application 2: Scalar implicature and speaker knowledge

##### Section overview

Capturing scalar implicature within the RSA framework might not induce waves of excitement. However, by implementing implicature-calculation within a formal model of communication, we can also capture its interactions with other pragmatic factors. reft:GoodmanStuhlmuller2013Impl explored a model of what happens when the speaker possibly only has partial knowledge about the state of the world. Below, we explore their model, taking into account the listener's beliefs about the speaker's (possibly uncertain) beliefs about the world. We then go beyond the work of reft:GoodmanStuhlmuller2013Impl and study also the way these higher-order listener beliefs about the speaker's knowledge may change when hearing an utterance.

##### Setting the scene

Suppose a speaker says: "Some of the apples are red." If you know that there are 3 apples in total, but that the speaker has only observed two of them, how likely do you think it is that 0, 1, 2 or 3 of the apples are red? This is the question that reft:GoodmanStuhlmuller2013Impl address (Fig. 1 below).

{% include figure.html 
file="../images/scalar.png" 
caption="Example communication scenario from Goodman and Stuhlmüller: How will the listener interpret the speaker’s utterance? How will this change if she knows that he can see only two of the objects?" 
number = "1"
width="500px" 
%}

Towards an implementation, let's introduce some terminology and some notation. The **total number** of apples is $$n$$, of which $$0 \le s \le n$$ are red. We call $$s$$ the **state** of the world. The speaker knows $$n$$ (as does the listener) but the speaker might not know the true state $$s$$, because she might only observe some of the apples' colors. Concretely, the speaker might only have seen (i.e., **accessed**) some of the apples; of those apples that were accessed, the speaker might have **observed** that some of them were red. The model of reft:GoodmanStuhlmuller2013Impl assumes that the listener knows how many apples the speaker saw (i.e., the speaker's access $$a$$). We will first look at this model, and then generalize to the case where the listener must also infer $$a$$ from the speaker's utterance.

##### The extended Scalar Implicature model 

In the extended Scalar Implicature model of reft:GoodmanStuhlmuller2013Impl, the pragmatic listener infers the true state $$s$$ of the world not only on the basis of the observed utterance, but also the speaker's epistemic access $$a$$:

$$P_{L_{1}}(s\mid u, a) \propto P_{S_{1}}(u\mid s, a) \cdot P(s)$$

where 

$$P_{S_{1}}(u\mid s, a) = \sum_o P_{S_{1}}(u \mid o, a) \cdot P(o
\mid s, a)$$

is obtained from marginalizing out the possible observations $$o$$. 

```python
from memo import memo
import jax
import jax.numpy as jnp
from enum import IntEnum
from memo import domain

# possible utterances
class Utterance(IntEnum):
  ALL = 0
  SOME = 1
  NONE = 2
  NULL = 3

base_rate = 0.62
NumRed = jnp.array([0,1,2,3])

# possible states of the world
class Apple(IntEnum):
  GREEN = 0
  RED = 1
  
State = domain(
  apple1=len(Apple),
  apple2=len(Apple),
  apple3=len(Apple)
)

@jax.jit
def num_red(s):
    return State.apple1(s) + State.apple2(s) + State.apple3(s)

@jax.jit
def meaning(utterance, state):
    return jnp.array([
        num_red(state) == 3,  # ALL: true if 3/3 apples are red
        num_red(state) >= 1,  # SOME: true if at least one is red
        num_red(state) == 0,  # NONE: true if 0/3 apples are red
      	True                  # NULL: always true
    ])[utterance]

@jax.jit
def obs_p(o, s, a): 
  s_array = jnp.array([State.apple1(s), State.apple2(s), State.apple3(s)])
  a_array = jnp.array([State.apple1(a), State.apple2(a), State.apple3(a)])
  N = jnp.sum(jnp.where(a_array, s_array, 0))
  return N == o 

@jax.jit
def binom_p(s):
  return base_rate**num_red(s) * (1 - base_rate)**(3 - num_red(s))

@memo  
def L0[_u: Utterance, _s: State]():
  listener: knows(_u)
  listener: chooses(s in State, wpp=meaning(_u, s))

  # add a tiny number to prevent NaNs when logging
  return Pr[listener.s == _s] + 1e-10

@memo
def S1[_o: NumRed, _a: State, _u: Utterance](alpha):  
  speaker: knows(_a)
  speaker: thinks[
    world: knows(_a),
    world: chooses(s in State, wpp = binom_p(s)),
    world: chooses(o in NumRed, wpp = obs_p(o, s, _a))
  ]

  speaker: observes [world.o] is _o
  speaker: chooses(u in Utterance, 
                   wpp=exp(alpha * E[log(L0[u, world.s]())]))
  return Pr[speaker.u == _u]

@memo
def L1[_n: NumRed, _a: State, _u: Utterance](alpha):
  listener: knows(_a)
  listener: thinks[
    speaker: knows(_a),
    speaker: given(s in State, wpp=binom_p(s)),
    speaker: given(o in NumRed, wpp = obs_p(o, s, _a)),
    speaker: chooses(u in Utterance, wpp=S1[o, _a, u](alpha))
  ]

  listener: observes [speaker.u] is _u
  listener: knows(_n)
  return listener[Pr[num_red(speaker.s) == _n]]

def test_listener(utt, access):
  outcomes = L1(3)
  for n in NumRed:
    print(f"P({n} | u={utt.name}, a={State._tuple(access)}) = {outcomes[n][access][utt]:.3f}")

test_listener(Utterance.SOME, 1)
```
{: data-executable="true" data-thebe-executable="true"}

We have to enrich the speaker model: first the speaker makes an observation $$o$$ of the true state $$s$$ with access $$a$$. On the basis of this observation, the speaker infers $$s$$.

> **Exercise:** See what happens when you change the red apple base rate.

Given potential uncertainty about the world state $$s$$, the speaker's probabilistic production rule has to be adapted from the simpler formulation in [Chapter I](01-introduction.html). This rule is now no longer a function of the true $$s$$ (because, with limited access, the speaker might not know $$s$$), but of what the speaker *believes* $$s$$ to be (i.e., the speaker's epistemic state, given the speaker's access). In the case at hand, the speaker's epistemic state $$P_{S_{1}}(\cdot \mid o,a) \in \Delta(S)$$ is given by access $$a$$ and observation $$o$$, as in the belief model implemented just above.

Even if the speaker is uncertain about the state $$s$$ after some partial observation $$o$$ and $$a$$, she would still seek to choose an utterance that maximizes information flow. There are several ways in which we can combine speaker uncertainty (in the form of a probability distribution over $$s$$) and the speaker's utility function, which remains unchanged from what we had before, so that utterances are chosen to minimize cost and maximize informativity:

$$U_{S_{1}}(u; s) = log(L_{0}(s\mid u)) - C(u)$$

The model implemented by reft:GoodmanStuhlmuller2013Impl assumes that the speaker samples a state $$s$$ from her belief distribution and then samples an utterance based on the usual soft-maximization of informativity for that sampled state $$s$$. The formulation of this choice rule looks cumbersome in mathematical notation but is particularly easy to implement. (Another variant that conservatively extends the vanilla RSA model's assumption of rational agency is implemented in the next section. See also the discussion in the exercises below.)

$$P_{S_{1}}(u\mid o, a) \propto P_{S_{1}}(s\mid o, a) \ P_{S_{1}}(u \mid s)$$

$$P_{S_{1}}(u \mid s) \propto  exp(\alpha[U(u; s)])$$


The intuition (which Goodman and Stuhlmüller validate experimentally) is that in cases where the speaker has partial knowledge access (say, she knows about only two out of three relevant apples), the listener will be less likely to calculate the implicature (because he knows that the speaker doesn't have the evidence to back up the strengthened meaning).

> **Exercises:** 
> 1. Check the predictions for the other possible knowledge states.
> 2. Compare the full-access predictions with the predictions from the simpler scalar implicature model above. Why are the predictions of the two models different? How can you get the model predictions to converge? (Hint: first try to align the predictions of the simpler model with those of the knowledge model, then try aligning the predictions of the knowledge model with those of the simpler model.)
> 3. Notice that the listener assigns some positive probability to the true state being 0, even when it is shared knowledge that the speaker saw 2 apples and said "some". Why is this puzzling? (Think about the Gricean Maxim of Quality demanding that speakers not say what they lack sufficient evidence for.) Look at the speaker choice function implemented above and explain why this behavior takes place.

We have seen how the RSA framework can implement the mechanism whereby utterance interpretations are strengthened. Through an interaction between what was said, what could have been said, and what all of those things literally mean, the model delivers scalar implicature. And by taking into account awareness of the speaker's knowledge, the model successfully *blocks* implicatures in those cases where listeners are unlikely to access them. 

##### Joint-inferences of world state and speaker competence

In this section we further extend the model of Goodman & Stuhlmüller to also consider the listener's uncertainty about $$a$$, the number of apples that the speaker actually observed. This has interesting theoretical implications.

Many traditional approaches to scalar implicature calculation follow what reft:Geurts2010:Quantity-Implic calls the **standard recipe**:

1. the speaker used *some*
2. one reason why the speaker did not use *all* instead is that she does not know whether it is true
3. the speaker is assumed to be epistemically competent, i.e., she knows whether it is "all" or "some but not all" [**Competence Assumption**] 
4. so, she knows that the *all* sentence is actually false
5. if the speaker knows it, it must be true (by veridicality of knowledge)

Crucially, the standard recipe requires the Competence Assumption to derive a strong scalar implicature about the way the world is. Without the competence assumption, we only derive the *weak epistemic implicature*: that the speaker does not know that *all* is true.

From a probabilistic perspective, this is way too simple. Probabilistic modeling, aided by probabilistic programming tools, lets us explore a richer and more intuitive picture. In this picture, the listener may have probabilistic beliefs about the degree to which the speaker is in possession of the relevant facts. While these gradient prior beliefs of the listener about the speaker's likely competence matter to the interpretation of an utterance, hearing an utterance may also dynamically change these beliefs.

As before, let $$n$$ be the total number of apples of which $$0 \le s \le n$$ are red. The speaker has **access** to $$0 \le a \le n$$ apples, of which she **observes** $$0 \le o \le a$$ to be red. Previously, we looked at a case where the listener knows $$a$$. Here, we look at the (more natural) case where the listener must infer the degree to which the speaker is knowledgable (competent) from prior knowledge and the speaker's utterance.

If the speaker communicates her belief state with a statement like "Some of the apples are red", the listener performs a **joint inference** of the true world state $$s$$, the access $$a$$ and the observation $$o$$:

$$P_{L_{1}}(s, a, o \mid u) \propto P_{S_{1}}(u\mid a, o) \cdot P(s,a,o)$$

```python
from memo import memo
import jax
import jax.numpy as jnp
from enum import IntEnum
from memo import domain

# possible utterances
class Utterance(IntEnum):
  ALL = 0
  SOME = 1
  NONE = 2
  NULL = 3

base_rate = 0.62
NumRed = jnp.array([0,1,2,3])

# possible states of the world
class Apple(IntEnum):
  GREEN = 0
  RED = 1
  
State = domain(
  apple1=len(Apple),
  apple2=len(Apple),
  apple3=len(Apple)
)

@jax.jit
def num_red(s):
    return State.apple1(s) + State.apple2(s) + State.apple3(s)

@jax.jit
def meaning(utterance, state):
    return jnp.array([
        num_red(state) == 3,  # ALL: true if 3/3 apples are red
        num_red(state) >= 1,  # SOME: true if at least one is red
        num_red(state) == 0,  # NONE: true if 0/3 apples are red
      	True
    ])[utterance]

@jax.jit
def obs_p(o, s, a): 
  s_array = jnp.array([State.apple1(s), State.apple2(s), State.apple3(s)])
  a_array = jnp.array([State.apple1(a), State.apple2(a), State.apple3(a)])
  N = jnp.sum(jnp.where(a_array, s_array, 0))
  return N == o 

@jax.jit
def binom_p(s):
  return base_rate**num_red(s) * (1 - base_rate)**(3 - num_red(s))

@memo  
def L0[_u: Utterance, _s: State]():
  listener: knows(_u)
  listener: chooses(s in State, wpp=meaning(_u, s))
  return Pr[listener.s == _s] + 1e-10

@memo
def S1[_o: NumRed, _a: State, _u: Utterance](alpha):  
  speaker: knows(_a)
  speaker: thinks[
    world: knows(_a),
    world: chooses(s in State, wpp = binom_p(s)),
    world: chooses(o in NumRed, wpp = obs_p(o, s, _a))
  ]
  speaker: observes [world.o] is _o

  speaker: chooses(u in Utterance, 
                   wpp=exp(alpha * E[log(L0[u, world.s]())]))
  return Pr[speaker.u == _u]

@memo
def L1[_a: State, _u: Utterance](alpha):
  listener: knows(_a)
  listener: thinks[
    speaker: given(a in State, wpp=1),
    speaker: given(s in State, wpp=binom_p(s)),
    speaker: given(o in NumRed, wpp = obs_p(o, s, a)),
    speaker: chooses(u in Utterance, wpp=S1[o, a, u](alpha))
  ]

  listener: observes [speaker.u] is _u
  return listener[Pr[num_red(speaker.a) == _a]]

def test_listener(utt):
  outcomes = L1(3)
  for a in NumRed:
    print(f"P({a} | u={utt.name}) = {outcomes[a][utt]:.3f}")

test_listener(Utterance.SOME)
```
{: data-executable="true" data-thebe-executable="true"}

This formulation of the pragmatic listener differs in two respects from the previous. First, the pragmatic listener also has a (possibly uncertain) prior belief about $$a$$, which we can think of as prior knowledge of the likely extent of the speaker's competence. Second, the new formulation also refers to a [hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution), which we use as a more general way of representing the speaker's (possibly partial) beliefs.

Here is how to understand the speaker belief model in terms of a hypergeometric distribution. We first look at the speaker's prior beliefs about $$s$$: what does a speaker believe about the world state $$s$$ after, say, having access to $$a$$ out of $$n$$ apples and seeing that $$o$$ of the accessed apples are red? These beliefs are given by a binomial distribution with a fixed base rate of redness: intuitively put, each apple has a chance `base_rate` of being red; how many red apples do we expect given that we have $$n$$ apples in total?

> Exercise: Play around with `total_apples` and `base_rate_red` to get good intuitions about the state prior for different parameters. (For which values of `total_apples` and `base_rate_red` would it be better to take more samples for a more precise visualization?)


A world state $$s$$ gives the true, actual number of red apples. If the world state was known to the speaker (and the total number of apples $$n$$), her beliefs for any value of $$o$$ for a given $$a$$ are given by a so-called [hypergeometric distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution). The hypergeometric distribution gives the probability of retrieving $$o$$ red balls when drawing $$a$$ balls without replacement from an urn which contains $$n$$ balls in total of which $$s$$ are red. (This distribution is not implemented in WebPPL, so we implement its probability mass function by hand. Although this is tedious and the previous coding-by-hand might be more insightful, this formulation allows for easier manipulation of $$o$$, $$a$$ and $$s$$ as actual numbers.)


The prior over states and the hypergeometric distribution combine to give the speaker's beliefs about world state $$s$$ given access $$a$$ and observation $$o$$, using Bayes rule (and knowledge of the total number of apples $$n$$):

$$P_{S_1}(s \mid a, o) \propto \text{Hypergeometric}(o \mid s, a, n) \ \text{Binomial}(s \mid \text{baserate, n}) $$



> **Exercises:** 
> 1. See what happens when you change the red apple base rate.
> 2. See what happens when you increase the number of total apples.
> 3. See what happens when you give the speaker more access and different numbers of observed red apples.


To combine the speaker's beliefs $$P_{S_{1}}(\cdot \mid o,a) \in \Delta(S)$$ about the world state,  reft:GoodmanStuhlmuller2013Impl assumed a simple sampling-based method: the speaker sampled a state and considered what to say for it. While this is cognitively plausible, a fully rational agent would rather soft-maximize the expected utility under the given uncertainty:

$$P_{S_{1}}(u \mid o, a) \propto \exp(\alpha \ \mathbb{E}_{P_{S_{1}}(s \mid o, a)}[U(u; s)])$$

The speaker's utility function remains unchanged:

$$U_{S_{1}}(u; s) = log(L_{0}(s\mid u)) - C(u)$$

An equivalent motivation for this speaker model is that the speaker chooses an utterance with the goal of minimizing the (Kullback-Leibler) divergence between her belief state $$P_{S_{1}}(\cdot \mid o,a)$$ and that of the literal listener $$P_{L_0}(\cdot \mid u)$$. Details are in [appendix chapter II](app-02-utilities.html).

There is one important bit to notice about this definition. In the vanilla RSA model of the previous chapter, the speaker will never say anything false. The present conservative extension makes it so that an uncertain speaker will never use an utterance whose truth the speaker is not absolutely convinced of. In other words, as long as $$P_{S_{1}}(\cdot \mid o,a)$$ puts positive probability on a state $$s$$ for which utterance $$u$$ is false, the speaker will *never* use $$u$$ in epistemic state $$\langle o, a \rangle$$. This is because $$\log P_{L_0}(s \mid u)$$ is negative infinity if $$u$$ is false of $$s$$ and so the expected utility (which is a weighted sum) will be negative infinity as well, unless $$P_{S_{1}}(s \mid o,a) = 0$$. As a consequence, we need to make sure in the model that the speaker always has something true to say for all pairs of $$a$$ and $$o$$. We do this by including a "null utterance", which is like saying nothing. (See also [chapter V](05-vagueness.html) and reft:PottsLassiter2016:Embedded-implic for similar uses of a "null utterance".) 

Adding a set of utterances, an utterance prior and the literal listener, we obtain a full speaker model.


> **Exercises:**
> 1. Test the speaker model for different parameter values. Also change `total_apples` and `base_rate_red`.
> 2. When does the speaker use the "null" utterance?

If the pragmatic listener does not know the number $$a$$ of apples that the speaker saw, the listener can nevertheless infer likely values for $$a$$, given an utterance. In fact, the listener can make a joint inference of $$s$$, $$a$$ and $$o$$, all of which are unknown to him, but all of which feed into the speaker's utterance probabilities. The posterior inference of $$a$$ is particularly interesting because it is a probabilistic inference of the speaker's competence, mediated by what the speaker said.

```python
from memo import memo
from enum import IntEnum

import jax
import jax.numpy as np
from jax.scipy.special import factorial
from jax.scipy.stats.binom import pmf as binompmf
binompmf = jax.jit(binompmf)

NN = 3  # total number of apples
N = np.arange(NN + 1)
base_rate = 0.62

class U(IntEnum):
    NONE = 0
    SOME = 1
    ALL = 2
    NULL = 3

@jax.jit  # hypergeometric pmf
def p_obs(s, o, a):
    # binomial coefficient
    def nck(n, k): return np.where((k < 0) | (k > n), 0, factorial(n) / factorial(k) / factorial(n - k))

    # probability of seeing $o$ red apples out of $a$ if $s/n$ are red
    return nck(a, o) * nck(NN - a, s - o) / nck(NN, s)

@memo  # literal listener
def L0[u: U, s: N]():
    listener: knows(u)
    listener: chooses(s in N, wpp=(
        (s == {NN}) if u == {U.ALL}  else
        (s > 0)     if u == {U.SOME} else
        (s == 0)    if u == {U.NONE} else
        1           
    ))
    return Pr[listener.s == s] + 1e-10 

@memo  # pragmatic speaker with limited access
def S1[a: N, o: N, u: U](alpha):
    speaker: knows(a)
    speaker: thinks[
        world: knows(a),
        world: chooses(s in N, wpp=binompmf(s, {NN}, {base_rate})),
        world: chooses(o in N, wpp=p_obs(s, o, a)),
    ]
    speaker: observes [world.o] is o

    speaker: chooses(u in U, wpp=exp(alpha * E[log(L0[u, world.s]())]))
    return Pr[speaker.u == u]

@memo
def L1[u: U, a: N](alpha):
    listener: knows(a)
    listener: thinks[
        speaker: given(a in N, wpp=binompmf(a, {NN}, .5)),
        speaker: given(s in N, wpp=binompmf(s, {NN}, {base_rate})),
        speaker: given(o in N, wpp=p_obs(s, o, a)),
        speaker: chooses(u in U, wpp=S1[a, o, u](alpha))
    ]
    listener: observes [speaker.u] is u

    return listener[Pr[speaker.a == a]]

L1(3.2)[Utterance.SOME, :]
```
{: data-executable="true" data-thebe-executable="true"}

> **Exercises:**
> 1. How would you describe the result of the code above? Does the pragmatic listener draw a scalar implicature from "some" to "some but not all"?
> 2. Does the pragmatic listener, after hearing "some", increase or decrease his belief in the speaker's competence?
> 3. What does the pragmatic listener infer about the speaker's competence when he hears "all" or "none"?
> 4. Speculate about how the listener's inferences about speaker competence might be influenced by the inclusion of further utterance alternatives.

#### Conclusion

In this chapter, we saw a simple model of scalar implicature calculation based on the vanilla RSA model. We then extended this model to also cover the speaker's uncertainty, and eventually also the listener's inferences about how likely the speaker is to be knowledgeable. 

Within the RSA framework, speakers try to optimize information transmission. You might then think that the framework has nothing to say about situations where speakers produce utterances that are *literally false*, as in  "I had to wait a million years to get a table last night!"
In the [next chapter](03-nonliteral.html), we'll see how expanding the range of communicative goals of the speaker can lead listeners to infer nonliteral interpretations of utterances.
