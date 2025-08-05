---
layout: chapter
title: Combining RSA and compositional semantics
description: "Jointly inferring parameters and interpretations"
---

### Chapter 4: Jointly inferring parameters and interpretations

One of the most remarkable aspects of natural language is its compositionality: speakers generate arbitrarily complex meanings by stitching together their smaller, meaning-bearing parts. The compositional nature of language has served as the bedrock of semantic (indeed, linguistic) theory since its modern inception; Montague demonstrates with his fragment how meaning gets constructed from a lexicon and some rules of composition. Since then, compositionality has continued to guide semantic inquiry: what are the meanings of the parts, and what is the nature of the mechanism that composes them? Put differently, what are the representations of the language we use, and what is the nature of the computational system that manipulates them?

So far, the models we have considered operate at the level of full utterances.  These models assume conversational agents who reason over propositional content to arrive at enriched interpretations: "I want the blue one," "Some of the apples are red," "The electric kettle cost $10,000 dollars," etc. Now, let's approach meaning from the opposite direction: building the literal interpretations (and our model of the world that verifies them) from the bottom up: [semantic parsing](http://dippl.org/examples/semanticparsing.html). The model constructs literal interpretations and verifying worlds from the semantic atoms of sentences. However, whereas the model explicitly targets compositional semantics, it stops at the level of the literal listener, the base of RSA reasoning. In what follows, we consider a different approach to approximating compositional semantics within the RSA framework.

What we want is a way for our models of language understanding to target sub-propositional aspects of meaning. We might wind up going the route of the fully-compositional but admittedly-unwieldy CCG semantic parser, but for current purposes an easier path presents itself: parameterizing our meaning function so that conversational agents can reason jointly over utterance interpretations and the parameters that fix them. To see how this move serves our aims, we consider the following application from reft:scontraspearl2021.

#### Application: Quantifier scope ambiguities

Quantifier scope ambiguities have stood at the heart of linguistic inquiry for nearly as long as the enterprise has existed in its current form. reft:montague1973 builds the possibility for scope-shifting into the bones of his semantics. reft:may1977 proposes the rule of QR, which derives scope ambiguities syntactically. Either of these efforts ensures that when you combine quantifiers like *every* and *all* with other logical operators like negation, you get an ambiguous sentence; the ambiguities correspond to the relative scope of these operators within the logical form (LF) of the sentence (whence the name "scope ambiguities").

- *Every apple isn't red.*
	- surface scope: ∀ > ¬; paraphrase: "none"
	- inverse scope: ¬ > ∀; paraphrease: "not all"

Rather than modeling the relative scoping of operators directly in the semantic composition, we can capture the possible meanings of these sentences---and, crucially, the active reasoning of speakers and listeners *about* these possible meanings---by assuming that the meaning of the utterance is evaluated relative to a scope interpretation parameter (surface vs. inverse). The meaning function thus takes an utterance, a world state, and an interpretation parameter `scope` (i.e., which interpretation the ambiguous utterance receives); it returns a truth value.

```python 
from memo import memo, domain
import jax
import jax.numpy as jnp
from enum import IntEnum

State = jnp.arange(4)
class U(IntEnum):
    NULL = 0
    EVERYNOT = 1

class Scope(IntEnum):
    SURFACE = 0
    INVERSE = 1

@jax.jit
def meaning(u, state, scope):
    return jnp.array([
      [1, 1],
      [state == 0, state < 3]
    ])[u][scope]

@memo
def L0[_u: U, _scope: Scope, _state: State]():
  listener: knows(_u, _scope)
  listener: chooses(state in State, wpp=meaning(_u, state, _scope))
  return Pr[listener.state == _state] + 1e-100

@memo
def S1[_state: State, _scope: Scope, _u: U](alpha): 
  speaker: knows(_state, _scope)
  speaker: chooses(u in U, wpp=exp(alpha * (log(L0[u, _scope, _state]()) - 1)))
  return Pr[speaker.u == _u]

@memo
def L1[_u: U, _scope: Scope, _state: State](alpha):
  listener: knows(_state, _scope)
  listener: thinks[
    speaker: given(state in State, wpp=1),
    speaker: given(scope in Scope, wpp=1),
    speaker: chooses(u in U, wpp=S1[state, scope, u](alpha))
  ]

  listener: observes [speaker.u] is _u
  return listener[Pr[speaker.state == _state, speaker.scope == _scope]]
     
posterior = L1(1)[U.EVERYNOT,:,:]
print(posterior)
```
{: data-executable="true" data-thebe-executable="true"}

The literal listener $$L_0$$ has prior uncertainty about the true state, *s*, and otherwise updates beliefs about *s* by conditioning on the meaning of *u* together with the intended scope. The interpretation variable (`scope`) is lifted, so that it will be actively reasoned about by the pragmatic listener. The pragmatic listener resolves the interpretation of an ambiguous utterance (determining what the speaker likely intended) while inferring the true state of the world.

> **Exercises:**
> 1. The pragmatic listener believes the `inverse` interpretation is more likely. Why?
> 2. Add some more utterances and check what happens to the interpretation of the ambiguous utterance.

As in the non-literal language models from the previous chapter, here we can add uncertainty about the topic of conversation, or QUD. This move recognizes that "Every apple isn't red" might be used to answer various questions. The listener might be interested to learn how many apples are red, or whether all of the apples are red, or whether none of them are, etc. Each question corresponds to a unique QUD; it's up to $$L_1$$ to decide which QUD is most likely given the utterance.

```python
from memo import memo, domain
import jax
import jax.numpy as jnp
from enum import IntEnum

State = jnp.arange(4)
class U(IntEnum):
    NULL = 0
    EVERYNOT = 1

class Scope(IntEnum):
    SURFACE = 0
    INVERSE = 1

class QUD(IntEnum):
    HOW_MANY = 0
    ALL_RED = 1

@jax.jit
def project(state, qud): 
    return jnp.array([
      state,           # QUD.HOW_MANY
      state == 3,      # QUD.ALL_RED
    ])[qud]

@jax.jit
def meaning(u, state, scope):
    return jnp.array([
      [1, 1],
      [state == 0, state < 3]
    ])[u][scope]

@memo
def L0[_u: U, _scope: Scope, _state: State]():
  listener: knows(_u, _scope)
  listener: chooses(state in State, wpp=meaning(_u, state, _scope))
  return Pr[listener.state == _state] + 1e-100

@memo
def S1[_qud: QUD, _state: State, _scope: Scope, _u: U](alpha): 
  speaker: knows(_state, _scope, _qud)
  speaker: thinks[
    world: knows(_state, _qud),
    world: chooses(state in State, wpp=(
      project(_state, _qud) == project(state, _qud)))
  ]
  speaker: chooses(u in U, wpp=exp(alpha * (log(E[L0[u, _scope, world.state]()]))))
  return Pr[speaker.u == _u]

@memo
def L1[_u: U, _scope: Scope, _state: State](alpha):
  listener: knows(_state, _scope)
  listener: thinks[
    speaker: given(state in State, wpp=1),
    speaker: given(scope in Scope, wpp=1),
    speaker: given(qud in QUD, wpp=1),
    speaker: chooses(u in U, wpp=S1[qud, state, scope, u](alpha))
  ]

  listener: observes [speaker.u] is _u
  return listener[Pr[speaker.state == _state, speaker.scope == _scope]]
     
posterior = L1(1)[U.EVERYNOT,:,:]
print(posterior)
```
{: data-executable="true" data-thebe-executable="true"}

> **Exercises:** 
> 1. What does the pragmatic listener infer about the QUD? Does this match your own intuitions? If not, how can you more closely align the model's predictions with your own?
> 2. Try adding a `none red?` QUD. What does this addition do to $$L_1$$'s inference about the state? Why?


Finally, we can add one more layer to our models: rather than predicting how a listener would interpret the ambiguous utterance, we can predict how a *speaker* would use it. In other words, we can derive predictions about whether the ambiguous utterance would be endorsed as a good description of a specific state of the world---a truth-value judgment. A speaker might see that two out of three apples are red. In this scenario, is the ambiguous utterance a good description? To answer this question, reft:scontraspearl2021 propose that the speaker should reason about how a listener would interpret the utterance. But our current speaker model isn't able to derive these predictions for us, given the variables that have been lifted to the level of $$L_1$$ (i.e., `scope` and `QUD`). In the case of lifted variables, we'll need another level of inference to model a *pragmatic* speaker, $$S_2$$.

This pragmatic speaker observes the state of the world and chooses an utterance that would best communicate this state to a pragmatic listener. The full model simply adds $$S_2$$ as an additional layer of inference.

```python
from memo import memo, domain
import jax
import jax.numpy as jnp
from enum import IntEnum

State = jnp.arange(4)
class U(IntEnum):
    NULL = 0
    EVERYNOT = 1

class Scope(IntEnum):
    SURFACE = 0
    INVERSE = 1

class QUD(IntEnum):
    HOW_MANY = 0
    ALL_RED = 1
    NONE_RED = 2

@jax.jit
def project(state, qud): 
    return jnp.array([
      state,           # QUD.HOW_MANY
      state == 3,      # QUD.ALL_RED
      state == 0,      # QUD.NONE_RED
    ])[qud]

@jax.jit
def meaning(u, state, scope):
    return jnp.array([
      [1, 1],
      [state == 0, state < 3]
    ])[u][scope]

@memo
def L0[_u: U, _scope: Scope, _state: State]():
  listener: knows(_u, _scope)
  listener: chooses(state in State, wpp=meaning(_u, state, _scope))
  return Pr[listener.state == _state] + 1e-100

@memo
def S1[_qud: QUD, _state: State, _scope: Scope, _u: U](alpha): 
  speaker: knows(_state, _scope, _qud)
  speaker: thinks[
    world: knows(_state, _qud),
    world: chooses(state in State, wpp=(
      project(_state, _qud) == project(state, _qud)))
  ]
  speaker: chooses(u in U, wpp=exp(alpha * (log(E[L0[u, _scope, world.state]()]))))
  return Pr[speaker.u == _u] + 1e-100

@memo
def L1[_u: U, _state: State](alpha):
  listener: knows(_state)
  listener: thinks[
    speaker: given(state in State, wpp=1),
    speaker: given(scope in Scope, wpp=1),
    speaker: given(qud in QUD, wpp=1),
    speaker: chooses(u in U, wpp=S1[qud, state, scope, u](alpha))
  ]

  listener: observes [speaker.u] is _u
  return listener[Pr[speaker.state == _state]] + 1e-100

@memo
def S2[_state: State, _u: U](alpha): 
  speaker: knows(_state)
  speaker: chooses(u in U, wpp=L1[u, _state](alpha))
  return Pr[speaker.u == _u]

posterior = S2(1)[2,:]
print(posterior)
```
{: data-executable="true" data-thebe-executable="true"}


> **Exercise:** What changes can you make to the model to get the speaker's endorsement to increase? Why do these changes have this effect?

Here we link to the [next chapter](05-vagueness.html).
