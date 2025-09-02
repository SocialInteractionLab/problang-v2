---
layout: chapter
title: Expanding our ontology
description: "Plural predication"
---

### Chapter 6: Plural predication

Our models so far have focused on identifying singular states of the world: a single object, a single price, or, in the case of the scalar implicature or scope ambiguity models, the single number corresponding to the numerosity of some state. Now, we extend our sights to pluralities: collections of objects. A plurality has members with properties (e.g., weights), but a plurality has collective properties of its own (e.g., total weight). To model the ambiguity inherent to plural predication---whether we are talking about the individual member properties or the collective property---we'll need to expand our ontology to include sets of objects.

We begin with some knowledge about the world, which contains objects (which have their own properties, e.g., weights). The `statePrior` recursively generates states with the appropriate number of objects; each state corresponds to a plurality.

~~~~
import jax
import jax.numpy as jnp
from memo import memo, domain
from enum import IntEnum
from jax.scipy.special import erf

# possible object weights
weights = jnp.array([2, 3, 4])
n_objects = 3
noise =  3

class U(IntEnum):
  NULL = 0
  EACHHEAVY = 1
  TOGETHERHEAVY = 2
  HEAVY = 3

class I(IntEnum):
  DISTRIBUTIVE = 0
  COLLECTIVE = 1

w = list(jnp.broadcast_to(weights, (n_objects, len(weights))))
states = jnp.transpose(jnp.array(jnp.meshgrid(*w))).reshape(-1, n_objects)
S = jnp.arange(len(states))

theta_dist = weights - 1
theta_coll = jnp.unique(jnp.sum(states, axis=1)) - 1
TD = jnp.arange(len(theta_dist))
TC = jnp.arange(len(theta_coll))

@jax.jit
def collective_meaning(state, theta):
    return 1 - (0.5 * (1 + erf((theta - jnp.sum(state)) / 
                              (noise * jnp.sqrt(2)))))

@jax.jit
def distributive_meaning(state, theta):
    return jnp.all(state > theta)
    
@jax.jit
def meaning(u, s, dt, ct, i) :
    return jnp.array([
        True,                                            # SILENCE
        distributive_meaning(states[s], theta_dist[dt]), # EACHHEAVY
        collective_meaning(states[s], theta_coll[ct]),   # TOGETHERHEAVY
        jnp.where(i,                                     # HEAVY (ambiguous)
            collective_meaning(states[s], theta_coll[ct]),
            distributive_meaning(states[s], theta_dist[dt])
        ),
    ])[u]

@jax.jit
def cost(u):
    return jnp.array([0, 2, 2, 1])[u]

@memo
def L0[u: U, td: TD, tc: TC, i: I, s: S]():
    listener: knows(u, td, tc, i)
    listener: given(s in S, wpp= meaning(u, s, td, tc, i))
    return Pr[listener.s == s] 

@memo
def S1[s: S, td: TD, tc: TC, i: I, u: U](alpha): 
    speaker: knows(s, td, tc, i)
    speaker: chooses(u in U, wpp= 
      exp(alpha * (log(L0[u, td, tc, i, s]()) - cost(u)))
    )
    return Pr[speaker.u == u]

@memo
def L1[u: U, s: S, i: I](alpha) :
    listener: knows(s, i)
    listener: thinks[
        speaker: given(s in S, wpp= 1),
        speaker: given(td in TD, wpp=1),
        speaker: given(tc in TC, wpp=1),
        speaker: given(i in I, wpp=0.8 if i == {I.COLLECTIVE} else 0.2),
        speaker: chooses(u in U, wpp=S1[s, td, tc, i, u](alpha))
    ]
    listener: observes [speaker.u] is u
    return listener[Pr[speaker.s == s, speaker.i == i]]

# print(list(zip(State,
#     L0()[U.HEAVY, 2, 3, 1, :])))

# list(zip(State,
#     L1(1)[U.HEAVY, :, :].sum(axis=1)))

print('p(collective) = ', L1(10)[U.HEAVY, :, :].sum(axis=0)[1])
~~~~
{: data-executable="true" data-thebe-executable="true"}

> **Exercise:** Visualize the state prior.

Now, to talk about the weight of a plurality of obejcts, given that we're dealing with scalar adjective semantics, we'll need to create a prior over threshold values. As before, these priors will be uniform over possible object weights. However, given that we can either talk about individual object weights or the weights of collections, we'll need two different threshold priors, one over possible individual object weights and another that scales up to possible collective weights.


> **Exercise:** Visualize the threshold priors.


The final piece of knowledge we need concerns utterances and a meaning function that will let us interpret them. A speaker can use the ambiguous utterances, "The objects are heavy," which receives either a distributive or a collective interpretation. For a slightly higher cost, the speaker can use unambiguous utterances: "The objects each are heavy" (distributive) or "The objects together are heavy" (collective). Lastly, the speaker has the option of saying nothing at all, the cheapest option.


> **Exercise:** Try out the meaning function on some utterances.

This model was designed to account for the possible noise in our estimation of collective properties. For example, when talking about the collective height of a plurality, our estimate of the collective property will depend on the physical arrangement of that property (i.e., how the objects are stacked); a listener might encounter the objects in a different arrangement that the speaker did, introducing noise in the estimation of the collective property. To model this noise, we parameterize the `collectiveInterpretation` so that as noise increases our estimate of the collective property departs from the actual value. The implementation of noise depends crucially on the [error function](https://en.wikipedia.org/wiki/Error_function), which we use to convert the difference between the collective property and the collective threshold into the probability that the collective property exceeds that threshold.



> **Exercise:** Check the predictions for the other values for `collectiveNoise`.

You might have guessed that we are dealing with a lifted-variable variant of RSA: the various interpretation parameters (i.e., `distTheta`, `collTheta`, and `isCollective`) get resolved at the level of the pragmatic listener:


The full model combines all of these ingredients in the RSA framework, with recursive reasoning about the likely state of the world:

> **Exercise:** Generate predictions from the $$S_1$$ speaker.

Finally, we add in a speaker knowledge manipulation: the speaker either has full access to the individual weights in the world state (i.e., `knowledge == true`), or the speaker only has access to the total weight of the world state (i.e., `knowledge == false`). On the basis of this knowledge, the speaker makes an observation of the world state, and generates a belief distribution of the states that could have led to the observation. The full model includes this belief manipulation so that the pragmatic listener takes into account the speaker's knowledge state while interpreting the speaker's utterance.

~~~~
import jax
import jax.numpy as jnp
from memo import memo, domain
from enum import IntEnum
from jax.scipy.special import erf

n_objects = 3
noise =  0.01

# possible object weights
weights = jnp.array([2, 3, 4])

class A(IntEnum):
  FULLOBS = 0
  SUMONLY = 1

class U(IntEnum):
  NULL = 0
  EACHHEAVY = 1
  TOGETHERHEAVY = 2
  HEAVY = 3

class I(IntEnum):
  DISTRIBUTIVE = 0
  COLLECTIVE = 1

w = list(jnp.broadcast_to(weights, (n_objects, len(weights))))
states = jnp.transpose(jnp.array(jnp.meshgrid(*w))).reshape(-1, n_objects)
S = jnp.arange(len(states))

theta_dist = weights - 1
theta_coll = jnp.unique(jnp.sum(states, axis=1)) - 1
TD = jnp.arange(len(theta_dist))
TC = jnp.arange(len(theta_coll))

@jax.jit
def collective_meaning(state, theta):
    return 1 - (0.5 * (1 + erf((theta - jnp.sum(state)) / 
                              (noise * jnp.sqrt(2)))))

@jax.jit
def distributive_meaning(state, theta):
    return jnp.all(state > theta)
    
@jax.jit
def meaning(u, s, dt, ct, i) :
    return jnp.array([
        True,                                            # SILENCE
        distributive_meaning(states[s], theta_dist[dt]), # EACHHEAVY
        collective_meaning(states[s], theta_coll[ct]),   # TOGETHERHEAVY
        jnp.where(i,                                     # HEAVY (ambiguous)
            collective_meaning(states[s], theta_coll[ct]),
            distributive_meaning(states[s], theta_dist[dt])
        ),
    ])[u]

@jax.jit
def cost(u):
    return jnp.array([0, 2, 2, 1])[u]

@jax.jit
def obseq(s1, s2, a):
  return jnp.array([
    s1 == s2,
    jnp.sum(states[s1]) == jnp.sum(states[s2])
  ])[a]
  
@memo
def L0[u: U, td: TD, tc: TC, i: I, s: S]():
    listener: knows(u, td, tc, i)
    listener: given(s in S, wpp=meaning(u, s, td, tc, i))
    return Pr[listener.s == s] + 1e-5

@memo
def S1[o: S, td: TD, tc: TC, i: I, a: A, u: U](alpha): 
    speaker: knows(o, a, td, tc, i)
    speaker: chooses(u in U, wpp=imagine[
      world: knows(o, a),
      world: given(s in S, wpp=obseq(s, o, a)),
      listener: knows(u, td, tc, i),
      listener: guesses(s in S, wpp=L0[u, td, tc, i, s]()),
      exp(alpha * (-KL[world.s | listener.s] - cost(u)))
    ])
    return Pr[speaker.u == u]

@memo
def L1[u: U, a: A, i: I, s: S](alpha) :
    listener: knows(s, a, i)
    listener: thinks[
      speaker: knows(a),
      speaker: given(s in S, wpp= 1),
      speaker: given(td in TD, wpp=1),
      speaker: given(tc in TC, wpp=1),
      speaker: given(i in I, wpp=0.8 if i == {I.COLLECTIVE} else 0.2),
      speaker: chooses(u in U, wpp=S1[s, td, tc, i, a, u](alpha))
    ]
    listener: observes [speaker.u] is u
    return listener[Pr[speaker.s == s, speaker.i == i]]

print('p(collective | sum-only) = ', L1(10)[U.HEAVY, A.SUMONLY, :, :].sum(axis=1)[1])
print('p(collective | full) = ', L1(10)[U.HEAVY, A.FULLOBS, :, :].sum(axis=1)[1])
~~~~
{: data-executable="true" data-thebe-executable="true"}

> **Exercise:**  Add an $$S_2$$ layer to the model.
