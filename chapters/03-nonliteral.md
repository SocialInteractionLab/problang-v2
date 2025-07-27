---
layout: chapter
title: Inferring the Question-Under-Discussion
description: "Non-literal language"
---

### Chapter 3: Non-literal language

The models we have so far considered strengthen the literal interpretations of our utterances: from "blue" to "blue circle" and from "some" to "some-but-not-all." Now, we consider what happens when we use utterances that are *literally* false. As we'll see, the strategy of strengthening interpretations by narrowing the set of worlds that our utterances describe will no longer serve to capture our meanings. Our secret ingredient will be the uncertainty conversational participants experience about the topic of conversation: the literal semantics will have different impacts depending on what the conversation is about.

#### Application 1: Hyperbole and the Question Under Discussion

If you hear that someone waited "a million years" for a table at a popular restaurant or paid "a thousand dollars" for a coffee at a hipster hangout, you are unlikely to conclude that the improbable literal meanings are true. Instead, you conclude that the diner waited a long time, or paid an exorbitant amount of money, *and that she is frustrated with the experience*. Whereas blue circles are compatible with the literal meaning of "blue," five-dollar coffees are not compatible with the literal meaning of "a thousand dollars." How, then, do we arrive at sensible interpretations when our words are literally false?

reft:kaoetal2014 propose that we model hyperbole understanding as pragmatic inference. Crucially, they propose that we recognize uncertainty about **communicative goals**: what Question Under Discussion (QUD) a speaker is likely addressing with their utterance. QUDs are modeled as summaries of the full world states, or *projections* of full world states onto the aspect(s) that are relevant for the Question Under Discussion. In the case study of hyperbolic language understanding, reft:kaoetal2014 propose that two aspects of the world are critical: the true state of the world and speakers' attitudes toward the true state of the world (e.g., the valence of their *affect*), which is modeled simply as a binary positive/negative variable (representing whether or not the speaker is upset). 

```python
from memo import memo, domain
import jax
import jax.numpy as jnp
from enum import IntEnum

# Define possible prices and valences
prices = jnp.array([50, 51, 500, 501, 1000, 1001, 5000, 5001, 10000, 10001])
Price = jnp.arange(len(prices))
Utterance = Price  # utterances are price values

class Valence(IntEnum):
  NEGATIVE = 0  # Speaker is upset
  POSITIVE = 1  # Speaker is not upset

# Define the full state space: price × valence
State = domain(
  price=len(Price),
  valence=len(Valence)
)

# Let's explore what a full state looks like
full_state = State.pack(price=9, valence=0)  # price=$10001, negative valence
print(f"Full state encoding: {full_state}")
print(f"Price component: ${prices[State.price(full_state)]}")
print(f"Valence component: {'negative' if State.valence(full_state) == 0 else 'positive'}")
```
{: data-executable="true" data-thebe-executable="true"}

In addition, the authors investigate the *pragmatic halo* effect, by considering a QUD that addresses *approximately* the exact price (`approxPrice`). Here are all of the QUDs we consider.

```python
# Define QUDs as an enumeration
class QUD(IntEnum):
    PRICE = 0
    VALENCE = 1
    PRICE_VALENCE = 2
    APPROX_PRICE = 3
    APPROX_PRICE_VALENCE = 4

# Helper function to round prices to nearest 10
@jax.jit
def approx(price):
    return 10 * jnp.round(price / 10)

@jax.jit
def project(s, qud):
    price_idx = State.price(s)
    valence_idx = State.valence(s)
    
    # Find closest price index for approximate price
    approx_price = approx(prices[price_idx])
    closest_idx = jnp.argmin(jnp.abs(prices - approx_price))
    
    return jnp.array([
        price_idx,                           # PRICE: just the price
        valence_idx,                         # VALENCE: just the valence  
        s,                                   # PRICE_VALENCE: full state
        closest_idx,                         # APPROX_PRICE: rounded price
        State.pack(closest_idx, valence_idx) # APPROX_PRICE_VALENCE: both
    ])[qud]

# Example: different QUD projections of the same state
example_state = State.pack(np.where(prices == 10001)[0], Valence.NEGATIVE)  # $10000, negative valence
print("Full state represents: $10001, negative valence")
print(f"Price QUD answer: price index {prices[project(example_state, QUD.PRICE)]}")
print(f"Valence QUD answer: {project(example_state, QUD.VALENCE)}")
print(f"ApproxPrice QUD answer: price index {prices[project(example_state, QUD.APPROX_PRICE)]}")
```

Accurately modeling world knowledge is key to getting appropriate inferences from the world. Kao et al. achieve this using **prior elicitation**, an empirical methodology for gathering precise quantitative information about interlocutors' relevant world knowledge. They do this to estimate the prior knowledge people carry about the price of an object (in this case, an *electric kettle*), as well as the probability of getting upset (i.e., experiencing a negatively-valenced affect) in response to a given price.

```python
@jax.jit
def state_p(s):
    """Prior probability of state s = (price, valence)"""
    
    # Prior probability of each price (from human experiments)
    price = State.price(s)
    price_p = jnp.array([
        0.4205, 0.3865,  # $50, $51
        0.0533, 0.0538,  # $500, $501
        0.0223, 0.0211,  # $1000, $1001
        0.0112, 0.0111,  # $5000, $5001
        0.0083, 0.0120   # $10000, $10001
    ])[price]   

    # Probability of negative valence given price (from human experiments)
    valence = State.valence(s)
    valence_p = jnp.array([
        [0.3173, 1-0.3173], [0.3173, 1-0.3173],  # $50, $51
        [0.7920, 1-0.7920], [0.7920, 1-0.7920],  # $500, $501
        [0.8933, 1-0.8933], [0.8933, 1-0.8933],  # $1000, $1001
        [0.9524, 1-0.9524], [0.9524, 1-0.9524],  # $5000, $5001
        [0.9864, 1-0.9864], [0.9864, 1-0.9864]   # $10000, $10001
    ])[price, valence]
    
    return price_p * valence_p

# Compute marginals
prior = jnp.zeros(len(State))
for s in range(len(State)):
    prior = prior.at[s].set(state_p(s))

# Reshape to (prices, valences) for easier marginalization
prior_2d = prior.reshape(len(prices), 2)

# Compute marginals
print(prior_2d.sum(axis=1))  # sum over valences
print(prior_2d.sum(axis=0))  # sum over prices
```

> **Exercise:** Use `Infer()` to visualize the joint distribution on price and valence. (Hint: You'll want to run inference over a function that returns an object like the following: `{price: aPrice, valence: aValence}`.)

Putting it all together, the literal listener updates these prior belief distributions by conditioning on the literal meaning of the utterance. 

$$P_{L_{0}}(s \mid u) \propto [\![u]\!](s) $$

This literal listener performs joint inference about the price and the valence in the full state. 
The speaker chooses an utterance to convey a particular answer of the QUD to the literal listener by margianalizing over states that project to the same cell of the partition induced by the QUD:

$$P_{S_{1}}(u \mid s, q) \propto \exp \left( \alpha \ (\log\sum_{s' : q(s') = q(s)} P_{L_{0}}(s' \mid u) - C(u)) \right)$$

To model hyperbole, Kao et al. posited that the pragmatic listener actually has uncertainty about what the QUD is, and jointly infers the price (and speaker valence) and the intended QUD from the utterance he receives. That is, the pragmatic listener simulates how the speaker would behave with various QUDs. (Notice that the code below marginalizes over different QUDs.)

$$P_{L_{1}}(s \mid u) \propto \sum_{q}  P_{S_{1}}(u \mid s, q) \ P(q) \ P(s)  $$

Here is the full model:

```python

@jax.jit
def meaning(u, s):
    return u == State.price(s)

# Cost function - precise numbers are more costly
@jax.jit
def cost(u):
    is_round = (prices[u] == approx(prices[u]))
    return jnp.where(is_round, 0.0, 1.0)  # cost of 1 for precise numbers

@memo
def L0[_u: Utterance, _s: State]():
  listener: knows(_u)
  listener: chooses(s in State, wpp=state_p(s) * meaning(_u, s))
  return Pr[listener.s == _s] + 1e-100

@memo
def S1[_s: State, _qud: QUD, _u: Utterance](alpha): 
  speaker: knows(_s, _qud)
  speaker: thinks[
    world: knows(_s, _qud),
    world: given(s in State, wpp = (
      project(_s, _qud) == project(s, _qud)
    ))
  ]
  speaker: chooses(u in Utterance, wpp = (
    exp(alpha * (log(E[L0[u, world.s]()]) - cost(u)))
  ))
  return Pr[speaker.u == _u]

@memo
def L1[_u: Utterance, _s: State](alpha):
  listener: knows(_s)
  listener: thinks[
    speaker: given(s in State, wpp=state_p(s)),
    speaker: given(qud in QUD, wpp=1),
    speaker: chooses(u in Utterance, wpp=S1[s, qud, u](alpha))
  ]

  listener: observes [speaker.u] is _u
  return listener[Pr[speaker.s == _s]]
  
# Look at marginals 
posterior = L1(1)[np.where(utterance == 10000)[0], :] # u=$10000
posterior_2d = posterior.reshape(len(prices), 2)  
print(posterior_2d.sum(axis=1))  # sum over valences
print(posterior_2d.sum(axis=0))  # sum over prices
```
{: data-executable="true" data-thebe-executable="true"}

> **Exercises:**
> 1. In the second code box, we looked at the joint *prior* distribution over price and valence. Compare that joint distribution with the listener interpretation of "`10000`". What is similar? What is different?
> 2. Try the `pragmaticListener` with the other possible utterances.
> 3. Check the predictions for a speaker who paid 501 and has a negative-valenced affect (i.e., valence is `true`) and wishes to only communicate affect. What are the three most likely utterances such a speaker would choose? (Hint: use `viz.table` to see all options ordered in terms of their probability; a histogram is not informative in this case.)
> 4. Look at the marginal distributions for "price" and "valence" of the pragmatic listener after hearing "`10000`". Do you find these intuitive? If not, how could the model possibly be amended to make it more intuitive?

By capturing the extreme (im)probability of kettle prices, together with the flexibility introduced by shifting communicative goals, the model is able to derive the inference that a speaker who comments on a "$10,000 kettle" likely intends to communicate that the kettle price was upsetting. The model thus captures some of the most flexible uses of language: what we mean when our utterances are literally false.

#### Application 2: Irony

The same machinery---actively reasoning about the QUD---has been used to capture other cases of non-literal language. [Kao and Goodman (2015)](http://cocolab.stanford.edu/papers/KaoEtAl2015-Cogsci.pdf) use this process to model ironic language, utterances whose intended meanings are opposite in polarity to the literal meaning. For example, if we are standing outside on a beautiful day and I tell you the weather is "terrible," you're unlikely to conclude that I intend to be taken literally. Instead, you will probably interpret the utterance ironically and conclude that I intended the opposite of what I uttered, namely that the weather is good and I'm happy about it. The following model implements this reasoning process by formalizing three possible conversational goals: communicating about the true state, communicating about the speaker's valence (i.e., whether they feel positively or negatively toward the state), and communicating about the speaker's arousal (i.e., how strongly they feel about the state).

~~~~
// There are three possible states the weather could be in: 
// terrible, ok, or amazing
var states = ['terrible', 'ok', 'amazing']

// Since we are in California, the prior over these states
// are the following. Once could also imagine this being 
// the prior in a certain context, e.g. when it's clearly
// sunny and nice out.
var statePrior = function() {
  categorical([1, 50, 50], states)
}

// Valence prior defined in terms of negative valence. 
// If the current state is terrible, it's extremely likely
// that the valence associated is negative. If it's ok, then 
// the valence could be negative or positive with equal 
// probability.
var valencePrior = function(state) {
  state === "terrible" ? flip(0.99) ? -1 : 1 :
  state === "ok" ? flip(0.5) ? -1 : 1 :
  state === "amazing" ? flip(0.01) ? -1 : 1 :
  true
}

// Define binary arousals (could model as continuous).
var arousals = ["low", "high"]

// Define goals and goal priors. Could want to communicate state of the world,    
// valence about it, or arousal (intensity of feeling) about it.
var goals = ["goalState", "goalValence", "goalArousal"]

var goalPrior = function() {
  categorical([1, 1, 1], goals)
}

// Assume possible utterances are identical to possible states
var utterances = states

// Assume cost of utterances is uniform.
var utterancePrior = function() {
  uniformDraw(utterances)
}

// Sample arousal given a state.
var arousalPrior = function(state) {
  state === "terrible" ? categorical([0.1, 0.9], arousals) :
  state === "ok" ? categorical([0.9, 0.1], arousals) :
  state === "amazing" ? categorical([0.1, 0.9], arousals) :
  true
}

// Literal interpretation is just whether utterance equals state
var literalInterpretation = function(utterance, state) {
  utterance === state
}

// A speaker's goal is satisfied if the listener infers the correct 
// and relevant information.
var goalState = function(goal, state, valence, arousal) {
  goal === "goalState" ? state :
  goal === "goalValence" ? valence :
  goal === "goalArousal" ? arousal :
  true
}

// Define a literal listener
var literalListener = function(utterance, goal) {
  Infer({model: function(){
    var state = uniformDraw(states)
    var valence = valencePrior(state)
    var arousal = arousalPrior(state)
    condition(literalInterpretation(utterance,state))
    return goalState(goal, state, valence, arousal)
  }})
}

// Define a speaker
var speaker = function(state, valence, arousal, goal) {
  Infer({model: function(){
    var utterance = utterancePrior()
    factor(1 * literalListener(utterance, 
                    goal).score(goalState(goal, 
                                          state, 
                                          valence, 
                                          arousal)))
    return utterance
  }})
}

// Define a pragmatic listener
var pragmaticListener = function(utterance) {
  Infer({model: function(){
    var state = statePrior()
    var valence = valencePrior(state)
    var arousal = arousalPrior(state)
    var goal = goalPrior()
    observe(speaker(state, valence, arousal, goal),utterance)
    return {state, valence, arousal}
  }})
}

viz.table(pragmaticListener("terrible"))

~~~~

#### Application 3: Metaphor

In yet another application, reft:kaoetal2014metaphor use a QUD manipulation to model metaphor, perhaps the most flagrant case of non-literal language use. If I call John a whale, you're unlikely to infer that he's an aquatic mammal. However, you probably will infer that John has qualities characteristic of whales (e.g., size, grace, majesty, etc.). The following model implements this reasoning process by aligning utterances (e.g., "whale", "person") with stereotypical features, then introducing uncertainty about which feature is currently the topic of conversation.


~~~~
// John could either be a whale or a person.
var categories = ["whale", "person"]

// It is extremely unlikely that John is actually a whale.
var categoriesPrior = function() {
  categorical([0.01, 0.99], categories)
}

// The speaker could either say "John is a whale" or "John is a person."
var utterances = ["whale", "person"]

// The utterances are equally costly.
var utterancePrior = function() {
  categorical([1,1], utterances)
}

// The features of John being considered are "large", "graceful",
// "majestic." Features are binary.
var featureSets = [
  {large : 1, graceful : 1, majestic : 1},
  {large : 1, graceful : 1, majestic : 0},
  {large : 1, graceful : 0, majestic : 1},
  {large : 1, graceful : 0, majestic : 0},
  {large : 0, graceful : 1, majestic : 1},
  {large : 0, graceful : 1, majestic : 0},
  {large : 0, graceful : 0, majestic : 1},
  {large : 0, graceful : 0, majestic : 0}
]

// information about feature priors (probabilistic world knowledge)
// obtained by an experimental study (see paper)
var featureSetPrior = function(category) {
  category === "whale" ? categorical([0.30592786494628, 0.138078454222818,
                                      0.179114768847673, 0.13098781834847,
                                      0.0947267162507846, 0.0531420411185539,
                                      0.0601520520596695, 0.0378702842057509],
                                     featureSets) :
  category === "person" ? categorical([0.11687632453038, 0.105787535267869,
                                       0.11568145784997, 0.130847056136141,
                                       0.15288225956497, 0.128098151176801,
                                       0.114694702836614, 0.135132512637255],
                                      featureSets) :
  true
}

// Speaker's possible goals are to communicate feature 1, 2, or 3
var goals = ["large", "graceful", "majestic"]

// Prior probability of speaker's goal is set to uniform but can
// change with context/QUD.
var goalPrior = function() {
  categorical([1,1,1], goals)
}

// Speaker optimality parameter
var alpha = 3

// Check if interpreted category is identical to utterance
var literalInterpretation = function(utterance, category) {
  utterance === category
}

// Check if goal is satisfied
var goalState = function(goal, featureSet) {
  goal === "large" ? featureSet.large :
  goal === "graceful" ? featureSet.graceful :
  goal === "majestic" ? featureSet.majestic :
  true
}

//  Define a literal listener
var literalListener = function(utterance, goal) {
  Infer({model: function() {
    var category = uniformDraw(categories)
    var featureSet = featureSetPrior(category)
    condition(literalInterpretation(utterance, category))
    return goalState(goal, featureSet)
  }})
}         

// Speaker model
var speaker = function(large, graceful, majestic, goal) {
  Infer({model: function() {
    var utterance = utterancePrior()
    factor(alpha *
           literalListener(utterance,goal).score(goalState(goal, {large : large, graceful : graceful, majestic : majestic})))
    return utterance
  }})
}

// Define a pragmatic listener
var pragmaticListener = function(utterance) {
  Infer({model: function() {
    var category = categoriesPrior()
    var featureSet = featureSetPrior(category)
    var large = featureSet.large
    var graceful = featureSet.graceful
    var majestic = featureSet.majestic
    var goal = goalPrior()
    observe(speaker(large, graceful, majestic, goal), utterance)
    return {category, large, graceful, majestic}
  }})
}

viz.table(pragmaticListener("whale"))

~~~~

All of the models we have considered so far operate at the level of full utterances, with conversational participants reasoning about propositions. In the [next chapter](04-ambiguity.html), we begin to look at what it would take to model reasoning about sub-propositional meaning-bearing elements within the RSA framework.
