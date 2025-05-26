---
layout: default
---

# Neuro-Symbolic Models of Language Use: A Computational Approach to Pragmatic Reasoning

**Note: This is a work in progress, being actively developed for a Summer 2025 workshop at the Stanford Social Interaction Lab.**

This course introduces a modern computational framework for understanding language use that bridges formal models of pragmatic reasoning with the rich representational capabilities of neural language models. As large language models (LLMs) demonstrate increasingly sophisticated linguistic behaviors, the need for principled, interpretable models of pragmatic reasoning becomes ever more critical. This course explores how neuro-symbolic approaches can provide both the formal rigor needed to understand the computational principles underlying communication and the flexibility to operate over natural language at scale.

Building on the foundations of the Rational Speech Act (RSA) framework, we develop models that integrate:
- **Symbolic reasoning** about speaker and listener intentions through recursive Bayesian inference
- **Neural representations** that capture real-world priors, contextual nuances, and the full complexity of natural language
- **Probabilistic programming** techniques that make these hybrid models both expressible and computationally tractable

The course employs `memo`, a modern probabilistic programming language designed for recursive reasoning about reasoning, which offers significant improvements in both expressiveness and computational efficiency over traditional approaches.

## Main content

{% assign sorted_pages = site.pages | sort:"name" %}

{% for p in sorted_pages %}
    {% if p.hidden %}
    {% else %}
        {% if p.layout == 'chapter' %}
1. **<a class="chapter-link" href="{{ site.baseurl }}{{ p.url }}">{{ p.title }}</a>**<br>
        <em>{{ p.description }}</em>
        {% endif %}
    {% endif %}
{% endfor %}

## Appendix

{% assign sorted_pages = site.pages | sort:"name" %}

{% for p in sorted_pages %}
    {% if p.hidden %}
    {% else %}
        {% if p.layout == 'appendix' %}
1. **<a class="chapter-link" href="{{ site.baseurl }}{{ p.url }}">{{ p.title }}</a>**<br>
        <em>{{ p.description }}</em>
        {% endif %}
    {% endif %}
{% endfor %}

## Citation

[Robert D. Hawkins]. Neuro-Symbolic Models of Language Use. Retrieved from https://neuroprag.org.

*Based on the original [Probabilistic Language Understanding](https://www.problang.org/) by G. Scontras, M. H. Tessler, and M. Franke.*

## Useful resources

- [Probabilistic Models of Cognition](http://probmods.org): An introduction to computational cognitive science
- [The ProbLang book](https://www.problang.org/): The WebPPL-based predecessor to this course
- [Pragmatic language interpretation as probabilistic inference](http://langcog.stanford.edu/papers_new/goodman-2016-underrev.pdf): A review of the RSA framework
- [memo documentation](https://github.com/kach/memo): The probabilistic programming language used throughout this book

## Acknowledgments

This webbook builds closely on the foundation laid by Scontras, Tessler, and Franke in their Probabilistic Language Understanding course. We are grateful for their pioneering work in making formal pragmatics accessible through probabilistic programming. We also thank Kartik Chandra and the memo development team for creating a tool that makes neuro-symbolic modeling both expressive and efficient.