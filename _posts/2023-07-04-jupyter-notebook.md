---
layout: post
title: Combining examples with text for better image search
date: 2024-06-28 08:57:00-0400
description: Combining examples with text hints to improve image searches
tags: machine-learning similarity-search retrieval AI
categories:
giscus_comments: true
related_posts: false
---

<!-- ## storyline
## looking for similar examples is an important operation for understanding your data, for retrieval augmented applications, etc. In the case of images, one may have an image of an object of interest, and then want to locate similar
## objects.

## previously, one can do better than just looking up nearest neighbors, using svm etc.
## at the same time, avances such as CLIP embeddings offer an alternative 'zero-shot' approach that is often
## better at locating scenes of interest than we can from querying by an example alone.

## we can do better than either method by providing text hints of whats relevant, ie, if we have an example, we can add a leverage both the example and multi-modality by providing a text hint of what seems relevant.

## can we also provide a hint of why something is not relevant? -->

TLDR: Nearest neighbor search from image examples is the default approach for image searches, but better approaches exist, and in the era of CLIP, combining image examples with text hints is easily possible and improves your results.

Semantic search is fundamental when working with large image datasets, common for developing vision based applications. Many data management products for AI such as [scale.ai nucleus](https://nucleus.scale.com/docs/find-similar-rare-samples) and others offer search features for this reason. The default approach to image search is to represent images as vectors, and vectors geometrically near each other correspond to semantically similar images.  When all vectors are of unit length, the vector dot product is a common similarity metric. We find similar images, given an example, by locating those with correspondingly high inner products.  Surprisingly, we can do better than that if all we have is a single example: Andrej Karpathy pointed out it is possible to improve results by not only using the image example, but instead using it to compute a slightly different vector

* Use the image example, $x_+$, as a positive example in a training set we will construct as follows.
* Pick elements at random from the database, and assign them a negative label, even if technically their label is unknown and we are mislabeling a few.
* Fit a model with linear weights, $w$,  such as a linear SVM, known as an "Exemplar SVM" on this synthetic training set.
* Now use the learned weight $w$ as the key for nearest neighbor lookup, instead of the original $x_+$.

The quality of the vector $w$ lookup turns out to be consistently better than only using the original vector $x_+$, all without any extra labels or human input, which I personally think is a really neat minimal working example of semi-supervised learning.   The reasoning of why this approach may help goes like this: while $x_+$ is itself a good lookup vector,  that would correctly classify itself as positive, the vector representation may include many superfluous details when compared to other vectors in our specific dataset. Consider the objectnet dataset: if every image is a combination of hand holding an object, and our example is that of a hand holding an apple, this vector will be close to both those an apple, or those with a hand holding other objects. However we want apples, not hands, thats the outstanding difference, and a classifier can pick this up from comparing our example vector to those in the rest of the database.  Nearest neighbor may produce high scores for an image of a hand holding a pear and one of a hand holding an apple simply because both have hands in them. When used as negative examples, though, we can learn to discount the part to do with hands and emphasize the one to do with apples.

<!-- This technique, ExemplarSVM (though also works with logistic regression) should be more widely known. In my own personal experience in working on systems for retrieval, I also found this and related techniques helpful and the results were robust across a number of searches and datasets. -->

<!-- A successful and very approachable use-case of "classic" semi-supervised learning, where we have labeled data (our example) as well as unlabeled database data, and we actually do better combining  Andrej Karpathy has also mentioned this strategy as being used in Tesla's internal tools for dealing with data operations, and also tweeted and wrote a small [notebook demo](https://github.com/karpathy/randomfun/commit/ae0363ada947b56e5484e2e4e755d2ec468c9687). -->

In the era of CLIP embeddings, it is possible to leverage a similar insight but from a multi-modal perspective: when a positive example is found, a textual description of what is relevant in it can help emphasize aspects of the image that matter. Models like CLIP are not only good at producing meaningful vector representations from images, but they are also very good at matching semantic information in text with that of the images. Instead of an image input, you can provide a textual description of what you are looking for (you can try both types of search in this [huggingface space](https://huggingface.co/spaces/vivien/clip))

I took the objectnet dataset, and computed CLIP embeddings for all (approx) 50k images. ObjectNet contains images partitioned into around 300 categories.  The results, in short show that exemplar SVM does indeed helps improve results when compared to nearest neighbor, but textual descriptions attain way higher accuracy without even a single example. Note image similarity searches are still useful when describing what we want is not necessarily easy, however measuring that is much harder, and this results sugggests textual descriptions may be a better starting point.

There is a way to get the best of both worlds: combining both the exemplar SVM with an extra loss term to stick close to a text query (a text hint in this case). The exact details of the loss function are not super relevant, I've seen different variations.

The results suggest that we can improve our searches by combining both examples, and text descriptions. Ideally, we would want to both provide examples, counter-examples, and text descriptions of what we want, as well as text descriptions of what we don't want, or text descriptions of why something is not relevant.

<!-- {::nomarkdown}
{% assign jupyter_path = "assets/jupyter/blog.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/blog.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown} -->