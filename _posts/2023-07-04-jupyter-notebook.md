---
layout: post
title: Combining images with text for better CLIP-based image search
date: 2024-06-28 08:57:00-0400
description: Combining images with text hints to improve image searches
tags: machine-learning similarity-search retrieval AI
categories:
giscus_comments: true
related_posts: false
---

*TLDR: Nearest neighbor search from image examples is the default approach for image searches, but better approaches exist. In the era of CLIP, text-based searches can be much more accurate than example-based searches. We can get the best of both worlds by combining image examples with text hints, and I show a simple way that works. As an aside, I found exemplar SVM was not as effective as I expected vs. simple NN lookup*

Semantic search is a key building block for working with image datasets.
The most direct way to implement it is via nearest neighbor vector search: vectors mathematically nearest to the lookup example are returned
The lookups can be fast because of vector indexing.

### Exemplar SVM

A more sophisticated approach is ExemplarSVM, explained by Andrej Karpathy, which I summarize:
1. Use the input image example, $$x_+$$, as a positive example in a training set.
2. Create negaive examples by sampling database elements at random, even if we may be mislabeling some.
3. Use this training set to fit a model with linear weights $$w$$, such as an SVM (support vector machine) model.
4. Use the learned weight $$w$$ as the key for nearest neighbor lookup instead of the original $$x_+$$.

This learned vector $$w$$ produces better quality results than the original vector $$x_+$$, without any extra labels or human input, which I personally think is a neat "hello world" example for old-school semi-supervised learning.

### Text based searches
Alternatively to starting searches with an image, dual image text models like CLIP make it possible to use text descriptions to start image searches.

CLIP is a very strong model that works very well for text-base search.

Often times I see demos where one can use a text search, and once examples are found, one can use that example for more lookups ([here's one example](https://huggingface.co/spaces/vivien/clip)),  but it is possible to combine both modalities: ie, provide an image as well as text describing the search, something I have not seen discussed.

The goal of this post is to quickly compare how well these approaches work on a test dataset, and show a simple modification that combines both example based and text based approaches.

### Combining both approaches

In particular, I found three surprising behaviors in this benchmark using CLIP embeddings:
1. Exemplar SVM was only marginally better than kNN (unexpected, I guess it takes away from being a good "hello world" example after all)
2. Text based search much better than example based search (the gap in accuracy was unexpected)
3. Combining them was clearly better than text search alone (unexpected given point 1)

One simple way to combine both approaches is to modify the SVM loss function
$$ \lambda \frac{1}{2}||w||^2 +  \sum_{i=0}^{n} \max(0, 1 - y_i\cdot(\mathbf{w} \cdot \mathbf {x_i} + b)) $$

to include a term for the query vector $$q$$ that encourages preserving a low cosine distance to the text vector.
$$ \lambda_q \cdot \left(1. - \frac{\mathbf{q} \cdot \mathbf {w}}{||\mathbf{q}||\cdot ||\mathbf{w}||}\right) $$

$$ \lambda = 10 $$ and $$ \lambda_q = 1000 $$ worked well.

I should highlight that using the text vector as just another example does not work well at all: I was better off sticking to text alone.
In my experience the cosine similarity betweeen text and image embeddings (from CLIP) are often low, in the .3-.4 range even when text and images visibly correspond to each other.
Text and image vectors really inhabit different regions of the vector space even while corresponding semantically, which may be part of the reason we cannot treat them the same way.

I tested these methods with a quick benchmark using the ObjectNet dataset, which includes 50k images, and 300 categories.
Each category has between 100 and 200 samples within the dataset.
I picked 10 samples from each category at random, using them as starting examples for example-based methods and used the remaining images were used as the test database.
The ground truth of the dataset can be used to compute average precision scores (higher is better) for each query example.
Average precision is a good way to evaluate rankings compared to pure precision or recall.
For exemplar SVM I used a sample of 1000 images from the test set to be used as the negative examples (varying this size did not vary results much, neither did varying the C regularization parameter for the SVM, where C=.3 was used)

The code is in this github repo

<style>
table {
  width: 100%;
  border-collapse: collapse;
}
/* th {
	border: 1px solid;
} */
</style>


| Search Method | Mean Average Precision (mAP) |
| :------------: | :--------------------------: |
| Exemplar SVM  |  0.099                       |
| Text          |   0.237                      |
| Combined example + text     | **0.251**                    |


### Where to go from here

Caption generating models provide a simple way to augment example based searches, so these kinds of text based hints may be possible to
add without requiring user input.

There is a possible bias in the dataset setup text base search: the dataset itself was deliberatly collected from a specified set of classes decided ahead of time,
This may mean that searches fit simple text descriptions by construction, making text searches look more useful than would be eg in a dataset collected in the wild.


and providing negative feedback on results should also be possible.
Ideally, we would want to both provide examples, counter-examples, and text descriptions of why something is incorrect.