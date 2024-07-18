---
layout: post
title: Combining images with text for better CLIP-based image search
date: 2024-07-17 08:57:00-0400
description: Combining images with text for better CLIP-based image search
tags: machine-learning similarity-search retrieval AI
categories:
giscus_comments: false
related_posts: false
related_publications: true

---

*TLDR: Nearest neighbor search from image examples is the default approach for image searches, but there are better approaches. Text-based searches using models like [CLIP](https://openai.com/index/clip/) can be much more accurate than example-based searches. We can combine the power of image examples with text hints, and I show a simple method to do it.*

Semantic search is a key building block for working with image datasets. You need it when selecting images with specific attributes for labeling, when seeking to understand corner cases qualitatively, when profiling model errors, when seeking to extend datasets based on specific needs, or when cleaning your data.  In all these cases you often need to locate relevant examples in your data, either based on similarity to some input image, or based on a description. Products like [Scale.ai Nucleus](https://nucleus.scale.com/docs/getting-started) and [Lilac.ml](https://docs.lilacml.com/datasets/dataset_explore.html#keyword-search) for example, offer [search](https://nucleus.scale.com/docs/basic-similarity-search) as a key feature.

The most direct way to implement these searches is via nearest neighbor vector search: vectors mathematically nearest to the vector representing the image example are returned as search results.
More sophisticated methods include Exemplar SVM and text-based searches. All these methods all eventually reduce to a nearest neighbor lookup and the differences lie on what vector is used.

### Exemplar SVM

ExemplarSVM is one such method, defined in [this paper](https://icml.cc/2012/papers/946.pdf) and highlighted by Andrej Karpathy in this [tweet](https://x.com/karpathy/status/1647025230546886658). It works as follows:
1. Use the input image example, $$\mathbf{x}$$, as a positive example in a training set.
2. Create negaive examples by sampling some database elements at random, even if we may be mislabeling some.
3. Use this training set to fit a model with linear weights $$\mathbf{w}$$, such as an SVM (support vector machine) model.
4. Use the learned weight $$\mathbf{w}$$ as the key for nearest neighbor lookup instead of the original $$\mathbf{x}$$.

The vector $$\mathbf{w}$$ produces (hopefully) better quality results than the original vector $$\mathbf{x}$$ without any extra labels or human input; this is a neat hello-world example for old-school semi-supervised learning.

### Text-based searches
Instead of using an image, we can also use CLIP to start searches with a text description.
CLIP maps both images and text to a vector space where semantically similar text and images have higher dot products, so one can use either to start a nearest neighbor lookup.
In practice CLIP works very well for text-based search;
the [CLIP paper](https://arxiv.org/pdf/2103.00020) shows CLIP zero-shot searches (ie. using text-based vectors) consistently deliver higher accuracy for classification than one-shot classifiers generated from an image example.

If you havent tried CLIP yourself, there are a few good web-demos of CLIP-powered image searches like [this one](https://huggingface.co/spaces/vivien/clip). The demo lets you use both text-based and image-based searches; you can use a search bar for text or click on images to find similar ones. As far as I can tell, it does not combine these modalities.

### Comparing these approaches

I compare how well these approaches work on a test dataset, and show a simple modification that combines both example-bsed and text-based approaches. I explain the benchmark details below, but the salient points of the benchmark results are the following:

1. Exemplar SVM was only marginally better than kNN (unexpected)
2. Text based search much better than example based search (the big gap in accuracy was unexpected, but the overall observation is consistent with the CLIP paper)
3. Combining both modalities was clearly better than text search alone (unexpected given point 1)

### Combining these approaches

One simple way to combine the text and image based approaches is to modify the SVM loss function

$$ \lambda \frac{1}{2}||\mathbf{w}||^2 +  \sum_{i=0}^{n} \max(0, 1 - y_i\cdot(\mathbf{w} \cdot \mathbf {x_i} + b))$$

by adding an extra term for the text query vector $$\mathbf{q}$$ that encourages preserving a low cosine distance to it:

$$ \lambda_q \cdot \left(1. - \frac{\mathbf{q} \cdot \mathbf {w}}{||\mathbf{q}||\cdot ||\mathbf{w}||}\right) $$

The text vector $$\mathbf{q}$$ needs to be handled differently from the image vectors $$\mathbf{x}$$;  for example, treating text vectors as if they were example images resulted in overall worse results than simply using the text vector alone.

I implemented this model with PyTorch to accomodate the custom loss function, and you can read it [in this file](https://github.com/orm011/playground/blob/main/playground/linear_model.py).

In the following experiments, setting  $$ \lambda = 10 $$ and $$\lambda_q = 1000 $$ worked well, and changing them less than an order of magnitude did not make a huge difference.
I tested the four different methods described so far on a quick benchmark based on the [ObjectNet dataset](https://objectnet.dev/).
ObjecNet includes 50K images assigned into 300 categories, I used each category as a test query.
I picked 10 positive samples at random for each category,  used them as starting vectors $$x$$ for example-based methods and used the remaining images as a test database.
For the exemplar SVM method and the combined method, I additionaly used a sample of 1000 images from the test set as the negative examples, following the steps above. The exact size of this sample set was not critical for the results.

For each query example we can compute average precision (AP) scores. The mean AP is the mean over all runs of the experiment. Average precision is a good way to evaluate rankings compared to pure precision or recall because it considers the full ordering of the results.

The code, data, and benchmarks are available in [this notebook and repo](https://github.com/orm011/playground/blob/main/svm_text_exp.ipynb), and I copy the results below from the notebook.

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
| Image-based Nearest Neighbor |  0.094 |
| Exemplar SVM  |  0.099                       |
| Text-based Nearest Neighbor          |   0.237                      |
| Combined exemplar SVM + text     | **0.251**                    |

<br>
It is surprising ExemplarSVM was only marginally better to image nearest neighbor.
Perhaps the bigger problem with exemplar SVM in this benchmark is that over the 300 categories, ExemplarSVM was better than kNN on about 58% of them, which may make it too unpredictable to be worth implementing in practice.
On the other hand, the combined approach works better than the text-based 80% of the time.
While not tested, more positive examples probably create a more consistent improvement.

It is possible the ObjectNet dataset makes text-based search appear stronger than it can be in the wild,  because the ObjectNet dataset itself was collected from a pre-specified set of easily stated classes; its contents cluster around 300 concepts, and these concepts by design correspond to objects with well known names.

### Extensions
I develop related ideas more deeply in {% cite seesaw %}, where the goal is not just evaluating one example but continuosly adapting during the search as a way to find images with less effort.

That said, there are a few cool things I'd love to explore more and if you know of good work, demos etc. in this area let me know in the comments.

*Caption generation:* integrating caption-generating models to augment example based searches with text descriptions transparently from the user. ChatGPT4v can easily generate captions for images, which could then be used.

*End-to-end retrieval models:* training an embedding model end-to-end to generate lookup vectors based on both images and text may result in better lookups. A few in-context image generation and editing models implicilty already do something close to this, I just haven't seen it used for retrieval.

*Conversational retrieval:* it would be great to provide a variety of negative feedback on results, explaining why something is not a good result.

