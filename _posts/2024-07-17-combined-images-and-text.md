---
layout: post
title: Combining images with text for better CLIP-based image search
date: 2024-07-17 08:57:00-0400
description: Combining images with text for better CLIP-based image search
tags: machine-learning similarity-search retrieval AI CLIP
categories:
giscus_comments: true
related_posts: false
related_publications: true
---

*Nearest neighbor search from image examples is the default approach for image searches, but there are better approaches, such as ExemplarSVM. Additionally, text-based searches using models like [CLIP](https://openai.com/index/clip/) can be much more accurate than example-based searches. How can we get the best of both worlds? I show a starter approach.*

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

One simple method to combine the text and image based approaches is to use a modified SVM.  Linear SVM is a linear model parametrized by a weight vector $$\mathbf{w} $$ and a scalar bias (which is not important here). It is trained by minimizing the hinge-loss function:

$$ \sum_{i=0}^{n} \max(0, 1 - y_i\cdot(\mathbf{w} \cdot \mathbf {x_i} + b)) + \lambda \frac{1}{2}||\mathbf{w}||^2  $$

The initial sum is over model prediction errors. The $$y_i$$, the synthetic labels, are $$+1$$ or $$-1$$. The second term is a penalty on making $$w$$ too large $$ \lambda $$ where $$\lambda$$ is a hyperparameter, which is one way to make prediction errors appear lower.

One simple method to incorporate text information is treating the text query vector $$\mathbf{q}$$ as if it was just another positive example, but this resulted in overall worse accuracy than simply using the text vector alone. Instead, we modify the loss function by adding a special extra term for the text query vector $$\mathbf{q}$$ that encourages preserving a low cosine distance to it:

$$ \lambda_q \left( 1 - \frac{\mathbf{q} \cdot \mathbf {w}}{||\mathbf{q}||\cdot ||\mathbf{w}||}\right) $$

where $$ \lambda_q $$ is a new hyperparamter weighting this term.

The exact functional form of the distance turns out to not be super-relevant, but the idea of tying $$w$$ to the query vector is consistently important in other experiments I've run. I implemented this model with PyTorch to accomodate the custom loss function, and you can read it [in this file](https://github.com/orm011/playground/blob/main/playground/linear_model.py).

In the following experiments, setting  $$ \lambda = 10 $$ and $$\lambda_q = 1000 $$ worked best, and alterting them below an order of magnitude did not make a big difference. I tested the four different methods described so far on a quick benchmark based on the [ObjectNet dataset](https://objectnet.dev/).
ObjecNet includes 50K images assigned into 300 categories, I used each category as a test query.
I picked 10 positive samples at random for each category,  used them as starting vectors $$x$$ for example-based methods and used the remaining images as a test database.
For the exemplar SVM method and the combined method, I additionaly used a sample of 1000 images from the test set as the negative examples, following the steps above. The exact size of this sample set was not critical for the results.

For each query example we compute average precision (AP) scores. The mean AP is the mean over all runs of the experiment (about 3K). Average precision is a good metric to evaluate rankings compared to pure precision or recall because it considers the full ordering of the results.

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
I was suprised that ExemplarSVM only marginally beat image-based nearest neighbor. Perhaps the bigger problem with exemplar SVM in this benchmark is that over the 300 categories, ExemplarSVM was better than kNN on less than 60% of them, which may make it too unpredictable to be worth implementing in practice.
On the other hand, the combined approach works better than the text-based 80% of the time.
For both approaches, more positive examples will probably create a more consistent improvement.

It is possible the ObjectNet dataset makes text-based search appear stronger than it can be in the wild,  because the ObjectNet dataset itself was collected from a pre-specified set of easily stated classes; its contents cluster around 300 concepts, and these concepts by design correspond to objects with well known names.

### Extensions
I develop related ideas more deeply in SeeSaw {% cite seesaw %}, a system that reduces the amount of feedback users need to provide in order to improve their image search results.  SeeSaw tackles this problem by leveraging different kinds of image representation and semi-supervised learning techniques, some show up as loss function modifications like the one above.

The experiments with text based and image based results also suggest text representations may be a better intermediate form than pure examples for some kinds of searches. Hence, now that GPT4V can easily generate captions for images, we may be able to use these as intermdiate search representations without requiring extra human input.

Beyond single lookup searches and simple binary feedback, it would be great to be able to provide a variety of verbal feedback on results, explaining why something is or is not a good result, and improve results that way. Current Visual Language models like GPT4V let you ask questions about input images, but retrieval over your own database of images is not yet an option.