---
title: Readings on text classification
date: 2018-04-23 12:00 UTC
tags: machine learning, text classification
---
<p>
At my work, I've been actively working on text classification problems. I began with simple Random Forest based model and now switched to using a hierarchical deep neural network for a domain specific problem. Meanwhile, I've been investigating a number of approaches which I've tested empirically and seem to work at large scale. Here are a few papers worth mentioning.
</p>
<br/>
<ul>
<li>Yoon Kim, Convolutional Neural Networks for sentence classification <a href="https://arxiv.org/abs/1408.5882" target="_blank">[arXiv:1408.5882]</a></li>
<li>Yang et. al., Hierarchical Attention Networks for Document Classification <a href="https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf" target="_blank">[CMU link]</a></li>
<li>Liu et. al, Deep Learning for Extreme Multi-label Text Classification <a href="https://dl.acm.org/citation.cfm?id=3080834" target="_blank">[ACM]</a></li>
<li>Johnson et. al., Deep Pyramid Convolutional Neural Networks for Text Categorization <a href="http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf" target="_blank">[Tencent AI]</a></li>
</ul>
<br/>

<p>
Until now, for my case, Kim's CNN approach has been the fastest approach in terms of training for me. This gives approximately the same accuracy as a hierarchical deep neural network that I trained earlier. Probably, the next thing I'm going to try is to add attention mechanism to improve the accuracy of the overall model.
</p>