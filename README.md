# Unsupervised-Pseudo-Multi-sense-Elimination

This is part of the code implementation for our submission ``Improving Multi-sense Word Embedding via Unsupervised Pseudo Sense Detection and Elimination'' to IJCNLP2017.

### Python Dependencies
```txt
gensim
argparse
json
torch
```

### Pseudo-multi-sense detection
Usage Example:
```python
from detector import PsdMulDetector
from gensim.models import KeyedVectors

global_model = KeyedVectors.load_word2vec_format(global_model_path)
local_model = KeyedVectors.load_word2vec_format(local_model_path)
detector = PsdMulDetector(global_model, local_model)
pseudo_multi_sense_pairs = detector.detect()
```
Please refer to our paper (will be released after published) for more information about global model and local model.

global model and local model should be stored in a gensim word2vec format for loading.


### Pseudo-multi-sense elimination by transformation matrix
Usage Example:
```bash
model=MSSG
dim=300D
method=compress  # pairwise or compress

python eliminator.py \
    --global-model ../../data/ijcnlp-vectors/wiki_vectors.$model.$dim.6K_global.vec \
    --local-model ../../data/ijcnlp-vectors/wiki_vectors.$model.$dim.6K_local.vec \
    --context-model ../../data/ijcnlp-vectors/wiki_vectors.$model.$dim.6K_context.vec \
    --store-path ../../data/ijcnlp-vectors/wiki_vectors.$model.$dim.6K_local.transform.$method \
    --pseudo-threshold 0.5 \
    --training-method $method
```
For more information, please refer to eliminator.py. 


### Self-Paced Corpus Tagging for Sense Discovery
Usage Example:
```bash
vectortype=wiki_vectors
model=MSSG
dim=50D
method=compress

python corpus_tagger.py \
    --global-model ../../data/ijcnlp-vectors/$vectortype.$model.$dim.6K_global.vec \
    --local-model ../../data/ijcnlp-vectors/$vectortype.$model.$dim.6K_local.vec \
    --context-model ../../data/ijcnlp-vectors/$vectortype.$model.$dim.6K_context.vec \
    --input-corpus ../../data/wiki/wiki.en.corpus \
    --output-corpus ../../data/wiki/wiki.en.corpus.wiki.tagged \
    --tag-threshold 0
```

### Miscellaneous
We slightly modified the code released by Neelakantan et al.(2015) for corpus tagging, and we will push the code after a while.

## References
[1] Neelakantan A, Shankar J, Passos A, et al. Efficient non-parametric estimation of multiple embeddings per word in vector space[J]. arXiv preprint arXiv:1504.06654, 2015.

[2] Shi H, Li C, Hu J. Real Multi-Sense or Pseudo Multi-Sense: An Approach to Improve Word Representation[J]. arXiv preprint arXiv:1701.01574, 2017.
