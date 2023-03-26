# woodscore

## Example
```
from sentence_transformers import SentenceTransformer, util
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from woodscore import Woodscore

model = SentenceTransformer('all-MiniLM-L6-v2')

cats = ['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)

newsgroups_test = fetch_20newsgroups(subset='test',
                                categories=cats)

b = 50

a = 10

predictions = np.zeros((len(newsgroups_test.target),))

wood_score = Woodscore(newsgroups_train.data, newsgroups_test.data,
                                newsgroups_test.target, predictions, model, b, a)
                                
wood_score.compute_metric()
```