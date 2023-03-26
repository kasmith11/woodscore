# Woodscore

This is an implementation of WoodScore proposed by  Mishra et al. which can found [here](https://arxiv.org/abs/2007.06898). By using Semantic Textual Similarity (STS), each test sample is weighted by the amount of out of distribution (OOD) characteristics that it contains. This weighting is then applied to a metric of choice with the original rationale that in order to preform well on this metric, models must generalize to test cases with higher levels of OOD characteristics.

While this metric is currently only set up for accuracy. It can also be exteneded to other metrics such as Pearson's Correlation Score, BLEU and F1 Score.

## Dependencies

Dependencies are listed within `requirements.txt` and can be installed with ```pip install -r requirements.txt```



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

## Citation
@misc{mishra2020evaluation,
      title={Our Evaluation Metric Needs an Update to Encourage Generalization}, 
      author={Swaroop Mishra and Anjana Arunkumar and Chris Bryan and Chitta Baral},
      year={2020},
      eprint={2007.06898},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}