## Online latent Dirichlet allocation

This module offers relatively fast implementations of a *trust-region method
for stochastic variational inference* and related algorithms applied to latent Dirichlet
allocation. These are realized in the following models:

* `OnlineLDA` (Hoffman et al., 2010; 2013; Theis & Hoffman, 2015)
* `BatchLDA` (Blei et al., 2013)
* `CumulativeLDA` (Broderick et al., 2013)

Additional features include adaptive learning rates (Ranganath et al., 2013) and automatic tuning
of hyperparameters via empirical Bayes.

### Requirements

* Python >= 2.7.3
* NumPy >= 1.6.1

I have tested the code with the versions above, but older versions might also work.

### Installation

	python setup.py build
	python setup.py install

To test whether the installation worked, you can run the following.

	nosetests code/trlda

### Example

```python
from trlda.models import OnlineLDA
from trlda.utils import load_documents

# create model
model = OnlineLDA(
	num_words=7000,
	num_topics=100,
	num_documents=1000000,
	alpha=.1,
	eta=.2)

# train model for 10 epochs with a batch size of 200
for epoch in range(10):
	for documents in load_documents('data_train.dat', 200):
		model.update_parameters(
			docs=documents,
			max_iter_tr=10,
			max_iter_inference=20,
			kappa=.7,
			tau=100.,
			update_alpha=True,
			update_eta=True)
```
