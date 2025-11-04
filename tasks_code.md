ToDo:  

- Include some pseudo-code and/or schema to clarify steps.

Ongoing/done(to be reviewed) :

- optimise GPU usage (high prio)
- apply different clustering methods and consensus
- https://github.com/collinleiber/ClustPy/blob/main/clustpy/deep/_data_utils.py#L101
- https://github.com/collinleiber/ClustPy/blob/main/clustpy/data/_utils.py#L99
- https://github.com/collinleiber/ClustPy

Done :
- handleDataLeakage : (done, to be reviewed / validate clusters with labels)
  - attributes in the AwA2Dataset.load_attributes does not stand for the symbolic tags
  - The load_predicates(pred_file) reads symbolic tag vectors (attributes) for each image instead of FOR EACH LABEL
  - SOLUTION : dict map each image with its tags using labels, i.e. image x with label y + label y has tags z = data is image x, tags z (label y is omitted from training)
  - input data : (.jpg, (vector of size 85))
  - Implement explicit train/test splits
- integrate consistency loss 
- Can train an autoencoder to extract embeddings.
- Can apply KMeans clustering to embeddings.
- can jointly learn symbolic attributes.
- Utility Functions for evaluating results.
- Add DDC Modules and Functions: Cluster-level explanations, Pairwise loss integration.
- Merge all pipelines : 
  - main_tags (tags in clustering)
  - main_tags2 (tags in clustering and autoencoder )
- Adapt the Autoencoder to include DDC functionalities: Integrating symbolic tags more tightly into the clustering 
- process by using them as additional features or constraints.
- fix luke error (PIL.UnidentifiedImageError: cannot identify image file/corrupted images.
- improve data availability and code portability 
- Utility Functions for loading tags.
- Can Process the AwA2 dataset.
- fix ssh to luke disk error.
- can flexibly choose between the full dataset and a sample dataset.
- can extract_embeddings to also extract symbolic tags representations
- fix : define additional constrained autoencoder loss
- fix : using combined features for clustering
- fix : sampled data conflicts accross version When versions run simultaneously different data samples are created/conflict.
- detailed logging
- feat : add a fully connected layer for tags in the model
- fix : remove recursive training