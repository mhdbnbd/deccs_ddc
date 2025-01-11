ToDo:  

-    #add NMI
- https://github.com/collinleiber/ClustPy/blob/main/clustpy/deep/_data_utils.py#L101
- https://github.com/collinleiber/ClustPy/blob/main/clustpy/data/_utils.py#L99
- https://github.com/collinleiber/ClustPy
- apply different clustering methods and consensus
- Validate model.
- Results evaluatuion and analysis.
- tests ( tag included dataset preprocessing + training)
- Include some pseudo-code and/or schema to clarify steps.

Ongoing/done(to be reviewed) :

- Can run on (no? gpu/ yes luke)
- Adapt the Autoencoder to include DDC functionalities: Integrating symbolic tags more tightly into the clustering process by using them as additional features or constraints.
- Add DDC Modules and Functions: Cluster-level explanations, Pairwise loss integration.
- main_tags (tags in clustering)
- main_tags2 (tags in clustering and autoencoder )
- Can train an autoencoder to extract embeddings.
- Can apply KMeans clustering to embeddings.
- can jointly learn symbolic attributes.
- Utility Functions for evaluating results.

Done :

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