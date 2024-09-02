ToDo:  
- https://github.com/collinleiber/ClustPy
- apply different clustering methods and consensus
- Data preprocessing : AwA, APY .
- Add DDC Modules and Functions: Cluster-level explanations, Pairwise loss integration.
- Validate model.
- Results evaluatuion and analysis.
- tests ( tag included dataset preprocessing + training)

Ongoing/done:  

- fix luke error (PIL.UnidentifiedImageError: cannot identify image file '/mnt/data/a11850068dm/deccs_ddc/Code/awa2/AwA2-data-sample/JPEGImages/bobcat/bobcat_10077.jpg')
- Include some pseudo-code and/or schema to clarify steps.
- Adapt the Autoencoder to include DDC functionalities: Integrating symbolic tags more tightly into the clustering process by using them as additional features or constraints.
- main_tags (tags in autoencoder)
- main_tags2 (tags in clustering)
- Can train an autoencoder to extract embeddings.
- Can apply KMeans clustering to embeddings.
- Can run on (gpu/luke) (to be tested)
- Update main to include new implementations functionalities.
- can return symbolig tags.
- can jointly learn symbolic attributes.
- Utility Functions for evaluating results.

Done:  

- improve data availability and code portability 
- Utility Functions for loading tags.
- Can Process the AwA2 dataset.
- fix ssh to luke disk error.
- can flexibly in choosing between the full dataset and a sample dataset.
- can extract_embeddings to also extract symbolic tags representations