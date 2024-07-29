ToDo:  

- fix ssh to luke disk error.
- apply different clustering methods and consensus
- Adapt the Autoencoder to include DDC functionalities: Integrating symbolic tags more tightly into the clustering process by using them as additional features or constraints.
- Data preprocessing : AwA, APY .
- Add DDC Modules and Functions: Cluster-level explanations, Pairwise loss integration.
- Utility Functions for loading tags, evaluating results.
- Validate model.
-  Include some pseudo-code and/or schema to clarify steps.
- Results evaluatuion and analysis.



Ongoing/done:  

- main_tags (tags in autoencoder)
- main_tags2 (tags in clustering)
- Can Processes the AwA2 dataset.
- Can train an autoencoder to extract embeddings.
- Can apply KMeans clustering to embeddings.
- can flexibly in choosing between the full dataset and a sample dataset.
- Can run on (gpu/luke) (to be tested)
- Update main to include new implementations functionalities.
- can return symbolig tags.
- can jointly learn symbolic attributes.
- can extract_embeddings to also extrac symbolic tags representations
- tests ( tag included dataset preprocessing + training)