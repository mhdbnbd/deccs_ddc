ToDo:  

- fix ssh to luke disk error.
- Adapt the Autoencoder to include DDC functionalities.
- Data preprocessing : AwA, APY .
- Add DDC Modules and Functions: Cluster-level explanations, Pairwise loss integration.
- Utility Functions for loading tags, evaluating results.
- validate model.
- results evaluatuion and analysis.



Ongoing/done:  

- Can Processes the AwA2 dataset.
- Can train an autoencoder to extract embeddings.
- Can apply KMeans clustering to embeddings.
- Flexibility in choosing between the full dataset and a sample dataset.
- Can run on (gpu/luke) (to be tested)
- Update main to include new implementations functionalities.
- modify AwA2Dataset class to return symbolig tags.
- modify the autoencoder to jointly learn symbolic attributes.
- modify extract_embeddings to also extrac symbolic tags representations
- tests ( tag included dataset preprocessing + training)