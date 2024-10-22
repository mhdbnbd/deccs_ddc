ToDo:  
-  fix: concatenating the embeddings with the symbolic tags (combined_features) but still using only the embeddings for clustering (kmeans.fit_predict(embeddings)
- log results in output file (loss, ...) : Ensure that logging is used consistently across both scripts. It’s good practice to log the shapes of intermediate outputs, like after extracting embeddings, to verify that the data flows correctly through your pipeline.

- When versions run simultaneously different data samples are created/conflict.
- how many clusters,  = n_labels/silhouette method ? clusters should be > labels (e.g animals with fur, animals that are small/sublabels) n_clusters = n_labels * (n_tags matchingeach label) = count of 1s in the matrix ?
- https://github.com/collinleiber/ClustPy
- apply different clustering methods and consensus
- Data preprocessing : AwA, APY .
- Add DDC Modules and Functions: Cluster-level explanations, Pairwise loss integration.
- Validate model.
- Results evaluatuion and analysis.
- tests ( tag included dataset preprocessing + training)

Ongoing/done:  

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

- fix luke error (PIL.UnidentifiedImageError: cannot identify image file/corrupted images.
- improve data availability and code portability 
- Utility Functions for loading tags.
- Can Process the AwA2 dataset.
- fix ssh to luke disk error.
- can flexibly in choosing between the full dataset and a sample dataset.
- can extract_embeddings to also extract symbolic tags representations