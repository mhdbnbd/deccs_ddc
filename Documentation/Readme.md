
To compile `main.tex` LaTeX document, follow these steps:

### Step-by-Step Instructions

1. **First compile to create the `.aux` file**:
    ```bash
    pdflatex main.tex
    ```

2. **Generate bibliography and cross-references**:
    ```bash
    bibtex main.aux
    ```

3. **Re-compile to include bibliography and update references**:
    ```bash
    pdflatex main.tex
    ```

4. **Final compile to ensure all references are correct**:
    ```bash
    pdflatex main.tex
    ```

### Directory Structure

Ensure the following files are included in your directory:

main.tex
chapters/
introduction.tex
literature_review.tex
methodology.tex
results.tex
discussion.tex
conclusion.tex
appendices/
appendix_a.tex
appendix_b.tex
figs/
Uni_Logo_2016.png
bibliography.bib

### Versions 

**main.py**

Method : processes the AwA2 dataset by creating a sample dataset, training a standard autoencoder, extracting embeddings,
and performing KMeans clustering on the embeddings.
Baseline model, showcasing standard autoencoder-based clustering 
without additional semantic information.

**main_tags.py**

Method : After extracting em beddings, they are concatenated with the symbolic tags associated with each image to form 
combined features. Then KMeans clustering is applied on the combined features (embeddings + symbolic tags). 
The combined features are standardized before clustering to ensure that all features contribute equally to the distance
metrics used in KMeans.

**main_tags.py**

An advanced integration of symbolic tags, not just in clustering but also in training, with an autoencoder 
(train_constrained_autoencoder) that includes an additional loss term that considers the symbolic tags during training. 


### Notes

- Ensure that the paths specified in `\includegraphics` and `\input` commands match the actual paths in your project directory.
- You might need to run the `pdflatex` command multiple times to ensure that all references and citations are updated correctly.

With these steps, you should be able to compile your thesis and generate a PDF output successfully.
