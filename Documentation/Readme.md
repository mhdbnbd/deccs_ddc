
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


### Notes

- Ensure that the paths specified in `\includegraphics` and `\input` commands match the actual paths in your project directory.
- You might need to run the `pdflatex` command multiple times to ensure that all references and citations are updated correctly.

With these steps, you should be able to compile your thesis and generate a PDF output successfully.

