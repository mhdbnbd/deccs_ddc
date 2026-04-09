## **Mapping Images to Symbolic Tags in the AwA2 Dataset**

"Or put simply we train models on tags that represent certain classes, we test our models directly on the classse"


The **Animals with Attributes 2 (AwA2) dataset** provides images, class labels, and symbolic attribute tags. The mapping process ensures that each image is correctly paired with its respective symbolic attributes.

### **1️ Mapping Class Names to Class IDs**
The file **`classes.txt`** defines class names and their corresponding IDs (starting from `1`):

1 antelope 2 grizzly+bear 3 killer+whale 4 beaver 5 dalmatian ...

Here, **Beaver is assigned ID `4`**.

### **2️ Mapping Images to Class Labels**
The file **`AwA2-labels.txt`** maps image filenames to **zero-indexed class labels**:

beaver/beaver_10070.jpg 3 beaver/beaver_10087.jpg 3 dolphin/dolphin_10313.jpg 49 ...

Since the labels in `AwA2-labels.txt` are **zero-indexed**, the **Beaver class (ID `4` in `classes.txt`) appears as `3`**.

### **3️ Mapping Class Labels to Symbolic Tag Vectors**
The file **`predicate-matrix-continuous.txt`** contains a **50-dimensional attribute vector** for each class:

Row 3 (Beaver)
-1.00 -1.00 -1.00 -1.00 12.34 0.00 0.00 0.00 16.11 9.19 ...

Row 49 (Dolphin)
39.25 1.39 0.00 74.14 3.75 0.00 0.00 0.00 1.25 0.00 ...

Each row **corresponds to the zero-indexed class label**:
- **Row `3` (Beaver) → Beaver's tag vector**
- **Row `49` (Dolphin) → Dolphin's tag vector**

### **4️ Mapping Attributes to Their Meaning**
The file **`predicates.txt`** defines what each index in the symbolic tag vector represents:

1 black 2 white 3 brown 4 furry ... 50 hibernate

For example, the **Beaver’s vector**:

-1.00 -1.00 -1.00 -1.00 12.34 0.00 0.00 0.00 16.11 9.19 ...

means:
- **"Gray" (attribute 5) = `12.34`** (indicating a grayish color).
- **"Patches" (attribute 9) = `16.11`** (suggesting a spotted pattern).

### **5️ Final Image-to-Tag Mapping**
When loading the dataset, the mapping is performed as:
1. Load **image paths** and **zero-indexed class labels** from `AwA2-labels.txt`.
2. Assign the **correct class ID** (e.g., `3` for Beaver).
3. Retrieve the **symbolic tag vector** for that class from `predicate-matrix-continuous.txt`.
4. Return `(image, symbolic_tag_vector)` for model training.

**This structured mapping ensures that each image is paired with the correct class-specific attributes, enabling supervised attribute-based learning.**
