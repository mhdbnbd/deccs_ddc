DECCS uses an autoencoder architecture for embedding data into a lower-dimensional space.



```python
class AwA2Dataset(Dataset)
```
custom dataset class AwA2Dataset to load images, labels, attributes, and symbolic tags from the AwA2 dataset.


```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder and decoder definition using convolutional and transposed convolutional layers
    def forward(self, x):
        # Forward pass through encoder and decoder
        return x
```
 The autoencoder has a series of convolutional layers for the encoder and transposed convolutional layers for the decoder.

Encoder:
- Uses Conv2d layers with ReLU activation to reduce the spatial dimensions and increase the feature depth.
- Strides of 2 and padding of 1 ensure that the spatial size is halved at each layer.
Decoder:
- Uses ConvTranspose2d layers with ReLU activation to upsample the feature maps back to the original image dimensions.
- The final layer uses a Sigmoid activation to map the output to the range [0, 1].
Training Parameters
- Loss Function: Mean Squared Error (MSE) is used to measure the reconstruction loss between the input and the output images.
- Optimizer: Adam optimizer with a learning rate of 1e-3, chosen for its efficiency in handling sparse gradients.


```python
class AwA2Dataset(Dataset)
```
the training loop for the autoencoder.


```python
def extract_embeddings(dataloader, model, use_gpu):
    # Setup device, switch model to evaluation mode
    with torch.no_grad():
        for images, _, _, _ in dataloader:
            # Forward pass through encoder to extract embeddings
    return embeddings
```
creating the dataset, training the autoencoder, extracting embeddings, and performing KMeans clustering.





```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # (N, 3, 128, 128) -> (N, 16, 64, 64)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (N, 16, 64, 64) -> (N, 32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (N, 32, 32, 32) -> (N, 64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # (N, 64, 16, 16) -> (N, 128, 8, 8)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # (N, 128, 8, 8) -> (N, 64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # (N, 64, 16, 16) -> (N, 32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # (N, 32, 32, 32) -> (N, 16, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # (N, 16, 64, 64) -> (N, 3, 128, 128)
            nn.Sigmoid(),
        )
```

symbolic tags are loaded alongside images and labels, but they are not actively used in the clustering process. 
They are simply matched with images to ensure each image has corresponding tags. The clustering is performed solely based 
on the embeddings extracted from the autoencoder. 

**A mapping mechanism to align AwA2 attributes with cluster-level explanations**

- (main_tags) One way to integrate symbolic tags is to concatenate them with the embeddings extracted from the autoencoder before
performing clustering. This approach ensures that the clustering algorithm considers both the learned embeddings and the symbolic tags.

- (main_tags2) Another approach is to use symbolic tags to guide the clustering process by enforcing constraints. This can be done by modifying the clustering objective to include a term that penalizes the clustering algorithm for assigning different clusters to data points with similar symbolic tags.
To achieve this, we can implement a custom clustering algorithm or modify the loss function used in the autoencoder 
training to include a term that considers the symbolic tags.