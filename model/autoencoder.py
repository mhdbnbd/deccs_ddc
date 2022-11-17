import torch
from torch import nn
from utils.train import fit_ae

class FullyConnectedBlock(nn.Module):
    """Feed Forward Neural Network
    Args:
        layers: list of the different layer sizes 
        batch_norm: bool, default=False, set True if you want to use torch.nn.BatchNorm1d
        dropout: float, default=None, set the amount of dropout you want to use.
        activation: activation function from torch.nn, default=None, set the activation function for the hidden layers, if None then it will be linear. 
        bias: bool, default=True, set False if you do not want to use a bias term in the linear layers
        output_fn: activation function from torch.nn, default=None, set the activation function for the last layer, if None then it will be linear. 

    Attributes:
        block: torch.nn.Sequential, feed forward neural network
    """
    def __init__(self, layers, batch_norm=False, dropout=None, activation_fn=None, bias=True, output_fn=None):
        super(FullyConnectedBlock, self).__init__()
        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.bias = bias
        self.activation_fn = activation_fn
        self.output_fn = output_fn

        fc_block_list = []
        for i in range(len(layers)-1):
            fc_block_list.append(nn.Linear(layers[i], layers[i+1], bias=self.bias))
            if self.batch_norm:
                fc_block_list.append(nn.BatchNorm1d(layers[i+1]))
            if self.dropout is not None:
                fc_block_list.append(nn.Dropout(self.dropout))
            if self.activation_fn is not None:
                # last layer is handled differently
                if (i != len(layers)-2):
                    fc_block_list.append(activation_fn())
                else:
                    if self.output_fn is not None:
                        fc_block_list.append(self.output_fn())

        self.block =  nn.Sequential(*fc_block_list)
    
    def forward(self, x):
        return self.block(x)

class Autoencoder(torch.nn.Module):
    """A feedforward symmetric autoencoder.
    
    Args:
        layers: list of the different layer sizes from input to embedding. The decoder is symmetric and goes in the same order from embedding to input.
        batch_norm: bool, default=False, set True if you want to use torch.nn.BatchNorm1d
        dropout: float, default=None, set the amount of dropout you want to use.
        activation: activation function from torch.nn, default=torch.nn.LeakyReLU, set the activation function for the hidden layers, if None then it will be linear. 
        bias: bool, default=True, set False if you do not want to use a bias term in the linear layers
        decoder_output_fn: activation function from torch.nn, default=None, set the activation function for the decoder output layer, if None then it will be linear. 
                           e.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1. 
    Attributes:
        encoder: encoder part of the autoencoder, responsible for embedding data points
        decoder: decoder part of the autoencoder, responsible for reconstructing data points from the embedding    
    """
    def __init__(self, layers, batch_norm=False, dropout=None, activation_fn=torch.nn.LeakyReLU, bias=True, decoder_output_fn=None):
        super(Autoencoder, self).__init__()
        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.bias = bias
        self.decoder_output_fn = decoder_output_fn
        
        self.encoder = FullyConnectedBlock(layers=self.layers, batch_norm=self.batch_norm, dropout=self.dropout, activation_fn=self.activation_fn, bias=self.bias, output_fn=None)
        
        # Inverts the list to make symmetric version of the encoder
        self.decoder = FullyConnectedBlock(layers=self.layers[::-1], batch_norm=self.batch_norm, dropout=self.dropout, activation_fn=self.activation_fn, bias=self.bias, output_fn=self.decoder_output_fn)

    
    def encode(self, x:torch.Tensor)->torch.Tensor:
        """
        Args:
            x: input data point, can also be a mini-batch of points
        
        Returns:
            embedded: the embedded data point with dimensionality embedding_size
        """
        return self.encoder(x)
    
    def decode(self, embedded:torch.Tensor)->torch.Tensor:
        """
        Args:
            embedded: embedded data point, can also be a mini-batch of embedded points
        
        Returns:
            reconstruction: returns the reconstruction of a data point
        """
        return self.decoder(embedded)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """ Applies both encode and decode function. 
        The forward function is automatically called if we call self(x).
        Args:
            x: input data point, can also be a mini-batch of embedded points
        
        Returns:
            reconstruction: returns the reconstruction of a data point
        """
        embedded = self.encode(x)
        reconstruction = self.decode(embedded)
        return reconstruction

    def fit(self, **kwargs):
        fit_ae(self, **kwargs)