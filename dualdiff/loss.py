import torch
from torchvision import transforms, models

# Define style weights for different layers
STYLE_WEIGHTS = {
    'relu1_2': 1.0 / 2.6,
    'relu2_2': 1.0 / 4.8,
    'relu3_3': 1.0 / 3.7,
    'relu4_3': 1.0 / 5.6,
    'relu5_3': 10.0 / 1.5
}

def get_features(image, model, layers=None):
    """
    Extract features from specific layers of a model for a given image.
    
    Args:
        image (torch.Tensor): Input image tensor.
        model (torch.nn.Module): Pretrained model (e.g., VGG).
        layers (dict): Mapping of layer indices to layer names.
    
    Returns:
        dict: A dictionary of features for the specified layers.
    """
    if layers is None:
        layers = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3',
            '29': 'relu5_3'
        }
    
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    """
    Compute the Gram matrix for a given tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, depth, height, width).
    
    Returns:
        torch.Tensor: Gram matrix of the input tensor.
    """
    b, d, h, w = tensor.size()
    tensor = tensor.view(b * d, h * w)  # Reshape tensor for matrix multiplication
    gram = torch.mm(tensor, tensor.t())  # Compute Gram matrix
    return gram

def gram_loss(style, target, model):
    """
    Compute the Gram loss (style loss) between a style image and a target image.
    
    Args:
        style (torch.Tensor): Style image tensor.
        target (torch.Tensor): Target image tensor.
        model (torch.nn.Module): Pretrained model (e.g., VGG).
    
    Returns:
        torch.Tensor: The computed Gram loss.
    """
    # Extract features for the style and target images
    style_features = get_features(style, model)
    target_features = get_features(target, model)
    
    # Compute Gram matrices for the style image
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Initialize total loss
    total_loss = 0
    
    # Compute the weighted Gram loss for each layer
    for layer, weight in STYLE_WEIGHTS.items():
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        
        # Compute the layer-specific Gram loss
        _, d, h, w = target_feature.shape
        layer_loss = weight * torch.mean((target_gram - style_gram) ** 2)
        total_loss += layer_loss / (d * h * w)
    
    return total_loss