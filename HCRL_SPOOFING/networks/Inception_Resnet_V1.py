#!/usr/bin/env python3
"""
Reduced Inception-ResNet V1 Model for Vehicle CAN Network Intrusion Detection

This implementation creates a lightweight version of Inception-ResNet V1 architecture
specifically optimized for processing 29x29 binary CAN frame matrices. The model combines:
- Inception modules for multi-scale feature extraction
- Residual connections for gradient flow and training stability
- Aggressive dimensionality reduction for computational efficiency
- Binary classification for normal vs attack traffic detection

Architecture Flow:
Input (29x29x1) → Stem → Inception-ResNet-A → Reduction-A → 
Inception-ResNet-B → Reduction-B → Global Pooling → Dense → Output (2 classes)

Key optimizations for CAN data:
- Reduced depth compared to standard Inception-ResNet
- Optimized for small input dimensions (29x29)
- Binary classification head for intrusion detection
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, AveragePooling2D,
                                     Concatenate, Add, Flatten, Dropout, Dense, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

###################################################
# Custom Training Callback for Batch-Level Monitoring
###################################################
class BatchLossHistory(Callback):
    """
    Custom Keras callback to record training loss at every batch iteration.
    
    This provides more granular monitoring than epoch-level tracking, allowing
    for detailed analysis of training dynamics and convergence behavior.
    Particularly useful for genetic algorithm experiments that need to track
    training progress over iterations rather than epochs.
    """
    
    def on_train_begin(self, logs=None):
        """
        Initialize tracking variables at the start of training.
        
        Args:
            logs: Training logs dictionary (unused but required by Keras)
        """
        self.batch_losses = []  # List to store (iteration, loss) tuples
        self.iterations = 0     # Counter for total training iterations
    
    def on_batch_end(self, batch, logs=None):
        """
        Record loss value after each training batch completes.
        
        Args:
            batch: Current batch number within the epoch
            logs: Dictionary containing batch metrics (loss, accuracy, etc.)
        """
        self.iterations += 1  # Increment global iteration counter
        # Store iteration number and corresponding loss value
        self.batch_losses.append((self.iterations, logs.get('loss')))

###################################################
# Stem Block: Initial Feature Extraction
###################################################
def stem_block(inputs):
    """
    Stem block for initial feature extraction from 29x29 CAN frame inputs.
    
    This block performs aggressive early feature extraction and dimensionality reduction:
    1. Extracts low-level features with small convolutions
    2. Reduces spatial dimensions while increasing channel depth
    3. Prepares features for subsequent Inception-ResNet blocks
    
    Architecture:
    - Conv2D(64, 3x3, valid) → 29x29x1 → 27x27x64
    - Conv2D(64, 3x3, same) → 27x27x64 → 27x27x64  
    - MaxPool2D(2x2, stride=2) → 27x27x64 → 13x13x64
    - Conv2D(128, 1x1, same) → 13x13x64 → 13x13x128
    
    Args:
        inputs: Input tensor of shape (batch_size, 29, 29, 1)
        
    Returns:
        Tensor of shape (batch_size, 13, 13, 128)
    """
    # First convolution with valid padding reduces spatial dimensions
    # 29x29x1 → 27x27x64 (removes 2 pixels due to valid padding)
    x = Conv2D(64, (3, 3), strides=1, padding='valid', activation='relu')(inputs)
    
    # Second convolution with same padding preserves spatial dimensions
    # 27x27x64 → 27x27x64 (maintains size, extracts more complex features)
    x = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(x)
    
    # Max pooling for spatial downsampling (critical for computational efficiency)
    # 27x27x64 → 13x13x64 (roughly halves spatial dimensions)
    x = MaxPooling2D((2, 2), strides=2, padding='valid')(x)
    
    # 1x1 convolution to increase channel depth without affecting spatial dimensions
    # 13x13x64 → 13x13x128 (doubles channel depth for richer feature representation)
    x = Conv2D(128, (1, 1), strides=1, padding='same', activation='relu')(x)
    
    return x

###################################################
# Inception-ResNet Block A: Multi-Scale Feature Extraction
###################################################
def inception_resnet_a_block(x, scale=0.1):
    """
    Inception-ResNet-A block combining multi-scale convolutions with residual connections.
    
    This block performs parallel convolutions at different scales to capture features
    at multiple receptive field sizes, then combines them with a residual connection
    for improved gradient flow and training stability.
    
    Architecture branches:
    - Branch 0: 1x1 conv (32 filters) → point-wise features
    - Branch 1: 1x1 conv → 3x3 conv (32 filters) → local spatial features  
    - Branch 2: 1x1 conv → 3x3 conv → 3x3 conv (64 filters) → larger spatial features
    
    The residual connection adds the scaled combined branches back to the input,
    enabling the network to learn incremental improvements to existing features.
    
    Args:
        x: Input tensor of shape (batch_size, height, width, channels)
        scale: Scaling factor for residual connection (0.1 for training stability)
        
    Returns:
        Tensor with same spatial dimensions but potentially different channel depth
    """
    # Branch 0: 1x1 convolution for point-wise feature extraction
    # Captures channel-wise interactions without spatial aggregation
    branch_0 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    
    # Branch 1: 1x1 → 3x3 convolution chain for local spatial features
    # 1x1 reduces channels, 3x3 captures local spatial patterns
    branch_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    branch_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch_1)
    
    # Branch 2: 1x1 → 3x3 → 3x3 convolution chain for larger receptive field
    # Sequential 3x3 convolutions effectively create a 5x5 receptive field
    # More efficient than direct 5x5 convolution
    branch_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    branch_2 = Conv2D(48, (3, 3), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch_2)
    
    # Concatenate all branches along channel dimension
    # Total channels: 32 + 32 + 64 = 128
    merged = Concatenate(axis=-1)([branch_0, branch_1, branch_2])
    
    # 1x1 convolution to match input channel dimensions for residual addition
    # This projection layer ensures dimensional compatibility
    up = Conv2D(tf.keras.backend.int_shape(x)[-1], (1, 1), padding='same')(merged)
    
    # Scale the residual branch for training stability
    # Scaling factor (0.1) prevents residual branch from dominating early in training
    up = Lambda(lambda s: s * scale)(up)
    
    # Residual connection: add scaled features to input
    # This enables gradient flow and allows learning of incremental improvements
    x = Add()([x, up])
    
    # Apply activation after residual addition
    # ReLU activation introduces non-linearity after feature combination
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

###################################################
# Reduction Block A: Spatial Downsampling with Feature Expansion
###################################################
def reduction_a_block(x):
    """
    Reduction-A block for spatial downsampling while expanding channel depth.
    
    This block reduces spatial dimensions (width/height) while increasing the number
    of feature channels. Multiple parallel branches ensure that information is
    preserved during downsampling through different aggregation strategies.
    
    Architecture branches:
    - Branch 0: Max pooling → preserves dominant features
    - Branch 1: Direct 3x3 conv with stride=2 → learned downsampling
    - Branch 2: 1x1 → 3x3 → 3x3 conv chain → complex feature extraction before downsampling
    
    Args:
        x: Input tensor (typically 13x13x128 from stem block)
        
    Returns:
        Tensor with reduced spatial dimensions and increased channels (6x6x448)
    """
    # Branch 0: Max pooling for dominant feature preservation
    # Stride=2 reduces spatial dimensions by half: 13x13 → 6x6
    # Preserves existing channel depth (128)
    branch_0 = MaxPooling2D((3, 3), strides=2, padding='valid')(x)
    
    # Branch 1: Direct convolution with stride=2 for learned downsampling
    # Simultaneously reduces spatial dimensions and extracts new features
    # 13x13x128 → 6x6x160
    branch_1 = Conv2D(160, (3, 3), strides=2, padding='valid', activation='relu')(x)
    
    # Branch 2: Multi-stage convolution chain for complex feature extraction
    # 1x1 conv reduces channels for computational efficiency
    branch_2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    # 3x3 conv with same padding maintains spatial dimensions
    branch_2 = Conv2D(160, (3, 3), strides=1, padding='same', activation='relu')(branch_2)
    # Final 3x3 conv with stride=2 for downsampling: 13x13 → 6x6
    branch_2 = Conv2D(160, (3, 3), strides=2, padding='valid', activation='relu')(branch_2)
    
    # Concatenate all branches along channel dimension
    # Total channels: 128 (branch_0) + 160 (branch_1) + 160 (branch_2) = 448
    x = Concatenate(axis=-1)([branch_0, branch_1, branch_2])
    
    return x

###################################################
# Inception-ResNet Block B: High-Level Feature Processing
###################################################
def inception_resnet_b_block(x, scale=0.1):
    """
    Inception-ResNet-B block for high-level feature extraction with asymmetric convolutions.
    
    This block operates on higher-level features (post-reduction) and uses asymmetric
    convolutions (1x7 and 7x1) to capture elongated patterns efficiently. The asymmetric
    approach is more parameter-efficient than square convolutions for certain patterns.
    
    Architecture branches:
    - Branch 0: 1x1 conv (192 filters) → channel-wise feature extraction
    - Branch 1: 1x1 → 1x7 → 7x1 conv chain → asymmetric spatial feature extraction
    
    Args:
        x: Input tensor of shape (batch_size, height, width, 448)
        scale: Scaling factor for residual connection (0.1 for stability)
        
    Returns:
        Tensor with same spatial dimensions and channel depth
    """
    # Branch 0: Simple 1x1 convolution for channel-wise feature transformation
    # Captures cross-channel interactions without spatial aggregation
    branch_0 = Conv2D(192, (1, 1), padding='same', activation='relu')(x)
    
    # Branch 1: Asymmetric convolution sequence for efficient spatial feature extraction
    # 1x1 convolution for dimensionality reduction
    branch_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    # 1x7 convolution captures horizontal patterns
    branch_1 = Conv2D(160, (1, 7), padding='same', activation='relu')(branch_1)
    # 7x1 convolution captures vertical patterns
    # This asymmetric approach is more efficient than 7x7 convolution
    branch_1 = Conv2D(192, (7, 1), padding='same', activation='relu')(branch_1)
    
    # Concatenate branches along channel dimension
    # Total channels: 192 + 192 = 384
    merged = Concatenate(axis=-1)([branch_0, branch_1])
    
    # 1x1 projection to match input channel dimensions for residual connection
    up = Conv2D(tf.keras.backend.int_shape(x)[-1], (1, 1), padding='same')(merged)
    
    # Apply scaling to residual branch for training stability
    up = Lambda(lambda s: s * scale)(up)
    
    # Residual connection: add scaled features to input
    x = Add()([x, up])
    
    # Apply activation after residual addition
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

###################################################
# Reduction Block B: Final Spatial Downsampling
###################################################
def reduction_b_block(x):
    """
    Reduction-B block for final spatial downsampling before global pooling.
    
    This block performs the final spatial reduction while dramatically increasing
    channel depth. It prepares features for global pooling by creating a very
    high-dimensional but spatially compact representation.
    
    Architecture branches:
    - Branch 0: Max pooling → preserves dominant features
    - Branch 1: Direct 3x3 conv with stride=2 → learned aggressive feature extraction
    
    Args:
        x: Input tensor of shape (batch_size, 6, 6, 448)
        
    Returns:
        Tensor of shape (batch_size, 2, 2, 896)
    """
    # Branch 0: Max pooling preserves strongest activations
    # 6x6x448 → 2x2x448
    branch_0 = MaxPooling2D((3, 3), strides=2, padding='valid')(x)
    
    # Branch 1: Aggressive feature extraction with large channel expansion
    # 6x6x448 → 2x2x448 (maintains input channel depth)
    # High channel count captures complex high-level patterns
    branch_1 = Conv2D(448, (3, 3), strides=2, padding='valid', activation='relu')(x)
    
    # Concatenate branches for maximum feature preservation
    # Total channels: 448 + 448 = 896
    x = Concatenate(axis=-1)([branch_0, branch_1])
    
    return x

###################################################
# Main Model Architecture Builder
###################################################
def build_reduced_inception_resnet(input_shape=(29, 29, 1), num_classes=2, dropout_rate=0.2):
    """
    Build the complete reduced Inception-ResNet model for CAN intrusion detection.
    
    This function assembles all components into a complete neural network optimized
    for binary classification of CAN network traffic (normal vs attack).
    
    Architecture Summary:
    1. Stem Block: 29x29x1 → 13x13x128 (initial feature extraction + reduction)
    2. Inception-ResNet-A: 13x13x128 → 13x13x128 (multi-scale feature extraction)
    3. Reduction-A: 13x13x128 → 6x6x448 (spatial reduction + channel expansion)
    4. Inception-ResNet-B: 6x6x448 → 6x6x448 (high-level asymmetric features)
    5. Reduction-B: 6x6x448 → 2x2x896 (final spatial reduction)
    6. Global Average Pooling: 2x2x896 → 1x1x896 (spatial aggregation)
    7. Classification Head: 896 → 2 (binary classification)
    
    Args:
        input_shape: Shape of input CAN frames (default: 29x29x1)
        num_classes: Number of output classes (default: 2 for binary classification)
        dropout_rate: Dropout rate for regularization (default: 0.2)
        
    Returns:
        Compiled Keras Model ready for training
    """
    # Define input layer for 29x29 binary CAN frame matrices
    inputs = Input(shape=input_shape)
    
    # Stage 1: Initial feature extraction and spatial reduction
    # 29x29x1 → 13x13x128
    x = stem_block(inputs)
    
    # Stage 2: Multi-scale feature extraction with residual connections
    # 13x13x128 → 13x13x128 (maintains spatial dimensions)
    x = inception_resnet_a_block(x, scale=0.1)
    
    # Stage 3: First major spatial reduction with channel expansion
    # 13x13x128 → 6x6x448
    x = reduction_a_block(x)
    
    # Stage 4: High-level feature extraction with asymmetric convolutions
    # 6x6x448 → 6x6x448 (maintains spatial dimensions)
    x = inception_resnet_b_block(x, scale=0.1)
    
    # Stage 5: Final spatial reduction with maximum channel expansion
    # 6x6x448 → 2x2x896
    x = reduction_b_block(x)
    
    # Stage 6: Global spatial aggregation
    # 2x2x896 → 1x1x896 (eliminates spatial dimensions entirely)
    x = AveragePooling2D((2, 2), padding='valid')(x)
    
    # Stage 7: Flatten for dense layer processing
    # 1x1x896 → 896-dimensional feature vector
    x = Flatten()(x)
    
    # Stage 8: Regularization to prevent overfitting
    # Randomly sets 20% of features to zero during training
    x = Dropout(dropout_rate)(x)
    
    # Stage 9: Final classification layer
    # 896 → 2 classes with softmax activation for probability distribution
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and return the complete model
    model = Model(inputs, outputs)
    return model

###################################################
# Model Wrapper Class for Training and Evaluation
###################################################
class Inception_Resnet_V1:
    """
    Wrapper class for the reduced Inception-ResNet model providing training and evaluation utilities.
    
    This class encapsulates the model architecture and provides methods for:
    - Model initialization with configurable hyperparameters
    - Training with batch-level loss tracking
    - Optional pre-trained weight loading
    - Model summary and inspection
    
    The class is designed to integrate seamlessly with the genetic algorithm
    adversarial attack framework and provides the batch-level loss tracking
    required for detailed training analysis.
    """
    
    def __init__(self, epochs=10, batch_size=32, load_weights=False):
        """
        Initialize the Inception-ResNet model with specified hyperparameters.
        
        Args:
            epochs: Number of training epochs (default: 10)
            batch_size: Batch size for training (default: 32)
            load_weights: Whether to load pre-trained weights (default: False)
        """
        # Store training hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Build the reduced Inception-ResNet architecture
        self.model = build_reduced_inception_resnet()
        
        # Optionally load pre-trained weights
        if load_weights:
            # Placeholder for weight loading - can be customized as needed
            # Example: self.model.load_weights('path_to_pretrained_weights.h5')
            pass

    def train(self, x_train, y_train, x_test, y_test, filename_prefix="", epochs_override=None):
        """
        Train the model with batch-level loss tracking for detailed analysis.
        
        This method compiles the model, trains it on the provided data, and captures
        detailed training metrics including per-batch loss values. This granular
        tracking is essential for genetic algorithm experiments and training analysis.
        
        Args:
            x_train: Training feature data (CAN frames)
            y_train: Training labels (0=normal, 1=attack)
            x_test: Test feature data (for validation during training)
            y_test: Test labels  
            filename_prefix: Prefix for saved model filename
            epochs_override: Override default epoch count if specified
            
        Returns:
            tuple: (training_history, batch_loss_list)
                - training_history: Keras training history object
                - batch_loss_list: List of (iteration, loss) tuples for each batch
        """
        # Use override epochs if provided, otherwise use instance default
        epochs_to_run = epochs_override if epochs_override is not None else self.epochs
        
        # Compile model with Adam optimizer and sparse categorical crossentropy loss
        # Adam optimizer: adaptive learning rate with momentum for stable training
        # Sparse categorical crossentropy: efficient for integer class labels
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Initialize custom callback for batch-level loss tracking
        batch_callback = BatchLossHistory()
        
        # Train the model with batch-level monitoring
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs_to_run,
            batch_size=self.batch_size,
            callbacks=[batch_callback]  # Capture per-batch metrics
        )
        
        # Save the trained model with custom filename prefix
        # This allows saving models for different attack types (DoS, Fuzzy, RPM)
        self.model.save(filename_prefix + 'final_model.h5')
        
        # Return both epoch-level and batch-level training metrics
        return history, batch_callback.batch_losses

    def summary(self):
        """
        Display model architecture summary including layer details and parameter counts.
        
        Returns:
            Model summary showing architecture, output shapes, and parameter counts
        """
        return self.model.summary()

###################################################
# Development and Testing Code
###################################################
# Uncomment the following lines for model architecture debugging and testing:
# if __name__ == "__main__":
#     # Create model instance with sample hyperparameters
#     instance = Inception_Resnet_V1(epochs=5, batch_size=32)
#     
#     # Display model architecture summary
#     instance.summary()
#     
#     # Optional: Test with dummy data
#     # import numpy as np
#     # x_dummy = np.random.rand(100, 29, 29, 1)
#     # y_dummy = np.random.randint(0, 2, 100)
#     # history, batch_losses = instance.train(x_dummy, y_dummy, x_dummy, y_dummy)
#     # print(f"Training completed. Final batch loss: {batch_losses[-1][1]:.4f}")
