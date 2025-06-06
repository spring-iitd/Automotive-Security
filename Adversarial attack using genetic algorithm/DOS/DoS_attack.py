#!/usr/bin/env python3
"""
Adversarial DoS Attack Generator using Genetic Algorithm

This script implements a genetic algorithm-based approach to generate adversarial attacks
against deep learning intrusion detection systems for vehicle CAN networks. The attack
focuses on DoS (Denial of Service) attack scenarios by modifying dummy rows in CAN frames
to evade detection while maintaining attack characteristics.

The genetic algorithm evolves adversarial examples through:
- Population-based search with crossover and mutation
- Fitness evaluation based on IDS confidence scores  
- Elitist selection to preserve best candidates
- Multi-generation evolution until successful evasion

Usage:
    python3 adversarial_dos_attack.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import os
import matplotlib.pyplot as plt
import itertools

###################################################
# Main Adversarial Attack Class
###################################################
class AdversarialDosAttack:
    def __init__(self, model_path, population_size=100, max_generations=75, mutation_rate=0.1):
        """
        Initialize the adversarial DoS attack generator.
        
        Args:
            model_path: Path to the trained IDS model (H5 format)
            population_size: Number of individuals in each genetic algorithm generation
            max_generations: Maximum number of generations to evolve before giving up
            mutation_rate: Probability of mutation occurring for each individual
        """
        # Load the pre-trained intrusion detection system model
        self.model = load_model(model_path)
        
        # Compile model to prevent TensorFlow warnings during prediction
        self.model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        
        # Store genetic algorithm hyperparameters
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        
        # Track which rows were originally all zeros (dummy rows)
        # These are the only rows we're allowed to modify
        self.original_dummy_rows = []
        
        # Load the test dataset containing DoS attack and normal CAN frames
        self.data = np.load('./CAN_DATA/DoS_test_data.npz')
        self.x_test = self.data['x_test']  # CAN frame data (29x29x1 binary matrices)
        self.y_test = self.data['y_test']  # Labels: 0=normal, 1=attack
        
    def find_dummy_rows(self, frame):
        """
        Identify rows in a CAN frame that are completely zero (dummy rows).
        
        These dummy rows represent unused CAN message slots and are the only
        parts of the frame we can modify without destroying the attack payload.
        
        Args:
            frame: 29x29x1 numpy array representing a CAN frame
            
        Returns:
            List of row indices that contain all zeros
        """
        return [i for i in range(29) if np.all(frame[i,:,0] == 0)]

    def mutate(self, frame):
        """
        Apply mutation to a frame by modifying only the original dummy rows.
        
        Mutation process:
        1. Select a random original dummy row with probability = mutation_rate
        2. Clear the entire row to ensure only one bit is set
        3. Set exactly one random bit (excluding priority bits 0-2) to 1
        
        This maintains the constraint that dummy rows can only contain
        sparse, single-bit modifications.
        
        Args:
            frame: 29x29x1 numpy array to mutate
            
        Returns:
            Mutated copy of the input frame
        """
        mutated = frame.copy()
        
        # Apply mutation with specified probability and only if dummy rows exist
        if random.random() < self.mutation_rate and self.original_dummy_rows:
            # Randomly select one of the original dummy rows to modify
            row_idx = random.choice(self.original_dummy_rows)
            
            # Clear the entire row first to ensure clean state
            mutated[row_idx, :, 0] = 0
            
            # Set exactly one bit to 1, avoiding first 3 priority bits (0-2)
            # Bits 3-28 represent the main CAN ID field that can be modified
            bit_to_flip = random.randint(3, 28)
            mutated[row_idx, bit_to_flip, 0] = 1
            
        return mutated

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent frames to create offspring.
        
        Crossover strategy:
        - Start with parent1 as the base (child inherits most characteristics)
        - For each original dummy row, randomly choose to inherit from parent2
        - Only swap rows that were originally dummy rows (preserves attack payload)
        
        This allows combining successful mutations from different parents
        while maintaining the integrity of the original attack structure.
        
        Args:
            parent1: First parent frame (29x29x1 numpy array)
            parent2: Second parent frame (29x29x1 numpy array)
            
        Returns:
            Child frame combining characteristics from both parents
        """
        child = parent1.copy()
        
        # For each original dummy row, decide whether to inherit from parent2
        for i in self.original_dummy_rows:
            if random.random() < 0.5:  # 50% chance to inherit from parent2
                child[i] = parent2[i]
                
        return child

    def calculate_confidence(self, frame):
        """
        Calculate the IDS confidence score for classifying a frame as an attack.
        
        The IDS model outputs probabilities for [normal, attack] classes.
        We return the attack confidence (index 1) since our goal is to
        minimize this score below 0.5 to achieve misclassification.
        
        Args:
            frame: 29x29x1 numpy array representing a CAN frame
            
        Returns:
            Float between 0-1 representing attack confidence score
        """
        # Expand dimensions to create batch of size 1 for model prediction
        frame_batch = np.expand_dims(frame, 0)
        
        # Get prediction probabilities [normal_prob, attack_prob]
        prediction = self.model.predict(frame_batch, verbose=0)
        
        # Return attack confidence (index 1)
        # Lower values indicate successful evasion
        return prediction[0][1]

    def generate_adversarial_attack(self, dummy_row_threshold=10, max_frames=100):
        """
        Main genetic algorithm to generate adversarial DoS attacks.
        
        Process:
        1. Load and filter frames based on dummy row count
        2. Create balanced dataset (70% attack, 30% benign)
        3. For each attack frame with sufficient dummy rows:
           - Initialize genetic algorithm population
           - Evolve through generations using crossover/mutation
           - Stop when successful evasion achieved (confidence < 0.5)
        4. Return complete adversarial dataset with labels
        
        Args:
            dummy_row_threshold: Minimum dummy rows required to process a frame
            max_frames: Maximum total frames to include in final dataset
            
        Returns:
            tuple: (final_test, y_test, orig_frame, generations_needed)
                - final_test: Adversarial frames ready for evaluation
                - y_test: Corresponding labels for the frames
                - orig_frame: Original frames before adversarial modification
                - generations_needed: List of generations required per attack frame
        """
        # Create copy of original test data for comparison
        orig = self.x_test.copy()
        
        # Initialize containers for final adversarial dataset
        final_test = []      # Adversarial/benign frames for evaluation
        orig_frame = []      # Original frames for comparison
        generations_needed = []  # Track genetic algorithm performance
        
        # Calculate balanced dataset composition (70% attack, 30% benign)
        attack_count = int(max_frames * 0.7)
        benign_count = max_frames - attack_count
        
        print(f"Using {attack_count} attack frames and {benign_count} benign frames")
        
        # === STEP 1: Add benign frames to dataset ===
        # Find indices of all benign frames (label = 0)
        benign_indices = np.where(self.y_test == 0)[0][:benign_count]
        
        # Handle case where insufficient benign frames exist
        if len(benign_indices) < benign_count:
            benign_count = len(benign_indices)
            print(f"Warning: Only {benign_count} benign frames available")
        
        # Add benign frames directly to final dataset (no modification needed)
        for i in benign_indices:
            final_test.append(self.x_test[i])
            orig_frame.append(orig[i])
        
        print("Starting genetic algorithm for adversarial attack generation...")
        
        # === STEP 2: Process attack frames with genetic algorithm ===
        attack_frames_processed = 0
        attack_indices = np.where(self.y_test == 1)[0]  # Find all attack frames
        
        for i in attack_indices:
            # Stop if we've processed enough attack frames
            if attack_frames_processed >= attack_count:
                break
                
            # Check if this frame has enough dummy rows to be worth attacking
            dummy_rows = self.find_dummy_rows(self.x_test[i])
            
            # Skip frames with insufficient dummy rows (can't be effectively modified)
            if len(dummy_rows) <= dummy_row_threshold:
                continue  # Skip this frame
                # Note: The next two lines are unreachable due to continue above
                final_test.append(self.x_test[i])
                orig_frame.append(orig[i])
                attack_frames_processed += 1
            else:
                # === STEP 3: Initialize genetic algorithm for this frame ===
                # Store which rows were originally dummy for mutation/crossover constraints
                self.original_dummy_rows = dummy_rows.copy()
                frame_copy = self.x_test[i].copy()
                
                # Pre-populate dummy rows with random single bits
                # This gives the genetic algorithm a better starting point
                for row_idx in dummy_rows:
                    bit_to_flip = random.randint(3, 28)  # Avoid priority bits
                    frame_copy[row_idx, bit_to_flip, 0] = 1
                
                # Create initial population by mutating the base frame
                # Temporarily set mutation rate to 1.0 to ensure all individuals are different
                mut_rate = self.mutation_rate  # Save current mutation rate
                self.mutation_rate = 1.0       # Force mutation for population diversity
                population = [self.mutate(frame_copy.copy()) for _ in range(self.population_size)]
                self.mutation_rate = mut_rate  # Restore original mutation rate
                
                # === STEP 4: Evolve population through generations ===
                success_generation = -1  # Track when successful attack was found
                
                for generation in range(self.max_generations):
                    print(f"Attack frame {attack_frames_processed+1}/{attack_count}, Generation {generation+1}/{self.max_generations}")
                    
                    # Evaluate fitness of all individuals in current population
                    scores = []
                    valid_individuals = []
                    success = False
                    
                    for individual in population:
                        # Calculate attack confidence score (lower = better)
                        score = self.calculate_confidence(individual)
                        scores.append(score)
                        valid_individuals.append(individual)
                        
                        # Check if successful evasion achieved (confidence < 0.5)
                        if score < 0.5:
                            print(f"Successful attack found in generation {generation+1}")
                            final_test.append(individual)
                            orig_frame.append(orig[i])
                            success_generation = generation + 1
                            success = True
                            break
                    
                    # If successful attack found, move to next frame
                    if success:
                        break
                    
                    # === STEP 5: Selection and reproduction for next generation ===
                    # Convert scores to numpy array and handle potential NaN values
                    scores = np.array(scores, dtype=float)
                    scores = np.nan_to_num(scores, nan=1.0)  # Replace NaN with worst score
                    
                    # Create inverse scores for fitness-proportional selection
                    # Lower confidence scores = higher fitness
                    inv_scores = 1.0 - scores
                    
                    # Calculate selection probabilities
                    total = inv_scores.sum()
                    num_positive = np.count_nonzero(inv_scores)
                    
                    # Handle edge case where all scores are very similar
                    if total <= 1e-12 or num_positive < 2:
                        # Use uniform selection if no clear fitness differences
                        selection_probs = np.ones_like(inv_scores) / len(inv_scores)
                    else:
                        # Use fitness-proportional selection
                        selection_probs = inv_scores / total
                    
                    # Create next generation population
                    indices = np.arange(len(valid_individuals))
                    new_pop = []
                    
                    # Elitism: Keep the best individual from current generation
                    best_idx = np.argmin(scores)
                    new_pop.append(valid_individuals[best_idx])
                    
                    # Generate remaining individuals through crossover and mutation
                    while len(new_pop) < self.population_size:
                        # Select two parents based on fitness
                        p1_idx, p2_idx = np.random.choice(
                            indices, size=2, p=selection_probs, replace=False
                        )
                        
                        # Create child through crossover
                        child = self.crossover(
                            valid_individuals[p1_idx], 
                            valid_individuals[p2_idx]
                        )
                        
                        # Apply mutation to child
                        child = self.mutate(child)
                        new_pop.append(child)
                    
                    # Replace current population with new generation
                    population = new_pop
                
                # === STEP 6: Handle end of genetic algorithm ===
                # If no successful attack found after all generations
                if success_generation == -1:
                    # Use the best candidate found (lowest confidence score)
                    best_idx = np.argmin(scores)
                    print(f"Best score achieved: {scores[best_idx]:.4f}")
                    final_test.append(valid_individuals[best_idx])
                    orig_frame.append(orig[i])
                    success_generation = self.max_generations
                
                # Record performance metrics
                generations_needed.append(success_generation)
                attack_frames_processed += 1
        
        # === STEP 7: Create final labeled dataset ===
        # Create labels array: 0 for benign frames, 1 for attack frames
        y_final = np.zeros(len(final_test))
        y_final[benign_count:] = 1  # Attack samples start after benign samples
        
        return np.array(final_test), y_final, np.array(orig_frame), generations_needed

    def save_adversarial_attack(self, frame, output_file="adversarial_dos_attack.npy"):
        """
        Save a single adversarial attack frame to disk.
        
        Args:
            frame: 29x29x1 numpy array to save
            output_file: Output filename (NPY format)
        """
        np.save(output_file, frame)
        print(f"Adversarial attack saved to {output_file}")

    def visualize_attack(self, original_frame, adversarial_frame, output_file="adversarial_dos_attack.png"):
        """
        Create side-by-side visualization comparing original and adversarial frames.
        
        Visualization shows binary CAN frames as images where:
        - Black pixels = 0 bits
        - White pixels = 1 bits
        
        This helps visualize what modifications the genetic algorithm made.
        
        Args:
            original_frame: Original attack frame before modification
            adversarial_frame: Modified frame designed to evade detection
            output_file: Output filename for the visualization image
        """
        plt.figure(figsize=(12, 6))
        
        # Plot original frame
        plt.subplot(1, 2, 1)
        plt.imshow(original_frame[:, :, 0], cmap='binary_r', vmin=0, vmax=1)
        plt.title("Original DoS Attack (Black=0, White=1)")
        
        # Plot adversarial frame
        plt.subplot(1, 2, 2)
        plt.imshow(adversarial_frame[:, :, 0], cmap='binary_r', vmin=0, vmax=1)
        plt.title("Adversarial DoS Attack")
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Visualization saved to {output_file}")

    def evaluate_attack_effectiveness(self, original_frame, adversarial_frame):
        """
        Compare IDS confidence scores for original vs adversarial frames.
        
        Prints detailed effectiveness analysis including:
        - Original attack confidence score
        - Adversarial attack confidence score  
        - Success determination (< 0.5 = successful evasion)
        
        Args:
            original_frame: Original attack frame
            adversarial_frame: Modified adversarial frame
        """
        original_score = self.calculate_confidence(original_frame)
        adversarial_score = self.calculate_confidence(adversarial_frame)
        
        print(f"Original attack confidence score: {original_score:.4f}")
        print(f"Adversarial attack confidence score: {adversarial_score:.4f}")
        print(f"Attack {'successful' if adversarial_score < 0.5 else 'unsuccessful'}")

###################################################
# Visualization and Analysis Functions
###################################################
def plot_confusion_matrix(cm, classes, suffix, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, filename=None):
    """
    Create and save a confusion matrix visualization with color coding and annotations.
    
    Args:
        cm: 2x2 confusion matrix array
        classes: List of class names ['Normal', 'Attack']
        suffix: String identifier for filename generation
        normalize: Whether to show percentages instead of raw counts
        title: Plot title
        cmap: Matplotlib colormap for visualization
        filename: Optional custom filename (auto-generated if None)
    """
    # Normalize confusion matrix to percentages if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure and display confusion matrix as colored image
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    # Set axis labels and tick marks
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add numerical annotations to each cell
    fmt = '.2f' if normalize else 'd'  # Format: decimals for percentages, integers for counts
    thresh = cm.max() / 2.  # Threshold for text color (white on dark, black on light)
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    # Finalize layout and save
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if filename is None:
        filename = f"confusion_matrix_{suffix}.png"
    plt.savefig(filename)
    plt.close()

###################################################
# Experimental Functions for Parameter Analysis
###################################################
def run_mutation_rate_experiment(model_path, max_frames=20):
    """
    Experimental function to test the effect of different mutation rates on genetic algorithm performance.
    
    Tests mutation rates from 0.1 to 1.0 and measures average generations needed
    to find successful adversarial attacks. This helps optimize the balance between:
    - Exploration (high mutation = more diversity)
    - Exploitation (low mutation = fine-tuning best solutions)
    
    Args:
        model_path: Path to the trained IDS model
        max_frames: Number of frames to test per mutation rate
        
    Returns:
        Dictionary mapping mutation rates to average generations needed
    """
    mutation_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}
    
    # Test each mutation rate independently
    for rate in mutation_rates:
        print(f"\n--- Testing mutation rate: {rate:.1f} ---")
        
        # Create attack generator with specific mutation rate
        attack = AdversarialDosAttack(
            model_path=model_path,
            population_size=100,      # Keep constant for fair comparison
            max_generations=75,       # Keep constant for fair comparison
            mutation_rate=rate        # Variable being tested
        )
        
        # Generate adversarial attacks and collect performance statistics
        _, _, _, generations_stats = attack.generate_adversarial_attack(
            dummy_row_threshold=10,   # Keep constant for fair comparison
            max_frames=max_frames
        )
        
        # Calculate average generations needed (lower = better performance)
        if generations_stats:
            avg_generations = np.mean(generations_stats)
            results[rate] = avg_generations
            print(f"Mutation rate {rate:.1f}: Average generations = {avg_generations:.2f}")
        else:
            # No successful attacks found
            results[rate] = 0
            print(f"Mutation rate {rate:.1f}: No successful attacks")
    
    return results

def run_dummy_row_experiment(model_path, max_frames=20):
    """
    Experimental function to test the effect of dummy row threshold on attack success.
    
    Tests different minimum dummy row requirements and measures performance.
    Higher thresholds mean:
    - More modification space available (easier attacks)
    - Fewer eligible frames (reduced dataset size)
    
    Args:
        model_path: Path to the trained IDS model
        max_frames: Number of frames to test per threshold
        
    Returns:
        Dictionary mapping dummy row thresholds to average generations needed
    """
    dummy_row_thresholds = [0, 5, 10, 15, 20, 25, 30]
    results = {}
    
    # Create attack object once to access dataset for analysis
    attack_obj = AdversarialDosAttack
