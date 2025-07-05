#!/usr/bin/env python3
"""
Adversarial Spoofing Attack Generator using Genetic Algorithm

This script implements a genetic algorithm-based approach to generate adversarial attacks
against deep learning intrusion detection systems for vehicle CAN networks. The attack
focuses on RPM/Spoofing attack scenarios by modifying ECU-controlled dummy rows in CAN frames
to evade detection while maintaining attack characteristics.

Key characteristics of Spoofing attacks:
- Uses ECU control information to identify modifiable dummy rows
- Only modifies rows where ECU control value = 1 (transmitter controlled)
- Similar to DoS attacks but with ECU-specific constraints
- Preserves RPM attack payload in non-dummy rows

The genetic algorithm evolves adversarial examples through:
- Population-based search with crossover and mutation
- Fitness evaluation based on IDS confidence scores  
- Elitist selection to preserve best candidates
- Multi-generation evolution until successful evasion

Usage:
    python3 adversarial_spoofing_attack.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import os
import matplotlib.pyplot as plt
import itertools

###################################################
# Main Adversarial Spoofing Attack Class
###################################################
class AdversarialSpoofAttack:
    def __init__(self, model_path, population_size=100, max_generations=75, mutation_rate=0.1):
        """
        Initialize the adversarial Spoofing attack generator.
        
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
        
        # Track which rows were originally ECU-controlled dummy rows
        # These are the only rows we're allowed to modify in spoofing attacks
        self.original_dummy_rows = []
        
        # Load the RPM/spoofing attack test dataset containing CAN frames and ECU control data
        self.data = np.load('./CAN_DATA/RPM_test_data.npz')
        self.x_test = self.data['x_test']      # CAN frame data (29x29x1 binary matrices)
        self.y_test = self.data['y_test']      # Labels: 0=normal, 1=attack
        self.ecu_control = self.data['ecu_control']  # ECU control values for each frame

    def find_dummy_rows(self, frame_idx):
        """
        Find and return indices of ECU-controlled dummy rows for a specific frame.
        
        For spoofing attacks, dummy rows are identified by ECU control values:
        - ECU control value = 0: Receiver controlled (cannot modify)
        - ECU control value = 1: Transmitter controlled (can modify)
        
        This is specific to RPM/spoofing attacks where ECU control information
        determines which parts of the CAN frame can be safely modified without
        destroying the attack payload.
        
        Args:
            frame_idx: Index of the frame in the dataset
            
        Returns:
            List of row indices where ECU control value = 1 (modifiable rows)
        """
        # Bounds checking to prevent index errors
        if frame_idx >= len(self.ecu_control):
            return []  # Return empty list if index is out of bounds
            
        # Find all rows where ECU control value is 1 (transmitter controlled)
        return [j for j in range(29) if self.ecu_control[frame_idx][j] == 1]

    def mutate(self, frame):
        """
        Apply mutation to a frame by modifying only the original ECU-controlled dummy rows.
        
        Spoofing attack mutation strategy:
        - Only modifies rows that were originally identified as ECU-controlled (value=1)
        - For each dummy row, applies mutation with probability = mutation_rate
        - Sets exactly one random bit to 1 in the selected row
        - Unlike fuzzy attacks, can set any bit position (0-28) including priority bits
        
        This maintains the constraint that only ECU-transmitter controlled portions
        of the frame can be modified, preserving the RPM attack characteristics.
        
        Args:
            frame: 29x29x1 numpy array representing a CAN frame
            
        Returns:
            Mutated copy of the input frame
        """
        mutated = frame.copy()
        
        # Apply mutation only to original ECU-controlled dummy rows
        for i in self.original_dummy_rows:
            # Apply mutation with specified probability per row
            if random.random() < self.mutation_rate:
                # Set one random bit (positions 0-28) to 1
                # Unlike DoS attacks, spoofing allows modification of priority bits (0-2)
                bit_to_flip = random.randint(0, 28)
                mutated[i, bit_to_flip, 0] = 1
                
        return mutated

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent frames to create offspring.
        
        Spoofing attack crossover strategy:
        - Start with parent1 as the base (child inherits most characteristics)
        - For each original ECU-controlled dummy row, randomly choose to inherit from parent2
        - Only swap rows that were originally ECU-controlled (preserves attack payload)
        - 50% chance per row to inherit from parent2
        
        This allows combining successful mutations from different parents
        while maintaining the integrity of the original RPM attack structure
        and ECU control constraints.
        
        Args:
            parent1: First parent frame (29x29x1 numpy array)
            parent2: Second parent frame (29x29x1 numpy array)
            
        Returns:
            Child frame combining characteristics from both parents
        """
        child = parent1.copy()
        
        # For each original ECU-controlled dummy row, decide whether to inherit from parent2
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

    def generate_adversarial_attack(self, dummy_row_threshold=1, max_frames=100):
        """
        Main genetic algorithm to generate adversarial Spoofing attacks.
        
        Process:
        1. Create balanced dataset (70% attack, 30% benign)
        2. Add benign frames directly (no modification needed)
        3. For each attack frame with sufficient ECU-controlled dummy rows:
           - Identify ECU-controlled dummy rows using ECU control data
           - Initialize genetic algorithm population
           - Evolve through generations using crossover/mutation
           - Stop when successful evasion achieved (confidence < 0.5)
        4. Return complete adversarial dataset with labels
        
        Args:
            dummy_row_threshold: Minimum ECU-controlled rows required to process a frame
            max_frames: Maximum total frames to include in final dataset
            
        Returns:
            tuple: (final_test, y_test, orig_frame, generations_needed)
                - final_test: Adversarial frames ready for evaluation
                - y_test: Corresponding labels for the frames
                - orig_frame: Original frames before adversarial modification
                - generations_needed: List of generations required per attack frame
        """
        # Calculate balanced dataset composition (70% attack, 30% benign)
        attack_count = int(max_frames * 0.7)
        benign_count = max_frames - attack_count
        
        print(f"Using {attack_count} attack frames and {benign_count} benign frames")
        
        # === STEP 1: Identify available frames by type ===
        # Find the first benign_count benign frames
        benign_indices = np.where(self.y_test == 0)[0][:benign_count]
        if len(benign_indices) < benign_count:
            benign_count = len(benign_indices)
            print(f"Warning: Only {benign_count} benign frames available")
        
        # Find the first attack_count attack frames
        attack_indices = np.where(self.y_test == 1)[0]
        if len(attack_indices) < attack_count:
            attack_count = len(attack_indices)
            print(f"Warning: Only {attack_count} attack frames available")
        
        # Initialize containers for final adversarial dataset
        orig_frame = []          # Original frames for comparison
        final_test = []          # Adversarial/benign frames for evaluation
        generations_needed = []  # Track genetic algorithm performance
        
        # === STEP 2: Add benign frames to dataset ===
        # Benign frames are added directly without modification
        for i in benign_indices:
            final_test.append(self.x_test[i])
            orig_frame.append(self.x_test[i])
        
        print("Starting genetic algorithm for adversarial attack generation...")
        
        # === STEP 3: Process attack frames with genetic algorithm ===
        attack_frames_processed = 0
        for i in attack_indices:
            # Stop if we've processed enough attack frames
            if attack_frames_processed >= attack_count:
                break
                
            # === STEP 4: Check ECU-controlled dummy row availability ===
            # Find ECU-controlled dummy rows for this specific frame
            dummy_rows = self.find_dummy_rows(i)
            
            # Skip frames with insufficient ECU-controlled dummy rows
            if len(dummy_rows) < dummy_row_threshold:
                continue  # Skip this frame
                # Note: The next two lines are unreachable due to continue above
                final_test.append(self.x_test[i])
                orig_frame.append(self.x_test[i])
                attack_frames_processed += 1
            else:
                # === STEP 5: Initialize genetic algorithm for this frame ===
                # Store which rows were originally ECU-controlled for mutation/crossover constraints
                self.original_dummy_rows = dummy_rows.copy()
                frame_copy = self.x_test[i].copy()
                
                # === STEP 6: Create initial population with high diversity ===
                # Temporarily set mutation rate to 1.0 to ensure all individuals are different
                mut_rate = self.mutation_rate  # Save current mutation rate
                self.mutation_rate = 1         # Force mutation for population diversity
                
                # Create diverse initial population through mutation
                population = [self.mutate(frame_copy.copy()) for _ in range(self.population_size)]
                
                # Restore original mutation rate for evolution
                self.mutation_rate = mut_rate
                
                # === STEP 7: Evolve population through generations ===
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
                            orig_frame.append(self.x_test[i])
                            success_generation = generation + 1
                            success = True
                            break
                    
                    # If successful attack found, move to next frame
                    if success:
                        break
                    
                    # === STEP 8: Selection and reproduction for next generation ===
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
                
                # === STEP 9: Handle end of genetic algorithm ===
                # If no successful attack found after all generations
                if success_generation == -1:
                    # Use the best candidate found (lowest confidence score)
                    best_idx = np.argmin(scores)
                    print(f"Best score achieved: {scores[best_idx]:.4f}")
                    final_test.append(valid_individuals[best_idx])
                    orig_frame.append(self.x_test[i])
                    success_generation = self.max_generations
                
                # Record performance metrics
                generations_needed.append(success_generation)
                attack_frames_processed += 1
        
        # === STEP 10: Create final labeled dataset ===
        # Create labels array: 0 for benign frames, 1 for attack frames
        y_final = np.zeros(len(final_test))
        y_final[benign_count:] = 1  # Attack samples start after benign samples
        
        return np.array(final_test), y_final, np.array(orig_frame), generations_needed

    def visualize_attack(self, original_frame, adversarial_frame, output_file="adversarial_spoofing_attack.png"):
        """
        Create side-by-side visualization comparing original and adversarial spoofing attack frames.
        
        Visualization shows binary CAN frames as images where:
        - Black pixels = 0 bits
        - White pixels = 1 bits
        
        This helps visualize the ECU-controlled modifications made by the genetic algorithm
        for spoofing attack evasion.
        
        Args:
            original_frame: Original spoofing attack frame before modification
            adversarial_frame: Modified frame designed to evade detection
            output_file: Output filename for the visualization image
        """
        plt.figure(figsize=(12, 6))
        
        # Plot original spoofing attack frame
        plt.subplot(1, 2, 1)
        plt.imshow(original_frame[:, :, 0], cmap='binary_r', vmin=0, vmax=1)
        plt.title("Original Spoof Attack (Black=0, White=1)")
        
        # Plot adversarial spoofing attack frame
        plt.subplot(1, 2, 2)
        plt.imshow(adversarial_frame[:, :, 0], cmap='binary_r', vmin=0, vmax=1)
        plt.title("Adversarial Spoof Attack")
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Visualization saved to {output_file}")

    def evaluate_attack_effectiveness(self, original_frame, adversarial_frame):
        """
        Compare IDS confidence scores for original vs adversarial spoofing attack frames.
        
        Prints detailed effectiveness analysis including:
        - Original attack confidence score
        - Adversarial attack confidence score  
        - Success determination (< 0.5 = successful evasion)
        
        Args:
            original_frame: Original spoofing attack frame
            adversarial_frame: Modified adversarial spoofing attack frame
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
    Experimental function to test the effect of different mutation rates on spoofing attack performance.
    
    Tests mutation rates from 0.1 to 1.0 and measures average generations needed
    to find successful adversarial spoofing attacks. This helps optimize the balance between:
    - Exploration (high mutation = more diversity in ECU-controlled rows)
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
        
        # Create spoofing attack generator with specific mutation rate
        attack = AdversarialSpoofAttack(
            model_path=model_path,
            population_size=100,      # Keep constant for fair comparison
            max_generations=75,       # Keep constant for fair comparison
            mutation_rate=rate        # Variable being tested
        )
        
        # Generate adversarial attacks and collect performance statistics
        _, _, _, generations_stats = attack.generate_adversarial_attack(
            dummy_row_threshold=1,    # Keep constant for fair comparison
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
    Experimental function to test the effect of ECU-controlled dummy row threshold on attack success.
    
    Tests different minimum ECU-controlled row requirements and measures performance.
    Higher thresholds mean:
    - More ECU-controlled modification space available (easier attacks)
    - Fewer eligible frames (reduced dataset size)
    
    Args:
        model_path: Path to the trained IDS model
        max_frames: Number of frames to test per threshold
        
    Returns:
        Dictionary mapping dummy row thresholds to average generations needed
    """
    dummy_row_thresholds = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    results = {}
    
    # Create attack object once to access dataset and ECU control data for analysis
    attack_obj = AdversarialSpoofAttack(model_path=model_path)
    
    # Test each ECU-controlled dummy row threshold
    for threshold in dummy_row_thresholds:
        print(f"\n--- Testing dummy row threshold: {threshold} ---")
        
        # === STEP 1: Find frames suitable for this threshold ===
        suitable_frames = []
        for i in range(min(len(attack_obj.x_test), len(attack_obj.ecu_control))):
            if attack_obj.y_test[i] == 1:  # Only consider attack frames
                # Find ECU-controlled dummy rows for this frame
                dummy_rows = attack_obj.find_dummy_rows(i)
                if len(dummy_rows) >= threshold:  # Frame has enough ECU-controlled rows
                    suitable_frames.append(i)
        
        print(f"Found {len(suitable_frames)} frames with dummy rows >= {threshold}")
        
        # Skip if no suitable frames found
        if len(suitable_frames) == 0:
            results[threshold] = 0
            print(f"Dummy row threshold {threshold}: No suitable frames found")
            continue
            
        # Limit to maximum frame count for consistent testing
        suitable_frames = suitable_frames[:max_frames]
        
        # === STEP 2: Create attack generator for this threshold ===
        attack = AdversarialSpoofAttack(
            model_path=model_path,
            population_size=100,
            max_generations=75,
            mutation_rate=0.3  # Fixed mutation rate for fair comparison
        )
        
        # === STEP 3: Test genetic algorithm on each suitable frame ===
        generations_list = []
        for idx, frame_idx in enumerate(suitable_frames):
            print(f"Processing frame {idx+1}/{len(suitable_frames)}")
            
            # Prepare frame for genetic algorithm
            frame = attack.x_test[frame_idx].copy()
            dummy_rows = attack.find_dummy_rows(frame_idx)
            attack.original_dummy_rows = dummy_rows.copy()
            
            # Create initial population
            population = [attack.mutate(frame.copy()) for _ in range(attack.population_size)]
            
            # === STEP 4: Run genetic algorithm ===
            success_generation = -1
            for generation in range(attack.max_generations):
                scores = []
                valid_individuals = []
                success = False
                
                # Evaluate population fitness
                for individual in population:
                    score = attack.calculate_confidence(individual)
                    scores.append(score)
                    valid_individuals.append(individual)
                    
                    # Check for successful evasion
                    if score < 0.5:
                        success_generation = generation + 1
                        success = True
                        break
                
                if success:
                    break
                
                # === STEP 5: Selection and reproduction ===
                # Convert scores and create selection probabilities
                scores = np.array(scores, dtype=float)
                scores = np.nan_to_num(scores, nan=1.0)
                inv_scores = 1.0 - scores
                
                total = inv_scores.sum()
                num_positive = np.count_nonzero(inv_scores)
                
                if total <= 1e-12 or num_positive < 2:
                    selection_probs = np.ones_like(inv_scores) / len(inv_scores)
                else:
                    selection_probs = inv_scores / total
                
                # Create next generation
                indices = np.arange(len(valid_individuals))
                new_pop = []
                
                # Elitism: keep best individual
                new_pop.append(valid_individuals[np.argmin(scores)])
                
                # Generate rest through crossover and mutation
                while len(new_pop) < attack.population_size:
                    p1_idx, p2_idx = np.random.choice(indices, size=2, p=selection_probs, replace=False)
                    child = attack.crossover(valid_individuals[p1_idx], valid_individuals[p2_idx])
                    child = attack.mutate(child)
                    new_pop.append(child)
                
                population = new_pop
            
            # Record result for this frame
            if success_generation == -1:
                success_generation = attack.max_generations
            
            generations_list.append(success_generation)
        
        # === STEP 6: Calculate average performance for this threshold ===
        if generations_list:
            avg_generations = np.mean(generations_list)
            results[threshold] = avg_generations
            print(f"Dummy row threshold {threshold}: Average generations = {avg_generations:.2f}")
        else:
            results[threshold] = 0
            print(f"Dummy row threshold {threshold}: No successful attacks")
    
    return results

###################################################
# Plotting Functions for Experimental Results
###################################################
def plot_mutation_rate_results(results):
    """
    Create line plot showing the relationship between mutation rate and spoofing attack performance.
    
    Lower average generations = better performance (faster convergence).
    
    Args:
        results: Dictionary mapping mutation rates to average generations needed
    """
    rates = list(results.keys())
    avgs = [results[r] for r in rates]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rates, avgs, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Mutation Rate')
    plt.ylabel('Average Number of Generations')
    plt.title('Effect of Mutation Rate on Spoofing Adversarial Attack Generations')
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.savefig('spoofing_mutation_rate_vs_generations.png')
    plt.close()
    
    print("Mutation rate experiment plot saved as 'spoofing_mutation_rate_vs_generations.png'")

def plot_dummy_row_results(results):
    """
    Create line plot showing the relationship between ECU-controlled dummy row threshold and attack performance.
    
    Shows how the amount of ECU-controlled modification space affects attack difficulty.
    
    Args:
        results: Dictionary mapping dummy row thresholds to average generations needed
    """
    thresholds = list(results.keys())
    avgs = [results[t] for t in thresholds]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, avgs, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Dummy Row Threshold')
    plt.ylabel('Average Number of Generations')
    plt.title('Effect of Dummy Row Threshold on Spoofing Adversarial Attack Generations')
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.savefig('spoofing_dummy_rows_vs_generations.png')
    plt.close()
    
    print("Dummy row experiment plot saved as 'spoofing_dummy_rows_vs_generations.png'")

###################################################
# Main Execution Function
###################################################
def main():
    """
    Main function that orchestrates the complete adversarial spoofing attack generation and evaluation process.
    
    Process:
    1. Load pre-trained IDS model for RPM/spoofing attacks
    2. Generate adversarial spoofing attack dataset (or load if exists)
    3. Evaluate model performance on adversarial examples
    4. Compute detailed performance metrics
    5. Create visualizations of misclassified examples
    6. Run parameter optimization experiments
    7. Generate performance analysis plots
    """
    # === STEP 1: Setup and Configuration ===
    model_path = "RPM_final_model.h5"
    
    # === STEP 2: Generate or Load Adversarial Dataset ===
    # Create spoofing attack generator instance with tuned parameters
    attack = AdversarialSpoofAttack(
        model_path=model_path,
        population_size=100,    # Population size for genetic algorithm
        max_generations=75,     # Maximum evolution generations
        mutation_rate=0.2       # Mutation probability per individual
    )
    
    # Check if adversarial dataset already exists to avoid regeneration
    if os.path.exists("adversarial_spoofing_attack.npz"):
        print("Adversarial attack already exists. Loading from file...")
        try:
            # Load pre-computed adversarial dataset
            data = np.load("adversarial_spoofing_attack.npz")
            final_test = data['final_test']  # Adversarial/benign frames
            y_test = data['y_test']          # True labels
            x_test = data['x_test']          # Original frames
        except Exception as e:
            print(f"Error loading file: {e}")
            # Regenerate the data if loading fails
            final_test, y_test, x_test, _ = attack.generate_adversarial_attack(dummy_row_threshold=1, max_frames=100)
            try:
                # Save newly generated dataset
                np.savez("adversarial_spoofing_attack.npz", 
                         final_test=final_test, 
                         y_test=y_test, 
                         x_test=x_test)
                print(f"Adversarial attack generated and saved to adversarial_spoofing_attack.npz")
            except Exception as e:
                print(f"Error saving file: {e}")
    else:
        # Generate new adversarial spoofing attack dataset
        try:
            print("Generating new adversarial spoofing attack dataset...")
            final_test, y_test, x_test, _ = attack.generate_adversarial_attack(dummy_row_threshold=1, max_frames=100)
            
            # Save generated dataset for future use
            np.savez("adversarial_spoofing_attack.npz", 
                     final_test=final_test, 
                     y_test=y_test, 
                     x_test=x_test)
            print(f"Adversarial attack generated and saved to adversarial_spoofing_attack.npz")
        except Exception as e:
            print(f"Error during attack generation or saving: {e}")
    
    # === STEP 3: Evaluate Model Performance on Adversarial Examples ===
    print("Evaluating model performance on adversarial spoofing attack dataset...")
    test_loss, test_accuracy = attack.model.evaluate(final_test, y_test, verbose=1)
    
    # Save basic evaluation metrics
    with open(f"adversarial_spoof_test.txt", "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_accuracy:.4f}\n")
    
    # === STEP 4: Generate Detailed Performance Analysis ===
    print("Computing detailed performance metrics...")
    
    # Get model predictions on adversarial dataset
    y_pred_prob = attack.model.predict(final_test)
    y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class predictions
    
    # Import required metrics from scikit-learn
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Compute confusion matrix and extract components
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()  # True Negative, False Positive, False Negative, True Positive
    
    # Calculate detailed performance metrics
    FNR = round(FN / (TP + FN), 4) if (TP + FN) > 0 else 0.0        # False Negative Rate
    ER = round((FP + FN) / (TN + FP + FN + TP), 4)                   # Error Rate
    precision = round(TP / (TP + FP), 4) if (TP + FP) > 0 else 0.0  # Precision
    recall = round(TP / (TP + FN), 4) if (TP + FN) > 0 else 0.0     # Recall (Sensitivity)
    f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0  # F1 Score
    
    # === STEP 5: Create Confusion Matrix Visualization ===
    plot_confusion_matrix(cm, classes=['Normal', 'Attack'], suffix="adv_spoof_attack", normalize=True,
                         title='Normalized Confusion Matrix')
    
    # === STEP 6: Save Comprehensive Evaluation Report ===
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])
    
    with open(f"evaluation_metrics_spoof_adv.txt", "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write(f"False Negative Rate (FNR): {FNR:.4f}\n")
        f.write(f"Error Rate (ER): {ER:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Results for spoof adversarial attack (test set) saved")
    
    # === STEP 7: Create Visualizations of Successful Attacks ===
    print("Creating visualizations of misclassified spoofing attack examples...")
    
    # Find and visualize up to 5 successful adversarial spoofing attacks
    # (cases where true label = attack but model predicted normal)
    cnt = 0
    for i in range(len(y_test)):
        if cnt == 5:  # Limit to 5 examples to avoid clutter
            break
            
        # Look for misclassified attack samples (successful evasions)
        if y_test[i] != y_pred[i] and y_test[i] == 1:
            # Create side-by-side comparison
            plt.figure(figsize=(12, 6))
            
            # Original spoofing attack frame
            plt.subplot(1, 2, 1)
            plt.imshow(x_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
            plt.title(f"Original Spoof Attack (True: {y_test[i]})")
            
            # Adversarial spoofing attack frame (successfully evaded detection)
            plt.subplot(1, 2, 2)
            plt.imshow(final_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
            plt.title(f"Adversarial Spoof Attack (Predicted: {y_pred[i]})")
            
            plt.tight_layout()
            plt.savefig(f"spoof_attack_comparison_{cnt}.png")
            plt.close()
            print(f"Attack comparison saved as spoof_attack_comparison_{cnt}.png")
            cnt += 1
    
    # === STEP 8: Run Parameter Optimization Experiments ===
    print("\n===== RUNNING MUTATION RATE EXPERIMENT =====")
    # Test different mutation rates to find optimal balance for spoofing attacks
    mutation_results = run_mutation_rate_experiment(model_path, max_frames=20)
    plot_mutation_rate_results(mutation_results)
    
    print("\n===== RUNNING DUMMY ROW THRESHOLD EXPERIMENT =====")
    # Test different ECU-controlled dummy row requirements to understand attack constraints
    dummy_row_results = run_dummy_row_experiment(model_path, max_frames=20)
    plot_dummy_row_results(dummy_row_results)

###################################################
# Script Entry Point
###################################################
if __name__ == "__main__":
    main()
