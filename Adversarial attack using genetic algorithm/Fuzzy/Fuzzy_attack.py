#!/usr/bin/env python3
"""
Adversarial Fuzzy Attack Generator using Genetic Algorithm

This script implements a genetic algorithm-based approach to generate adversarial attacks
against deep learning intrusion detection systems for vehicle CAN networks. The attack
focuses on Fuzzy attack scenarios by modifying CAN frame bits to evade detection while
maintaining attack characteristics.

Key differences from DoS attacks:
- Fuzzy attacks can modify any bit in the frame (not limited to dummy rows)
- Uses bit-flipping mutations across the entire 29x29 frame
- Applies multiple mutations per frame for increased perturbation
- More aggressive crossover strategy (pixel-by-pixel inheritance)

The genetic algorithm evolves adversarial examples through:
- Population-based search with crossover and mutation
- Fitness evaluation based on IDS confidence scores  
- Elitist selection to preserve best candidates
- Multi-generation evolution until successful evasion

Usage:
    python3 adversarial_fuzzy_attack.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import os
import matplotlib.pyplot as plt
import itertools

###################################################
# Main Adversarial Fuzzy Attack Class
###################################################
class AdversarialFuzzyAttack:
    def __init__(self, model_path, population_size=100, max_generations=75, mutation_rate=0.1):
        """
        Initialize the adversarial Fuzzy attack generator.
        
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
        
        # Load the fuzzy attack test dataset containing CAN frames
        self.data = np.load('./CAN_DATA/Fuz_test_data.npz')
        self.x_test = self.data['x_test']  # CAN frame data (29x29x1 binary matrices)
        self.y_test = self.data['y_test']  # Labels: 0=normal, 1=attack

    def mutate(self, frame):
        """
        Apply mutation to a frame by randomly flipping bits across the entire frame.
        
        Fuzzy attack mutation strategy:
        - Unlike DoS attacks, fuzzy attacks can modify any part of the frame
        - Applies multiple bit flips per mutation operation for stronger perturbation
        - Uses bit-flipping (0->1, 1->0) rather than just setting bits to 1
        - More aggressive approach suitable for fuzzy attack characteristics
        
        Args:
            frame: 29x29x1 numpy array representing a CAN frame
            
        Returns:
            Mutated copy of the input frame
        """
        mutated = frame.copy()
        
        # Apply mutation with specified probability
        if random.random() < self.mutation_rate:
            # For fuzzy attacks, apply multiple mutations per frame
            # This creates more significant perturbations compared to single-bit DoS modifications
            for _ in range(3):  # Apply 3 random bit flips per mutation event
                # Select random position anywhere in the 29x29 frame
                row_idx = random.randint(0, 28)
                bit_to_flip = random.randint(0, 28)
                
                # Flip the bit: 0 becomes 1, 1 becomes 0
                # This preserves existing attack patterns while adding noise
                mutated[row_idx, bit_to_flip, 0] = 1 - mutated[row_idx, bit_to_flip, 0]
                
        return mutated

    def crossover(self, parent1, parent2):
        """
        Perform pixel-by-pixel crossover between two parent frames.
        
        Fuzzy attack crossover strategy:
        - More fine-grained than DoS attacks (pixel-level vs row-level)
        - Each bit position has 50% chance to inherit from either parent
        - Creates diverse offspring by mixing successful mutations at bit level
        - Suitable for fuzzy attacks where any bit can contribute to evasion
        
        Args:
            parent1: First parent frame (29x29x1 numpy array)
            parent2: Second parent frame (29x29x1 numpy array)
            
        Returns:
            Child frame combining bit-level characteristics from both parents
        """
        child = parent1.copy()
        
        # For each position in the 29x29 frame
        for i in range(29):
            for j in range(29):
                # 50% chance to inherit this bit from parent2
                if random.random() < 0.5:
                    child[i, j, 0] = parent2[i, j, 0]
                    
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

    def generate_adversarial_attack(self, max_frames=100):
        """
        Main genetic algorithm to generate adversarial Fuzzy attacks.
        
        Process:
        1. Create balanced dataset (70% attack, 30% benign)
        2. Add benign frames directly (no modification needed)
        3. For each attack frame:
           - Apply initial random perturbations across the frame
           - Initialize genetic algorithm population
           - Evolve through generations using bit-level crossover/mutation
           - Stop when successful evasion achieved (confidence < 0.5)
        4. Return complete adversarial dataset with labels
        
        Args:
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
        # Find indices of benign frames (label = 0)
        benign_indices = np.where(self.y_test == 0)[0][:benign_count]
        if len(benign_indices) < benign_count:
            benign_count = len(benign_indices)
            print(f"Warning: Only {benign_count} benign frames available")
        
        # Find indices of attack frames (label = 1)
        attack_indices = np.where(self.y_test == 1)[0][:attack_count]
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
        for idx, i in enumerate(attack_indices):
            print(f"Processing attack frame {idx+1}/{len(attack_indices)}")
            
            # === STEP 4: Initialize frame with random perturbations ===
            # Apply initial fuzzy-style modifications to give GA a starting point
            frame_copy = self.x_test[i].copy()
            
            # For fuzzy attacks, randomly perturb bits across the entire frame
            for row_idx in range(29):
                # 50% chance to modify each row
                if random.random() < 0.5:
                    # Set one random bit in this row (avoiding priority bits 0-2)
                    bit_to_flip = random.randint(3, 28)
                    frame_copy[row_idx, bit_to_flip, 0] = 1
            
            # === STEP 5: Initialize genetic algorithm population ===
            # Temporarily increase mutation rate to ensure population diversity
            mut_rate = self.mutation_rate  # Save current mutation rate
            self.mutation_rate = 1         # Force mutation for all initial individuals
            
            # Create diverse initial population through mutation
            population = [self.mutate(frame_copy.copy()) for _ in range(self.population_size)]
            
            # Restore original mutation rate for evolution
            self.mutation_rate = mut_rate
            
            # === STEP 6: Evolve population through generations ===
            success_generation = -1  # Track when successful attack was found
            
            for generation in range(self.max_generations):
                print(f"Frame {idx+1}/{len(attack_indices)}, Generation {generation+1}/{self.max_generations}")
                
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
                
                # === STEP 7: Selection and reproduction for next generation ===
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
                    
                    # Create child through pixel-level crossover
                    child = self.crossover(
                        valid_individuals[p1_idx], 
                        valid_individuals[p2_idx]
                    )
                    
                    # Apply mutation to child
                    child = self.mutate(child)
                    new_pop.append(child)
                
                # Replace current population with new generation
                population = new_pop
            
            # === STEP 8: Handle end of genetic algorithm ===
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
        
        # === STEP 9: Create final labeled dataset ===
        # Create labels array: 0 for benign frames, 1 for attack frames
        y_final = np.zeros(len(final_test))
        y_final[benign_count:] = 1  # Attack samples start after benign samples
        
        return np.array(final_test), y_final, np.array(orig_frame), generations_needed

    def visualize_attack(self, original_frame, adversarial_frame, output_file="adversarial_fuzzy_attack.png"):
        """
        Create side-by-side visualization comparing original and adversarial fuzzy attack frames.
        
        Visualization shows binary CAN frames as images where:
        - Black pixels = 0 bits
        - White pixels = 1 bits
        
        This helps visualize the bit-level modifications made by the genetic algorithm
        for fuzzy attack evasion.
        
        Args:
            original_frame: Original fuzzy attack frame before modification
            adversarial_frame: Modified frame designed to evade detection
            output_file: Output filename for the visualization image
        """
        plt.figure(figsize=(12, 6))
        
        # Plot original fuzzy attack frame
        plt.subplot(1, 2, 1)
        plt.imshow(original_frame[:, :, 0], cmap='binary_r', vmin=0, vmax=1)
        plt.title("Original Fuzzy Attack (Black=0, White=1)")
        
        # Plot adversarial fuzzy attack frame
        plt.subplot(1, 2, 2)
        plt.imshow(adversarial_frame[:, :, 0], cmap='binary_r', vmin=0, vmax=1)
        plt.title("Adversarial Fuzzy Attack")
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Visualization saved to {output_file}")

    def evaluate_attack_effectiveness(self, original_frame, adversarial_frame):
        """
        Compare IDS confidence scores for original vs adversarial fuzzy attack frames.
        
        Prints detailed effectiveness analysis including:
        - Original attack confidence score
        - Adversarial attack confidence score  
        - Success determination (< 0.5 = successful evasion)
        
        Args:
            original_frame: Original fuzzy attack frame
            adversarial_frame: Modified adversarial fuzzy attack frame
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
    Experimental function to test the effect of different mutation rates on fuzzy attack performance.
    
    Tests mutation rates from 0.1 to 1.0 and measures average generations needed
    to find successful adversarial fuzzy attacks. For fuzzy attacks, this helps optimize:
    - Exploration vs exploitation balance
    - Bit-flipping frequency across the entire frame
    - Population diversity maintenance
    
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
        
        # Create fuzzy attack generator with specific mutation rate
        attack = AdversarialFuzzyAttack(
            model_path=model_path,
            population_size=100,      # Keep constant for fair comparison
            max_generations=75,       # Keep constant for fair comparison
            mutation_rate=rate        # Variable being tested
        )
        
        # Generate adversarial attacks and collect performance statistics
        _, _, _, generations_stats = attack.generate_adversarial_attack(max_frames=max_frames)
        
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

def plot_mutation_rate_results(results):
    """
    Create line plot showing the relationship between mutation rate and fuzzy attack performance.
    
    For fuzzy attacks, higher mutation rates may be more effective due to:
    - Need for significant frame modifications
    - Bit-flipping approach benefits from higher rates
    - More aggressive perturbation requirements
    
    Args:
        results: Dictionary mapping mutation rates to average generations needed
    """
    rates = list(results.keys())
    avgs = [results[r] for r in rates]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rates, avgs, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Mutation Rate')
    plt.ylabel('Average Number of Generations')
    plt.title('Effect of Mutation Rate on Fuzzy Adversarial Attack Generations')
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.savefig('fuzzy_mutation_rate_vs_generations.png')
    plt.close()
    
    print("Mutation rate experiment plot saved as 'fuzzy_mutation_rate_vs_generations.png'")

###################################################
# Main Execution Function
###################################################
def main():
    """
    Main function that orchestrates the complete adversarial fuzzy attack generation and evaluation process.
    
    Process:
    1. Load pre-trained IDS model for fuzzy attacks
    2. Generate adversarial fuzzy attack dataset (or load if exists)
    3. Evaluate model performance on adversarial examples
    4. Compute detailed performance metrics
    5. Create visualizations of misclassified examples
    6. Run mutation rate optimization experiments
    7. Generate performance analysis plots
    """
    # === STEP 1: Setup and Configuration ===
    model_path = "Fuzzy_final_model.h5"
    
    # === STEP 2: Generate or Load Adversarial Dataset ===
    # Create fuzzy attack generator instance with tuned parameters
    attack = AdversarialFuzzyAttack(
        model_path=model_path,
        population_size=100,    # Population size for genetic algorithm
        max_generations=75,     # Maximum evolution generations
        mutation_rate=0.2       # Mutation probability per individual
    )
    
    # Check if adversarial dataset already exists to avoid regeneration
    if os.path.exists("adversarial_fuzzy_attack.npz"):
        print("Adversarial attack already exists. Loading from file...")
        try:
            # Load pre-computed adversarial dataset
            data = np.load("adversarial_fuzzy_attack.npz")
            final_test = data['final_test']  # Adversarial/benign frames
            y_test = data['y_test']          # True labels
            x_test = data['x_test']          # Original frames
        except Exception as e:
            print(f"Error loading file: {e}")
            # Regenerate the data if loading fails
            final_test, y_test, x_test, _ = attack.generate_adversarial_attack(max_frames=100)
            try:
                # Save newly generated dataset
                np.savez("adversarial_fuzzy_attack.npz", 
                         final_test=final_test, 
                         y_test=y_test, 
                         x_test=x_test)
                print(f"Adversarial attack generated and saved to adversarial_fuzzy_attack.npz")
            except Exception as e:
                print(f"Error saving file: {e}")
    else:
        # Generate new adversarial fuzzy attack dataset
        try:
            print("Generating new adversarial fuzzy attack dataset...")
            final_test, y_test, x_test, _ = attack.generate_adversarial_attack(max_frames=100)
            
            # Save generated dataset for future use
            np.savez("adversarial_fuzzy_attack.npz", 
                     final_test=final_test, 
                     y_test=y_test, 
                     x_test=x_test)
            print(f"Adversarial attack generated and saved to adversarial_fuzzy_attack.npz")
        except Exception as e:
            print(f"Error during attack generation or saving: {e}")
    
    # === STEP 3: Evaluate Model Performance on Adversarial Examples ===
    print("Evaluating model performance on adversarial fuzzy attack dataset...")
    test_loss, test_accuracy = attack.model.evaluate(final_test, y_test, verbose=1)
    
    # Save basic evaluation metrics
    with open(f"adversarial_fuzzy_test.txt", "w") as f:
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
    plot_confusion_matrix(cm, classes=['Normal', 'Attack'], suffix="adv_fuzzy_attack", normalize=True,
                         title='Normalized Confusion Matrix')
    
    # === STEP 6: Save Comprehensive Evaluation Report ===
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])
    
    with open(f"evaluation_metrics_fuzzy_adv.txt", "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write(f"False Negative Rate (FNR): {FNR:.4f}\n")
        f.write(f"Error Rate (ER): {ER:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Results for fuzzy adversarial attack (test set) saved")
    
    # === STEP 7: Create Visualizations of Successful Attacks ===
    print("Creating visualizations of misclassified fuzzy attack examples...")
    
    # Find and visualize up to 5 successful adversarial fuzzy attacks
    # (cases where true label = attack but model predicted normal)
    cnt = 0
    for i in range(len(y_test)):
        if cnt == 5:  # Limit to 5 examples to avoid clutter
            break
            
        # Look for misclassified attack samples (successful evasions)
        if y_test[i] != y_pred[i] and y_test[i] == 1:
            # Create side-by-side comparison
            plt.figure(figsize=(12, 6))
            
            # Original fuzzy attack frame
            plt.subplot(1, 2, 1)
            plt.imshow(x_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
            plt.title(f"Original Fuzzy Attack (True: {y_test[i]})")
            
            # Adversarial fuzzy attack frame (successfully evaded detection)
            plt.subplot(1, 2, 2)
            plt.imshow(final_test[i][:, :, 0], cmap='binary_r', vmin=0, vmax=1)
            plt.title(f"Adversarial Fuzzy Attack (Predicted: {y_pred[i]})")
            
            plt.tight_layout()
            plt.savefig(f"fuzzy_attack_comparison_{cnt}.png")
            plt.close()
            print(f"Attack comparison saved as fuzzy_attack_comparison_{cnt}.png")
            cnt += 1
    
    # === STEP 8: Run Parameter Optimization Experiments ===
    print("\n===== RUNNING MUTATION RATE EXPERIMENT =====")
    # Test different mutation rates to find optimal balance for fuzzy attacks
    mutation_results = run_mutation_rate_experiment(model_path, max_frames=20)
    plot_mutation_rate_results(mutation_results)

###################################################
# Script Entry Point
###################################################
if __name__ == "__main__":
    main()
