#!/usr/bin/env python3
"""
Script to generate comprehensive plots for LEON research paper
Creates publication-quality figures for experimental results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def plot_gmrl_performance(csv_file, output_file):
    """
    Plot 1: GMRL Performance Over Iterations
    Shows train and test GMRL convergence
    """
    df = pd.read_csv(csv_file)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines
    ax.plot(df['iteration'], df['train_gmrl'], 
            marker='o', linewidth=2.5, markersize=7, 
            label='Train GMRL', color='#2E86AB', alpha=0.8)
    ax.plot(df['iteration'], df['test_gmrl'], 
            marker='s', linewidth=2.5, markersize=7, 
            label='Test GMRL', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Training Iteration', fontweight='bold')
    ax.set_ylabel('GMRL (Geometric Mean Relative Latency)', fontweight='bold')
    ax.set_title('LEON Model Performance: GMRL Convergence', fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add horizontal line at GMRL=1.0 (PostgreSQL baseline)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, 
               alpha=0.5, label='PostgreSQL Baseline')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_iteration_time_breakdown(csv_file, output_file):
    """
    Plot 2: Iteration Time Breakdown
    Shows dptime, gmrl_test_time, and total iteration time
    """
    df = pd.read_csv(csv_file)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = df['iteration']
    width = 0.25
    
    # Convert to minutes for better readability
    dptime_min = df['dptime'] / 60
    gmrl_time_min = df['gmrl_test_time'] / 60
    other_time_min = (df['iteration_time'] - df['dptime'] - df['gmrl_test_time']) / 60
    
    ax.bar(x - width, dptime_min, width, label='DP Search Time', color='#F77F00', alpha=0.8)
    ax.bar(x, gmrl_time_min, width, label='GMRL Test Time', color='#06A77D', alpha=0.8)
    ax.bar(x + width, other_time_min, width, label='Training Time', color='#4361EE', alpha=0.8)
    
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Time (minutes)', fontweight='bold')
    ax.set_title('Training Time Breakdown per Iteration', fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_training_data_growth(csv_file, output_file):
    """
    Plot 3: Training Data Growth
    Shows growth of training pairs and experience
    """
    df = pd.read_csv(csv_file)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['iteration'], df['trainpair_num'], 
            marker='o', linewidth=2.5, markersize=7,
            label='Training Pairs', color='#E63946', alpha=0.8)
    ax.plot(df['iteration'], df['experience_num'], 
            marker='s', linewidth=2.5, markersize=7,
            label='Experience Pool', color='#06A77D', alpha=0.8)
    ax.plot(df['iteration'], df['best_exp_num'], 
            marker='^', linewidth=2.5, markersize=7,
            label='Best Experience', color='#4361EE', alpha=0.8)
    
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Training Data Accumulation Over Iterations', fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_model_performance_grid(csv_file, samples_file, output_file):
    """
    Plot 4: Model Performance Metrics Grid (2x2)
    Shows loss, accuracy, train_time, test_time for each model level
    """
    df = pd.read_csv(csv_file)
    samples_df = pd.read_csv(samples_file)
    
    # Merge with samples data
    df = df.merge(samples_df, on='model_level', how='left')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Across Join Levels (Final Epoch)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    colors = {
        'loss': '#E63946',
        'accuracy': '#06A77D',
        'train_time': '#F77F00',
        'test_time': '#4361EE'
    }
    
    # Plot 1: Loss
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['model_level'], df['loss'], color=colors['loss'], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Model Level (Number of Joins)', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training Loss per Model Level', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Accuracy
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['model_level'], df['accuracy'], color=colors['accuracy'], alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Model Level (Number of Joins)', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Prediction Accuracy per Model Level', fontweight='bold')
    ax2.set_ylim([0.7, 1.0])
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Training Time
    ax3 = axes[1, 0]
    bars3 = ax3.bar(df['model_level'], df['train_time'], color=colors['train_time'], alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Model Level (Number of Joins)', fontweight='bold')
    ax3.set_ylabel('Time (seconds)', fontweight='bold')
    ax3.set_title('Training Time per Epoch', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Test Time
    ax4 = axes[1, 1]
    bars4 = ax4.bar(df['model_level'], df['test_time'], color=colors['test_time'], alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Model Level (Number of Joins)', fontweight='bold')
    ax4.set_ylabel('Time (seconds)', fontweight='bold')
    ax4.set_title('Inference Time per Epoch', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_model_convergence(convergence_file, samples_file, output_file):
    """
    Plot 5: Model Convergence Analysis
    Shows convergence epoch and training samples for each model
    """
    conv_df = pd.read_csv(convergence_file)
    samples_df = pd.read_csv(samples_file)
    
    # Merge data
    df = conv_df.merge(samples_df, on='model_level', how='left')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Training Convergence Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Convergence Epoch
    bars1 = ax1.bar(df['model_level'], df['converged_epoch'], 
                    color='#A23B72', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Model Level (Number of Joins)', fontweight='bold')
    ax1.set_ylabel('Convergence Epoch', fontweight='bold')
    ax1.set_title('Epochs to Convergence', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Training Samples
    bars2 = ax2.bar(df['model_level'], df['num_samples'], 
                    color='#2E86AB', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Model Level (Number of Joins)', fontweight='bold')
    ax2.set_ylabel('Number of Training Samples', fontweight='bold')
    ax2.set_title('Training Dataset Size', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_learning_curves(all_epochs_file, output_file):
    """
    Plot 6: Learning Curves for Selected Models
    Shows loss and accuracy over epochs for key model levels
    """
    df = pd.read_csv(all_epochs_file)
    
    # Select representative models (e.g., levels 2, 5, 8, 11)
    selected_levels = [2, 5, 8, 11]
    df_selected = df[df['model_level'].isin(selected_levels)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Learning Curves for Representative Model Levels', 
                 fontsize=14, fontweight='bold')
    
    colors = ['#E63946', '#F77F00', '#06A77D', '#4361EE']
    
    # Plot 1: Loss curves
    for i, level in enumerate(selected_levels):
        level_data = df_selected[df_selected['model_level'] == level]
        ax1.plot(level_data['epoch'], level_data['loss'], 
                marker='o', linewidth=2, markersize=5,
                label=f'Level {level}', color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Loss Convergence', fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Accuracy curves
    for i, level in enumerate(selected_levels):
        level_data = df_selected[df_selected['model_level'] == level]
        ax2.plot(level_data['epoch'], level_data['accuracy'], 
                marker='s', linewidth=2, markersize=5,
                label=f'Level {level}', color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Accuracy Improvement', fontweight='bold')
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def generate_summary_statistics(data_dir):
    """Generate and print summary statistics"""
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    # GMRL results
    gmrl_df = pd.read_csv(data_dir / 'gmrl_data.csv')
    print(f"\n1. GMRL Performance:")
    print(f"   Final Train GMRL: {gmrl_df['train_gmrl'].iloc[-1]:.4f}")
    print(f"   Final Test GMRL: {gmrl_df['test_gmrl'].iloc[-1]:.4f}")
    print(f"   Best Train GMRL: {gmrl_df['train_gmrl'].min():.4f}")
    print(f"   Best Test GMRL: {gmrl_df['test_gmrl'].min():.4f}")
    print(f"   Speedup vs PostgreSQL: {1/gmrl_df['test_gmrl'].iloc[-1]:.2f}x")
    
    # Model metrics
    model_df = pd.read_csv(data_dir / 'model_final_metrics.csv')
    print(f"\n2. Model Performance:")
    print(f"   Number of models trained: {len(model_df)}")
    print(f"   Average accuracy: {model_df['accuracy'].mean():.4f}")
    print(f"   Best accuracy: {model_df['accuracy'].max():.4f}")
    print(f"   Average loss: {model_df['loss'].mean():.4f}")
    
    # Training data
    stats_df = pd.read_csv(data_dir / 'training_stats.csv')
    print(f"\n3. Training Data:")
    print(f"   Final training pairs: {stats_df['trainpair_num'].iloc[-1]:,}")
    print(f"   Total experience collected: {stats_df['experience_num'].iloc[-1]:,}")
    
    # Time metrics
    iter_df = pd.read_csv(data_dir / 'iteration_metrics.csv')
    total_time = iter_df['iteration_time'].sum()
    print(f"\n4. Training Time:")
    print(f"   Total training time: {total_time/3600:.2f} hours")
    print(f"   Average iteration time: {iter_df['iteration_time'].mean()/60:.2f} minutes")
    print(f"   Total DP search time: {iter_df['dptime'].sum()/3600:.2f} hours")
    
    print("="*80)


def main():
    data_dir = Path('.')  # Use current directory (report/)
    output_dir = data_dir
    
    print("Generating research paper plots...")
    print("="*80)
    
    # Check if CSV files exist
    required_files = ['gmrl_data.csv', 'iteration_metrics.csv', 'training_stats.csv',
                     'model_final_metrics.csv', 'model_convergence.csv', 
                     'model_samples.csv', 'all_epochs.csv']
    
    for file in required_files:
        if not (data_dir / file).exists():
            print(f"Error: {file} not found!")
            print("Please run extract_training_data.py first.")
            return
    
    # Generate all plots
    print("\n1. Generating GMRL performance plot...")
    plot_gmrl_performance(data_dir / 'gmrl_data.csv', 
                         output_dir / 'plot1_gmrl_performance.png')
    
    print("\n2. Generating iteration time breakdown...")
    plot_iteration_time_breakdown(data_dir / 'iteration_metrics.csv',
                                 output_dir / 'plot2_time_breakdown.png')
    
    print("\n3. Generating training data growth plot...")
    plot_training_data_growth(data_dir / 'training_stats.csv',
                             output_dir / 'plot3_data_growth.png')
    
    print("\n4. Generating model performance grid...")
    plot_model_performance_grid(data_dir / 'model_final_metrics.csv',
                               data_dir / 'model_samples.csv',
                               output_dir / 'plot4_model_performance.png')
    
    print("\n5. Generating model convergence analysis...")
    plot_model_convergence(data_dir / 'model_convergence.csv',
                          data_dir / 'model_samples.csv',
                          output_dir / 'plot5_convergence.png')
    
    print("\n6. Generating learning curves...")
    plot_learning_curves(data_dir / 'all_epochs.csv',
                        output_dir / 'plot6_learning_curves.png')
    
    # Generate summary statistics
    generate_summary_statistics(data_dir)
    
    print("\n" + "="*80)
    print("✓ All plots generated successfully!")
    print(f"\nPlots saved in: {output_dir}")
    print("\nGenerated plots:")
    print("  1. plot1_gmrl_performance.png - GMRL convergence over iterations")
    print("  2. plot2_time_breakdown.png - Training time breakdown")
    print("  3. plot3_data_growth.png - Training data accumulation")
    print("  4. plot4_model_performance.png - Model metrics grid (2x2)")
    print("  5. plot5_convergence.png - Convergence analysis")
    print("  6. plot6_learning_curves.png - Learning curves for key models")
    print("="*80)


if __name__ == '__main__':
    main()
