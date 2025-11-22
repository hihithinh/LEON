#!/usr/bin/env python3
"""
Script to extract comprehensive training metrics from LEON running_log.txt
Extracts all relevant data for research paper reporting
"""

import re
import csv
from pathlib import Path
from collections import defaultdict


def extract_gmrl_data(log_file):
    """Extract train_gmrl and test_gmrl data per iteration"""
    train_gmrl_pattern = r'\[INFO\] train_gmrl =\[(.*?)\]'
    test_gmrl_pattern = r'\[INFO\] test_gmrl =\[(.*?)\]'
    
    gmrl_data = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Find all train_gmrl entries
        train_matches = list(re.finditer(train_gmrl_pattern, content))
        test_matches = list(re.finditer(test_gmrl_pattern, content))
        
        for i, (train_match, test_match) in enumerate(zip(train_matches, test_matches)):
            train_values = [float(v.strip()) for v in train_match.group(1).split(',')]
            test_values = [float(v.strip()) for v in test_match.group(1).split(',')]
            
            # Get the last value (current iteration's GMRL)
            gmrl_data.append({
                'iteration': i,
                'train_gmrl': train_values[-1],
                'test_gmrl': test_values[-1]
            })
    
    return gmrl_data


def extract_iteration_metrics(log_file):
    """Extract iteration-level metrics: dptime, iteration_time, gmrl_test_time"""
    dptime_pattern = r'\[INFO\] dptime = ([\d.]+)'
    iter_time_pattern = r'\[INFO\] \[ITERATION\] Iteration (\d+) completed in ([\d.]+)s'
    gmrl_test_time_pattern = r'\[INFO\] GMRL test time =([\d.]+)s'
    
    iteration_metrics = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        dptime_matches = list(re.finditer(dptime_pattern, content))
        iter_time_matches = list(re.finditer(iter_time_pattern, content))
        gmrl_test_matches = list(re.finditer(gmrl_test_time_pattern, content))
        
        for i, (dp_match, iter_match, gmrl_match) in enumerate(zip(dptime_matches, iter_time_matches, gmrl_test_matches)):
            iteration_metrics.append({
                'iteration': i,
                'dptime': float(dp_match.group(1)),
                'iteration_time': float(iter_match.group(2)),
                'gmrl_test_time': float(gmrl_match.group(1))
            })
    
    return iteration_metrics


def extract_training_data_stats(log_file):
    """Extract trainpair num, experience num, best exp num per iteration"""
    pattern = r'trainpair num =(\d+),now experience num = (\d+),best exp num  = (\d+)'
    
    training_stats = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                training_stats.append({
                    'iteration': len(training_stats),
                    'trainpair_num': int(match.group(1)),
                    'experience_num': int(match.group(2)),
                    'best_exp_num': int(match.group(3))
                })
    
    return training_stats


def extract_all_epoch_metrics(log_file):
    """Extract loss, accuracy, train_time, test_time for ALL epochs of ALL models"""
    pattern = r'\[TRAINING\] Epoch (\d+), Model (\d+): loss=([\d.]+), accuracy=([\d.]+), train_time=([\d.]+)s, test_time=([\d.]+)s'
    
    epoch_metrics = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch_metrics.append({
                    'epoch': int(match.group(1)),
                    'model_level': int(match.group(2)),
                    'loss': float(match.group(3)),
                    'accuracy': float(match.group(4)),
                    'train_time': float(match.group(5)),
                    'test_time': float(match.group(6))
                })
    
    return epoch_metrics


def extract_model_convergence(log_file):
    """Extract convergence epoch for each model"""
    pattern = r'\[TRAINING\] Model (\d+) converged at epoch (\d+)'
    
    convergence_data = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                convergence_data.append({
                    'model_level': int(match.group(1)),
                    'converged_epoch': int(match.group(2))
                })
    
    return convergence_data


def extract_model_samples(log_file):
    """Extract number of training samples for each model"""
    pattern = r'\[TRAINING\] Starting training for model level (\d+) with (\d+) samples'
    
    samples_data = {}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                model_level = int(match.group(1))
                samples = int(match.group(2))
                # Keep only first occurrence (iteration 0)
                if model_level not in samples_data:
                    samples_data[model_level] = samples
    
    return [{'model_level': k, 'num_samples': v} for k, v in sorted(samples_data.items())]


def get_final_model_metrics(epoch_metrics):
    """Get final (best) metrics for each model from all epochs"""
    model_final = {}
    
    for metric in epoch_metrics:
        model_level = metric['model_level']
        # Keep the last epoch's metrics for each model
        model_final[model_level] = metric
    
    return [v for k, v in sorted(model_final.items())]


def save_to_csv(data, output_file, fieldnames):
    """Save data to CSV file"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"✓ Saved {len(data)} rows to {output_file}")


def main():
    log_file = Path('../_1116-142828/running_log.txt')
    output_dir = Path('../_1116-142828')
    
    if not log_file.exists():
        print(f"Error: {log_file} not found!")
        return
    
    print(f"Reading log file: {log_file}")
    print("=" * 80)
    
    # 1. Extract GMRL data
    print("\n1. Extracting GMRL metrics (train & test)...")
    gmrl_data = extract_gmrl_data(log_file)
    save_to_csv(gmrl_data, output_dir / 'gmrl_data.csv', 
                ['iteration', 'train_gmrl', 'test_gmrl'])
    
    # 2. Extract iteration metrics
    print("\n2. Extracting iteration-level metrics...")
    iteration_metrics = extract_iteration_metrics(log_file)
    save_to_csv(iteration_metrics, output_dir / 'iteration_metrics.csv',
                ['iteration', 'dptime', 'iteration_time', 'gmrl_test_time'])
    
    # 3. Extract training data statistics
    print("\n3. Extracting training data statistics...")
    training_stats = extract_training_data_stats(log_file)
    save_to_csv(training_stats, output_dir / 'training_stats.csv',
                ['iteration', 'trainpair_num', 'experience_num', 'best_exp_num'])
    
    # 4. Extract all epoch metrics
    print("\n4. Extracting all epoch metrics for all models...")
    all_epoch_metrics = extract_all_epoch_metrics(log_file)
    save_to_csv(all_epoch_metrics, output_dir / 'all_epochs.csv',
                ['epoch', 'model_level', 'loss', 'accuracy', 'train_time', 'test_time'])
    
    # 5. Extract final model metrics
    print("\n5. Extracting final model metrics...")
    final_metrics = get_final_model_metrics(all_epoch_metrics)
    save_to_csv(final_metrics, output_dir / 'model_final_metrics.csv',
                ['epoch', 'model_level', 'loss', 'accuracy', 'train_time', 'test_time'])
    
    # 6. Extract model convergence info
    print("\n6. Extracting model convergence information...")
    convergence_data = extract_model_convergence(log_file)
    save_to_csv(convergence_data, output_dir / 'model_convergence.csv',
                ['model_level', 'converged_epoch'])
    
    # 7. Extract model samples
    print("\n7. Extracting model training samples...")
    samples_data = extract_model_samples(log_file)
    save_to_csv(samples_data, output_dir / 'model_samples.csv',
                ['model_level', 'num_samples'])
    
    print("\n" + "=" * 80)
    print("✓ Data extraction complete!")
    print(f"\nGenerated CSV files in {output_dir}:")
    print("  - gmrl_data.csv: GMRL performance per iteration")
    print("  - iteration_metrics.csv: Time metrics per iteration")
    print("  - training_stats.csv: Training data statistics")
    print("  - all_epochs.csv: All training epochs for all models")
    print("  - model_final_metrics.csv: Final metrics for each model")
    print("  - model_convergence.csv: Convergence epochs")
    print("  - model_samples.csv: Training sample counts")
    print("=" * 80)


if __name__ == '__main__':
    main()
