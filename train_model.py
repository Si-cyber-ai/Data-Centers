"""
Main Training Script for DQN Agent

Trains the reinforcement learning agent on the data center cooling task.
"""

import argparse
import yaml
import os
from datetime import datetime

from rl_agent.training_pipeline import TrainingPipeline


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train DQN agent for data center cooling optimization"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of training episodes (overrides config)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory to save logs'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate trained agent instead of training'
    )
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("AI-Based Data Center Cooling Optimization")
    print("DQN Training Pipeline")
    print("=" * 70)
    print(f"\nConfiguration: {args.config}")
    print(f"Grid Size: {config['simulation']['grid_size']}")
    print(f"Action Space: {config['rl']['action_dim']} discrete actions")
    print(f"Workload Pattern: {config['workload']['synthetic_pattern']}")
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        pipeline.agent.load_checkpoint(args.resume)
    
    if args.evaluate:
        # Evaluation mode
        print("\n" + "=" * 70)
        print("EVALUATION MODE")
        print("=" * 70)
        
        if not args.resume:
            print("Warning: No checkpoint specified. Using untrained agent.")
        
        eval_results = pipeline.evaluate(
            num_episodes=args.eval_episodes,
            render=False
        )
        
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"Average Temperature: {eval_results['avg_temperature']:.2f}°C")
        print(f"Total Violations: {eval_results['total_violations']}")
        print(f"Success Rate: {eval_results['success_rate']*100:.1f}%")
        
    else:
        # Training mode
        print("\n" + "=" * 70)
        print("TRAINING MODE")
        print("=" * 70)
        
        num_episodes = args.episodes if args.episodes else config['rl']['training_episodes']
        print(f"Training for {num_episodes} episodes...")
        
        start_time = datetime.now()
        
        # Train agent
        training_summary = pipeline.train(num_episodes=num_episodes)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Total Episodes: {training_summary['total_episodes']}")
        print(f"Training Duration: {duration/60:.1f} minutes")
        print(f"Final Average Reward: {training_summary['final_avg_reward']:.2f}")
        print(f"Final Average Temperature: {training_summary['final_avg_temp']:.2f}°C")
        print(f"Total Violations: {training_summary['total_violations']}")
        print(f"Best Episode: {training_summary['best_episode']} "
              f"(Reward: {training_summary['best_reward']:.2f})")
        print(f"Final Epsilon: {training_summary['final_epsilon']:.4f}")
        
        print(f"\nModel saved to: {args.checkpoint_dir}/dqn_final.pth")
        print(f"Training plots saved to: {args.log_dir}/")
        
        # Evaluate trained agent
        print("\n" + "=" * 70)
        print("FINAL EVALUATION")
        print("=" * 70)
        
        eval_results = pipeline.evaluate(num_episodes=10)
        print(f"Final Evaluation Reward: {eval_results['avg_reward']:.2f}")
        print(f"Final Success Rate: {eval_results['success_rate']*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("Training pipeline completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
