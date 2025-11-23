text = """
import wandb
import json
import hashlib
import os
import sys
from pathlib import Path


def build_config_from_wandb(wandb_config):
    """Build a nested config dictionary from flat wandb.config parameters."""
    config = {}
    
    # Global parameters
    config['dataset'] = wandb_config.get('dataset', 'cifar100')
    config['model'] = wandb_config.get('model', 'resnet18')
    config['pretrained'] = wandb_config.get('pretrained', False)
    config['epochs'] = wandb_config.get('epochs', 100)
    config['batch_size'] = wandb_config.get('batch_size', 128)
    config['max_batch_size'] = wandb_config.get('max_batch_size', 512)
    config['min_batch_size'] = wandb_config.get('min_batch_size', 1)
    config['lr'] = wandb_config.get('lr', 0.01)
    config['optimizer'] = wandb_config.get('optimizer', 'sgd')
    config['scheduler'] = wandb_config.get('scheduler')
    config['batch_schedule'] = wandb_config.get('batch_schedule', 'none')
    config['early_stopping'] = wandb_config.get('early_stopping', True)
    config['use_amp'] = wandb_config.get('use_amp', True)
    config['label_smoothing'] = wandb_config.get('label_smoothing', 0.0)
    
    # Optimizer configs - always include all with defaults, override if provided
    config['optimizer_configs'] = {}
    
    # SGD
    config['optimizer_configs']['sgd'] = {
        'momentum': wandb_config.get('optimizer_configs.sgd.momentum', 0.9),
        'nesterov': wandb_config.get('optimizer_configs.sgd.nesterov', False),
        'weight_decay': wandb_config.get('optimizer_configs.sgd.weight_decay', 5e-4)
    }
    
    # Adam
    config['optimizer_configs']['adam'] = {
        'betas': wandb_config.get('optimizer_configs.adam.betas', [0.9, 0.999]),
        'weight_decay': wandb_config.get('optimizer_configs.adam.weight_decay', 0.0)
    }
    
    # AdamW
    config['optimizer_configs']['adamw'] = {
        'betas': wandb_config.get('optimizer_configs.adamw.betas', [0.9, 0.999]),
        'weight_decay': wandb_config.get('optimizer_configs.adamw.weight_decay', 0.01)
    }
    
    # Muon
    config['optimizer_configs']['muon'] = {
        'base_optimizer': wandb_config.get('optimizer_configs.muon.base_optimizer', 'sgd'),
        'momentum': wandb_config.get('optimizer_configs.muon.momentum', 0.85),
        'nesterov': wandb_config.get('optimizer_configs.muon.nesterov', True)
    }
    
    # SAM
    config['optimizer_configs']['sam'] = {
        'base_optimizer': wandb_config.get('optimizer_configs.sam.base_optimizer', 'sgd'),
        'rho': wandb_config.get('optimizer_configs.sam.rho', 0.05),
        'adaptive': wandb_config.get('optimizer_configs.sam.adaptive', False),
        'momentum': wandb_config.get('optimizer_configs.sam.momentum', 0.9),
        'weight_decay': wandb_config.get('optimizer_configs.sam.weight_decay', 1e-4)
    }
    
    # Scheduler configs - include if scheduler is set
    config['scheduler_configs'] = {}
    
    # StepLR
    config['scheduler_configs']['steplr'] = {
        'step_size': wandb_config.get('scheduler_configs.steplr.step_size', 30),
        'gamma': wandb_config.get('scheduler_configs.steplr.gamma', 0.1),
        'last_epoch': wandb_config.get('scheduler_configs.steplr.last_epoch', -1)
    }
    
    # ReduceLROnPlateau
    config['scheduler_configs']['reducelronplateau'] = {
        'factor': wandb_config.get('scheduler_configs.reducelronplateau.factor', 0.5),
        'patience': wandb_config.get('scheduler_configs.reducelronplateau.patience', 5),
        'threshold': wandb_config.get('scheduler_configs.reducelronplateau.threshold', 1e-4),
        'threshold_mode': wandb_config.get('scheduler_configs.reducelronplateau.threshold_mode', 'rel'),
        'cooldown': wandb_config.get('scheduler_configs.reducelronplateau.cooldown', 0),
        'min_lr': wandb_config.get('scheduler_configs.reducelronplateau.min_lr', 1e-6),
        'eps': wandb_config.get('scheduler_configs.reducelronplateau.eps', 1e-8)
    }
    
    # Batch scheduler configs - include all with defaults
    config['batch_scheduler_config'] = {}
    
    # Linear
    config['batch_scheduler_config']['linear'] = {
        'start_factor': wandb_config.get('batch_scheduler_config.linear.start_factor', 1.0),
        'end_factor': wandb_config.get('batch_scheduler_config.linear.end_factor', 4.0),
        'milestone': wandb_config.get('batch_scheduler_config.linear.milestone', 250)
    }
    
    # Exponential
    config['batch_scheduler_config']['exponential'] = {
        'step_size': wandb_config.get('batch_scheduler_config.exponential.step_size', 10),
        'gamma': wandb_config.get('batch_scheduler_config.exponential.gamma', 2.0)
    }
    
    # Step
    config['batch_scheduler_config']['step'] = {
        'step_size': wandb_config.get('batch_scheduler_config.step.step_size', 10),
        'gamma': wandb_config.get('batch_scheduler_config.step.gamma', 2.0)
    }
    
    # Plateau
    config['batch_scheduler_config']['plateau'] = {
        'mode': wandb_config.get('batch_scheduler_config.plateau.mode', 'max'),
        'factor': wandb_config.get('batch_scheduler_config.plateau.factor', 2.0),
        'patience': wandb_config.get('batch_scheduler_config.plateau.patience', 10),
        'threshold': wandb_config.get('batch_scheduler_config.plateau.threshold', 1e-4),
        'threshold_mode': wandb_config.get('batch_scheduler_config.plateau.threshold_mode', 'rel'),
        'cooldown': wandb_config.get('batch_scheduler_config.plateau.cooldown', 0)
    }
    
    # Early stopping config
    config['early_stopping_config'] = {
        'patience': wandb_config.get('early_stopping_config.patience', 10),
        'min_delta': wandb_config.get('early_stopping_config.min_delta', 0)
    }
    
    # Transforms - include all with defaults
    config['transforms'] = {
        'train': {},
        'test': {}
    }
    
    # Train transforms
    config['transforms']['train']['random_resized_crop'] = {
        'enabled': wandb_config.get('transforms.train.random_resized_crop.enabled', False),
        'p': wandb_config.get('transforms.train.random_resized_crop.p', 1.0),
        'size': wandb_config.get('transforms.train.random_resized_crop.size', [32, 32]),
        'scale': wandb_config.get('transforms.train.random_resized_crop.scale', [0.8, 0.95])
    }
    
    config['transforms']['train']['random_translate'] = {
        'enabled': wandb_config.get('transforms.train.random_translate.enabled', False),
        'p': wandb_config.get('transforms.train.random_translate.p', 0.4),
        'translate': wandb_config.get('transforms.train.random_translate.translate', [0.1, 0.2])
    }
    
    config['transforms']['train']['random_rotation'] = {
        'enabled': wandb_config.get('transforms.train.random_rotation.enabled', False),
        'p': wandb_config.get('transforms.train.random_rotation.p', 0.4),
        'degrees': wandb_config.get('transforms.train.random_rotation.degrees', 15)
    }
    
    config['transforms']['train']['random_horizontal_flip'] = {
        'enabled': wandb_config.get('transforms.train.random_horizontal_flip.enabled', False),
        'p': wandb_config.get('transforms.train.random_horizontal_flip.p', 0.5)
    }
    
    config['transforms']['train']['random_vertical_flip'] = {
        'enabled': wandb_config.get('transforms.train.random_vertical_flip.enabled', False),
        'p': wandb_config.get('transforms.train.random_vertical_flip.p', 0.4)
    }
    
    config['transforms']['train']['cutmix_mixup'] = {
        'enabled': wandb_config.get('transforms.train.cutmix_mixup.enabled', False),
        'p': wandb_config.get('transforms.train.cutmix_mixup.p', 1.0),
        'alpha': wandb_config.get('transforms.train.cutmix_mixup.alpha', 0.2),
        'max_ramp_epochs': wandb_config.get('transforms.train.cutmix_mixup.max_ramp_epochs', 1)
    }
    
    # Test transforms
    config['transforms']['test']['resized_crop'] = {
        'enabled': wandb_config.get('transforms.test.resized_crop.enabled', False),
        'crop_size': wandb_config.get('transforms.test.resized_crop.crop_size', 28),
        'size': wandb_config.get('transforms.test.resized_crop.size', [32, 32])
    }
    
    config['transforms']['test']['fixed_affine'] = {
        'enabled': wandb_config.get('transforms.test.fixed_affine.enabled', False),
        'translations': wandb_config.get('transforms.test.fixed_affine.translations', [[3, 3], [-3, -3]])
    }
    
    config['transforms']['test']['random_rotation'] = {
        'enabled': wandb_config.get('transforms.test.random_rotation.enabled', False),
        'degrees': wandb_config.get('transforms.test.random_rotation.degrees', 15)
    }
    
    config['transforms']['test']['random_horizontal_flip'] = {
        'enabled': wandb_config.get('transforms.test.random_horizontal_flip.enabled', False)
    }
    
    config['transforms']['test']['random_vertical_flip'] = {
        'enabled': wandb_config.get('transforms.test.random_vertical_flip.enabled', False)
    }
    
    return config


def generate_config_hash(config_dict):
    """Generate a random hash for the config file name."""
    config_str = json.dumps(config_dict, sort_keys=True)
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()[:16]


def main():
    wandb.login(key="af71e9ae5a91896c01f2d861c535adab454bbe8c")

    # Initialize wandb for the sweep
    wandb.init()
    
    # Build config from wandb.config
    config = build_config_from_wandb(wandb.config)
    
    # Generate a unique hash for this config
    config_hash = generate_config_hash(config)
    config_filename = f"config-sweep-{config_hash}.json"
    config_path = Path(config_filename)
    
    # Write config to JSON file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created config file: {config_path}")
    
    try:
        # Import and call main from main.py
        # We need to modify sys.argv to pass the config path
        original_argv = sys.argv.copy()
        sys.argv = ['main.py', '-c', str(config_path.stem)]
        
        # Import main function
        from main import main as train_main
        
        # Run training
        val_acc = train_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        # Log final validation accuracy (it should already be logged during training, but log it again for clarity)
        wandb.log({'final_val_acc': val_acc})
        wandb.summary['final_val_acc'] = val_acc
        
        print(f"Final Validation Accuracy: {val_acc:.2f}%")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        # Clean up config file
        if config_path.exists():
            config_path.unlink()
            print(f"Cleaned up config file: {config_path}")
        
        # Finish wandb run
        wandb.finish()


if __name__ == '__main__':
    main()


"""

with open("sweep.py", "w") as f:
    f.write(text)