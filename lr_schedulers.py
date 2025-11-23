import torch.optim.lr_scheduler as lr_scheduler


def get_lr_scheduler(scheduler_name: str, optimizer, **kwargs):
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'steplr':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        last_epoch = kwargs.get('last_epoch', -1)
        return lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch
        )
    
    elif scheduler_name == 'reducelronplateau':
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 10)
        threshold = kwargs.get('threshold', 1e-4)
        threshold_mode = kwargs.get('threshold_mode', 'rel')
        cooldown = kwargs.get('cooldown', 0)
        min_lr = kwargs.get('min_lr', 0)
        eps = kwargs.get('eps', 1e-8)
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps
        )
    
    return None