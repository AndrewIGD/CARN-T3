from bs_scheduler import LinearBS, StepBS, IncreaseBSOnPlateau

def get_batch_size_scheduler(scheduler_name: str, dataloader, **kwargs):
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'linear':
        start_factor = kwargs.get('start_factor', 1.0)
        end_factor = kwargs.get('end_factor', 1.0)
        milestone = kwargs.get('milestone', 5)
        max_batch_size = kwargs.get('max_batch_size', None)
        min_batch_size = kwargs.get('min_batch_size', 1)
        return LinearBS(
            dataloader,
            start_factor=start_factor,
            end_factor=end_factor,
            milestone=milestone,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size
        )
    
    elif scheduler_name == 'exponential' or scheduler_name == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 2.0)
        max_batch_size = kwargs.get('max_batch_size', None)
        min_batch_size = kwargs.get('min_batch_size', 1)
        return StepBS(
            dataloader,
            step_size=step_size,
            gamma=gamma,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size
        )
    
    elif scheduler_name == 'plateau':
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 2.0)
        patience = kwargs.get('patience', 10)
        threshold = kwargs.get('threshold', 1e-4)
        threshold_mode = kwargs.get('threshold_mode', 'rel')
        cooldown = kwargs.get('cooldown', 0)
        max_batch_size = kwargs.get('max_batch_size', None)
        min_batch_size = kwargs.get('min_batch_size', 1)
        return IncreaseBSOnPlateau(
            dataloader,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size
        )
    
    return None
