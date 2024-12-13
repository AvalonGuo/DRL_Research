
class PPO:
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    Code: This implementation(concise version) borrows code from Stable-baseline3(https://github.com/DLR-RM/stable-baselines3).
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, MultiInputPolicy).
    """
    def __init__(
        self,
        policy: str,
        batch_size: int = 64,
        n_steps: int=2048,
        normalize_advantage:bool=True,

        ):
        if normalize_advantage:
            assert(batch_size>1),"`batch_size` must be greater than 1."
            