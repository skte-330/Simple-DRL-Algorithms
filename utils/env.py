import gymnasium as gym
import ale_py

def make_env(env_id: str, seed: int | None = None) -> gym.Env:
    print(f"Creating environment {env_id}.")
    env = gym.make(env_id)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def _make_base_atari_env(env_id: str, repeat_action_probability: float = 0.0):
    """
    Gymnasium Atari envs are usually created as ALE/<Game>-v5.
    frameskip=1 is used because AtariPreprocessing performs frame skipping itself.
    """
    try:
        return gym.make(
            env_id,
            frameskip=1,
            repeat_action_probability=repeat_action_probability,
        )
    except TypeError:
        # Some non-ALE or older envs may not accept these keyword arguments.
        return gym.make(env_id)


def make_atari_env(env_id: str, seed: int | None = None, repeat_action_probability: float = 0.0):
    print(f"Creating Atari environment {env_id}.")

    env = _make_base_atari_env(env_id, repeat_action_probability)

    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("Atari DQN only supports discrete action spaces.")

    env = gym.wrappers.AtariPreprocessing(env)

    # Gymnasium version compatibility: new versions use FrameStackObservation.
    if hasattr(gym.wrappers, "FrameStackObservation"):
        env = gym.wrappers.FrameStackObservation(env, 4)
    else:
        env = gym.wrappers.FrameStack(env, 4)

    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env