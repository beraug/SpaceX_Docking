import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)
    
register(
    id='SpaceX_PPO_PROLONETS-v0',
    entry_point='gym_spacex.envs:SpaceXPpoProlonetsEnv',
)

register(
    id='SpaceX_PPO_PROLONETS-v1',
    entry_point='gym_spacex.envs:SpaceXPpoProlonetsEnv1',
)