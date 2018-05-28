from gym.envs.registration import register

register(
    id='heat-v0',
    entry_point='gym_heat_flow.envs:HeatEnv',
)
register(
    id='heat-extrahard-v0',
    entry_point='gym_heat_flow.envs:HeatExtraHardEnv',
)