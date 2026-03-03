from legged_gym.envs.zqsa01.zqsa01_config import ZqSA01CfgPPO


class ZqSA01ResidualCfgPPO(ZqSA01CfgPPO):
    class policy(ZqSA01CfgPPO.policy):
        base_ckpt_path = None
        freeze_base = True
        residual_scale = 1.0
        residual_clip = 1.0
        residual_last_layer_gain = 0.01

    class algorithm(ZqSA01CfgPPO.algorithm):
        learning_rate = 5e-5
        entropy_coef = 0.0005

    class runner(ZqSA01CfgPPO.runner):
        policy_class_name = 'ResidualActorCritic'
        experiment_name = 'zqsa01_residual_ppo'
        max_iterations = 80000
