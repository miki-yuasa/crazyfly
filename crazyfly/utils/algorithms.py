from policy.layers.ac_networks import GaussianMLPActor, MLPCritic
from policy import PPO, TRPO


def ppo_policy(args_cli, state_dim: int, action_dim: int):
    actor = GaussianMLPActor(
        input_dim=state_dim,
        hidden_dim=args_cli.actor_fc_dim,
        action_dim=action_dim,
        device=args_cli.device,
    )
    critic = MLPCritic(state_dim, hidden_dim=args_cli.critic_fc_dim)

    policy = PPO(
        actor=actor,
        critic=critic,
        actor_lr=args_cli.actor_lr,
        critic_lr=args_cli.critic_lr,
        eps_clip=args_cli.eps_clip,
        entropy_scaler=args_cli.entropy_scaler,
        target_kl=args_cli.target_kl,
        gamma=args_cli.gamma,
        gae=args_cli.gae,
        K=args_cli.K_epochs,
        num_minibatch=args_cli.num_minibatch,
        device=args_cli.device,
    )

    return policy
