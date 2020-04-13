"""Config for DQN on Pong-No_FrameSkip-v4.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""

agent = dict(
    type="DQNAgent",
    hyper_params=dict(
        gamma=0.99,
        tau=5e-3,
        buffer_size=int(1e4),  # openai baselines: int(1e4)
        batch_size=32,  # openai baselines: 32
        update_starts_from=int(1e4),  # openai baselines: int(1e4)
        multiple_update=1,  # multiple learning updates
        train_freq=4,  # in openai baselines, train_freq = 4
        gradient_clip=10.0,  # dueling: 10.0
        n_step=3,
        w_n_step=1.0,
        w_q_reg=0.0,
        per_alpha=0.6,  # openai baselines: 0.6
        per_beta=0.4,
        per_eps=1e-6,
        # Distributional Q function
        use_dist_q="IQN",
        # NoisyNet
        use_noisy_net=False,
        std_init=0.5,
        # Epsilon Greedy
        max_epsilon=0.0,
        min_epsilon=0.0,  # openai baselines: 0.01
        epsilon_decay=1e-6,  # openai baselines: 1e-7 / 1e-1
    ),
    network_cfg=dict(
        hidden_sizes=[512],
        cnn_cfg=dict(
            input_sizes=[4, 32, 64],
            output_sizes=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            paddings=[1, 0, 0],
        ),
    ),
    backbone_cfg=dict(
        type="CNN",
        params=dict(
            input_sizes=[4, 32, 64],
            output_sizes=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            paddings=[1, 0, 0],
        ),
    ),
    head_cfg=dict(
        type="IQNMLP",
        params=dict(
            # NoisyNet
            use_noisy_net=True,
            std_init=0.5,
            hidden_sizes=[512],
        ),
    ),
    optim_cfg=dict(
        lr_dqn=1e-4,  # dueling: 6.25e-5, openai baselines: 1e-4
        weight_decay=0.0,  # this makes saturation in cnn weights
        adam_eps=1e-8,  # rainbow: 1.5e-4, openai baselines: 1e-8
    ),
)
