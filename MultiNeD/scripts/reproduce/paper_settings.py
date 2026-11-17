"""Paper training settings shared by the CLI and the two interactive examples."""
from common.cd8_model_components import DriftModelConfig, PKConfig, ReactionEffectConfig
from training.drift_training import DriftTrainConfig
from training.coupled_dynamics import ReactionCoupledConfig, build_z_transform_params


def _paper_pk_config():
    return PKConfig(
        enabled=True,
        pd1_dose_days=(1.0, 4.0),
        b41bb_dose_days=(2.0,),
        pd1_dose_amount=1.0,
        b41bb_dose_amount=1.0,
        delta_pd1=0.23,
        delta_b41bb=0.1,
        normalize_to_unit_peak=True,
    )


def drift_configs():
    model_config = DriftModelConfig(
        z_dim=5,
        hidden_dim=256,
        n_layers=3,
        max_drift=10.0,
        pk=_paper_pk_config(),
    )
    train_config = DriftTrainConfig(
        n_epochs=800,
        n_max=256,
        lr=0.001,
        weight_decay=1e-05,
        sinkhorn_epsilon=0.08,
        sinkhorn_iters=60,
        loss_mode='local_rollout',
        rollout_loss_weight=0.5,
        rollout_n_max=256,
        n_steps_per_day=10,
        drift_l2_weight=1e-05,
        lr_patience=20,
        lr_factor=0.5,
        min_lr=5e-07,
        print_every=10,
        seed=2026,
        device='auto',
    )
    return model_config, train_config


def reaction_configs(group_data):
    z_transform_params = build_z_transform_params(group_data, q_low=0.01, q_high=0.99)
    pk_config = _paper_pk_config()
    reaction_effect_config = ReactionEffectConfig(
        f_b_coeff=1.0,
        g_b_coeff=1.0,
        g_pd1_coeff=1.0,
        pd1_k=1.0,
        b41bb_k=1.0,
        eps=1e-08,
        learnable=True,
    )
    reaction_config = ReactionCoupledConfig(
        n_epochs=600,
        n_particles=512,
        dt=0.1,
        lr=0.01,
        drift_lr=0.001,
        weight_decay=1e-05,
        loss_mode='local_rollout',
        z_loss_weight=1,
        volume_loss_weight=2,
        train_drift=True,
        sinkhorn_epsilon=0.08,
        sinkhorn_iters=60,
        reaction_effect=reaction_effect_config,
        z_transform_params=z_transform_params,
        stim_k=150.0,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=30,
        min_lr=1e-07,
        device='auto',
        print_every=20,
    )
    # Killing and source rates are on the normalized abundance scale, n(day 1) = 1.
    parameters = {'tumor_r': {'mode': 'optimize', 'bounds': (0.005, 2.0), 'init': 0.31},
     'tumor_k': {'mode': 'optimize', 'bounds': (2000.0, 6000.0), 'init': 4000.0},
     'killing': {'mode': 'optimize', 'bounds': (0.01, 0.3), 'init': 0.1},
     'source_rate': {'mode': 'optimize', 'bounds': (0.0, 0.8), 'init': 0.2},
     'lambda_act': {'mode': 'optimize', 'bounds': (0.0, 15.0), 'init': 0.3},
     'death_rate': {'mode': 'fixed', 'bounds': (0.005, 1.0), 'value': 0.18},
     'stim_k': {'mode': 'optimize', 'bounds': (30.0, 500.0), 'init': 150.0},
     'f_b_coeff': {'mode': 'optimize', 'bounds': (0.0, 15.0), 'init': 3.0},
     'g_b_coeff': {'mode': 'optimize', 'bounds': (0.0, 15.0), 'init': 3.0},
     'g_pd1_coeff': {'mode': 'optimize', 'bounds': (0.0, 15.0), 'init': 1.5},
     'K_P': {'mode': 'fixed', 'bounds': (0.1, 1.0), 'value': 0.5},
     'K_B': {'mode': 'fixed', 'bounds': (0.1, 1.0), 'value': 0.5}}
    return reaction_config, pk_config, parameters
