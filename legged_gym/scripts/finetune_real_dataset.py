import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from rsl_rl.rsl_rl.modules.residual_actor_critic import ResidualActorCritic


def _load_real_dataset_module():
    module_path = Path(__file__).resolve().parents[1] / 'utils' / 'real_dataset.py'
    spec = importlib.util.spec_from_file_location('real_dataset', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _infer_hidden_dims(model_state_dict, prefix):
    linear_layers = []
    for key, value in model_state_dict.items():
        if key.startswith(prefix) and key.endswith('.weight'):
            idx_str = key[len(prefix) : -len('.weight')]
            if idx_str.isdigit() and value.ndim == 2:
                linear_layers.append((int(idx_str), value.shape[0]))

    if len(linear_layers) < 2:
        raise RuntimeError('Unable to infer hidden dims from checkpoint for prefix: ' + prefix)

    linear_layers.sort(key=lambda x: x[0])
    return [out_dim for _, out_dim in linear_layers[:-1]]


def _infer_model_dims(model_state_dict):
    actor_w0 = model_state_dict['actor.0.weight']
    obs_dim = int(actor_w0.shape[1])

    actor_weights = [
        (k, v) for k, v in model_state_dict.items() if k.startswith('actor.') and k.endswith('.weight')
    ]
    actor_weights.sort(key=lambda x: int(x[0].split('.')[1]))
    action_dim = int(actor_weights[-1][1].shape[0])

    critic_obs_dim = obs_dim
    if 'critic.0.weight' in model_state_dict:
        critic_obs_dim = int(model_state_dict['critic.0.weight'].shape[1])

    actor_hidden_dims = _infer_hidden_dims(model_state_dict, 'actor.')
    critic_hidden_dims = actor_hidden_dims
    if any(k.startswith('critic.') and k.endswith('.weight') for k in model_state_dict.keys()):
        critic_hidden_dims = _infer_hidden_dims(model_state_dict, 'critic.')

    return obs_dim, critic_obs_dim, action_dim, actor_hidden_dims, critic_hidden_dims


def parse_args():
    parser = argparse.ArgumentParser(description='Offline BC finetuning for residual actor')
    parser.add_argument('--base_ckpt_path', type=str, required=True)
    parser.add_argument('--dataset_npz', type=str, required=True)
    parser.add_argument('--out_ckpt', type=str, required=True)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


def main():
    args = parse_args()

    real_dataset_mod = _load_real_dataset_module()
    dataloader = real_dataset_mod.make_dataloader(
        args.dataset_npz,
        batch_size=args.batch_size,
        shuffle=True,
    )

    base_checkpoint = torch.load(args.base_ckpt_path, map_location='cpu')
    if 'model_state_dict' not in base_checkpoint:
        raise KeyError("Base checkpoint must include 'model_state_dict'")
    base_state_dict = base_checkpoint['model_state_dict']

    obs_dim, critic_obs_dim, action_dim, actor_hidden_dims, critic_hidden_dims = _infer_model_dims(
        base_state_dict
    )

    model = ResidualActorCritic(
        obs_dim=obs_dim,
        critic_obs_dim=critic_obs_dim,
        action_dim=action_dim,
        base_ckpt_path=args.base_ckpt_path,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        residual_hidden_dims=actor_hidden_dims,
        freeze_base=True,
        map_location='cpu',
    ).to(args.device)
    model.train()

    optimizer = torch.optim.Adam(
        model.residual_actor.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    global_step = 0
    loss_history = []

    for epoch in range(args.epochs):
        for batch in dataloader:
            obs = batch['obs'].to(args.device)
            target_action = batch['action'].to(args.device)

            pred_action = model.actor(obs)
            loss = F.mse_loss(pred_action, target_action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(float(loss.detach().cpu()))
            global_step += 1

            if global_step % args.log_interval == 0:
                print(
                    f'[step {global_step:07d}] epoch={epoch:04d} '
                    f'loss={loss.item():.8f}'
                )

    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'base_ckpt_path': args.base_ckpt_path,
        'epochs': args.epochs,
    }
    torch.save(payload, out_path)
    print(f'Saved finetuned checkpoint to: {out_path}')


if __name__ == '__main__':
    main()
