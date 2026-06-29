#!/usr/bin/env python3
"""
HW2, Часть 1: Single-GPU baseline.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    default_data_collator,
)

from common import LocalTimer, get_mem_stats, load_and_preprocess_data

LOGGER = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='HW2 Part 1: Single-GPU baseline')
    parser.add_argument('-e', '--experiment-name', default=None)
    parser.add_argument('-d', '--dataset-name', default=None, required=True)
    parser.add_argument('--dataset-subset', default=None)
    parser.add_argument('-m', '--model-name', default=None, required=True)
    parser.add_argument('--save-dir', default='outputs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num-epochs', default=1, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--ckpt-freq', default=500, type=int)
    parser.add_argument('-s', '--seq-length', default=512, type=int)

    parser.add_argument(
        '--dtype',
        default='bf16',
        choices=['fp32', 'bf16'],
    )
    parser.add_argument(
        '--activation-checkpointing',
        action='store_true',
    )
    parser.add_argument(
        '--grad-accum',
        default=1,
        type=int,
    )
    return parser


def main(args: argparse.Namespace) -> None:  # noqa: C901, PLR0915
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s:%(message)s',
        level=logging.INFO,
    )

    LOGGER.info(f'Config: dtype={args.dtype}, activation_checkpointing={args.activation_checkpointing}')
    LOGGER.info(f'batch_size={args.batch_size}, seq_length={args.seq_length}')
    LOGGER.info(
        f'Effective global_batch_size = {args.batch_size * args.seq_length * args.grad_accum} tokens/step'
    )

    device = torch.device('cuda')
    dtype = torch.float32 if args.dtype == 'fp32' else torch.bfloat16

    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    with device:
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f'Параметров в модели: {n_params:,}')
    LOGGER.info(
        f'Теоретическая память параметров: '
        f'{n_params * (4 if args.dtype == "fp32" else 2) / 1024**3:.3f} GB'
    )
    LOGGER.info(f'Память после инициализации: {get_mem_stats(device)["curr_alloc_gb"]:.3f} GB')

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()
        LOGGER.info('Activation checkpointing ВКЛЮЧЁН')

    data = load_and_preprocess_data(
        args.model_name,
        args.seq_length,
        args.dataset_name,
        args.dataset_subset,
        config,
    )
    data = data.train_test_split(test_size=0.05, seed=args.seed)
    train_data = data['train']
    eval_data = data['test']
    LOGGER.info(f'{len(train_data)} обучающих примеров, {len(eval_data)} валидационных')

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=default_data_collator,
    )
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=default_data_collator,
    )
    LOGGER.info(
        f'{len(dataloader)} батчей на эпоху (train), '
        f'{len(eval_dataloader)} батчей (eval)'
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    # Папка для чекпоинтов
    is_experiment = args.experiment_name is not None
    exp_dir: Path = Path(args.save_dir)
    if is_experiment:
        exp_dir = exp_dir / args.experiment_name

    state = {
        'epoch': 0,
        'global_step': 0,
        'epoch_step': 0,
        'running_loss': 0.0,
    }

    if is_experiment and (exp_dir / 'state.json').exists():
        def _load_to_device(p: str | Path) -> dict[str, Any]:
            return torch.load(p, map_location=device, weights_only=True)

        model.load_state_dict(_load_to_device(exp_dir / 'model.pt'))
        optimizer.load_state_dict(_load_to_device(exp_dir / 'optimizer.pt'))
        lr_scheduler.load_state_dict(_load_to_device(exp_dir / 'lr_scheduler.pt'))
        with (exp_dir / 'state.json').open() as f:
            state = json.load(f)
        LOGGER.info(f'Возобновление с шага {state["global_step"]}')
    elif is_experiment:
        exp_dir.mkdir(parents=True, exist_ok=True)

    timers = {k: LocalTimer(device) for k in ['data', 'forward', 'backward', 'update']}
    all_tokens_per_s = []

    for state['epoch'] in range(state['epoch'], args.num_epochs):  # noqa: B020
        LOGGER.info(f'--- Эпоха {state["epoch"]} ---')
        model.train()

        progress_bar = tqdm.tqdm(range(len(dataloader)))
        if state['epoch_step'] > 0:
            progress_bar.update(state['epoch_step'])

        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            with timers['data'], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            if i_step < state['epoch_step']:
                continue

            with timers['forward']:
                outputs = model(**batch)
                del batch

            with timers['backward']:
                (outputs.loss / args.grad_accum).backward()

            if (i_step + 1) % args.grad_accum == 0 or (i_step + 1) == len(dataloader):
                with timers['update']:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                state['global_step'] += 1

                if state['global_step'] % args.log_freq == 0:
                    tok_per_step = args.batch_size * args.seq_length * args.grad_accum
                    ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())
                    info = {
                        'global_step': state['global_step'],
                        'lr': lr_scheduler.get_last_lr()[0],
                        'running_loss': state['running_loss'] / (args.log_freq * args.grad_accum),
                        'epoch': state['epoch'],
                        'epoch_progress': state['epoch_step'] / len(dataloader),
                        **get_mem_stats(device),
                        'tokens_per_s': 1000 * tok_per_step / ms_per_step,
                        'time/total_ms': ms_per_step,
                        **{f'time/{k}_ms': t.avg_elapsed_ms() for k, t in timers.items()},
                    }
                    LOGGER.info(info)
                    all_tokens_per_s.append(info['tokens_per_s'])
                    torch.cuda.reset_peak_memory_stats(device)
                    state['running_loss'] = 0.0
                    for t in timers.values():
                        t.reset()

            state['epoch_step'] += 1
            state['running_loss'] += outputs.loss.item()
            progress_bar.update(1)

            if is_experiment and state['global_step'] % args.ckpt_freq == 0:
                LOGGER.info('Сохраняем чекпоинт')
                torch.save(optimizer.state_dict(), exp_dir / 'optimizer.pt')
                torch.save(model.state_dict(), exp_dir / 'model.pt')
                torch.save(lr_scheduler.state_dict(), exp_dir / 'lr_scheduler.pt')
                with (exp_dir / 'state.json').open('w') as fp:
                    json.dump(state, fp)

        # Валидация в конце эпохи
        model.eval()
        losses = []
        for _, batch in enumerate(eval_dataloader):
            batch = {k: v.to(device=device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            losses.append(outputs.loss.item())

        eval_loss = torch.mean(torch.tensor(losses)).item()
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float('inf')

        LOGGER.info(
            f'[RESULT] epoch={state["epoch"]} '
            f'val_ppl={perplexity:.2f} '
            f'val_loss={eval_loss:.4f} '
            f'peak_mem_gb={get_mem_stats(device)["peak_alloc_gb"]:.3f} '
            f'avg_throughput={sum(all_tokens_per_s)/len(all_tokens_per_s):.0f}'
        )

        state['epoch_step'] = 0


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)