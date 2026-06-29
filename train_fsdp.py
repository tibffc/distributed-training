#!/usr/bin/env python3

import argparse
import logging
import math
import os
 
import torch
import tqdm
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, default_data_collator
 
from common import LocalTimer, get_mem_stats, load_and_preprocess_data, rank0_first
 
LOGGER = logging.getLogger(__name__)
 
 
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-name', required=True)
    parser.add_argument('--dataset-subset', default=None)
    parser.add_argument('-m', '--model-name', required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num-epochs', default=1, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('-s', '--seq-length', default=512, type=int)
    parser.add_argument('--strategy', default='FULL_SHARD',
                        choices=['NO_SHARD', 'SHARD_GRAD_OP', 'FULL_SHARD'])
    parser.add_argument('--cpu-offload', action='store_true')
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--grad-accum', default=1, type=int)
    parser.add_argument('--warmup-steps', default=20, type=int,
                        help='Число шагов прогрева, не учитываемых в среднем throughput')
    return parser
 
 
@record
def main(args):
    rank = int(os.getenv('RANK', '0'))
    local_rank = rank % torch.cuda.device_count()
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(rank=rank, world_size=world_size, device_id=device)
 
    logging.basicConfig(
        format=f'[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s',
        level=logging.INFO,
    )
 
    LOGGER.info(f'rank={rank} world_size={world_size} strategy={args.strategy}')
    LOGGER.info(f'global_batch_size = {args.batch_size * args.seq_length * world_size} tokens/step')
 
    torch.manual_seed(args.seed)
 
    with rank0_first(), device:
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
 
    LOGGER.info(f'Параметров: {sum(p.numel() for p in model.parameters()):,}')
 
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()
 
    if args.strategy == 'NO_SHARD':
        model = DistributedDataParallel(model, device_ids=[local_rank],
                                        bucket_cap_mb=500, gradient_as_bucket_view=True)
        LOGGER.info(f'DDP. Mem: {get_mem_stats(device)["curr_alloc_gb"]:.3f} GB')
    else:
        from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy
 
        mesh = init_device_mesh('cuda', (world_size,), mesh_dim_names=('dp',))
 
        offload_policy = CPUOffloadPolicy() if args.cpu_offload else None
        reshard_after_forward = (args.strategy == 'FULL_SHARD')
 
        fsdp_kwargs = {'mesh': mesh, 'reshard_after_forward': reshard_after_forward}
        if offload_policy is not None:
            fsdp_kwargs['offload_policy'] = offload_policy
 
        for module in model.modules():
            if type(module).__name__ in ('GPTNeoXLayer',):
                fully_shard(module, **fsdp_kwargs)
 
        fully_shard(model, **fsdp_kwargs)
        LOGGER.info(f'FSDP2 ({args.strategy}). Mem: {get_mem_stats(device)["curr_alloc_gb"]:.3f} GB')
 
    with rank0_first():
        data = load_and_preprocess_data(
            args.model_name, args.seq_length,
            args.dataset_name, args.dataset_subset, config,
        )
    data = data.train_test_split(test_size=0.05, seed=args.seed)
 
    dataloader = DataLoader(
        data['train'], batch_size=args.batch_size, num_workers=1,
        prefetch_factor=2, collate_fn=default_data_collator,
        sampler=DistributedSampler(data['train'], shuffle=True, drop_last=True),
    )
    eval_dataloader = DataLoader(
        data['test'], batch_size=args.batch_size, drop_last=True,
        num_workers=1, prefetch_factor=2, collate_fn=default_data_collator,
    )
    LOGGER.info(f'{len(dataloader)} train батчей, {len(eval_dataloader)} eval батчей')
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=args.lr * 1e-2)
 
    state = {'epoch': 0, 'global_step': 0, 'epoch_step': 0, 'running_loss': 0.0}
    timers = {k: LocalTimer(device) for k in ['data', 'forward', 'backward', 'update']}
    all_tokens_per_s = []
 
    for state['epoch'] in range(args.num_epochs):
        LOGGER.info(f'Эпоха {state["epoch"]}')
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        dataloader.sampler.set_epoch(state['epoch'])
        progress_bar = tqdm.tqdm(range(len(dataloader)), disable=(rank != 0))
 
        for i_step, batch in enumerate(dataloader):
            with timers['data'], torch.no_grad():
                batch = {k: v.to(device=device) for k, v in batch.items()}

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
                    tok_per_step = args.batch_size * args.seq_length * world_size * args.grad_accum
                    ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values()) * args.grad_accum
                    info = {
                        'global_step': state['global_step'],
                        'lr': lr_scheduler.get_last_lr()[0],
                        'running_loss': state['running_loss'] / (args.log_freq * args.grad_accum),
                        'strategy': args.strategy,
                        **get_mem_stats(device),
                        'tokens_per_s': 1000 * tok_per_step / ms_per_step,
                    }
                    if rank == 0:
                        LOGGER.info(info)
                        if state['global_step'] > args.warmup_steps:
                            all_tokens_per_s.append(info['tokens_per_s'])
                    torch.cuda.reset_peak_memory_stats(device)
                    state['running_loss'] = 0.0

                for t in timers.values():
                    t.reset()

            state['running_loss'] += outputs.loss.item()
            progress_bar.update(1)

 
        dist.barrier()
        model.eval()
        losses = []
        for batch in eval_dataloader:
            batch = {k: v.to(device=device) for k, v in batch.items()}
            with torch.no_grad():
                if args.strategy == 'NO_SHARD':
                    outputs = model.module(**batch)
                else:
                    outputs = model(**batch)
            losses.append(outputs.loss.item())
 
        eval_loss = torch.tensor(losses).mean().item()
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float('inf')
 
        if rank == 0:
            avg_tps = sum(all_tokens_per_s) / len(all_tokens_per_s) if all_tokens_per_s else 0
            LOGGER.info(
                f'[RESULT] epoch={state["epoch"]} strategy={args.strategy} '
                f'val_ppl={perplexity:.2f} val_loss={eval_loss:.4f} '
                f'peak_mem_gb={get_mem_stats(device)["peak_alloc_gb"]:.3f} '
                f'avg_throughput={avg_tps:.0f} '
            )
        dist.barrier()
 
    dist.destroy_process_group()
 
 
if __name__ == '__main__':
    main(get_parser().parse_args())