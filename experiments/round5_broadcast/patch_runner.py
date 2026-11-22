from pathlib import Path

p = Path('/root/autodl-tmp/fedvade_project/reproduction_runs/round5_broadcast/runner.py')
s = p.read_text()

def one(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)

one("return replace_once(source, point, point + \"\"\"", "source = replace_once(source, point, point + \"\"\"")
one("seconds=_now() - observed_started)\n\"\"\")", "seconds=_now() - observed_started)\n            _capture(epoch + 1, server, clients, server_vade_param, pre)\n\"\"\")")
one("\"\"\")\n\n\ndef fedvade(args, copy, result):", "\"\"\")\n    if mode == 'post_server':\n        source = replace_once(source,\n            \"                c['vade'].load_state_dict(server_vade_param, strict=False)\",\n            \"                c['vade'].load_state_dict({k: server['vade'].state_dict()[k] for k in server_vade_param}, strict=False)\")\n    return source\n\n\ndef fedvade(args, copy, result):")
one("'rounds': 12 if args.smoke else 300", "'rounds': 2 if args.smoke else 300")
one("'pretrain_rounds': 3 if args.smoke else 50", "'pretrain_rounds': 1 if args.smoke else 50")
one("    generator = torch.Generator(device='cpu').manual_seed(1000 + args.seed)", """    def capture(epoch, server, clients, aggregate, predictions):
        if epoch not in ((1, 2) if args.smoke else (1, 150, 300)):
            return
        server_state = server['vade'].state_dict()
        keys = tuple(aggregate)
        assert keys and not {'rho_', 'mu_c', 'log_sigma2_c'}.intersection(keys)
        actual = 'post_server' if args.schedule == 'post_server' else 'author'
        client_deltas_to_aggregate = [
            max((client['vade'].state_dict()[key].detach().cpu() - aggregate[key]).abs().max().item()
                for key in keys) for client in clients]
        client_deltas_to_server = [
            max((client['vade'].state_dict()[key] - server_state[key]).abs().max().item()
                for key in keys) for client in clients]
        aggregate_server_delta = max((server_state[key].detach().cpu() - aggregate[key]).abs().max().item()
                                     for key in keys)
        assert max(client_deltas_to_aggregate if actual == 'author' else client_deltas_to_server) < 1e-7
        assert all(torch.isfinite(v).all() for v in server_state.values())
        assert all(all(torch.isfinite(v).all() for v in c['vade'].state_dict().values()) for c in clients)
        nc = len(server_state['rho_'])
        record = {
            'round': epoch, 'mode': actual,
            'aggregate_to_post_server_max_abs': aggregate_server_delta,
            'client_to_aggregate_max_abs': client_deltas_to_aggregate,
            'client_to_post_server_max_abs': client_deltas_to_server,
            'server_gmm_weights': torch.softmax(server_state['rho_'].detach().cpu(), 0).tolist(),
            'client_gmm_weights': [torch.softmax(c['vade'].rho_.detach().cpu(), 0).tolist() for c in clients],
            'prediction_occupancy': (np.bincount(np.asarray(predictions, dtype=int), minlength=nc) / len(predictions)).tolist(),
            'client_optimizer_states': [len(c['optimizer'].state) for c in clients],
            'server_optimizer_states': len(server['optimizer'].state),
        }
        with (args.out / 'parameter_flow.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\\n')
        if not args.smoke and epoch in (150, 300):
            checkpoint = {
                'round': epoch, 'mode': actual, 'seed': args.seed,
                'server': {k: v.detach().cpu().clone() for k, v in server_state.items()},
                'clients': [{k: v.detach().cpu().clone() for k, v in c['vade'].state_dict().items()} for c in clients],
                'server_optimizer': server['optimizer'].state_dict(),
                'client_optimizers': [c['optimizer'].state_dict() for c in clients],
                'server_scheduler': server['lr_s'].state_dict(),
                'client_schedulers': [c['lr_s'].state_dict() for c in clients],
                'torch_rng': torch.get_rng_state(),
                'numpy_rng': np.random.get_state(),
            }
            torch.save(checkpoint, args.out / f'round_{epoch}_full.pt')

    generator = torch.Generator(device='cpu').manual_seed(1000 + args.seed)""")
one("namespace = {'_observe': observe, '_now': now,", "namespace = {'_observe': observe, '_now': now, '_capture': capture,")
one("if not args.smoke:\n        result['round_150']", "if not args.smoke:\n        result['round_150']")
one("choices=('author', 'epoch')", "choices=('author', 'post_server')")
p.write_text(s)
