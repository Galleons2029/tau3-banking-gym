"""Separate current queue/accounting/transport timing; preserve historical timeouts."""

import argparse
import json
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


def stats(values):
    """Small deterministic quantile summary in seconds."""
    values = sorted(values)
    return {} if not values else {'count': len(values), 'median': statistics.median(values),
                                 'p95': values[int((len(values) - 1) * .95)], 'max': values[-1]}


def historical(item):
    """Read native timing without touching the frozen teacher trajectory."""
    native = json.loads((Path(item['evidence']).parent / 'result.json').read_text())['simulation']
    measurements = defaultdict(list)
    previous = None
    for message in native['messages']:
        current = datetime.fromisoformat(message['timestamp']) if message.get('timestamp') else None
        if previous and current and message['role'] in ('assistant', 'user'):
            measurements[message['role'] + '_turn_wall'].append((current - previous).total_seconds())
        if message.get('generation_time_seconds') is not None:
            measurements[message['role'] + '_completion_including_budget_pool'].append(message['generation_time_seconds'])
        previous = current or previous
    return measurements


def main():
    """Run two small budgeted model probes; do not resample any teacher slot."""
    from tau3.synthesis.targeted.budget import Budget
    from tau3.synthesis.targeted.models import RoundConfig
    from tau3.utils import llm_utils
    from tau3.worldgen.v2.pipeline import write_json

    parser = argparse.ArgumentParser()
    parser.add_argument('--round-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.round_dir.resolve()
    output = root / 'gap-recovery-v1/timeout-diagnostics.json'
    if output.exists():
        return
    config = RoundConfig.model_validate_json((root / 'config.json').read_text())
    timings = []

    class TimedBudget(Budget):
        """Time existing accounting without changing its locking or limit."""

        def reserve(self, *args, **kwargs):
            started = time.monotonic()
            try:
                return super().reserve(*args, **kwargs)
            finally:
                timing['reserve_seconds'] = time.monotonic() - started

        def complete(self, *args, **kwargs):
            started = time.monotonic()
            try:
                return super().complete(*args, **kwargs)
            finally:
                timing['settle_seconds'] = time.monotonic() - started

    for model in config.settings.agent_models:
        timing = {'model': model}
        marker = root / 'gap-recovery-v1' / ('probe-' + model.split('/')[-1] + '.json')
        if marker.exists():
            timings.append(json.loads(marker.read_text()))
            continue
        write_json(marker, {'status': 'STARTED', 'model': model})

        def transport(**kwargs):
            started = time.monotonic()
            try:
                return llm_utils.completion(**kwargs)
            finally:
                timing['transport_seconds'] = time.monotonic() - started

        started = time.monotonic()
        try:
            TimedBudget(root, config).call('latency_diagnostic_v1', transport, model=model,
                messages=[{'role': 'user', 'content': 'Reply with exactly OK.'}],
                **{**config.settings.llm_args, 'max_tokens': 128, 'temperature': 0})
            timing['status'] = 'COMPLETE'
        except Exception as exc:
            timing.update(status='INCONCLUSIVE', error=type(exc).__name__)
        timing['total_seconds'] = time.monotonic() - started
        timing['pool_and_other_seconds'] = timing['total_seconds'] - sum(timing.get(k, 0) for k in ('transport_seconds', 'reserve_seconds', 'settle_seconds'))
        write_json(marker, timing)
        timings.append(timing)
    items = json.loads((root / 'gap-triage/timeout_experiment.json').read_text())
    aggregate = defaultdict(list)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for values in pool.map(historical, items):
            for key, samples in values.items():
                aggregate[key].extend(samples)
    write_json(output, {'historical_slots': len(items), 'historical': {k: stats(v) for k, v in aggregate.items()},
                        'current_probes': timings, 'teacher_resampled': False,
                        'limitation': 'Historical completion time includes pool, accounting and transport; old logs cannot uniquely separate them. Current short probes do not reproduce 256-concurrency load or long-context latency.'})
    print(json.dumps(json.loads(output.read_text())), flush=True)


if __name__ == '__main__':
    main()
