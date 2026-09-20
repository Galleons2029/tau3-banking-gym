"""Bounded audit-only recovery; never resample an existing teacher capture."""

import argparse
import hashlib
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def read(path):
    """Read a JSON artifact."""
    return json.loads(path.read_text())


def pending_category(reviews):
    """Only retry unknown verdicts; conclusive rejection stops the whole item."""
    return bool(reviews) and any(r['status'] == 'INCONCLUSIVE' for r in reviews) and all(
        r['status'] in ('PASS', 'INCONCLUSIVE') for r in reviews
    )


def archive(directory, paths):
    """Preserve exact bytes before mutation, refusing inconsistent recovery inputs."""
    from tau3.worldgen.v2.pipeline import write_json

    hashes = {}
    for path in paths:
        if not path.exists():
            continue
        target = directory / 'original' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Original evidence changed: ' + str(path))
        if not target.exists():
            target.write_bytes(data)
        hashes[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    write_json(directory / 'original-hashes.json', hashes)
    return hashes


def recover_teacher(root, config, item, output):
    """Rejudge only inconclusive audits on a normal, immutable conversation."""
    from math import prod

    from tau3.data_model.simulation import SimulationRun
    from tau3.data_model.tasks import RewardType, Task
    from tau3.synthesis.targeted.budget import Budget, ask
    from tau3.synthesis.targeted.quality import review_quality
    from tau3.synthesis.targeted.workflow import clean_sample
    from tau3.synthesis.world_sft import configure, successful
    from tau3.worldgen.v2.audit import AuditSession
    from tau3.worldgen.v2.pipeline import write_json
    from tau3.worldgen.v2.runtime import digest
    from tau3.worldgen.v2.semantic import StrictSemanticEvaluator, audited_judge

    directory = root / 'formal/slots' / f"{item['slot']:06d}"
    trial = directory / 'teacher' / str(item['trial'])
    quality_path = trial / 'quality.json'
    if hashlib.sha256(quality_path.read_bytes()).hexdigest() != item['quality_sha256']:
        raise ValueError('Triage evidence changed')
    old = read(quality_path)
    if old['status'] != 'INCONCLUSIVE':
        raise ValueError('Conclusive judgments cannot be retried')
    native, capture = read(trial / 'result.json'), read(trial / 'capture.json')
    if native['capture_hash'] != digest(capture) or native['identity'] != capture['identity']:
        raise ValueError('Capture binding mismatch')
    simulation = SimulationRun.model_validate(native['simulation'])
    if simulation.termination_reason.value not in ('user_stop', 'agent_stop'):
        raise ValueError('Premature conversation cannot be audit-recovered')
    archive(output, [quality_path, trial / 'result.json', trial / 'capture.json', trial / 'started.json'])
    record = read(directory / 'task.json')
    world = Path(record['world'])
    settings = configure(world, root / 'settings.json')
    task = Task.model_validate_json((world / 'tasks' / f"{record['task_id']}.json").read_text())
    reward = simulation.reward_info
    unknown = [c.justification.startswith('INCONCLUSIVE:') for c in reward.nl_assertions or []]
    if any(unknown):
        if not all(unknown):
            raise ValueError('Mixed conclusive and unknown assertions require individual recovery')
        session = AuditSession(world, output / 'judge', settings, 6, 'teacher-audit-recovery-v1')
        with Budget(root, config).installed('teacher_judge_recovery_v1'):
            with audited_judge(session, settings.judge_model):
                revised = StrictSemanticEvaluator.calculate_reward(task, simulation.messages)
        if any(c.justification.startswith('INCONCLUSIVE:') for c in revised.nl_assertions or []):
            raise ValueError('Recovery judge remains inconclusive')
        reward.nl_assertions = revised.nl_assertions
        reward.reward_breakdown[RewardType.NL_ASSERTION] = revised.reward
        reward.reward = prod(reward.reward_breakdown[k] for k in reward.reward_basis)
        reward.info = {**(reward.info or {}), 'nl': revised.info, 'audit_recovery': str(output.resolve())}
        native['simulation'] = simulation.model_dump(mode='json')
        write_json(output / 'regraded-result.json', native)
    result = {k: old[k] for k in ('trial', 'seed', 'task_id')}
    result.update(status='COMPLETE', environment_success=successful(simulation), sft_qualified=False,
                  native_hash=digest(native), capture_hash=digest(capture),
                  recovery={'version': 1, 'teacher_resampled': False, 'evidence': str(output.resolve())})
    if result['environment_success']:
        sample = clean_sample(capture['sample'])
        evidence, protocol = {}, {}

        def caller(root, config, stage, *args, **kwargs):
            return ask(root, config, stage + '_gap_recovery_v1', *args, **kwargs)

        review = review_quality(root, config, {
            'user_instructions': task.user_scenario.instructions,
            'visible_context': sample['messages'],
            'conversation': simulation.model_dump(mode='json')['messages'],
        }, evidence, protocol, caller)
        result.update(quality=review.model_dump(), quality_evidence=evidence,
                      quality_protocol=protocol, sft_qualified=review.passed)
        if review.passed:
            write_json(trial / 'sample.json', sample)
            result['sample_hash'] = digest(sample)
    # Publish only a complete, evidence-bound recovery. Capture/seed never change.
    write_json(trial / 'result.json', native)
    write_json(quality_path, result)
    write_json(directory / 'trials.json', [read(directory / 'teacher' / str(i) / 'quality.json') for i in range(4)])
    return result


def recover_category(root, config, item, output):
    """Preserve independent PASS reviews and retry unknown reviewers once each."""
    from tau3.synthesis.targeted.budget import Budget
    from tau3.synthesis.targeted.workflow import collect_slot, generate_slot, load_plan
    from tau3.worldgen.v2.category_review import review_category
    from tau3.worldgen.v2.pipeline import initial_database, write_json
    from tau3.worldgen.v2.runtime import digest
    from tau3.worldgen.v2.specs import WorldSpec

    directory = root / 'formal/slots' / f"{item['slot']:06d}"
    world = directory / 'candidate_0/world'
    spec = WorldSpec.model_validate_json((world / 'spec.json').read_text())
    initial = initial_database(spec)
    for category in spec.categories:
        path = world / 'private/category_reviews' / f'{category.id}.json'
        reviews = read(path)
        if not pending_category(reviews):
            raise ValueError('Not eligible for category protocol recovery')
        archive(output, [path, world / 'progress.json', world / 'spec.json'])
        recovered = []
        for review in reviews:
            if review['status'] == 'INCONCLUSIVE':
                with Budget(root, config).installed('category_gap_recovery_v1'):
                    review = review_category(category, spec.clock, initial, review['model'], config.settings.llm_args)
            recovered.append(review)
        write_json(output / 'reviews.json', recovered)
        if not all(r['status'] == 'PASS' for r in recovered):
            return {'status': 'NOT_ADMITTED', 'reviews': recovered}
        write_json(path, recovered)
    plan = load_plan(root)
    slot = next(s for s in plan.slots if s.index == item['slot'])
    generated = generate_slot(str(root), config.model_dump(), slot.model_dump(), 'formal', digest(plan.model_dump()))
    write_json(output / 'generation.json', generated)
    if generated.get('status') != 'VALID':
        return generated
    trials = collect_slot(str(root), config.model_dump(), slot.model_dump(), 'formal', digest(plan.model_dump()))
    return {'status': 'VALID', 'trials': trials}


def worker(root_string, kind, item):
    """An isolated process owns every world and durable attempt marker."""
    from tau3.synthesis.targeted.models import RoundConfig
    from tau3.utils.llm_concurrency import file_lock
    from tau3.worldgen.v2.pipeline import write_json

    root = Path(root_string)
    config = RoundConfig.model_validate_json((root / 'config.json').read_text())
    output = root / 'gap-recovery-v1' / kind / f"{item['slot']:06d}"
    with file_lock(output / 'lock'):
        if (output / 'result.json').exists():
            return read(output / 'result.json')
        if (output / 'started.json').exists():
            return {'slot': item['slot'], 'status': 'INTERRUPTED', 'reason': 'Manual evidence review required; no automatic repeated calls'}
        write_json(output / 'started.json', {'item': item, 'teacher_resampled': False})
        try:
            result = (recover_teacher if kind == 'teacher' else recover_category)(root, config, item, output)
        except Exception as exc:
            result = {'status': 'INCONCLUSIVE', 'error': type(exc).__name__, 'reason': str(exc)[:1000]}
        result = {'slot': item['slot'], 'kind': kind, **result}
        write_json(output / 'result.json', result)
        return result


def main():
    """Recover only classified protocol gaps, with the existing global LLM cap."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    root = args.round_dir.resolve()
    output = root / 'gap-recovery-v1'
    output.mkdir(parents=True, exist_ok=True)
    for name in ('formal/report.json', 'formal/export.json', 'budget.json'):
        source = root / name
        target = output / 'baseline' / name
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    tasks = [('teacher', i) for i in read(root / 'gap-triage/audit_only.json')]
    tasks += [('category', i) for i in read(root / 'gap-triage/category-evidence.json')
              if all(pending_category(reviews) for reviews in i['category_reviews'])]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
        futures = [pool.submit(worker, str(root), kind, item) for kind, item in tasks]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps({k: result[k] for k in ('slot', 'kind', 'status', 'sft_qualified', 'reason') if k in result}), flush=True)


if __name__ == '__main__':
    main()
