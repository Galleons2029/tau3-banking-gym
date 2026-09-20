"""Recover six isolated audit protocol gaps without retrying valid verdicts."""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from recover_targeted_gaps import archive, read


def worker(root_string, index, kind):
    """Refresh only a proven malformed/failed request, then replay strict gates."""
    from tau3.data_model.message import AssistantMessage, SystemMessage, UserMessage
    from tau3.synthesis.targeted.budget import Budget
    from tau3.synthesis.targeted.models import RoundConfig
    from tau3.synthesis.targeted.workflow import collect_slot, generate_slot, load_plan
    from tau3.utils.llm_concurrency import file_lock
    from tau3.worldgen.v2.audit import AuditIncomplete, AuditSession
    from tau3.worldgen.v2.blind import run_blind
    from tau3.worldgen.v2.json_output import json_llm_args, parse_object
    from tau3.worldgen.v2.pipeline import write_json
    from tau3.worldgen.v2.runtime import digest
    from tau3.worldgen.v2.semantic import parse_verdicts

    root = Path(root_string)
    output = root / 'gap-recovery-v1' / kind / f'{index:06d}'
    with file_lock(output / 'lock'):
        if (output / 'result.json').exists():
            return read(output / 'result.json')
        if (output / 'started.json').exists():
            return {'slot': index, 'status': 'INTERRUPTED'}
        write_json(output / 'started.json', {'slot': index, 'teacher_resampled': False})
        try:
            config = RoundConfig.model_validate_json((root / 'config.json').read_text())
            directory = root / 'formal/slots' / f'{index:06d}'
            world = directory / 'candidate_0/world'
            audit = directory / 'candidate_0/audit' / kind
            report = read(audit / 'report.json')
            bad_rows = [r for r in report['results'] if r['status'] != 'PASS']
            if len(bad_rows) != 1 or bad_rows[0]['status'] != 'INCONCLUSIVE':
                raise ValueError('Only a single inconclusive cell is eligible')
            pending = []
            for path in (audit / 'calls').glob('*.json'):
                record = read(path)
                if kind == 'calibration' and record['status'] != 'COMPLETE':
                    pending.append((path, record))
                elif kind == 'blind':
                    payload = json.loads(record['request']['messages'][-1]['content'])
                    if 'public_format_feedback' in payload:
                        required = ('intent_satisfied', 'evidence_sufficient', 'state_correct', 'permissions_correct', 'calculations_complete')
                        try:
                            response = parse_object(record.get('response'))
                            valid = all(type(response.get(k)) is bool for k in required) and isinstance(response.get('issues'), list)
                        except (ValueError, TypeError):
                            valid = False
                        if not valid:
                            pending.append((path, record))
            if len(pending) != 1:
                raise ValueError(f'Expected one failed protocol request; got {len(pending)}')
            path, original = pending[0]
            archive(output, [path, audit / 'report.json', audit / 'audit.json'])
            session = AuditSession(world, output / 'audit', config.settings, 6, 'admission-protocol-recovery-v1')
            classes = {'system': SystemMessage, 'user': UserMessage, 'assistant': AssistantMessage}
            request = original['request']
            with Budget(root, config).installed('calibration'):
                reply = session.generate(model=request['model'], messages=[classes[m['role']](**m) for m in request['messages']], **json_llm_args(config.settings.llm_args))
            response = parse_object(reply.content)
            if kind == 'calibration':
                payload = json.loads(request['messages'][1]['content'])
                parse_verdicts(reply.content, len(payload['assertions']))
            elif not all(type(response.get(k)) is bool for k in required) or not isinstance(response.get('issues'), list):
                raise ValueError('Recovered public review still malformed')
            revised = read(output / 'audit/calls' / path.name)
            revised['attempts'] = original.get('attempts', []) + revised['attempts']
            revised['recovery_evidence'] = str(output.resolve())
            write_json(path, revised)
            # A negative repaired public verdict must not trigger new solutions.
            if kind == 'blind':
                previous = AuditSession.generate

                def cache_only(self, *, model, messages, **kwargs):
                    key = digest({'model': model, 'messages': [{'role': m.role, 'content': m.content} for m in messages]})
                    if not (self.output / 'calls' / f'{key}.json').exists():
                        raise AuditIncomplete('Recovery stops before resampling a valid negative solution/review')
                    return previous(self, model=model, messages=messages, **kwargs)

                AuditSession.generate = cache_only
                try:
                    checked = run_blind(world, audit, settings_path=root / 'settings.json', max_calls=64)
                finally:
                    AuditSession.generate = previous
                if checked['status'] != 'PASS':
                    raise AuditIncomplete('Repaired protocol did not establish blind validity')
            plan = load_plan(root)
            slot = next(s for s in plan.slots if s.index == index)
            generated = generate_slot(str(root), config.model_dump(), slot.model_dump(), 'formal', digest(plan.model_dump()))
            write_json(output / 'generation.json', generated)
            result = generated
            if generated.get('status') == 'VALID':
                trials = collect_slot(str(root), config.model_dump(), slot.model_dump(), 'formal', digest(plan.model_dump()))
                result = {'status': 'VALID', 'trials': trials}
        except Exception as exc:
            result = {'status': 'INCONCLUSIVE', 'reason': f'{type(exc).__name__}: {exc}'[:1000]}
        result = {'slot': index, 'kind': kind, **result}
        write_json(output / 'result.json', result)
        return result


def main():
    """Run the six classified gaps with independent world processes."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-dir', type=Path, required=True)
    args = parser.parse_args()
    cases = [(126, 'blind'), (1147, 'blind')] + [(i, 'calibration') for i in (422, 714, 2157, 2579)]
    with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
        jobs = [pool.submit(worker, str(args.round_dir.resolve()), i, kind) for i, kind in cases]
        for job in as_completed(jobs):
            result = job.result()
            print(json.dumps({k: result[k] for k in ('slot', 'kind', 'status', 'reason') if k in result}), flush=True)


if __name__ == '__main__':
    main()
