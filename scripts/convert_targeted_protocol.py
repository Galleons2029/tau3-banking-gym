"""Versioned selector-protocol conversion with captured evidence and dual replay.

Business operations and policies remain V2 operations. This adapter aligns public
selector names, not the complete official banking execution environment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import re
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

SELECTORS = {
    'unlock_discoverable_agent_tool': 'agent_tool_name',
    'call_discoverable_agent_tool': 'agent_tool_name',
    'give_discoverable_user_tool': 'discoverable_tool_name',
    'call_discoverable_user_tool': 'discoverable_tool_name',
}
VERSION = 'targeted-selector-protocol-v1'
PROTOCOL = '''\n\nDiscoverable-tool calling protocol:
- Assistant: unlock_discoverable_agent_tool(agent_tool_name=the exact documented name).
- Assistant: call_discoverable_agent_tool(agent_tool_name=that unlocked name, arguments=a JSON object encoded as a string).
- Assistant: give_discoverable_user_tool(discoverable_tool_name=the exact customer-owned tool name).
- Customer only: call_discoverable_user_tool(discoverable_tool_name=the granted name, arguments=a JSON object encoded as a string).
Read each actual tool schema. Preserve all business argument names and values. Do not call customer-owned tools as the assistant.
'''


def canonical(value):
    """Stable JSON for comparisons and hashes."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def sha(value):
    """Hash text or exact file bytes."""
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def atomic(path, value):
    """Atomically save a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read(path):
    """Retry transient shared-filesystem read errors without repeating model calls."""
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except OSError:
            if attempt == 2:
                raise
            time.sleep(.1 * (attempt + 1))


def rename_arguments(name, arguments, reverse=False):
    """Rename only the outer selector; preserve inner JSON bytes and failed calls."""
    result = copy.deepcopy(arguments)
    if name not in SELECTORS:
        return result
    old, new = ('capability', SELECTORS[name])
    if reverse:
        old, new = new, old
    if old in result:
        if new in result:
            raise ValueError('Ambiguous selector collision')
        result[new] = result.pop(old)
    return result


def transform_text(text, aliases, role, source_call=None):
    """Rewrite explicit protocol references, never ordinary capability prose."""
    if role == 'system':
        text = text.replace('tell the user the exact capability name',
                            'tell the user the exact discoverable tool name')
        text += PROTOCOL

    def explicit(match):
        alias = match['alias']
        if alias not in aliases:
            raise ValueError('Unresolved explicit capability instruction: ' + alias)
        key = 'discoverable_tool_name' if aliases[alias] == 'user' else 'agent_tool_name'
        return key + match['tail']

    text = re.sub(r'\bcapability(?P<tail>:\s*`(?P<alias>[^`]+)`)', explicit, text)
    # These are exact schema-error payloads, not arbitrary words in retrieved text.
    if role == 'tool' and source_call in SELECTORS and 'missing ' in text and 'positional argument' in text:
        text = text.replace("'capability'", repr(SELECTORS[source_call]))
    if re.search(r'["\']capability["\']\s*:', text) or re.search(r'\bcapability\s*=', text):
        raise ValueError('Unresolved structured capability reference')
    return text


def transform_row(row, aliases, failed_ids):
    """Return converted and control rows with identical selective supervision."""
    converted, control = copy.deepcopy(row), copy.deepcopy(row)
    changes = []
    call_names = {}
    masked = []
    for i, message in enumerate(row['messages']):
        calls = message.get('tool_calls') or []
        loss = message['role'] == 'assistant' and not any(c['id'] in failed_ids for c in calls)
        if message['role'] == 'assistant' and not loss:
            masked.append(i)
        for target in (converted, control):
            target['loss_mask'][i] = int(loss)
            if message['role'] == 'assistant':
                target['messages'][i]['weight'] = int(loss)
        for j, call in enumerate(calls):
            name = call['function']['name']
            call_names[call['id']] = name
            args = json.loads(call['function']['arguments'])
            mapped = rename_arguments(name, args)
            if rename_arguments(name, mapped, reverse=True) != args:
                raise ValueError('Selector round-trip failed')
            if args != mapped:
                converted['messages'][i]['tool_calls'][j]['function']['arguments'] = json.dumps(mapped, ensure_ascii=False)
                changes.append({'path': f'/messages/{i}/tool_calls/{j}/function/arguments', 'operation': 'outer_selector_rename'})
        text = message.get('content', '')
        new_text = transform_text(text, aliases, message['role'], call_names.get(message.get('tool_call_id')))
        if new_text != text:
            converted['messages'][i]['content'] = new_text
            changes.append({'path': f'/messages/{i}/content', 'operation': 'protocol_text', 'before_hash': sha(text), 'after_hash': sha(new_text)})
        if 'reasoning_content' in message:
            raise ValueError('Use visible-context source without reasoning')
    for i, tool in enumerate(converted['tools']):
        function = tool['function']
        name = function['name']
        if name not in SELECTORS:
            continue
        params = function['parameters']
        new = SELECTORS[name]
        if 'capability' not in params['properties'] or new in params['properties']:
            raise ValueError('Unexpected source wrapper schema')
        definition = params['properties'].pop('capability')
        definition['title'] = new.replace('_', ' ').title()
        params['properties'][new] = definition
        params['required'] = [new if k == 'capability' else k for k in params.get('required', [])]
        changes.append({'path': f'/tools/{i}/function/parameters', 'operation': 'selector_schema_rename'})
    return converted, control, {'changes': changes, 'masked_assistant_indices': masked}


def validate(row):
    """Check pairing, schema fidelity of supervised calls and assistant-only loss."""
    from jsonschema import Draft202012Validator

    tools = {t['function']['name']: t['function'] for t in row['tools']}
    validators = {}
    for name, tool in tools.items():
        schema = copy.deepcopy(tool['parameters'])
        schema['additionalProperties'] = False
        Draft202012Validator.check_schema(schema)
        validators[name] = Draft202012Validator(schema)
    if len(row['messages']) != len(row['loss_mask']):
        raise ValueError('Mask length mismatch')
    pending, seen = set(), set()
    for message, loss in zip(row['messages'], row['loss_mask'], strict=True):
        role = message['role']
        if role not in {'system', 'assistant', 'user', 'tool'} or (loss and role != 'assistant'):
            raise ValueError('Private or non-assistant supervised content')
        if role == 'assistant':
            if pending:
                raise ValueError('Unpaired tool results before assistant turn')
            for call in message.get('tool_calls') or []:
                if call['id'] in seen:
                    raise ValueError('Duplicate tool call id')
                seen.add(call['id'])
                pending.add(call['id'])
                name = call['function']['name']
                arguments = json.loads(call['function']['arguments'])
                if loss:
                    if name not in validators:
                        raise ValueError('Supervised unknown tool')
                    validators[name].validate(arguments)
        elif role == 'tool':
            identifier = message.get('tool_call_id')
            if identifier not in pending:
                raise ValueError('Tool result without call')
            pending.remove(identifier)
    if pending:
        raise ValueError('Unfinished tool calls')


class ProtocolAdapter:
    """Public selector-compatible signatures backed by unchanged V2 operations."""

    def __init__(self, agent, user):
        self.agent, self.user = agent, user

    def unlock_discoverable_agent_tool(self, agent_tool_name):
        """Unlock the exact documented agent tool."""
        return self.agent.unlock_discoverable_agent_tool(agent_tool_name)

    def call_discoverable_agent_tool(self, agent_tool_name, arguments):
        """Call an unlocked tool without modifying its business arguments."""
        return self.agent.call_discoverable_agent_tool(agent_tool_name, arguments)

    def give_discoverable_user_tool(self, discoverable_tool_name):
        """Grant the exact user-owned tool, retaining original V2 grant semantics."""
        return self.agent.give_discoverable_user_tool(discoverable_tool_name)

    def call_discoverable_user_tool(self, discoverable_tool_name, arguments):
        """Execute an actually granted user tool."""
        return self.user.call_discoverable_user_tool(discoverable_tool_name, arguments)


def normalized(value):
    """Compare semantic JSON results without changing training result bodies."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def outcome(function, arguments):
    """Preserve real error outcomes instead of repairing failed actions."""
    try:
        return False, function(**arguments)
    except Exception as exc:
        return True, f'Error: {exc}'


def error_identity(value):
    """Ignore only class qualification and renamed selector in Python errors."""
    for new in set(SELECTORS.values()):
        value = value.replace(repr(new), "'capability'")
    return re.sub(r'(?:WorldTools|WorldUserTools|ProtocolAdapter)\.', '', value)


@lru_cache(maxsize=8)
def world_material(world_string, expected_world_hash):
    """Bind source world artifacts and cache immutable definitions per worker."""
    from tau3.worldgen.v2.pipeline import artifact_hashes, check_certificate
    from tau3.worldgen.v2.runtime import WorldDB, digest
    from tau3.worldgen.v2.specs import WorldSpec

    world = Path(world_string)
    check_certificate(world)
    if digest(artifact_hashes(world)) != expected_world_hash:
        raise ValueError('World hash mismatch')
    spec = WorldSpec.model_validate_json((world / 'spec.json').read_text())
    initial = WorldDB.model_validate_json((world / 'db.json').read_text())
    return spec, initial


def dual_replay(row, native, world):
    """Replay both actors, compare every state/result, then check original goals."""
    from tau3.worldgen.v2.environment import WorldTools, WorldUserTools
    from tau3.worldgen.v2.runtime import OperationRuntime, digest, goal_errors

    spec, initial = world_material(str(world), row['metadata']['world_hash'])
    agent = WorldTools(spec, initial.model_copy(deep=True), initial)
    user = WorldUserTools(agent)
    target_agent = WorldTools(spec, initial.model_copy(deep=True), initial)
    target_user = WorldUserTools(target_agent)
    adapter = ProtocolAdapter(target_agent, target_user)
    runtime = OperationRuntime(spec, initial)
    aliases = {alias: runtime.operations[op][1].actor for alias, op in runtime.aliases.items()}
    pending = {}
    failed_ids = set()
    counts = Counter()
    for message in native['simulation']['messages']:
        for call in message.get('tool_calls') or []:
            actor = call.get('requestor', message['role'])
            key = (actor, call['id'])
            pending.setdefault(key, deque()).append(call)
        if message['role'] != 'tool':
            continue
        key = (message.get('requestor', 'assistant'), message['id'])
        if key not in pending or not pending[key]:
            raise ValueError('Native tool response lacks actor-bound call')
        call = pending[key].popleft()
        name, args = call['name'], call['arguments']
        actor = key[0]
        content = message.get('content') or ''
        is_error = bool(message.get('error')) or content.startswith('Error:')
        if is_error and actor == 'assistant':
            failed_ids.add(call['id'])
        if name in {'KB_search', 'grep'}:
            # Retrieval evidence is retained verbatim; no fresh retrieval is claimed.
            counts['retrieval_results_preserved'] += 1
            continue
        original_tools = agent if actor == 'assistant' else user
        target_tools = target_agent if actor == 'assistant' else target_user
        if not hasattr(original_tools, name):
            raise ValueError('Unsupported replay tool: ' + name)
        old_error, old = outcome(getattr(original_tools, name), args)
        changed = rename_arguments(name, args)
        if name in SELECTORS:
            expected_actor = 'user' if name == 'call_discoverable_user_tool' else 'assistant'
            if actor != expected_actor:
                raise ValueError('Wrapper actor mismatch')
            new_error, new = outcome(getattr(adapter, name), changed)
        else:
            new_error, new = outcome(getattr(target_tools, name), changed)
        if old_error != new_error:
            raise ValueError('Replay exception outcome changed')
        if old_error:
            if error_identity(old) != error_identity(new):
                raise ValueError('Replay error changed')
            if error_identity(old) != error_identity(content):
                raise ValueError('Original captured error does not replay: ' + name)
        elif normalized(old) != normalized(new) or normalized(old) != normalized(content):
            raise ValueError('Replay result differs from native capture: ' + name)
        if agent.db != target_agent.db or agent.unlocked != target_agent.unlocked or agent.user_grants != target_agent.user_grants:
            raise ValueError('State/grant divergence after call')
        counts['replayed_calls'] += 1
        counts['user_calls'] += actor == 'user'
        counts['error_calls'] += old_error
    if any(queue for queue in pending.values()):
        raise ValueError('Incomplete native results')
    scenario = next(s for c in spec.categories for s in c.scenarios if 'task_' + s.id == row['task_id'])
    errors = goal_errors(initial, target_agent.db, scenario.goals, scenario.user_id, spec)
    if errors:
        raise ValueError('Original target state not achieved: ' + str(errors))
    return aliases, failed_ids, {'status': 'PASS', **counts, 'final_state_hash': digest(agent.db.model_dump()), 'grant_hash': digest(sorted(agent.user_grants)), 'unlocked_hash': digest(sorted(agent.unlocked))}


def general(row):
    """Export visible OpenAI messages with explicit loss and no private metadata."""
    tools = copy.deepcopy(row['tools'])
    for tool in tools:
        tool['function']['strict'] = tool['function'].get('strict', False)
    messages = []
    for message, loss in zip(row['messages'], row['loss_mask'], strict=True):
        value = {k: v for k, v in message.items() if k in {'role', 'content', 'tool_calls', 'tool_call_id', 'name'}}
        if not value.get('tool_calls'):
            value.pop('tool_calls', None)
        value['loss'] = bool(loss)
        messages.append(value)
    return {'tools': tools, 'messages': messages}


def process_group(source_rows, output_string, identity_hash):
    """Validate one world and produce independently resumable conversion shards."""
    from tau3.synthesis.targeted.workflow import clean_sample
    from tau3.worldgen.v2.runtime import digest

    output = Path(output_string)
    slot = Path(source_rows[0][1]['metadata']['evidence']).parents[1].name
    directory = output / 'shards' / slot
    report_path = directory / 'report.json'
    if report_path.exists():
        cached = read(report_path)
        if cached.get('identity_hash') == identity_hash and cached.get('status') == 'PASS':
            if all(sha((directory / name).read_bytes()) == h for name, h in cached['files'].items()):
                return cached
        raise ValueError('Recovery shard changed; use a new output directory')
    directory.mkdir(parents=True, exist_ok=True)
    converted_rows, control_rows, audit_rows = [], [], []
    for line_index, row in source_rows:
        trial = Path(row['metadata']['evidence'])
        native, capture, quality = (read(trial / name) for name in ('result.json', 'capture.json', 'quality.json'))
        if not quality.get('sft_qualified') or quality.get('status') != 'COMPLETE':
            raise ValueError('Source trajectory no longer qualified')
        if digest(native) != quality['native_hash'] or digest(capture) != quality['capture_hash'] or native['capture_hash'] != digest(capture):
            raise ValueError('Native/capture evidence mismatch')
        sample = clean_sample(capture['sample'])
        if digest(sample) != quality['sample_hash'] or any(row[k] != v for k, v in sample.items()):
            raise ValueError('Export is not the actual teacher capture')
        if native['identity'] != capture['identity'] or native['identity'] != read(trial / 'started.json'):
            raise ValueError('Teacher seed/capture identity mismatch')
        world = Path(read(trial.parents[1] / 'task.json')['world'])
        aliases, failed_ids, replay = dual_replay(row, native, world)
        converted, control, mapping = transform_row(row, aliases, failed_ids)
        validate(converted)
        validate(control)
        mapped_metadata = {**converted['metadata'], 'protocol_conversion': {'version': VERSION, 'source_line': line_index + 1, 'source_row_sha256': sha(canonical(row)), 'derived_not_original_capture': True}}
        converted['metadata'] = mapped_metadata
        converted_rows.append(converted)
        control_rows.append(control)
        audit_rows.append({'source_line': line_index + 1, 'task_id': row['task_id'], 'trial': row['metadata']['trial'], 'seed': row['metadata']['seed'], 'source_row_hash': sha(canonical(row)), 'converted_row_hash': sha(canonical(converted)), 'native_hash': quality['native_hash'], 'capture_hash': quality['capture_hash'], 'replay': replay, **mapping})
    files = {}
    for name, rows in [('converted.jsonl', converted_rows), ('control.jsonl', control_rows), ('mapping.jsonl', audit_rows)]:
        text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        temporary = directory / (name + '.tmp')
        temporary.write_text(text)
        temporary.replace(directory / name)
        files[name] = sha(text)
    report = {'status': 'PASS', 'slot': slot, 'identity_hash': identity_hash, 'rows': len(source_rows), 'files': files, 'masked_turns': sum(len(r['masked_assistant_indices']) for r in audit_rows), 'replayed_calls': sum(r['replay']['replayed_calls'] for r in audit_rows), 'user_calls': sum(r['replay']['user_calls'] for r in audit_rows)}
    atomic(report_path, report)
    return report


def safe_group(*args):
    """Persist failures as data instead of losing them through process pickling."""
    try:
        return process_group(*args)
    except Exception as exc:
        rows, output, identity = args
        slot = Path(rows[0][1]['metadata']['evidence']).parents[1].name
        result = {'status': 'FAIL', 'slot': slot, 'reason': f'{type(exc).__name__}: {exc}'}
        atomic(Path(output) / 'failures' / f'{slot}.json', result)
        return result


def main():
    """Convert the frozen corpus and publish only after all rows pass replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit-worlds', type=int)
    args = parser.parse_args()
    if args.output.resolve() == args.source.parent.resolve():
        raise ValueError('Output must be a separate version directory')
    source_bytes = args.source.read_bytes()
    identity = {'version': VERSION, 'source': str(args.source.resolve()), 'source_sha256': sha(source_bytes), 'converter_sha256': sha(Path(__file__).read_bytes()), 'runtime': os.environ.get('PYTHONPATH'), 'limit_worlds': args.limit_worlds, 'reasoning': 'excluded', 'failed_assistant_turns': 'masked_in_both_arms'}
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / 'identity.json'
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError('Output version is frozen; use a new output directory')
    atomic(identity_path, identity)
    identity_hash = sha(canonical(identity))
    groups = {}
    for index, line in enumerate(source_bytes.decode().splitlines()):
        row = json.loads(line)
        slot = Path(row['metadata']['evidence']).parents[1].name
        if args.limit_worlds and slot not in groups and len(groups) >= args.limit_worlds:
            continue
        groups.setdefault(slot, []).append((index, row))
    del source_bytes
    reports, failures = {}, []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(safe_group, rows, str(args.output), identity_hash) for rows in groups.values()]
        for future in as_completed(futures):
            report = future.result()
            reports[report['slot']] = report
            if report['status'] != 'PASS':
                failures.append(report)
            if len(reports) % 20 == 0 or len(reports) == len(groups) or report['status'] != 'PASS':
                progress = {'status': 'VALIDATING', 'worlds': len(groups), 'processed': len(reports), 'failures': failures, 'passed_rows': sum(r.get('rows', 0) for r in reports.values()), 'elapsed_seconds': time.monotonic() - started}
                atomic(args.output / 'progress.json', progress)
                print(json.dumps({k: v for k, v in progress.items() if k != 'failures'} | {'failed_worlds': len(failures)}), flush=True)
    if failures:
        atomic(args.output / 'report.json', {'status': 'FAIL', 'failures': failures, 'published': False})
        raise SystemExit(1)
    destinations = {'sft.jsonl': 'converted.jsonl', 'control.sft.jsonl': 'control.jsonl', 'mapping.jsonl': 'mapping.jsonl', 'general_agent.jsonl': 'converted.jsonl', 'control.general_agent.jsonl': 'control.jsonl'}
    hashes = {}
    for target, shard in destinations.items():
        temporary = args.output / (target + '.tmp')
        with temporary.open('w') as stream:
            for slot in sorted(groups):
                path = args.output / 'shards' / slot / shard
                if sha(path.read_bytes()) != reports[slot]['files'][shard]:
                    raise ValueError('Shard changed before publication')
                for line in path.open():
                    stream.write(json.dumps(general(json.loads(line)), ensure_ascii=False) + '\n' if 'general_agent' in target else line)
        hashes[target] = sha(temporary.read_bytes())
        temporary.replace(args.output / target)
    if sha(args.source.read_bytes()) != identity['source_sha256']:
        raise ValueError('Source changed during conversion')
    report = {'status': 'PASS', 'identity': identity, 'worlds': len(groups), 'rows': sum(r['rows'] for r in reports.values()), 'masked_assistant_turns': sum(r['masked_turns'] for r in reports.values()), 'replayed_calls_per_arm': sum(r['replayed_calls'] for r in reports.values()), 'user_calls_per_arm': sum(r['user_calls'] for r in reports.values()), 'hashes': hashes, 'elapsed_seconds': time.monotonic() - started, 'scope': 'V2 business-state/result/grant equivalence under renamed public selectors; retrieval results preserved, no new LLM semantic review or native tokenizer validation; not an official banking task conversion.'}
    atomic(args.output / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
