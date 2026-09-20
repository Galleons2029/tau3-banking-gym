"""Keep recovery export running independently and publish an explicit final status."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def process_identity(pid):
    """Avoid mistaking a reused process ID for the recovery worker."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def atomic(path, data):
    """Atomically publish watcher progress."""
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    """Export now, periodically, and once after all recovery jobs finish."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-dir', type=Path, required=True)
    parser.add_argument('--pids', type=int, nargs='+', required=True)
    args = parser.parse_args()
    root = args.round_dir.resolve()
    identities = {p: process_identity(p) for p in args.pids}
    export = None
    last_export = float('-inf')
    final_started = False
    exit_code = None
    while True:
        live = [p for p, identity in identities.items() if identity is not None and process_identity(p) == identity]
        if export is not None and export.poll() is not None:
            exit_code = export.returncode
            export = None
            if final_started:
                atomic(root / 'gap-recovery-v1/watcher.json', {'status': 'RECOVERY_FINISHED' if exit_code == 0 else 'EXPORT_FAILED', 'export_exit_code': exit_code, 'batch_complete': False, 'note': 'Recovery completion does not imply 3000-task batch completion.', 'updated_at': time.time()})
                break
        if export is None and (not live or time.monotonic() - last_export >= 600):
            final_started = not live
            with open('/tmp/targeted-gap-export-v1.log', 'a') as log:
                export = subprocess.Popen([sys.executable, 'scripts/run_targeted_continuous.py', '--round-dir', str(root), '--heartbeat', '/tmp/targeted-gap-export-heartbeat.json', '--mode', 'export'], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=os.environ.copy())
            last_export = time.monotonic()
        atomic(root / 'gap-recovery-v1/watcher.json', {'status': 'RUNNING', 'live_processes': live, 'export_pid': export.pid if export else None, 'last_export_exit_code': exit_code, 'final_export': final_started, 'updated_at': time.time()})
        time.sleep(15)


if __name__ == '__main__':
    main()
