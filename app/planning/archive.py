"""Append-only, hash-chained trajectory with atomic checkpoint files."""
from copy import deepcopy
from pathlib import Path
import json
import os
from app.planning.state import PlanningState, canonical, digest


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


class Trajectory:
    def __init__(self, path, initial=None):
        self.path = Path(path)
        if (self.path / 'initial.json').exists():
            self.state, self.rows = reconstruct(self.path)
        else:
            if initial is None:
                raise ValueError('initial snapshot required')
            self.state = PlanningState(initial)
            self.rows = []
            write(self.path / 'initial.json', self.state.snapshot)

    def apply(self, event):
        before = digest(self.state.snapshot)
        if not self.state.apply(event):
            return False
        row = {'seq': len(self.rows), 'event': deepcopy(event), 'before': before,
               'after': deepcopy(self.state.snapshot),
               'previous_digest': self.rows[-1]['digest'] if self.rows else digest(read(self.path/'initial.json'))}
        row['digest'] = digest(row)
        with (self.path/'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(canonical(row)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        self.rows.append(row)
        return True


def reconstruct(path):
    path = Path(path)
    initial = read(path/'initial.json')
    state = PlanningState(initial)
    rows = []
    previous = digest(initial)
    if (path/'events.jsonl').exists():
        for line in (path/'events.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            unsigned = {k:v for k,v in row.items() if k != 'digest'}
            if row['seq'] != len(rows) or row['previous_digest'] != previous or digest(unsigned) != row['digest']:
                raise ValueError('corrupt event chain')
            if row['before'] != digest(state.snapshot):
                raise ValueError('before-state mismatch')
            # Playback reproduces an archived decision, rather than asking the
            # current mathematical model whether that old decision is valid.
            state.apply(row['event'], validate_plan=False)
            if state.snapshot != row['after']:
                raise ValueError('state reducer incompatibility or corrupted timeline')
            rows.append(row)
            previous = row['digest']
    return state, rows

