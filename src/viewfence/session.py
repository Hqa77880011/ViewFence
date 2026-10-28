"""Persist exact linked-task beliefs and write-ahead state for the finite sandbox."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sqlite3

from .core import normalize, run_case


class SessionStore:
    """Stop on uncertain execution rather than resetting the observer's belief."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS sessions '
                        '(id TEXT PRIMARY KEY, belief TEXT, actual_state TEXT, world INTEGER, '
                        'policy TEXT, busy INTEGER, pending INTEGER)')
        self.db.execute('CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, result TEXT)')

    def run(self, case, method='viewfence'):
        if method not in ('viewfence', 'one_step'):
            raise ValueError('Durable sessions support the two full-observation safe controllers')
        session = str(case.get('session_id', case['id']))
        request = session + ':' + str(case['id'])
        policy = json.dumps([case['secrets'], case['alpha'], case.get('support_k')], sort_keys=True)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT belief,actual_state,world,policy,busy FROM sessions WHERE id=?', (session,)).fetchone()
            if row is not None and row[4]:
                raise RuntimeError('Uncertain prior execution: linked service is disabled without recovery evidence')
            previous = self.db.execute('SELECT result FROM results WHERE id=?', (request,)).fetchone()
            if previous:
                self.db.commit()
                return json.loads(previous[0])
            if row is None:
                states = case.get('initial_states', ['0'] * len(case['prior']))
                belief = normalize({(w, str(s)): Fraction(str(p)) for w, (s, p) in enumerate(zip(states, case['prior']))})
                actual_state = str(states[case['actual_world']])
                exact = json.dumps([[w, s, str(p)] for w, s, p in belief])
                self.db.execute('INSERT INTO sessions VALUES (?, ?, ?, ?, ?, 1, 0)',
                                (session, exact, actual_state, case['actual_world'], policy))
            else:
                if row[2] != case['actual_world'] or row[3] != policy:
                    raise ValueError('A linked session cannot replace its world identity or privacy policy')
                belief = tuple((w, s, Fraction(p)) for w, s, p in json.loads(row[0]))
                actual_state = row[1]
                self.db.execute('UPDATE sessions SET busy=1 WHERE id=?', (session,))
            self.db.commit()
        except (sqlite3.Error, ValueError, RuntimeError):
            self.db.rollback()
            raise

        def checkpoint(phase, posterior, state):
            exact = json.dumps([[w, s, str(p)] for w, s, p in posterior])
            with self.db:
                self.db.execute('UPDATE sessions SET belief=?,actual_state=?,pending=? WHERE id=?',
                                (exact, state, int(phase == 'reserve'), session))

        result = run_case(case, method, initial_belief=belief, actual_state=actual_state, step_hook=checkpoint)
        with self.db:
            self.db.execute('UPDATE sessions SET belief=?,actual_state=?,busy=0,pending=0 WHERE id=?',
                            (json.dumps(result['exact_posterior']), result['actual_state'], session))
            self.db.execute('INSERT INTO results VALUES (?, ?)', (request, json.dumps(result)))
        return result

    def close(self):
        self.db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='JSONL in linked-task order')
    parser.add_argument('--state-db', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    Path(args.state_db).parent.mkdir(parents=True, exist_ok=True)
    store = SessionStore(args.state_db)
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines() if line.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        for case in rows:
            stream.write(json.dumps(dict(id=case['id'], result=store.run(case))) + '\n')
    store.close()


if __name__ == '__main__':
    main()
