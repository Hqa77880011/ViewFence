import random
import unittest

from viewfence import core


class DisclosureTests(unittest.TestCase):
    def test_rational_kernel_executes_and_survives_json_roundtrip(self):
        import json
        case = next(core.generate(random.Random(17), 1))
        for action in case['actions']:
            for table in action['worlds']:
                for rows in table.values():
                    for row in rows:
                        row['probability'] = '1/1'
        result = core.run_case(case, 'viewfence')
        self.assertTrue(result['completed'])
        self.assertEqual(result, core.run_case(json.loads(json.dumps(case)), 'viewfence'))

    def test_side_channel_is_checked_even_when_body_is_constant(self):
        case = next(core.generate(random.Random(17), 1))
        action = case['actions'][2]
        for world, table in enumerate(action['worlds']):
            table['0'][0]['event'].update(body='public', status=str(case['secrets'][world]))
        case['actions'] = [action]
        self.assertFalse(core.run_case(case, 'viewfence')['completed'])
        self.assertTrue(core.run_case(case, 'body_only')['violation'])

    def test_continuation_planning_avoids_greedy_dead_end(self):
        case = next(core.generate(random.Random(17), 1))
        trap = case['actions'][2]
        for table in trap['worlds']:
            table['0'][0].update(next_state='dead')
            table['0'][0]['event']['body'] = 'public'
        case['actions'] = [trap, case['actions'][0], case['actions'][1]]
        case['progress_values'] = {'dead': 0.9, 'inspected': 0.1}
        case['horizon'] = case['call_budget'] = 2
        self.assertTrue(core.run_case(case, 'viewfence')['completed'])
        self.assertFalse(core.run_case(case, 'one_step')['completed'])

    def test_durable_retry_and_exact_prior_retention(self):
        import tempfile
        from pathlib import Path
        from viewfence.session import SessionStore
        case = next(core.generate(random.Random(17), 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.sqlite'
            store = SessionStore(path)
            first = store.run(case)
            store.close()
            store = SessionStore(path)
            self.assertEqual(first, store.run(case))
            store.db.execute('UPDATE sessions SET busy=1')
            store.db.commit()
            with self.assertRaises(RuntimeError):
                store.run(case)
            store.close()

    def test_terminal_contract_reward_drives_selection(self):
        case = next(core.generate(random.Random(17), 1))
        action = case['actions'][2]
        action['reward'] = 0
        for world in action['worlds']:
            world['0'][0]['event']['body'] = 'complete'
        case['actions'] = [action]
        result = core.run_case(case, 'viewfence')
        self.assertTrue(result['completed'])
        self.assertFalse(result['violation'])
        self.assertEqual(result['reward'], 1)

    def test_stochastic_bayes_update_is_exact(self):
        from fractions import Fraction
        belief = ((0, 's', Fraction(1, 2)), (1, 's', Fraction(1, 2)))
        action = {'worlds': [
            [{'event': {'body': 'x'}, 'probability': '1/4'},
             {'event': {'body': 'y'}, 'probability': '3/4'}],
            [{'event': {'body': 'x'}, 'probability': '3/4'},
             {'event': {'body': 'y'}, 'probability': '1/4'}],
        ]}
        branches = core.branches(belief, action)
        self.assertEqual(branches[0][1], Fraction(1, 2))
        self.assertEqual(branches[0][2][0][2], Fraction(1, 4))


if __name__ == '__main__':
    unittest.main()
