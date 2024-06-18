import argparse
import json
import os
from ppo_daac_idaac.arguments import parser


def generate_train_cmds(
        params, envs, num_trials=1, start_index=0, newlines=False,
        xpid_generator=None, xpid_prefix=''):
    separator = ' \\\n' if newlines else ' '

    cmds = []

    start_seed = params['seed']
    for e in envs:
        params['env_name'] = e
        if xpid_generator:
            params['xpid'] = xpid_generator(params, xpid_prefix)

        for t in range(num_trials):
            params['seed'] = start_seed + t + start_index

            cmd = [f'python train.py']

            trial_idx = t + start_index
            for k, v in params.items():
                if type(v) == bool:
                    if v:
                        cmd.append(f'--{k}')
                    else:
                        pass
                    continue
                if k == 'xpid':
                    v = f'{v}-{trial_idx}'

                cmd.append(f'--{k}={v}')

            cmd = separator.join(cmd)

            cmds.append(cmd)

    return cmds


def generate_all_params_for_grid(grid, defaults={}):
    def update_params_with_choices(prev_params, param, choices):
        updated_params = []
        for v in choices:
            for p in prev_params:
                updated = p.copy()
                updated[param] = v
                updated_params.append(updated)

        return updated_params

    all_params = [{}]
    for param, choices in grid.items():
        all_params = update_params_with_choices(all_params, param, choices)

    full_params = []
    for p in all_params:
        d = defaults.copy()
        d.update(p)
        full_params.append(d)

    return full_params


def parse_args():
    parser = argparse.ArgumentParser(description='Make commands')

    parser.add_argument(
        '--dir',
        type=str,
        default='slurm/exp_config/',
        help='Path to directory with .json configs')

    parser.add_argument(
        '--json',
        type=str,
        default=None,
        help='Name of .json config for hyperparameter search-grid')

    parser.add_argument(
        '--all_envs',
        action='store_true',
        help='Whether to generate commands for all environments')

    parser.add_argument(
        '--num_trials',
        type=int,
        default=1,
        help='Name of .json config for hyperparameter search-grid')

    parser.add_argument(
        '--no_linebreaks',
        action='store_true',
        help='Whether to include linebreaks in output.')

    parser.add_argument(
        '--start_index',
        default=0,
        type=int,
        help='Starting trial index of runs')

    parser.add_argument(
        '--count',
        action='store_true',
        help='Print number of generated commands at the end of output.')

    return parser.parse_args()


# def xpid_from_params(p, prefix=''):
#     prefix_str = '' if prefix == '' else f'{prefix}'
#     env_prefix = p['env_name']
#     algo = p['algo']
#     adv_coef = p['adv_loss_coef']
#
#     return f'{prefix_str}-e:{env_prefix}-alg:{algo}-adv:{adv_coef}'

ENV_NAMES = ['plunder', 'chaser', 'miner', 'climber', 'bigfish', 'dodgeball', 'maze', 'leaper',
             'fruitbot', 'bossfight', 'jumper', 'ninja', 'starpilot', 'coinrun', 'heist', 'caveflyer']


if __name__ == '__main__':
    args = parse_args()

    # Default parameters
    params = vars(parser.parse_args([]))

    json_filename = args.json
    if not json_filename.endswith('.json'):
        json_filename += '.json'

    grid_path = os.path.join(os.path.expandvars(os.path.expanduser(args.dir)), json_filename)
    config = json.load(open(grid_path))
    grid = config['grid']
    # xpid_prefix = '' if 'xpid_prefix' not in config else config['xpid_prefix']

    # Generate all parameter combinations within grid, using defaults for fixed params
    all_params = generate_all_params_for_grid(grid, defaults=params)

    # Print all commands
    count = 0
    for p in all_params:
        cmds = generate_train_cmds(p,
                                   envs=ENV_NAMES if args.all_envs else [p['env_name']],
                                   num_trials=args.num_trials,
                                   start_index=args.start_index,
                                   newlines=not args.no_linebreaks,
                                   xpid_generator=None)

        for c in cmds:
            print(c)
            if not args.no_linebreaks:
                print('\n')
            count += 1

    if args.count:
        print(f'Generated {count} commands')
