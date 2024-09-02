# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2023 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

import logging
import os.path
import socket
import cProfile
import warnings
import getpass
import subprocess
from pandas.errors import SettingWithCopyWarning

from openquake.baselib import performance, general, parallel, slurm
from openquake.hazardlib import valid
from openquake.commonlib import logs, datastore, readinput
from openquake.calculators import base, views
from openquake.engine.engine import create_jobs, run_jobs

calc_path = None  # set only when the flag --slowest is given


# called when profiling
def _run(job_ini, concurrent_tasks, pdb, reuse_input, loglevel, exports,
         params, user_name, host=None):
    global calc_path
    if 'hazard_calculation_id' in params:
        hc_id = int(params['hazard_calculation_id'])
        if hc_id < 0:  # interpret negative calculation ids
            calc_ids = logs.get_calc_ids()
            try:
                params['hazard_calculation_id'] = calc_ids[hc_id]
            except IndexError:
                raise SystemExit(
                    'There are %d old calculations, cannot '
                    'retrieve the %s' % (len(calc_ids), hc_id))
        else:
            params['hazard_calculation_id'] = hc_id
    dic = readinput.get_params(job_ini, params)
    # set the logs first of all
    log = logs.init(dic, log_level=getattr(logging, loglevel.upper()),
                    user_name=user_name, host=host)
    logs.dbcmd('update_job', log.calc_id,
               {'status': 'executing', 'pid': os.getpid()})
    with log, performance.Monitor('total runtime', measuremem=True) as monitor:
        calc = base.calculators(log.get_oqparam(), log.calc_id)
        if reuse_input:  # enable caching
            calc.oqparam.cachedir = datastore.get_datadir()
        calc.run(concurrent_tasks=concurrent_tasks, pdb=pdb, exports=exports)

    logging.info('Total time spent: %s s', monitor.duration)
    logging.info('Memory allocated: %s', general.humansize(monitor.mem))
    calc_path, _ = os.path.splitext(calc.datastore.filename)  # used below
    return calc


def main(job_ini,
         pdb=False,
         reuse_input=False,
         *,
         slowest: int = None,
         hc: int = None,
         param=(),
         concurrent_tasks: int = None,
         exports: valid.export_formats = '',
         loglevel='info',
         nodes: int = 1):
    """
    Run a calculation
    """
    # os.environ['OQ_DISTRIBUTE'] = 'processpool'
    warnings.filterwarnings("error", category=SettingWithCopyWarning)
    user_name = getpass.getuser()
    try:
        host = socket.gethostname()
    except Exception:  # gaierror
        host = None
    if param:
        params = {}
        for par in param:
            k, v = par.split('=', 1)
            params[k] = v
    else:
        params = {}
    if hc:
        params['hazard_calculation_id'] = str(hc)
    if slowest:
        prof = cProfile.Profile()
        prof.runctx('_run(job_ini[0], None, pdb, reuse_input, loglevel, '
                    'exports, params, host)', globals(), locals())
        pstat = calc_path + '.pstat'
        prof.dump_stats(pstat)
        print('Saved profiling info in %s' % pstat)
        data = performance.get_pstats(pstat, slowest)
        print(views.text_table(data, ['ncalls', 'cumtime', 'path'],
                               ext='org'))
        return
    dics = [readinput.get_params(ini) for ini in job_ini]
    for dic in dics:
        dic.update(params)
        dic['exports'] = ','.join(exports)
        if concurrent_tasks:
            dic['concurrent_tasks'] = ct = str(concurrent_tasks)
        elif 'concurrent_tasks' not in dic:
            ct = 2 * parallel.Starmap.num_cores * nodes
            dic['concurrent_tasks'] = str(ct)
    jobs = create_jobs(dics, loglevel, hc_id=hc,
                       user_name=user_name, host=host, multi=False)
    job_id = jobs[0].calc_id
    dist = parallel.oq_distribute()
    if dist == 'slurm' and 'job_id' not in params:
        slurm.start_workers(nodes, job_id)
        slurm.wait_workers(nodes, job_id)
        run_args = [' '.join(job_ini), '-l', loglevel]
        if hc:
            run_args.extend(['--hc', str(hc)])
        if concurrent_tasks:
            run_args.extend(['-c', str(concurrent_tasks)])
        else:
            run_args.extend(['-c', str(ct)])
        export = ','.join(exports)
        if export:
            run_args.extend(['-e', export])
        run_args.extend(['-p', f'job_id={job_id}'])
        run_args.extend(param)
        try:
            cmd = ['srun', '--cpus-per-task', '16', '--time', '24:00:00'] + \
                slurm.submit_cmd[1:] + run_args
            subprocess.run(cmd)
        finally:
            slurm.stop_workers(job_id)
    else:
        if dist == 'slurm':
            parallel.Starmap.CT = concurrent_tasks
        run_jobs(jobs)
    return job_id


main.job_ini = dict(help='calculation configuration file '
                    '(or files, space-separated)', nargs='+')
main.pdb = dict(help='enable post mortem debugging', abbrev='-d')
main.reuse_input = dict(help='reuse source model and exposure')
main.slowest = dict(help='profile and show the slowest operations')
main.hc = dict(help='previous calculation ID')
main.param = dict(help='override parameters with TOML syntax', nargs='*')
main.concurrent_tasks = dict(help='hint for the number of tasks to spawn')
main.exports = dict(help='export formats as a comma-separated string')
main.loglevel = dict(help='logging level',
                     choices='debug info warn error critical'.split())
main.nodes=dict(help='number of worker nodes to start')
