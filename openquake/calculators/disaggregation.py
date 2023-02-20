# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2023 GEM Foundation
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

"""
Disaggregation calculator core functionality
"""

import logging
import numpy

from openquake.baselib import parallel, hdf5
from openquake.baselib.general import (
    AccumDict, pprod, agg_probs, shortlist)
from openquake.baselib.python3compat import encode
from openquake.hazardlib import stats
from openquake.hazardlib.calc import disagg
from openquake.hazardlib.contexts import (
    read_cmakers, read_src_mutex, read_ctx_by_grp)
from openquake.commonlib import util, calc
from openquake.calculators import getters
from openquake.calculators import base

POE_TOO_BIG = '''\
Site #%d: you are trying to disaggregate for poe=%s.
However the source model produces at most probabilities
of %.7f for rlz=#%d, IMT=%s.
The disaggregation PoE is too big or your model is wrong,
producing too small PoEs.'''
U8 = numpy.uint8
U16 = numpy.uint16
U32 = numpy.uint32
F32 = numpy.float32


def _iml4(rlzs, iml_disagg, imtls, poes_disagg, curves):
    # an ArrayWrapper of shape (N, M, P, Z)
    N, Z = rlzs.shape
    P = len(poes_disagg)
    M = len(imtls)
    arr = numpy.empty((N, M, P, Z))
    acc = AccumDict(accum=[])  # site, imt, poe -> rlzs
    for m, imt in enumerate(imtls):
        for (s, z), rlz in numpy.ndenumerate(rlzs):
            curve = curves[s][z]
            if poes_disagg == (None,):
                arr[s, m, 0, z] = imtls[imt]
            elif curve:
                rlz = rlzs[s, z]
                max_poe = curve[imt].max()
                arr[s, m, :, z] = calc.compute_hazard_maps(
                    curve[imt], imtls[imt], poes_disagg)
                for iml, poe in zip(arr[s, m, :, z], poes_disagg):
                    if iml == 0:
                        acc[s, imt, poe].append(rlz)
                    elif poe > max_poe:
                        logging.warning(
                            POE_TOO_BIG, s, poe, max_poe, rlz, imt)
    for (s, imt, poe), zero_rlzs in acc.items():
        logging.warning('Cannot disaggregate for site %d, %s, '
                        'poe=%s, rlzs=%s: the hazard is zero',
                        s, imt, poe, shortlist(zero_rlzs))
    return hdf5.ArrayWrapper(arr, {'rlzs': rlzs})


def compute_disagg(dstore, ctxt, sitecol, cmaker, bin_edges, src_mutex, wdic,
                   monitor):
    """
    :param dstore:
        a DataStore instance
    :param ctxt:
        a context array
    :param sitecol:
        a site collection
    :param cmaker:
        a ContextMaker instance
    :param bin_edges:
        a tuple of bin edges (mag, dist, lon, lat, eps, trt)
    :param src_mutex:
        a dictionary src_id -> weight, usually empty
    :param wdic:
        dictionary rlz -> weight, empty for individual realizations
    :param monitor:
        monitor of the currently running job
    :returns:
        one 6D matrix of rates per site and realization
    """
    mon0 = monitor('disagg mean_std', measuremem=False)
    mon1 = monitor('disagg by eps', measuremem=False)
    mon2 = monitor('composing pnes', measuremem=False)
    mon3 = monitor('disagg matrix', measuremem=False)
    out = []
    for site in sitecol:
        try:
            dis = disagg.Disaggregator([ctxt], site, cmaker, bin_edges)
        except disagg.FarAwayRupture:
            continue
        with dstore:
            iml3 = dstore['hmap4'][dis.sid]
            rlzs = dstore['best_rlzs'][dis.sid]
        for magi in range(dis.Ma):
            try:
                dis.init(magi, src_mutex, mon0, mon1, mon2, mon3)
            except disagg.FarAwayRupture:
                continue
            res = {'trti': cmaker.trti, 'magi': dis.magi, 'sid': dis.sid}
            for z, rlz in enumerate(rlzs):
                iml2 = iml3[:, :, z]
                if rlz not in dis.g_by_rlz or iml2.sum() == 0:
                    continue  # do not disaggregate
                rates6D = disagg.to_rates(dis.disagg6D(iml2, rlz))
                if wdic:  # compute mean rates and store them in the 0 key
                    if 0 not in res:
                        res[0] = 0
                    res[0] += rates6D * wdic[rlz]
                else:  # store the rates in the rlz key
                    res[rlz] = rates6D
            out.append(res)
    return out


def get_outputs_size(shapedic, disagg_outputs, Z):
    """
    :returns: the total size of the outputs
    """
    tot = AccumDict(accum=0)
    for out in disagg_outputs:
        tot[out] = 8
        for key in out.lower().split('_'):
            tot[out] *= shapedic[key]
    return tot * shapedic['N'] * shapedic['M'] * shapedic['P'] * Z


def output_dict(shapedic, disagg_outputs, Z):
    N, M, P = shapedic['N'], shapedic['M'], shapedic['P']
    dic = {}
    for out in disagg_outputs:
        shp = tuple(shapedic[key] for key in out.lower().split('_'))
        dic[out] = numpy.zeros((N,) + shp + (M, P, Z))
    return dic


def logdebug(allrlzs, N, grp_id):
    msg = 'Sending task with %d/%d sites for grp_id=%d, %d rlzs'
    num_rlzs = sum(len(rlzs) for rlzs in allrlzs)
    logging.debug(msg, len(allrlzs), N, grp_id, num_rlzs)


@base.calculators.add('disaggregation')
class DisaggregationCalculator(base.HazardCalculator):
    """
    Classical PSHA disaggregation calculator
    """
    precalc = 'classical'
    accept_precalc = ['classical', 'disaggregation']

    def pre_checks(self):
        """
        Checks on the number of sites, atomic groups and size of the
        disaggregation matrix.
        """
        if self.N >= 32768:
            raise ValueError('You can disaggregate at max 32,768 sites')
        few = self.oqparam.max_sites_disagg
        if self.N > few:
            raise ValueError(
                'The number of sites is to disaggregate is %d, but you have '
                'max_sites_disagg=%d' % (self.N, few))
        self.oqparam.mags_by_trt = self.datastore['source_mags']
        all_edges, shapedic = disagg.get_edges_shapedic(
            self.oqparam, self.sitecol, self.R)
        *b, trts = all_edges
        T = len(trts)
        shape = [len(bin) - 1 for bin in
                 (b[0], b[1], b[2][0], b[3][0], b[4])] + [T]
        matrix_size = numpy.prod(shape)  # 6D
        if matrix_size > 1E6:
            raise ValueError(
                'The disaggregation matrix is too large '
                '(%d elements): fix the binning!' % matrix_size)

    def execute(self):
        """Performs the disaggregation"""
        return self.full_disaggregation()

    def get_curve(self, sid, rlzs):
        """
        Get the hazard curves for the given site ID and realizations.

        :param sid: site ID
        :param rlzs: a matrix of indices of shape Z
        :returns: a list of Z arrays of PoEs
        """
        poes = []
        pcurve = self.pgetter.get_pcurve(sid)
        for z, rlz in enumerate(rlzs):
            pc = pcurve.extract(rlz)
            if z == 0:
                self.curves.append(pc.array[:, 0])
            poes.append(pc.convert(self.oqparam.imtls))
        return poes

    def full_disaggregation(self):
        """
        Run the disaggregation phase.
        """
        oq = self.oqparam
        try:
            full_lt = self.full_lt
        except AttributeError:
            full_lt = self.datastore['full_lt']
        ws = [rlz.weight for rlz in full_lt.get_realizations()]
        if oq.rlz_index is None and oq.num_rlzs_disagg == 0:
            oq.num_rlzs_disagg = len(ws)  # 0 means all rlzs
        self.oqparam.mags_by_trt = self.datastore['source_mags']
        edges, self.shapedic = disagg.get_edges_shapedic(
            oq, self.sitecol, self.R)
        logging.info(self.shapedic)
        self.save_bin_edges(edges)
        self.full_lt = self.datastore['full_lt']
        self.poes_disagg = oq.poes_disagg or (None,)
        self.imts = list(oq.imtls)
        self.M = len(self.imts)
        dstore = (self.datastore.parent if self.datastore.parent
                  else self.datastore)
        nrows = len(dstore['_poes/sid'])
        self.pgetter = getters.PmapGetter(
            dstore, ws, [(0, nrows + 1)], oq.imtls, oq.poes)

        # build array rlzs (N, Z)
        if oq.rlz_index is None:
            Z = oq.num_rlzs_disagg
            rlzs = numpy.zeros((self.N, Z), int)
            if self.R > 1:
                for sid in self.sitecol.sids:
                    pcurve = self.pgetter.get_pcurve(sid)
                    mean = getters.build_stat_curve(
                        pcurve, oq.imtls, stats.mean_curve, ws)
                    # get the closest realization to the mean
                    rlzs[sid] = util.closest_to_ref(
                        pcurve.array.T, mean.array)[:Z]
            self.datastore['best_rlzs'] = rlzs
        else:
            Z = len(oq.rlz_index)
            rlzs = numpy.zeros((self.N, Z), int)
            for z in range(Z):
                rlzs[:, z] = oq.rlz_index[z]
            self.datastore['best_rlzs'] = rlzs

        assert Z <= self.R, (Z, self.R)
        self.Z = Z
        self.rlzs = rlzs
        self.curves = []

        if oq.iml_disagg:
            # no hazard curves are needed
            self.poe_id = {None: 0}
            curves = [[None for z in range(Z)] for s in range(self.N)]
        else:
            self.poe_id = {poe: i for i, poe in enumerate(oq.poes_disagg)}
            curves = [self.get_curve(sid, rlzs[sid])
                      for sid in self.sitecol.sids]
        s = self.shapedic
        logging.info('Building N * M * P * Z = {:_d} intensities'.format(
                     s['N'] * s['M'] * s['P'] * s['Z']))
        self.iml4 = _iml4(rlzs, oq.iml_disagg, oq.imtls,
                          self.poes_disagg, curves)
        if self.iml4.array.sum() == 0:
            raise SystemExit('Cannot do any disaggregation: zero hazard')
        self.datastore['hmap4'] = self.iml4
        self.datastore['poe4'] = numpy.zeros_like(self.iml4.array)
        self.sr2z = {(s, r): z for s in self.sitecol.sids
                     for z, r in enumerate(rlzs[s])}
        return self.compute()

    def compute(self):
        """
        Submit disaggregation tasks and return the results
        """
        oq = self.oqparam
        dstore = (self.datastore.parent if self.datastore.parent
                  else self.datastore)
        logging.info("Reading contexts")
        cmakers = read_cmakers(dstore)
        src_mutex_by_grp = read_src_mutex(dstore)
        ctx_by_grp = read_ctx_by_grp(dstore)
        totctxs = sum(len(ctx) for ctx in ctx_by_grp.values())
        logging.info('Read {:_d} contexts'.format(totctxs))
        self.datastore.swmr_on()
        smap = parallel.Starmap(compute_disagg, h5=self.datastore.hdf5)
        # IMPORTANT!! we rely on the fact that the classical part
        # of the calculation stores the ruptures in chunks of constant
        # grp_id, therefore it is possible to build (start, stop) slices;
        # we are NOT grouping by operator.itemgetter('grp_id', 'magi'):
        # that would break the ordering of the indices causing an incredibly
        # worse performance, but visible only in extra-large calculations!

        # compute the total weight of the contexts and the maxsize
        totweight = sum(cmakers[grp_id].Z * len(ctx)
                        for grp_id, ctx in ctx_by_grp.items())
        maxsize = int(numpy.ceil(totweight / (oq.concurrent_tasks or 1)))
        logging.debug(f'{totweight=}, {maxsize=}')

        s = self.shapedic
        if self.Z == 1 or oq.individual_rlzs:
            weights = None
        else:
            weights = self.datastore['weights'][:]
        mutex_by_grp = self.datastore['mutex_by_grp'][:]
        for grp_id, ctxt in ctx_by_grp.items():
            cmaker = cmakers[grp_id]
            src_mutex, rup_mutex = mutex_by_grp[grp_id]
            src_mutex = src_mutex_by_grp.get(grp_id, {})
            if rup_mutex:
                raise NotImplementedError(
                    'Disaggregation with mutex ruptures')

            # build rlz weight dictionary
            if weights is None:
                wdic = {}  # don't compute means
            else:
                wdic = {}
                for rlzs in cmaker.gsims.values():
                    for rlz in rlzs:
                        wdic[rlz] = weights[rlz]

            # submit tasks
            ntasks = len(ctxt) * cmaker.Z / maxsize
            if ntasks < 2 or src_mutex or rup_mutex:
                # do not split (see case_11)
                smap.submit((self.datastore, ctxt, self.sitecol, cmaker,
                             self.bin_edges, src_mutex, wdic))
            elif self.N > ntasks:
                # split context by tiles (see test_disagg_case_multi)
                for tile in self.sitecol.split(ntasks):
                    smap.submit((self.datastore, ctxt, tile, cmaker,
                                 self.bin_edges, src_mutex, wdic))
            else:
                # split by magnitude (see case_1)
                for ctx in disagg.split_by_magbin(
                        ctxt, self.bin_edges[0]).values():
                    smap.submit((self.datastore, ctx, self.sitecol, cmaker,
                                 self.bin_edges, src_mutex, wdic))

        shape8D = (s['trt'], s['mag'], s['dist'], s['lon'], s['lat'], s['eps'],
                   s['M'], s['P'])
        acc = AccumDict(accum=numpy.zeros(shape8D))
        results = smap.reduce(self.agg_result, acc)
        return results  # s, r -> array 8D

    def agg_result(self, acc, results):
        """
        Collect the results coming from compute_disagg into self.results.

        :param acc: dictionary s, r -> array8D
        :param result: dictionary with the result coming from a task
        """
        with self.monitor('aggregating disagg matrices'):
            for res in results:
                trti = res.pop('trti')
                magi = res.pop('magi')
                sid = res.pop('sid')
                for rlz, arr in res.items():
                    acc[sid, rlz][trti, magi] += arr
        return acc

    def post_execute(self, results):
        """
        Save all the results of the disaggregation. NB: the number of results
        to save is #sites * #rlzs * #disagg_poes * #IMTs.

        :param results:
            a dictionary sid, rlz -> 8D disagg matrix
        """
        # the DEBUG dictionary is populated only for OQ_DISTRIBUTE=no
        for sid, pnes in disagg.DEBUG.items():
            print('site %d, mean pnes=%s' % (sid, pnes))
        with self.monitor('saving disagg results'):
            logging.info('Extracting and saving the PMFs')
            if self.Z == 1 or self.oqparam.individual_rlzs:
                res = {(s, self.sr2z[s, r]): results[s, r] for s, r in results}
                self.save_disagg_results(res, 'disagg-rlzs')
            else:  # save mean PMFs
                self.save_disagg_results(results, 'disagg-stats')

    def save_bin_edges(self, all_edges):
        """
        Save disagg-bins
        """
        *self.bin_edges, self.trts = all_edges
        b = self.bin_edges

        def ll_edges(bin_no):
            # lon/lat edges for the sites, bin_no can be 2 or 3
            num_edges = len(b[bin_no][0])
            arr = numpy.zeros((self.N, num_edges))
            for sid, edges in b[bin_no].items():
                arr[sid] = edges
            return arr
        self.datastore['disagg-bins/Mag'] = b[0]
        self.datastore['disagg-bins/Dist'] = b[1]
        self.datastore['disagg-bins/Lon'] = ll_edges(2)
        self.datastore['disagg-bins/Lat'] = ll_edges(3)
        self.datastore['disagg-bins/Eps'] = b[4]
        self.datastore['disagg-bins/TRT'] = encode(self.trts)

    def save_disagg_results(self, results, name):
        """
        Save the computed PMFs in the datastore.

        :param results:
            a dict s, z -> 8D-matrix of shape (T, Ma, D, E, Lo, La, M, P)
        :param name:
            the string "disagg-rlzs" or "disagg-stats"
        """
        oq = self.oqparam
        if name.endswith('rlzs'):
            Z = self.shapedic['Z']
        else:
            Z = len(oq.hazard_stats())
        out = output_dict(self.shapedic, oq.disagg_outputs, Z)
        count = numpy.zeros(len(self.sitecol), U16)
        _disagg_trt = numpy.zeros(self.N, [(trt, float) for trt in self.trts])
        vcurves = []  # hazard curves with a vertical section for large poes
        best_rlzs = self.datastore['best_rlzs'][:]  # (shape N, Z)
        for (s, z), mat8 in sorted(results.items()):
            mat8 = disagg.to_probs(mat8)
            mat7 = agg_probs(*mat8)  # shape (Ma, D, E, Lo, La, M, P)
            for key in oq.disagg_outputs:
                if key == 'TRT':
                    out[key][s, ..., z] = disagg.pmf_map[key](mat8)  # (T,M,P)
                elif key.startswith('TRT_'):
                    proj = disagg.pmf_map[key[4:]]
                    out[key][s, ..., z] = [proj(m7) for m7 in mat8]
                else:
                    out[key][s, ..., z] = disagg.pmf_map[key](mat7)

            # display some warnings if needed
            for m, imt in enumerate(self.imts):
                for p, poe in enumerate(self.poes_disagg):
                    mat6 = mat8[..., m, p]  # shape (T, Ma, D, Lo, La, E)
                    if m == 0 and poe == self.poes_disagg[-1]:
                        _disagg_trt[s] = tuple(
                            pprod(mat8[..., 0, 0], axis=(1, 2, 3, 4, 5)))
                    if name.endswith('-rlzs'):
                        poe_agg = pprod(mat6, axis=(0, 1, 2, 3, 4, 5))
                        self.datastore['poe4'][s, m, p, z] = poe_agg
                        if poe and abs(1 - poe_agg/poe) > .1 and not count[s]:
                            # warn only once per site
                            msg = ('Site #%d, IMT=%s, rlz=#%d: poe_agg=%s is '
                                   'quite different from the expected poe=%s,'
                                   ' perhaps not enough levels')
                            logging.warning(msg,  s, imt, best_rlzs[s, z],
                                            poe_agg, poe)
                            vcurves.append(self.curves[s])
                            count[s] += 1
        self.datastore[name] = out
        # below a dataset useful for debugging, at minimum IMT and maximum RP
        self.datastore['_disagg_trt'] = _disagg_trt
        if len(vcurves):
            NML1 = len(vcurves), self.M, oq.imtls.size // self.M
            self.datastore['_vcurves'] = numpy.array(vcurves).reshape(NML1)
            self.datastore['_vcurves'].attrs['sids'] = numpy.where(count)[0]

        # check null realizations in the single site case, see disagg/case_2
        if name.endswith('-rlzs'):
            for (s, z), r in numpy.ndenumerate(best_rlzs):
                lst = []
                for key in out:
                    if out[key][s, ..., z].sum() == 0:
                        lst.append(key)
                if lst:
                    logging.warning('No %s contributions for site=%d, rlz=%d',
                                    lst, s, r)
