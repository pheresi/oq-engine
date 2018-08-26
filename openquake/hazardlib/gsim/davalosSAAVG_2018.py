# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2017 GEM Foundation
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

from __future__ import division

import numpy as np

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import SA, SAAVG

class DavalosMiranda2018SAAVG(GMPE):
    """
    Implements GMPE developed by Hector Davalos and Eduardo Miranda
    Coded by Pablo Heresi
    """
    #: Supported tectonic region type is active shallow crust
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure type is spectral inelastic displacement
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        SAAVG
    ])

    #: Supported intensity measure component is random horizontal
    #: :attr:`~openquake.hazardlib.const.IMC.RANDOM_HORIZONTAL`,
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RANDOM_HORIZONTAL

    #: Supported standard deviation types are inter-event, intra-event
    #: and total
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL,
        const.StdDev.INTER_EVENT,
        const.StdDev.INTRA_EVENT
    ])

    #: No sites parameters are required
    REQUIRES_SITES_PARAMETERS = set()

    #: Required rupture parameters are magnitude
    REQUIRES_RUPTURE_PARAMETERS = set(('mag',))

    #: Required distance measure is Rjb
    REQUIRES_DISTANCES = set(('rjb',))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        
        #: imt_aux is for searching inside the coefficient table
        imt_aux = SA(damping=5, period=imt.period)
        CoeffsTable = self.Coeffs
        C = CoeffsTable[imt_aux]

        mean = (self._get_magnitude_scaling_term(C, rup.mag) +
                self._get_path_scaling(C, dists, rup.mag))
        stddevs = self._get_stddevs(C, stddev_types, dists)
        return mean, stddevs

    def _get_magnitude_scaling_term(self, C, mag):
        """
        Returns the magnitude scaling term
        """
        dmag = mag - C['Mh']
        if mag <= C['Mh']:
            mag_term = C['b2'] * dmag + C['b3'] * dmag ** 2
        else:
            mag_term = C['b4'] * dmag
        return C['b1'] + mag_term

    def _get_path_scaling(self, C, dists, mag):
        """
        Returns the path scaling term
        """
        R = np.sqrt((dists.rjb ** 2.0) + (C['h'] ** 2.0))
        return (C['a1'] + C['a2'] * (mag - self.CONSTS['Mref'])) * \
            np.log(R / self.CONSTS['Rref'])

    def _get_stddevs(self, C, stddev_types, dists):
        """
        Returns the aleatory uncertainty terms
        """
        stddevs = []
        tau = C['tau']
        phi = C['phi']
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                stddevs.append(np.sqrt((tau ** 2.0)+(phi ** 2.0)) + 
                                                             dists.rjb * 0)
            elif stddev_type == const.StdDev.INTRA_EVENT:
                stddevs.append(phi + dists.rjb * 0)
            elif stddev_type == const.StdDev.INTER_EVENT:
                stddevs.append(tau + dists.rjb * 0)
        return stddevs

    CONSTS = {'Rref': 1.0, 'Mref': 6.9}

    Coeffs = CoeffsTable(sa_damping=5, table='''
imt 	b1 	b2 	b3 	b4 	Mh 	a1 	a2 	h 	phi  tau
0.1 3.30940 -0.87510 -1.86940 -0.56300 5.80000 -1.16840000 0.24909100 14.96520122 0.440 0.380
0.2 2.50380 -0.47340 -1.62680 -0.40360 5.80000 -1.00390000 0.23830000 10.69220000 0.430 0.380
0.3 1.80370 -0.34710 -1.13560 -0.32100 6.00000 -0.88790000 0.22635900 7.70891171 0.420 0.380
0.4 1.34800 -0.06250 -1.00300 -0.28240 6.00000 -0.80790000 0.21553600 6.18182172 0.412 0.380
0.5 1.04960 0.12920 -0.92520 -0.25350 6.00000 -0.76550000 0.20507500 5.36918125 0.410 0.380
0.6 0.81460 0.26450 -0.87360 -0.20550 6.00000 -0.74300000 0.19497600 4.94743126 0.400 0.390
0.7 0.60260 0.39260 -0.83670 -0.15820 6.00000 -0.72570000 0.18523900 4.70875986 0.410 0.400
0.8 0.43090 0.50250 -0.82470 -0.10800 6.00000 -0.71900000 0.17586400 4.53382474 0.410 0.400
0.9 0.24990 0.59000 -0.76610 -0.05270 6.00000 -0.70940000 0.16685100 4.36763608 0.420 0.420
1.0 0.07990 0.64530 -0.73010 0.00480 6.00000 -0.70330000 0.15820000 4.19860000 0.410 0.430
1.5 -0.35400 0.82910 -0.54660 0.07150 6.20000 -0.68960000 0.12040000 3.96580000 0.430 0.410
2.0 -0.68500 0.79200 -0.31420 0.20727 6.40000 -0.69050000 0.09160000 3.94000000 0.460 0.440
2.5 -0.87650 0.88390 -0.31160 0.30400 6.40000 -0.70840000 0.07190000 3.92010000 0.470 0.450
3.0 -0.94450 0.91230 -0.30620 0.30600 6.40000 -0.72320000 0.06390000 3.91020000 0.480 0.390


        ''')


    


