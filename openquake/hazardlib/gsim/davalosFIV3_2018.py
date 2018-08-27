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
from openquake.hazardlib.imt import FIV3
from openquake.hazardlib import imt as imt_module

class DavalosMiranda2018FIV3(GMPE):
    """
    Implements GMPE developed by Hector Davalos and Eduardo Miranda
    Coded by Pablo Heresi
    """
    #: Supported tectonic region type is active shallow crust
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure type is spectral inelastic displacement
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        FIV3
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
        imt_aux = imt_module.SA(damping=5, period=imt.period)
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
        if mag < C['Mh']:
            mag_term = C['e2'] * dmag + C['e3'] * dmag ** 2
        else:
            mag_term = C['e4'] * dmag
        return C['e1'] + mag_term

    def _get_path_scaling(self, C, dists, mag):
        """
        Returns the path scaling term
        """
        R = np.sqrt((dists.rjb ** 2.0) + (C['h'] ** 2.0))
        scaling = (C['c1'] + C['c2'] * mag) * \
            np.log(R / self.CONSTS['Rref'])
        return scaling

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

    CONSTS = {"Rref": 1.0}

    Coeffs = CoeffsTable(sa_damping=5, table='''
imt e1  e2  e3  e4  Mh  c1  c2  h   phi   tau
0.1 4.72582 1.03189 -0.09857 -0.06401 6.3 -1.56891 0.11974 3.97851 0.40936 0.34436
0.2 5.29304 1.10644 -0.05793 -0.06563 6.3 -1.5618 0.11975 3.82241 0.41238 0.34294
0.3 5.6266 1.19485 -0.02242 -0.05528 6.3 -1.53627 0.1168 3.70312 0.41815 0.34774
0.4 5.79263 1.23938 -0.02827 -0.03378 6.3 -1.50031 0.11206 3.60844 0.4237 0.34965
0.5 5.93035 1.36771 0.03192 -0.00999 6.3 -1.46082 0.10666 3.52616 0.43005 0.35345
0.6 5.996 1.47055 0.08295 0.03027 6.3 -1.39447 0.09753 3.43339 0.43654 0.35855
0.7 6.03333 1.57767 0.14604 0.07794 6.3 -1.32281 0.08732 3.40135 0.44224 0.36633
0.8 6.05279 1.70062 0.2382 0.11974 6.3 -1.26907 0.07952 3.3985 0.44702 0.37029
0.9 6.04351 1.83569 0.30062 0.18663 6.3 -1.18507 0.06758 3.34919 0.45173 0.37419
1 6.04607 1.91478 0.33218 0.23173 6.25 -1.14208 0.06114 3.36557 0.45607 0.37779
1.1 6.04638 2.00532 0.38141 0.28042 6.25 -1.09715 0.05407 3.38165 0.45961 0.37923
1.2 6.04878 2.01646 0.35532 0.33197 6.25 -1.06070624 0.04805 3.45098 0.46244 0.3766
1.3 6.04247 2.01082 0.34256 0.37312 6.25 -1.02997 0.04304 3.46387 0.46422 0.38605
1.4 6.02077 1.99822 0.30428 0.42084 6.25 -0.98684 0.03656 3.50879 0.46651 0.38643
1.5 5.99127 2.04396 0.32327 0.46923 6.2 -0.94789 0.03059 3.54682 0.46788 0.38564
1.6 5.97201 2.02518 0.28902 0.50172 6.2 -0.93546 0.0285 3.62664 0.46928 0.38365
1.7 5.94443 2.02284 0.261 0.55236 6.2 -0.88461 0.02069 3.7155 0.46978 0.38329
1.8 5.93507 2.00359 0.24658 0.57091 6.2 -0.87424 0.01907 3.72828 0.47066 0.38521
1.9 5.94288 1.98512 0.20707 0.58878 6.2 -0.866327002 0.0175 3.78186 0.47183 0.3855
2 5.95519 2.04387 0.25549 0.58437 6.2 -0.88869 0.020358812 3.84159 0.47183 0.38453
2.1 5.97738 2.10643 0.31478 0.56644 6.2 -0.9305 0.02621 3.88533 0.47165 0.38122
2.2 5.98856 2.15499 0.35297 0.56251 6.2 -0.93233581 0.02631 3.91098 0.47104 0.37807
2.3 5.99238 2.15922 0.36045 0.56167 6.2 -0.94458 0.028019804 3.89806 0.47036 0.37373
2.4 5.99329 2.18873 0.38107 0.56054 6.2 -0.93036 0.02621 3.84778 0.47082 0.37023
2.5 5.9783 2.13982 0.33689 0.57346 6.2 -0.932720447 0.02681 3.84639 0.47066 0.36833
2.6 5.98797 2.16061 0.34475 0.56864 6.2 -0.92767 0.02613 3.93301 0.47085 0.36712
2.7 5.9678 2.14717 0.33381 0.58829 6.2 -0.90425 0.02304 3.92928 0.47068 0.36687
2.8 5.94425 2.15631 0.34162 0.61709 6.2 -0.87551 0.018994212 3.92539 0.46987 0.36584
2.9 5.95032 2.25409 0.42452 0.61119 6.2 -0.86323 0.01722 3.96451 0.46941 0.36693
3 5.92577 2.22839 0.42062 0.62478 6.2 -0.86435 0.01761 3.97649 0.47049 0.36914
        ''')


    


