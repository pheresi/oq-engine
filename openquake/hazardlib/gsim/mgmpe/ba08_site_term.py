# The Hazard Library
# Copyright (C) 2012-2025 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Module :mod:`openquake.hazardlib.mgmpe.ba08_site_term` implements
:class:`~openquake.hazardlib.mgmpe.ba08_site_term.BA08SiteTerm`
"""

import copy
import numpy as np
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA
from openquake.hazardlib.gsim.base import GMPE, registry
from openquake.hazardlib.gsim.boore_atkinson_2008 import BooreAtkinson2008
from openquake.hazardlib.gsim.atkinson_boore_2006 import (
    _get_site_amplification_linear,
    _get_site_amplification_non_linear,
    _get_pga_on_rock) 


def _get_ba08_site_term(imt, ctx):
    """
    Get the site amplification term as applied within the
    Boore and Atkinson 2008 GMM.
    """
    # Get vs30 and some coeffs
    vs30 = ctx.vs30
    C_PGA = BooreAtkinson2008.COEFFS[PGA()]
    C_SR = BooreAtkinson2008.COEFFS_SOIL_RESPONSE[imt]
    
    # Compute PGA on rock
    ctx.rake = np.full_like(ctx.vs30, 60)
    pga4nl = _get_pga_on_rock(C_PGA, ctx)

    # Get linear
    linear = _get_site_amplification_linear(vs30, C_SR)
    
    # Get non-linear
    non_linear = _get_site_amplification_non_linear(vs30, pga4nl, C_SR)

    return linear + non_linear


class BA08SiteTerm(GMPE):
    """
    Implements a modified GMPE class that can be used to account for local
    soil conditions in the estimation of ground motion using the site term
    from :class:`openquake.hazardlib.gsim.boore_atkinson_2008.BooreAtkinson2008`.
    
    The BA08SiteTerm can be applied to any GMPE that natively uses the vs30
    parameter or, if vs30 is not used, the GMPE must specify a reference
    velocity (i.e. DEFINED_FOR_REFERENCE_VELOCITY) between 730 and 790 m/s
    (applying +/- 30 m/s bounds to the reference velocity of 760 m/s defined
    in BA08).

    :param gmpe_name:
        The name of a GMPE class
    """
    # Parameters
    REQUIRES_SITES_PARAMETERS = {'vs30'}
    REQUIRES_DISTANCES = set()
    REQUIRES_RUPTURE_PARAMETERS = set()
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = ''
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set()
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {const.StdDev.TOTAL}
    DEFINED_FOR_TECTONIC_REGION_TYPE = ''
    DEFINED_FOR_REFERENCE_VELOCITY = None

    def __init__(self, gmpe_name, **kwargs):
        self.gmpe = registry[gmpe_name]()
        self.set_parameters()
        
        # Check if this GMPE has the necessary requirements
        req = 'DEFINED_FOR_REFERENCE_VELOCITY'
        if not (hasattr(self.gmpe, req)
                or
                'vs30' in self.gmpe.REQUIRES_SITES_PARAMETERS):
            msg = f'{self.gmpe} does not use vs30 or lacks a defined reference velocity'
            raise AttributeError(msg)
        if 'vs30' not in self.gmpe.REQUIRES_SITES_PARAMETERS:
            self.REQUIRES_SITES_PARAMETERS |= {'vs30'}
        
        # Check compatibility of reference velocity
        if not hasattr(self.gmpe, req):
            msg = f'The original GMPE must have the {req} parameter'
            raise ValueError(msg)
        if not (self.gmpe.DEFINED_FOR_REFERENCE_VELOCITY >= 730
                and
                self.gmpe.DEFINED_FOR_REFERENCE_VELOCITY <= 790):
            msg = 'DEFINED_FOR_REFERENCE_VELOCITY outside of range'
            raise ValueError(msg)

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        # Make sites with ref bedrock vs30
        rup_rock = copy.copy(ctx)
        rup_rock.vs30 = np.full_like(ctx.vs30, 760.)

        # Compute mean on bedrock
        self.gmpe.compute(rup_rock, imts, mean, sig, tau, phi)
        
        # Compute and apply the site term for each IMT
        for m, imt in enumerate(imts):
            mean[m] += _get_ba08_site_term(imt, ctx)