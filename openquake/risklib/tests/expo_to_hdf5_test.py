# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# 
# Copyright (C) 2024, GEM Foundation
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
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.
import os
from openquake.commonlib.datastore import read
from openquake.risklib.expo_to_hdf5 import main


def test_main():
    # only tests that it runs
    expo_xml = os.path.join(os.path.dirname(__file__), 'data', 'grm_exposure.xml')
    out = main([expo_xml])
    with read(out) as dstore:
        assets = list(dstore['assets/ASSET_ID'])
        assert assets == [b'TWNRes_0', b'TWNRes_1', b'TWNRes_2', b'TWNRes_3',
                          b'TWNRes_4', b'TWNRes_5', b'TWNRes_6', b'TWNRes_7',
                          b'TWNRes_8', b'TWNRes_9']

