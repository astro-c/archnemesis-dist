

from typing import TYPE_CHECKING, Self, IO

import numpy as np
import matplotlib.pyplot as plt


from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

import archnemesis.Data.constants as const
from archnemesis.enum import AtmosphericProfileTypeEnum
from archnemesis.Data.gas_data import svp_coefficients

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0
    from archnemesis.Atmosphere_0 import Atmosphere_0

    nx = 'number of elements in state vector'
    m = 'an undetermined number, but probably less than "nx"'
    mx = 'synonym for nx'
    mparam = 'the number of parameters a model has'
    nparam = 'the number of parameters a model has'
    NCONV = 'number of spectral bins'
    NGEOM = 'number of geometries'
    NX = 'number of elements in state vector'
    NDEGREE = 'number of degrees in a polynomial'
    NWINDOWS = 'number of spectral windows'




class Model66(PreRTModelBase):
    """
    Two associated cloud profiles, one at a variable PCLOUD,
    where the gas vmr first drops, and then at the condensation
    level.
    """

    id: int = 66

    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            atm_profile_type : AtmosphericProfileTypeEnum,
            #   ENUM that tells us what kind of atmospheric profile this model instance represents
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries, atm_profile_type)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model.
        # NOTE: It is best to define these in the same order and with the
        # same names as they are saved to the state vector, and use the same
        # names and ordering when they are passed to the `self.calculate(...)` 
        # class method.
        self.parameters = (
            ModelParameter("deep_vmr", slice(0, 1), "Deep gas abundance", "RATIO"),
            ModelParameter("mid_vmr", slice(1, 2), "Middle gas abundance above 1st cloud", "RATIO"),
            ModelParameter("frac_scale_height", slice(2, 3), "Fractional scale height above 1st cloud", "km"),
            ModelParameter("humidity", slice(3, 4), "Relative humidity", "RATIO"),
            ModelParameter("scale_height", slice(4, 5), "Scale height of the RH above condensation", "km"),
            ModelParameter("cloud1_opacity", slice(5, 6), "1st cloud opacity", ""),
            ModelParameter("cloud1_sh", slice(6, 7), "1st cloud scale height", "km"),
            ModelParameter("cloud1_pressure", slice(7, 8), "1st cloud base pressure", "atm"),
            ModelParameter("cloud2_opacity", slice(8, 9), "2nd cloud opacity", ""),
            ModelParameter("cloud2_sh", slice(9, 10), "2nd cloud scale height", "km"),
        )

        return

    @classmethod
    def calculate(
            cls,
            atm: "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon

            atm_profile_type: AtmosphericProfileTypeEnum,
            #   ENUM of atmospheric profile type we are altering.

            atm_profile_idx: int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

            deep_vmr: float,
            #   Deep gas abundance

            mid_vmr: float,
            #   Middle gas abundance at P < PCLOUD

            fsh: float,
            #   Fractional scale height above PCLOUD (km)
            
            rh: float,
            #   Relative humidity

            scale_height: float,
            #   Scale height of the RH above condensation (km)
            
            cloud1_tau: float,
            #   Cloud 1 opacity ()
            
            cloud1_sh: float,
            #   Cloud 1 scale height (km)
            
            cloud1_pressure: float,
            #   Cloud 1 base pressure (atm)
            
            cloud2_tau: float,
            #   Cloud 2 opacity ()
            
            cloud2_sh: float,
            #   Cloud 2 scale height (km)
            
            cloud1_id: int,
            #   Cloud ID for cloud 1

            cloud2_id: int,
            #   Cloud ID for cloud 2

            phaze1: float,
            #   Haze 1 pressure level (atm)

            whaze1: float,
            #   Haze 1 width (atm)

            phaze2: float,
            #   Haze 2 pressure level (atm)

            whaze2: float,
            #   Haze 2 width (atm)

            MakePlot: bool = False,
        ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            FUNCTION NAME : model66()

            DESCRIPTION :
                Function defining the model parameterisation 11 in NEMESIS.
                Condensing gas profile (no cloud). Profile is deep constant until
                limited by saturation vapour pressure (SVP) * RH.

            INPUTS :

                atm :: Python class defining the atmosphere

                atm_profile_type :: AtmosphericProfileTypeEnum
                    ENUM of atmospheric profile type we are altering.

                atm_profile_idx :: int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                deep_vmr :: float
                    Deep gas abundance

                mid_vmr :: float
                    Middle VMR above cloud 1

                fsh :: float
                    Fractional scale height above cloud 1 (km)

                rh :: float
                    Relative humidity

                scale_height :: float
                    Scale height of the RH above condensation (km)

                cloud1_tau :: float
                    Cloud 1 opacity ()

                cloud1_sh :: float
                    Cloud 1 scale height (km)

                cloud1_pressure :: float
                    Cloud 1 base pressure (atm)

                cloud2_tau :: float
                    Cloud 2 opacity ()

                cloud2_sh :: float
                    Cloud 2 scale height (km)

                cloud1_id :: int
                    Cloud ID for cloud 1

                cloud2_id :: int
                    Cloud ID for cloud 2

                phaze1 :: float
                    Haze 1 pressure level (atm)

                whaze1 :: float
                    Haze 1 width (atm)

                phaze2 :: float
                    Haze 2 pressure level (atm)

                whaze2 :: float
                    Haze 2 width (atm)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmosphere class
                xmap(2,npro) :: Matrix of relating funtional derivatives to
                                elements in state vector

            MODIFICATION HISTORY : Michelle Colantoni (29/05/2026)
        """

        if atm_profile_type != AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            _msg = f"Model id={cls.id} is only defined for gas VMR profiles."
            _lgr.error(_msg)
            raise ValueError(_msg)

        # Renaming variables to match FORTRAN code
        xdeep = deep_vmr
        xmid = mid_vmr
        yfsh = fsh
        xrh = rh
        xscale = scale_height
        xcdeep1 = cloud1_tau
        xwidc1 = cloud1_sh
        pcloud = cloud1_pressure
        xcdeep2 = cloud2_tau
        xwidc2 = cloud2_sh

        # Initialising arrays
        p_atm = np.array(atm.P) / 101325.0  # Convert from Pa to atm
        T = np.array(atm.T)
        h = np.array(atm.H) * 1e-3  # Convert from m to km
        x1 = np.zeros(atm.NP)
        xmap = np.zeros((10, atm.NP))
        q = np.zeros(atm.NP)
        nd = np.zeros(atm.NP)
        od = np.zeros(atm.NP)
        x2 = np.zeros(atm.NP)
        x3 = np.zeros(atm.NP)
        xnow = np.zeros(atm.NP)

        # Calculating the actual atmospheric scale height in each level
        R = const.R
        scale = R * atm.T / (atm.MOLWT * atm.GRAV) * 1e-3  # Convert to km

        yfac = (1.0 - yfsh) / yfsh

        ifla1 = None
        ifla2 = None
        khaze1 = None
        khaze2 = None

        jspec1 = abs(cloud1_id)
        jspec2 = abs(cloud2_id)
        jpar1 = atm.NVMR + 1 + jspec1
        jpar2 = atm.NVMR + 1 + jspec2
        _lgr.debug(f"Model 66, JPARs = {jpar1}, {jpar2}")

        gas_id = int(atm.ID[atm_profile_idx])

        if gas_id in svp_coefficients:
            a, b, c, d = svp_coefficients[gas_id]
        else:
            _msg = f"Error in model 66 :: no SVP coefficients for gas id={gas_id}"
            _lgr.error(_msg)
            raise ValueError(_msg)

        for i in range(atm.NP):

            if p_atm[i] > pcloud:
                xnow = xdeep
            else:
                if ifla1 is None:
                    xnow = xmid
                    ifla1 = i
                else:
                    delh = h[i] - h[i - 1]
                    xnow = x1[i - 1] * np.exp(-delh * yfac / scale[i])
            
            p1 = p_atm[i] * xnow

            ps = np.exp(a + b / T[i] + c * T[i] + d * T[i] * T[i])

            if p1 < ps and ifla2 == None:
                x1[i] = xnow
            else:
                if ifla2 == None:
                    ifla2 = i
                    hcond = h[ifla2]
                    pcond = p_atm[ifla2]

                delh = h[i] - hcond
                xrh1 = xrh * np.exp(-delh / xscale)
                x1[i] = xrh1 * ps / p_atm[i]

            if i < atm.NP - 1:
                if p_atm[i] >= phaze1 and p_atm[i + 1] < phaze1:
                    khaze1 = i

                if p_atm[i] >= phaze2 and p_atm[i + 1] < phaze2:
                    khaze2 = i

            if i > 0 and ifla2 != None and p_atm[i] < 0.3 and x1[i] > x1[i - 1]:
                    x1[i] = x1[i - 1]

        # Update atmosphere VMR
        atm.VMR[:, atm_profile_idx] = x1[:]
        # MC_NOTE : CHECK IF CORRECT TO UPDATE THE VMR PROFILE IN-PLACE

        _lgr.debug("Model 66 diagnostics")
        # _lgr.debug(f"pcloud, pcond = {pcloud}, {pcond}")
        _lgr.debug(f"Condensing clouds = {jspec1}, {jspec2}")
        _lgr.debug(f"{ifla1=},{ifla2=}")
        _lgr.debug(f"{p_atm[ifla1]=},{p_atm[ifla2]=}")
        _lgr.debug(f"{phaze1=},{khaze1=}")
        _lgr.debug(f"{phaze2=},{khaze2=}")

        if ifla2 is None:
            ifla2 = 0
            pcond = p_atm[ifla2]

        xwid1 = 0.05
        xod = 0.0
        y0 = -np.log(pcloud)
        yhaze1 = -np.log(phaze1)

        for j in range(atm.NP):
            y = -np.log(p_atm[j])

            if j <= ifla1:
                q[j] = np.exp(-((y - y0) / xwid1) ** 2)
            else:
                q[j] = np.exp(-(y - y0) / xwidc1)

            if khaze1 != None:
                xfac = np.exp(-((y - yhaze1) / whaze1) ** 2)
                xfac = 1.0 - xfac

                if j > khaze1:
                    xfac = 0.0

                q[j] = q[j] * xfac

            xmolwt = atm.MOLWT[i]

            rho = (0.1013 * xmolwt / R) * (p_atm[j] / T[j])

            nd[j] = q[j] * rho
            od[j] = nd[j] * scale[j] * 1e5

            xod = xod + od[j]

            x2[j] = np.float32(q[j])

            x2[j] = max(x2[j], 1e-36)

        xwid1 = 0.05
        xod = 0.0
        y0 = -np.log(pcond)
        yhaze2 = -np.log(phaze2)

        for j in range(atm.NP):

            y = -np.log(p_atm[j])

            if j <= ifla2:
                q[j] = np.exp(-((y - y0) / xwid1) ** 2)
            else:
                q[j] = np.exp(-(y - y0) / xwidc2)

            if khaze2 != None:
                xfac = np.exp(-((y - yhaze2) / whaze2) ** 2)
                xfac = 1.0 - xfac

                if j > khaze2:
                    xfac = 0.0

                q[j] = q[j] * xfac

            xmolwt = atm.MOLWT[i]

            rho = (0.1013 * xmolwt / R) * (p_atm[j] / T[j])

            nd[j] = q[j] * rho
            od[j] = nd[j] * scale[j] * 1e5

            xod = xod + od[j]

            x3[j] = np.float32(q[j])

            x3[j] = max(x3[j], 1e-36)

        # Update cloud aerosols
        atm.DUST[:, cloud1_id] = x2[:]
        atm.DUST[:, cloud2_id] = x3[:]
        # MC_NOTE : CHECK IF CORRECT TO UPDATE THE DUST PROFILES LIKE THIS

        if MakePlot == True:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
            ax1.semilogy(atm.VMR[:, atm_profile_idx], p_atm)
            ax1.set_ylim(p_atm.max(), p_atm.min())
            ax1.set_xlabel("VMR")
            ax1.set_ylabel("Pressure (atm)")
            ax1.grid()

            ax2.plot(atm.T, p_atm)
            ax2.set_ylim(p_atm.max(), p_atm.min())
            ax2.set_xlabel("Temperature (K)")
            ax2.set_ylabel("Pressure (atm)")
            ax2.grid()

            plt.tight_layout()
            plt.show()

        return atm, xmap

    @classmethod
    def from_apr_to_state_vector(
            cls,
            variables: "Variables_0",
            f: IO,
            varident: np.ndarray[[3], int],
            varparam: np.ndarray[["mparam"], float],
            ix: int,
            lx: np.ndarray[["mx"], int],
            x0: np.ndarray[["mx"], float],
            sx: np.ndarray[["mx", "mx"], float],
            inum: np.ndarray[["mx"], int],
            npro: int,
            ngas: int,
            ndust: int,
            nlocations: int,
            runname: str,
            sxminfac: float,
        ) -> Self:
        ix_0 = ix
        # *** model 66 - Condensing gas, two clouds *******
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")  # Use "!" as comment character in *.apr files
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        xmid = tmp[0]
        emid = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        yfsh = tmp[0]
        efsh = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        xrh = tmp[0]
        erh = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        xscale = tmp[0]
        escale = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        xcdeep1 = tmp[0]
        ecdeep1 = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        xwidc1 = tmp[0]
        ewidc1 = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        pcloud = tmp[0]
        epcloud = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        xcdeep2 = tmp[0]
        ecdeep2 = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        xwidc2 = tmp[0]
        ewidc2 = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype=int)
        jspec1 = tmp[0]
        jspec2 = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        phaze1 = tmp[0]
        whaze1 = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        phaze2 = tmp[0]
        whaze2 = tmp[1]

        varparam[0] = jspec1 - 1
        varparam[1] = jspec2 - 1
        varparam[2] = phaze1
        varparam[3] = whaze1
        varparam[4] = phaze2
        varparam[5] = whaze2

        # deep abundance
        if xdeep > 0.0:
            x0[ix] = np.log(xdeep)
            lx[ix] = 1
        else:
            raise ValueError(f"Model66 xdeep must be > 0 for log retrieval, got {xdeep}")

        err = edeep / xdeep
        sx[ix, ix] = err**2.0

        ix += 1

        # mid abundance
        if xmid > 0.0:
            x0[ix] = np.log(xmid)
            lx[ix] = 1
        else:
            raise ValueError(f"Model66 xmid must be > 0 for log retrieval, got {xmid}")

        err = emid / xmid
        sx[ix, ix] = err**2.0

        ix += 1

        # fractional scale height
        if yfsh > 0.0:
            x0[ix] = np.log(yfsh)
            lx[ix] = 1
        else:
            raise ValueError(f"Model66 yfsh must be > 0 for log retrieval, got {yfsh}")

        err = efsh / yfsh
        sx[ix, ix] = err**2.0

        ix += 1

        # relative humidity
        if xrh > 0.0:
            x0[ix] = np.log(xrh)
            lx[ix] = 1
        else:
            raise ValueError(f"Model66 xrh must be > 0 for log retrieval, got {xrh}")

        err = erh / xrh
        sx[ix, ix] = err**2.0

        ix += 1

        # scale height
        if xscale > 0.0:
            x0[ix] = np.log(xscale)
            lx[ix] = 1
        else:
            raise ValueError(f"Model66 xscale must be > 0 for log retrieval, got {xscale}")

        err = escale / xscale
        sx[ix, ix] = err**2.0

        ix += 1

        # cloud 1 opacity
        if xcdeep1 > 0.0:
            x0[ix] = np.log(xcdeep1)
            lx[ix] = 1
        else:
            raise ValueError(f"Model66 xcdeep1 must be > 0 for log retrieval, got {xcdeep1}")

        err = ecdeep1 / xcdeep1
        sx[ix, ix] = err**2.0

        ix += 1

        # cloud 1 scale height
        if xwidc1 > 0.0:
            x0[ix] = np.log(xwidc1)
            lx[ix] = 1
        else:
            raise ValueError(f"Model66 xwidc1 must be > 0 for log retrieval, got {xwidc1}")

        err = ewidc1 / xwidc1
        sx[ix, ix] = err**2.0

        ix += 1

        # cloud 1 base pressure
        if pcloud > 0.0:
            x0[ix] = np.log(pcloud)
            lx[ix] = 1
        else:
            raise ValueError(f"Model66 pcloud must be > 0 for log retrieval, got {pcloud}")

        err = epcloud / pcloud
        sx[ix, ix] = err**2.0

        ix += 1

        # cloud 2 opacity
        if xcdeep2 > 0.0:
            x0[ix] = np.log(xcdeep2)
            lx[ix] = 1
        else:
            raise ValueError(f"Model66 xcdeep2 must be > 0 for log retrieval, got {xcdeep2}")

        err = ecdeep2 / xcdeep2
        sx[ix, ix] = err**2.0

        ix += 1

        # cloud 2 scale height
        if xwidc2 > 0.0:
            x0[ix] = np.log(xwidc2)
            lx[ix] = 1
        else:
            raise ValueError(f"Model66 xwidc2 must be > 0 for log retrieval, got {xwidc2}")

        err = ewidc2 / xwidc2
        sx[ix, ix] = err**2.0

        ix += 1

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, model_classification[1])

    @classmethod
    def from_bookmark(
            cls,
            variables: "Variables_0",
            varident: np.ndarray[[3], int],
            varparam: np.ndarray[["mparam"], float],
            ix: int,
            npro: int,
            ngas: int,
            ndust: int,
            nlocations: int,
        ) -> Self:
        ix_0 = ix
        # ******** profile defined by two clouds ********
        if varident[2] != cls.id:
            raise ValueError("error in Model66.from_bookmark() :: wrong model id")

        ix = ix + 13

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, model_classification[1])

    def calculate_from_subprofretg(
            self,
            forward_model: "ForwardModel_0",
            ix: int,
            ipar: int,
            ivar: int,
            xmap: np.ndarray,
        ) -> None:
        # Model66. Two associated cloud profiles
        # ***************************************************************

        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)

        xdeep, xmid, yfsh, xrh, xscale, xcdeep1, xwidc1, pcloud, xcdeep2, xwidc2 = (
            self.get_parameter_values_from_state_vector(
                forward_model.Variables.XN, forward_model.Variables.LX
            )
        )

        jspec1 = int(forward_model.Variables.VARPARAM[ivar, 0])
        jspec2 = int(forward_model.Variables.VARPARAM[ivar, 1])
        phaze1 = forward_model.Variables.VARPARAM[ivar, 2]
        whaze1 = forward_model.Variables.VARPARAM[ivar, 3]
        phaze2 = forward_model.Variables.VARPARAM[ivar, 4]
        whaze2 = forward_model.Variables.VARPARAM[ivar, 5]

        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            xdeep,
            xmid,
            yfsh,
            xrh,
            xscale,
            xcdeep1,
            xwidc1,
            pcloud,
            xcdeep2,
            xwidc2,
            jspec1,
            jspec2,
            phaze1,
            whaze1,
            phaze2,
            whaze2,
            MakePlot=False,
        )

        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0 : forward_model.AtmosphereX.NP] = xmap1

        return