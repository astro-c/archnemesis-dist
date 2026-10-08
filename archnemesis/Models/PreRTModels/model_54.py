

from typing import TYPE_CHECKING, Self, IO

import numpy as np
import matplotlib.pyplot as plt


from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

import archnemesis.Data.constants as const
from archnemesis.enum import AtmosphericProfileTypeEnum

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


class Model54(PreRTModelBase):
    """
    New cloud profile based on Galileo probe nephelometer main cloud.
    Features a cloud centred at a specified pressure, variable FWHM
    below the specified pressure, defined total opacity, while density
    falls off as an exponential above the base pressure.
    """

    id: int = 54

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
            ModelParameter("opacity", slice(0, 1), "Total opacity", ""),
            ModelParameter("peak_pressure", slice(1, 2), "Pressure where the distribution peaks", "atm"),
            ModelParameter("above_width", slice(2, 3), "Width above the cloud peak", "ln(atm)"),
            ModelParameter("below_width", slice(3, 4), "Width below the cloud peak", "ln(atm)"),
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

            opacity: float,
            #   Total opacity

            peak_pressure: float,
            #   Pressure where the distribution peaks (atm)
            
            above_width: float,
            #   Width above the cloud peak (ln(atm))
            
            below_width: float,
            #   Width below the cloud peak (ln(atm))
            
            phaze: float,
            #   Haze pressure level (atm)

            whaze: float,
            #   Haze width (atm)

            MakePlot: bool = False,
        ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            FUNCTION NAME : model54()

            DESCRIPTION :
                Function defining the model parameterisation 54 in NEMESIS.
                New cloud profile based on Galileo probe nephelometer main cloud.

            INPUTS :

                atm :: Python class defining the atmosphere

                atm_profile_type :: AtmosphericProfileTypeEnum
                    ENUM of atmospheric profile type we are altering.

                atm_profile_idx :: int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                opacity :: float
                    Opacity

                peak_pressure :: float
                    Pressure where the distribution peaks (atm)

                above_width :: float
                    Width above the cloud peak (ln(atm))

                below_width :: float
                    Width below the cloud peak (ln(atm))

                phaze :: float
                    Haze pressure level (atm)

                whaze :: float
                    Haze width (atm)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmosphere class
                xmap(4,npro) :: Matrix of relating funtional derivatives to
                                elements in state vector

            MODIFICATION HISTORY : Michelle Colantoni (29/05/2026)
        """

        # Renaming variables to match FORTRAN code
        xdeep = opacity
        pknee = peak_pressure
        xwid = above_width
        xwid1 = below_width

        # Initialising arrays
        p_atm = np.array(atm.P) / 101325.0  # Convert from Pa to atm
        T = np.array(atm.T)
        xmap = np.zeros((4, atm.NP))
        xmolwt = np.array(atm.MOLWT) * 1e3  # Convert from kg/mol to g/mol
        q = np.zeros(atm.NP)
        nd = np.zeros(atm.NP)
        od = np.zeros(atm.NP)
        xfac = np.zeros(atm.NP)

        # Calculate atmospheric properties
        R = const.R
        scale = R * atm.T / (atm.MOLWT * atm.GRAV) * 1e-3  # Convert to km

        y0 = -np.log(pknee)
        yhaze = np.log(phaze)

        k=-1
        khaze=-1

        for j in range(atm.NP):
            if p_atm[j] >= pknee and p_atm[j+1] < pknee:
                k=j 
            if p_atm[j] >= phaze and p_atm[j+1] < phaze:
                khaze=j

        if k < 0:
            _msg = f"Model {cls.id} error: Cannot find KNEE."
            _lgr.error(_msg)
            raise ValueError(_msg)

        xod=0.

        for j in range(atm.NP):
            y=-np.log(p_atm[j])          

            if j <= k:
                q[j] = np.exp(-((y-y0)/xwid1)**2)	
            else:
                q[j] = np.exp(-(y-y0)/xwid)

            xfac = np.exp(-((y-yhaze)/whaze)**2)
            xfac = 1.0 - xfac

            if j > khaze:
                xfac=0.0

            q[j] = q[j]*xfac
            xmolwt = atm.MOLWT[j] * 1e3 
            rho = (0.1013*xmolwt/R)*(p_atm[j]/T[j])
            nd[j] = q[j]*rho 
            od[j] = nd[j]*scale[j]*1e5
            xod=xod+od[j]

        x1=np.float32(q)

        xod = xod*0.25

        for j in range(atm.NP):
            x1[j]=np.float32(q[j]*xdeep/xod)
            y=np.log(p_atm[j])          
            x1[j] = max(x1[j], 1e-36)

            # VARIDENT(IVAR,1).EQ.0 in Fortran
            if atm_profile_type == AtmosphericProfileTypeEnum.TEMPERATURE:
                xmap[0,j]=x1[j]/xdeep
            else:
                xmap[0,j]=x1[j]

        if atm_profile_type == AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            # MC_NOTE : NEED TO CONVERT UNITS?????
            atm.DUST[:, atm_profile_idx] = x1
            # MC_NOTE : CHECK IF CORRECT TO UPDATE THE DUST PROFILE IN-PLACE
        else:
            _msg = f"Model id={cls.id} is only defined for aerosol density profiles."
            _lgr.error(_msg)
            raise ValueError(_msg)
        
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
        # *** model 54 *******
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")  # Use "!" as comment character in *.apr files
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        pknee = tmp[0]
        eknee = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        xwid = tmp[0]
        ewid = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        xwid1 = tmp[0]
        ewid1 = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        phaze = tmp[0]
        whaze = tmp[1]

        varparam[0] = phaze
        varparam[1] = whaze

        # opacity
        if xdeep > 0.0:
            x0[ix] = np.log(xdeep)
            lx[ix] = 1
        else:
            raise ValueError(f"Model54 xdeep must be > 0 for log retrieval, got {xdeep}")

        err = edeep / xdeep
        sx[ix, ix] = err**2.0

        ix += 1

        # pressure at distribution peak
        if pknee > 0.0:
            x0[ix] = np.log(pknee)
            lx[ix] = 1
        else:
            raise ValueError(f"Model54 pknee must be > 0 for log retrieval, got {pknee}")

        err = eknee / pknee
        sx[ix, ix] = err**2.0

        ix += 1

        # width above cloud peak
        if xwid > 0.0:
            x0[ix] = np.log(xwid)
            lx[ix] = 1
        else:
            raise ValueError(f"Model54 xwid must be > 0 for log retrieval, got {xwid}")

        err = ewid / xwid
        sx[ix, ix] = err**2.0

        ix += 1

        # width below cloud peak
        if xwid1 > 0.0:
            x0[ix] = np.log(xwid1)
            lx[ix] = 1
        else:
            raise ValueError(f"Model54 xwid1 must be > 0 for log retrieval, got {xwid1}")

        err = ewid1 / xwid1
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
        # ******** profile defined by opacity and peak pressure ********
        # ******** with above and below peak width ********
        if varident[2] != cls.id:
            raise ValueError("error in Model54.from_bookmark() :: wrong model id")

        ix = ix + 5

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, model_classification[1])

        return cls(ix_0, ix - ix_0, model_classification[1])

    def calculate_from_subprofretg(
            self,
            forward_model: "ForwardModel_0",
            ix: int,
            ipar: int,
            ivar: int,
            xmap: np.ndarray,
        ) -> None:
        # Model 54. Density falls off as an exponential above the base pressure
        # ***************************************************************

        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)

        xdeep, pknee, xwid, xwid1 = self.get_parameter_values_from_state_vector(
            forward_model.Variables.XN, forward_model.Variables.LX
        )

        phaze = forward_model.Variables.VARPARAM[ivar, 0]
        whaze = forward_model.Variables.VARPARAM[ivar, 1]

        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            xdeep,
            pknee,
            xwid,
            xwid1,
            phaze,
            whaze,
            MakePlot=False,
        )

        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0 : forward_model.AtmosphereX.NP] = xmap1

        return