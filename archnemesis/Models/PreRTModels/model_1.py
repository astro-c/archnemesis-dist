

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




class Model1(PreRTModelBase):
    """
    Condensing gas, but no associated cloud. Model requires
    the deep gas abundance and the desired relative humidity above the
    condensation level only or at all levels.
    """

    id: int = 1

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
            ModelParameter("frac_scale_height", slice(1, 2), "Fractional scale height above 1st cloud", "km"),
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
            
            fsh: float,
            #   Fractional scale height (km)
            
            pknee: float,
            #   Knee pressure (atm)
            
            MakePlot: bool = False,
        ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            FUNCTION NAME : model1()

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

                fsh :: float
                    Fractional scale height (km)

                pknee :: float
                    Knee pressure (atm)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmosphere class
                xmap(2,npro) :: Matrix of relating funtional derivatives to
                                elements in state vector

            MODIFICATION HISTORY : Michelle Colantoni (29/05/2026)
        """

        # Renaming variables to match FORTRAN code
        xdeep = deep_vmr
        xfsh = fsh

        # Initialising arrays
        p_atm = np.array(atm.P) / 101325.0  # Convert from Pa to atm
        h = np.array(atm.H) * 1e-3  # Convert from m to km
        x1 = np.zeros(atm.NP)
        xmap = np.zeros((2, atm.NP))

        # Calculate atmospheric properties
        R = const.R
        scale = R * atm.T / (atm.MOLWT * atm.GRAV) * 1e-3  # Convert to km

        # To calculate the crossover point
        def verint(x, y, xin):
            n = len(x)

            if x[0] < x[-1]:
                for i in range(n):
                    if x[i] > xin:
                        break
            else:
                for i in range(n):
                    if x[i] < xin:
                        break

            if i == 0:
                i = 1
            if x[i - 1] == x[i]:
                yout = y[i]
            else:
                yout = y[i - 1] + (y[i] - y[i - 1]) * (xin - x[i - 1]) / (x[i] - x[i - 1])

            if np.isnan(yout):
                yout = y[i]

            return yout

        xfac = (1.0 - xfsh) / xfsh
        dxfac = -1.0 / xfsh

        hknee = verint(p_atm, h, pknee)
        jfsh = 0

        for j in range(atm.NP):
            x1[j] = xdeep
            # VARIDENT(IVAR,1).EQ.0 in Fortran
            if atm_profile_type == AtmosphericProfileTypeEnum.TEMPERATURE:
                xmap[0, j] = 1.0
            else:
                xmap[0, j] = x1[j]

            if p_atm[j] < pknee:
                if jfsh == 0:
                    delh = h[j] - hknee
                else:
                    delh = h[j] - h[j - 1]

                x1[j] = x1[j - 1] * np.exp(-delh * xfac / scale[j])
                xmap[0, j] = xmap[0, j - 1] * np.exp(-delh * xfac / scale[j])
                xmap[1, j] = (-delh / scale[j]) * dxfac * x1[j - 1] * np.exp(
                    -delh * xfac / scale[j]
                ) + xmap[1, j - 1] * np.exp(-delh * xfac / scale[j])

                jfsh = 1

                x1[j] = max(x1[j], 1e-36)

        if atm_profile_type == AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            # Update atmosphere VMR -- no conversion necessary as unitless
            atm.VMR[:, atm_profile_idx] = x1[:]
        else:
            _msg = f"Model id={cls.id} is only defined for gas VMR profiles."
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
        # *** model 1 - Variable profile *******
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=1, dtype="float")  # Use "!" as comment character in *.apr files
        pknee = tmp[0]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        xfsh = tmp[0]
        efsh = tmp[1]

        varparam[0] = pknee

        # deep abundance / temperature(?)
        if varident[0] == 0:
            x0[ix] = xdeep
            lx[ix] = 0
        else:
            if xdeep < 0.0:
                raise ValueError(f"Model1 xdeep must be > 0 for log retrieval, got {xdeep}")
            x0[ix] = np.log(xdeep)
            lx[ix] = 1

        err = edeep / xdeep
        sx[ix, ix] = err**2.0

        ix += 1
        # fractional scale height
        if xfsh > 0.0:
            x0[ix] = np.log(xfsh)
            lx[ix] = 1
        else:
            raise ValueError(f"Model1 xfsh must be > 0 for log retrieval, got {xfsh}")

        err = efsh / xfsh
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
        # ******** profile defined by deep abundance and knee pressure ********
        # ******** and the fractional scale height ********
        if varident[2] != cls.id:
            raise ValueError("error in Model1.from_bookmark() :: wrong model id")

        ix = ix + 3

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
        # Model 1. Variable profile defined by deep abundance and knee pressure and the fractional scale height
        # ***************************************************************

        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)

        xdeep, xfsh = self.get_parameter_values_from_state_vector(
            forward_model.Variables.XN, forward_model.Variables.LX
        )

        pknee = forward_model.Variables.VARPARAM[ivar, 0]

        atm, xmap1 = self.calculate(
            atm, 
            atm_profile_type, 
            atm_profile_idx, 
            xdeep, 
            xfsh, 
            pknee, 
            MakePlot=False,
        )

        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0 : forward_model.AtmosphereX.NP] = xmap1

        return
    