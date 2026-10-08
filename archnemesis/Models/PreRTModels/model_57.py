

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


class Model57(PreRTModelBase):
    """
    Gas profile where the abundance is constant up to a knee pressure and then falls
    linearly with respect to log pressure to a second abundance defined at a lower
    stratospheric pressure.
    """

    id: int = 57

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
            ModelParameter("strato_vmr", slice(1, 2), "Stratospheric gas abundance", "RATIO"),
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
            
            strato_vmr: float,
            #   Middle gas abundance
            
            pknee: float,
            #   Knee pressure

            pstrat: float,
            #   Stratospheric pressure

            MakePlot: bool = False,
        ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            FUNCTION NAME : model57()

            DESCRIPTION :
                Function defining the model parameterisation 57 in NEMESIS.
                Gas profile where the abundance is constant up to a knee pressure
                and then falls linearly with respect to log pressure to a second
                abundance defined at a lower stratospheric pressure.

            INPUTS :

                atm :: Python class defining the atmosphere

                atm_profile_type :: AtmosphericProfileTypeEnum
                    ENUM of atmospheric profile type we are altering.

                atm_profile_idx :: int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                knee_pressure :: float,
                    Knee pressure level (atm)

                strato_pressure :: float,
                    Stratospheric pressure level (atm)

                deep_vmr :: float
                    Deep gas abundance

                strato_vmr :: float
                    Middle gas abundance


            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmosphere class
                xmap(2,npro) :: Matrix of relating funtional derivatives to
                                elements in state vector

            MODIFICATION HISTORY : Michelle Colantoni (29/05/2026)
        """

        # MC_NOTE : CHECK IF NECESSARY
        if atm_profile_type != AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            _msg = f"Model id={cls.id} is only defined for gas VMR profiles."
            _lgr.error(_msg)
            raise ValueError(_msg)

        # Renaming variables to match FORTRAN code
        xdeep = deep_vmr
        xmid = strato_vmr

        # Initialising arrays
        p_atm = np.array(atm.P) / 101325.0  # Convert from Pa to atm
        x1 = np.zeros(atm.NP)
        xmap = np.zeros((2, atm.NP))

        x1[:] = xdeep
        xmap[0, :] = xdeep

        mask = p_atm < pknee

        if np.any(mask):
            dy = np.log(p_atm) - np.log(pknee)
            dx = np.log(pstrat) - np.log(pknee)
            f = dy / dx

            x1[mask] = np.exp((1 - f[mask]) * np.log(xdeep) + f[mask] * np.log(xmid))
            xmap[0, mask] = (1 - f[mask]) * x1[mask]
            xmap[1, mask] = f[mask] * x1[mask]

        x1[x1 < 1e-36] = 1e-36

        # Update atmosphere VMR
        atm.VMR[:, atm_profile_idx] = x1[:]
        # MC_NOTE : CHECK IF CORRECT TO UPDATE THE VMR PROFILE IN-PLACE

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
        # *** model 57 - Constant VMR up to knee, linear fall to stratospheric pressure *******
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")  # Use "!" as comment character in *.apr files
        pknee = tmp[0]
        pstrat = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromstring(f.readline().rsplit("!", 1)[0], sep=" ", count=2, dtype="float")
        xmid = tmp[0]
        emid = tmp[1]

        varparam[0] = pknee
        varparam[1] = pstrat

        # deep abundance
        if xdeep > 0.0:
            x0[ix] = np.log(xdeep)
            lx[ix] = 1
        else:
            raise ValueError(f"Model57 xdeep must be > 0 for log retrieval, got {xdeep}")

        err = edeep / xdeep
        sx[ix, ix] = err**2.0

        ix += 1

        # mid abundance
        if xmid > 0.0:
            x0[ix] = np.log(xmid)
            lx[ix] = 1
        else:
            raise ValueError(f"Model57 xmid must be > 0 for log retrieval, got {xmid}")

        err = emid / xmid
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
        # ******** profile defined by knee and strat pressure ********
        # ******** with deep and mid abundance ********
        if varident[2] != cls.id:
            raise ValueError("error in Model57.from_bookmark() :: wrong model id")

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
        # Model 57. VMR is constant up to a knee pressure
        # ***************************************************************

        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)

        xdeep, xmid = self.get_parameter_values_from_state_vector(
            forward_model.Variables.XN, forward_model.Variables.LX
        )

        pknee = forward_model.Variables.VARPARAM[ivar, 0]
        pstrat = forward_model.Variables.VARPARAM[ivar, 1]

        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            xdeep,
            xmid,
            pknee,
            pstrat,
            MakePlot=False,
        )

        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0 : forward_model.AtmosphereX.NP] = xmap1

        return