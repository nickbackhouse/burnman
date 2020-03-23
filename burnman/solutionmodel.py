# This file is part of BurnMan - a thermoelastic and thermodynamic toolkit for the Earth and Planetary Sciences
# Copyright (C) 2012 - 2017 by the BurnMan team, released under the GNU
# GPL v2 or later.

from __future__ import absolute_import

import numpy as np
from . import processchemistry
from . import constants

# Try to import the jit from numba.  If it is
# not available, just go with the standard
# python interpreter
try:
    import os
    if 'NUMBA_DISABLE_JIT' in os.environ and int(os.environ['NUMBA_DISABLE_JIT']) == 1:
        raise ImportError("NOOOO!")
    from numba import jit
except ImportError:
    def jit(fn):
        return fn


@jit
def _ideal_activities_fct(molar_fractions, endmember_occupancies, n_endmembers, n_occupancies, site_multiplicities, endmember_configurational_entropies):
    site_occupancies = np.dot(molar_fractions, endmember_occupancies)
    a = np.power(site_occupancies,
                 endmember_occupancies * site_multiplicities).prod(-1)
    normalisation_constants = np.exp(endmember_configurational_entropies /
                                     constants.gas_constant)
    return normalisation_constants * a

@jit
def _non_ideal_hessian_fct(phi, molar_fractions, n_endmembers, alpha, W):
    q = np.eye(n_endmembers) - phi*np.ones((n_endmembers, n_endmembers))
    sum_pa = np.sum(np.dot(molar_fractions, alpha))
    hess = np.einsum('m, i, ij, jk, mk->im', -alpha/sum_pa, -alpha, q, W, q)
    hess += hess.T
    return hess

@jit
def _non_ideal_interactions_fct(phi, molar_fractions, n_endmembers, alpha, W):
    # -sum(sum(qi.qj.Wij*)
    # equation (2) of Holland and Powell 2003
    q = np.eye(n_endmembers) - phi*np.ones((n_endmembers, n_endmembers))
    # The following are equivalent to
    # np.einsum('i, ij, jk, ik->i', -self.alphas, q, self.Wx, q)
    Wint = -alpha * (q.dot(W)*q).sum(-1)
    return Wint

@jit
def _non_ideal_hessian_subreg(p, n_endmembers, W):
    hess = np.zeros((n_endmembers, n_endmembers))
    for l in range(n_endmembers):
        for m in range(n_endmembers):
            for i in range(n_endmembers):
                for j in range(n_endmembers):
                    dil = 1. if i==l else 0.
                    djl = 1. if j==l else 0.
                    dim = 1. if i==m else 0.
                    djm = 1. if j==m else 0.


                    hess[l,m] += W[i,j]*((djl*djm*p[i] - dil*dim*p[j]) +
                                         (dil*djm + djl*dim) * (p[j] - p[i]) +
                                         (djl + djm)*(p[i]*(p[i] - 2.*p[j])) -
                                         (dil + dim)*(p[j]*(p[j] - 2.*p[i])) +
                                         3.*p[i]*p[j]*(p[j] - p[i]) +
                                         (((dil + dim) - p[i]) *
                                          ((djl + djm) - p[j]) + p[i]*p[j])/2.)
    return hess

@jit
def _non_ideal_interactions_subreg(p, n_endmembers, W):
    Wint = np.zeros(n_endmembers)
    for l in range(n_endmembers):
        for i in range(n_endmembers):
            for j in range(n_endmembers):
                dil = 1. if i==l else 0.
                djl = 1. if j==l else 0.
                Wint[l] += W[i,j]/2. * ((p[i] - dil)*(p[j] - djl)*(2.*(p[i] - p[j]) - 1.) +
                                        djl*p[i]*p[i] - dil*p[j]*p[j])
    return Wint

def logish(x, eps=1.e-5):
    """
    2nd order series expansion of log(x) about eps: log(eps) - sum_k=1^infty (f_eps)^k / k
    Prevents infinities at x=0
    """
    f_eps = 1. - x/eps
    mask = x>eps
    ln = np.where(x<=eps, np.log(eps) - f_eps - f_eps*f_eps/2., 0.)
    ln[mask] = np.log(x[mask])
    return ln

def inverseish(x, eps=1.e-5):
    """
    1st order series expansion of 1/x about eps: 2/eps - x/eps/eps
    Prevents infinities at x=0
    """
    mask = x>eps
    oneoverx = np.where(x<=eps, 2./eps - x/eps/eps, 0.)
    oneoverx[mask] = 1./x[mask]
    return oneoverx


class SolutionModel(object):

    """
    This is the base class for a solution model,  intended for use
    in defining solid solutions and performing thermodynamic calculations
    on them.  All minerals of type :class:`burnman.SolidSolution` use
    a solution model for defining how the endmembers in the solid solution
    interact.

    A user wanting a new solution model should define the functions included
    in the base class. All of the functions in the base class return zero,
    so if the user-defined solution model does not implement them,
    they essentially have no effect, and the Gibbs free energy and molar
    volume of a solid solution will be equal to the weighted arithmetic
    averages of the different endmember values.
    """

    def __init__(self):
        """
        Does nothing.
        """
        pass

    def excess_gibbs_free_energy(self, pressure, temperature, molar_fractions):
        """
        Given a list of molar fractions of different phases,
        compute the excess Gibbs free energy of the solution.
        The base class implementation assumes that the excess gibbs
        free energy is zero.

        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the solution model. [Pa]

        temperature : float
            Temperature at which to evaluate the solution. [K]

        molar_fractions : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        G_excess : float
            The excess Gibbs free energy
        """
        return np.dot(np.array(molar_fractions), self.excess_partial_gibbs_free_energies(pressure, temperature, molar_fractions))

    def excess_magnetic_gibbs_free_energy( self, pressure, temperature, molar_fractions):
            """
            Given a list of molar fractions of different phases,
            compute the excess magnetic Gibbs free energy of the solution.
            The base class implementation assumes that the excess magnetic gibbs
            free energy is zero.

            Parameters
            ----------
            pressure : float
                Pressure at which to evaluate the solution model. [Pa]

            temperature : float
                Temperature at which to evaluate the solution. [K]

            molar_fractions : list of floats
                List of molar fractions of the different endmembers in solution

            Returns
            -------
            G_excess : float
                The excess Gibbs free energy
            """
            return 0.

    def excess_volume(self, pressure, temperature, molar_fractions):
        """
        Given a list of molar fractions of different phases,
        compute the excess volume of the solution.
        The base class implementation assumes that the excess volume is zero.

        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the solution model. [Pa]

        temperature : float
            Temperature at which to evaluate the solution. [K]

        molar_fractions : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        V_excess : float
            The excess volume of the solution
        """
        return np.dot(molar_fractions,
                      self.excess_partial_volumes(pressure, temperature, molar_fractions))

    def excess_entropy(self, pressure, temperature, molar_fractions):
        """
        Given a list of molar fractions of different phases,
        compute the excess entropy of the solution.
        The base class implementation assumes that the excess entropy is zero.

        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the solution model. [Pa]

        temperature : float
            Temperature at which to evaluate the solution. [K]

        molar_fractions : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        S_excess : float
            The excess entropy of the solution
        """
        return np.dot(molar_fractions,
                      self.excess_partial_entropies(pressure, temperature, molar_fractions))

    def excess_enthalpy(self, pressure, temperature, molar_fractions):
        """
        Given a list of molar fractions of different phases,
        compute the excess enthalpy of the solution.
        The base class implementation assumes that the excess enthalpy is zero.

        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the solution model. [Pa]

        temperature : float
            Temperature at which to evaluate the solution. [K]

        molar_fractions : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        H_excess : float
            The excess enthalpy of the solution
        """
        return (self.excess_gibbs_free_energy(pressure, temperature, molar_fractions) +
                temperature*self.excess_entropy(pressure, temperature, molar_fractions))

    def excess_partial_gibbs_free_energies(self, pressure, temperature, molar_fractions):
        """
        Given a list of molar fractions of different phases,
        compute the excess Gibbs free energy for each endmember of the solution.
        The base class implementation assumes that the excess gibbs
        free energy is zero.

        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the solution model. [Pa]

        temperature : float
            Temperature at which to evaluate the solution. [K]

        molar_fractions : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        partial_G_excess : numpy array
            The excess Gibbs free energy of each endmember
        """
        return np.zeros_like(molar_fractions)

    def excess_partial_gibbs_free_energies(self, pressure, temperature, molar_fractions):
        """
        Given a list of molar fractions of different phases,
        compute the excess Gibbs free energy for each endmember of the solution.
        The base class implementation assumes that the excess gibbs
        free energy is zero.

        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the solution model. [Pa]

        temperature : float
            Temperature at which to evaluate the solution. [K]

        molar_fractions : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        partial_G_excess : numpy array
            The excess Gibbs free energy of each endmember
        """
        return np.zeros_like(molar_fractions)

    def excess_partial_entropies(self, pressure, temperature, molar_fractions):
        """
        Given a list of molar fractions of different phases,
        compute the excess entropy for each endmember of the solution.
        The base class implementation assumes that the excess entropy
        is zero (true for mechanical solutions).

        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the solution model. [Pa]

        temperature : float
            Temperature at which to evaluate the solution. [K]

        molar_fractions : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        partial_S_excess : numpy array
            The excess entropy of each endmember
        """
        return np.zeros_like(molar_fractions)

    def excess_partial_volumes(self, pressure, temperature, molar_fractions):
        """
        Given a list of molar fractions of different phases,
        compute the excess volume for each endmember of the solution.
        The base class implementation assumes that the excess volume
        is zero.

        Parameters
        ----------
        pressure : float
            Pressure at which to evaluate the solution model. [Pa]

        temperature : float
            Temperature at which to evaluate the solution. [K]

        molar_fractions : list of floats
            List of molar fractions of the different endmembers in solution

        Returns
        -------
        partial_V_excess : numpy array
            The excess volume of each endmember
        """
        return np.zeros_like(np.array(molar_fractions))

class MechanicalSolution (SolutionModel):

    """
    An extremely simple class representing a mechanical solution model.
    A mechanical solution experiences no interaction between endmembers.
    Therefore, unlike ideal solutions there is no entropy of mixing;
    the total gibbs free energy of the solution is equal to the
    dot product of the molar gibbs free energies and molar fractions
    of the constituent materials.
    """

    def __init__(self, endmembers):
        self.n_endmembers = len(endmembers)
        self.formulas = [e[1] for e in endmembers]

    def excess_gibbs_free_energy(self, pressure, temperature, molar_fractions):
        return 0.

    def excess_volume(self, pressure, temperature, molar_fractions):
        return 0.

    def excess_entropy(self, pressure, temperature, molar_fractions):
        return 0.

    def excess_enthalpy(self, pressure, temperature, molar_fractions):
        return 0.

    def excess_partial_gibbs_free_energies(self, pressure, temperature, molar_fractions):
        return np.zeros_like(molar_fractions)

    def excess_partial_volumes(self, pressure, temperature, molar_fractions):
        return np.zeros_like(molar_fractions)

    def excess_partial_entropies(self, pressure, temperature, molar_fractions):
        return np.zeros_like(molar_fractions)

    def activity_coefficients(self, pressure, temperature, molar_fractions):
        return np.ones_like(molar_fractions)

    def activities(self, pressure, temperature, molar_fractions):
        return np.ones_like(molar_fractions)



class IdealSolution (SolutionModel):

    """
    A very simple class representing an ideal solution model.
    Calculate the excess gibbs free energy and entropy due to configurational
    entropy, excess volume is equal to zero.
    """

    def __init__(self, endmembers):
        self.n_endmembers = len(endmembers)
        self.formulas = [e[1] for e in endmembers]

        # Process solid solution chemistry
        processchemistry.process_solution_chemistry(self)

        self._calculate_endmember_configurational_entropies()

    def _calculate_endmember_configurational_entropies(self):
        S_conf = -constants.gas_constant * (self.site_multiplicities *
                                            self.endmember_occupancies *
                                            logish(self.endmember_occupancies)).sum(-1)
        self.endmember_configurational_entropies = S_conf

    def excess_partial_gibbs_free_energies(self, pressure, temperature, molar_fractions):
        return self._ideal_excess_partial_gibbs(temperature, molar_fractions)

    def excess_partial_entropies(self, pressure, temperature, molar_fractions):
        return self._ideal_excess_partial_entropies(temperature, molar_fractions)

    def excess_partial_volumes(self, pressure, temperature, molar_fractions):
        return np.zeros((self.n_endmembers))

    def gibbs_hessian(self, pressure, temperature, molar_fractions):
        hess_S = self._ideal_entropy_hessian(temperature, molar_fractions)
        return -temperature*hess_S

    def entropy_hessian(self, pressure, temperature, molar_fractions):
        hess_S = self._ideal_entropy_hessian(temperature, molar_fractions)
        return hess_S

    def volume_hessian(self, pressure, temperature, molar_fractions):
        return np.zeros((len(molar_fractions), len(molar_fractions)))

    def _configurational_entropy(self, molar_fractions):
        site_occupancies = np.dot(molar_fractions, self.endmember_occupancies)
        conf_entropy = - constants.gas_constant * (site_occupancies *
                                                   self.site_multiplicities *
                                                   logish(site_occupancies)).sum(-1)
        return conf_entropy

    def _ideal_excess_partial_gibbs(self, temperature, molar_fractions):
        return -temperature*self._ideal_excess_partial_entropies(temperature, molar_fractions)

    def _ideal_excess_partial_entropies(self, temperature, molar_fractions):
        return -constants.gas_constant * self._log_ideal_activities(molar_fractions)

    def _ideal_entropy_hessian(self, temperature, molar_fractions):
        hessian = -constants.gas_constant * self._log_ideal_activity_derivatives(molar_fractions)
        return hessian

    def _log_ideal_activities(self, molar_fractions):
        site_occupancies = np.dot(molar_fractions, self.endmember_occupancies)
        lna = (self.endmember_occupancies * self.site_multiplicities *
               logish(site_occupancies)).sum(-1)
        normalisation_constants = (self.endmember_configurational_entropies /
                                   constants.gas_constant)

        return lna + normalisation_constants

    def _log_ideal_activity_derivatives(self, molar_fractions):
        site_occupancies = np.dot(molar_fractions, self.endmember_occupancies)
        n_site_atoms = self.endmember_occupancies[0].dot(self.site_multiplicities)
        dlnadp = np.einsum('ij, j, mj',
                           self.endmember_occupancies,
                           self.site_multiplicities*inverseish(site_occupancies),
                           self.endmember_occupancies) - n_site_atoms
        return dlnadp


    def _ideal_activities(self, molar_fractions):
        return _ideal_activities_fct(molar_fractions,
                                     self.endmember_occupancies,
                                     self.n_endmembers,
                                     self.n_occupancies,
                                     self.site_multiplicities,
                                     self.endmember_configurational_entropies)

    def activity_coefficients(self, pressure, temperature, molar_fractions):
        return np.ones_like(molar_fractions)

    def activities(self, pressure, temperature, molar_fractions):
        return self._ideal_activities(molar_fractions)


class AsymmetricRegularSolution (IdealSolution):

    """
    Solution model implementing the asymmetric regular solution model formulation as described in :cite:`HP2003`.
    """

    def __init__(self, endmembers, alphas, energy_interaction, volume_interaction=None, entropy_interaction=None):

        self.n_endmembers = len(endmembers)

        # Create array of van Laar parameters
        self.alphas = np.array(alphas)

        # Create 2D arrays of interaction parameters
        self.We = np.triu(2. / (self.alphas[:, np.newaxis] + self.alphas), 1)
        self.We[np.triu_indices(self.n_endmembers, 1)] *= np.array([i for row in energy_interaction
                                                                for i in row])

        if entropy_interaction is not None:
            self.Ws = np.triu(2. / (self.alphas[:, np.newaxis] + self.alphas), 1)
            self.Ws[np.triu_indices(self.n_endmembers, 1)] *= np.array([i for row in entropy_interaction
                                                                        for i in row])
        else:
            self.Ws = np.zeros((self.n_endmembers, self.n_endmembers))

        if volume_interaction is not None:
            self.Wv = np.triu(2. / (self.alphas[:, np.newaxis] + self.alphas), 1)
            self.Wv[np.triu_indices(self.n_endmembers, 1)] *= np.array([i for row in volume_interaction
                                                                        for i in row])
        else:
            self.Wv = np.zeros((self.n_endmembers, self.n_endmembers))


        # initialize ideal solution model
        IdealSolution.__init__(self, endmembers)

    def _phi(self, molar_fractions):
        phi = self.alphas*molar_fractions
        phi = np.divide(phi, np.sum(phi))
        return phi


    def _non_ideal_interactions(self, W, molar_fractions):
        # -sum(sum(qi.qj.Wij*)
        # equation (2) of Holland and Powell 2003
        phi = self._phi(molar_fractions)
        return _non_ideal_interactions_fct(phi, molar_fractions, self.n_endmembers, self.alphas, W)

    def _non_ideal_excess_partial_gibbs(self, pressure, temperature, molar_fractions):
        Eint = self._non_ideal_interactions(self.We, molar_fractions)
        Sint = self._non_ideal_interactions(self.Ws, molar_fractions)
        Vint = self._non_ideal_interactions(self.Wv, molar_fractions)
        return Eint - temperature * Sint + pressure * Vint

    def excess_partial_gibbs_free_energies(self, pressure, temperature, molar_fractions):
        ideal_gibbs = IdealSolution._ideal_excess_partial_gibbs(
            self, temperature, molar_fractions)
        non_ideal_gibbs = self._non_ideal_excess_partial_gibbs(
            pressure, temperature, molar_fractions)
        return ideal_gibbs + non_ideal_gibbs

    def excess_partial_entropies(self, pressure, temperature, molar_fractions):
        ideal_entropies = IdealSolution._ideal_excess_partial_entropies(
            self, temperature, molar_fractions)
        non_ideal_entropies = self._non_ideal_interactions(self.Ws, molar_fractions)
        return ideal_entropies + non_ideal_entropies

    def excess_partial_volumes(self, pressure, temperature, molar_fractions):
        return self._non_ideal_interactions(self.Wv, molar_fractions)

    def gibbs_hessian(self, pressure, temperature, molar_fractions):
        ideal_entropy_hessian = IdealSolution._ideal_entropy_hessian(self, temperature, molar_fractions)
        phi = self._phi(molar_fractions)
        nonideal_gibbs_hessian = _non_ideal_hessian_fct(phi, molar_fractions,
                                                        self.n_endmembers, self.alphas,
                                                        self.We - temperature*self.Ws + pressure*self.Wv)

        return nonideal_gibbs_hessian - temperature*ideal_entropy_hessian

    def entropy_hessian(self, pressure, temperature, molar_fractions):
        ideal_entropy_hessian = IdealSolution._ideal_entropy_hessian(self, temperature, molar_fractions)
        phi = self._phi(molar_fractions)
        nonideal_entropy_hessian = _non_ideal_hessian_fct(phi, molar_fractions,
                                                          self.n_endmembers, self.alphas,
                                                          self.Ws)
        return ideal_entropy_hessian + nonideal_entropy_hessian

    def volume_hessian(self, pressure, temperature, molar_fractions):
        phi = self._phi(molar_fractions)
        return _non_ideal_hessian_fct(phi, molar_fractions,
                                      self.n_endmembers, self.alphas,
                                      self.Wv)

    def activity_coefficients(self, pressure, temperature, molar_fractions):
        if temperature > 1.e-10:
            return np.exp(self._non_ideal_excess_partial_gibbs(pressure, temperature, molar_fractions) / (constants.gas_constant * temperature))
        else:
            raise Exception("Activity coefficients not defined at 0 K.")

    def activities(self, pressure, temperature, molar_fractions):
        return IdealSolution.activities(self, pressure, temperature, molar_fractions) * self.activity_coefficients(pressure, temperature, molar_fractions)


class SymmetricRegularSolution (AsymmetricRegularSolution):

    """
    Solution model implementing the symmetric regular solution model. This is simply a special case of the :class:`burnman.solutionmodel.AsymmetricRegularSolution` class.
    """

    def __init__(self, endmembers, energy_interaction, volume_interaction=None, entropy_interaction=None):
        alphas = np.ones(len(endmembers))
        AsymmetricRegularSolution.__init__(
            self, endmembers, alphas, energy_interaction, volume_interaction, entropy_interaction)


class SubregularSolution (IdealSolution):

    """
    Solution model implementing the subregular solution model formulation as described in :cite:`HW1989`.
    """

    def __init__(self, endmembers, energy_interaction, volume_interaction=None, entropy_interaction=None):

        self.n_endmembers = len(endmembers)

        # Create 2D arrays of interaction parameters
        self.We = np.zeros(shape=(self.n_endmembers, self.n_endmembers))
        self.Ws = np.zeros(shape=(self.n_endmembers, self.n_endmembers))
        self.Wv = np.zeros(shape=(self.n_endmembers, self.n_endmembers))

        # setup excess enthalpy interaction matrix
        for i in range(self.n_endmembers):
            for j in range(i + 1, self.n_endmembers):
                self.We[i][j] = energy_interaction[i][j - i - 1][0]
                self.We[j][i] = energy_interaction[i][j - i - 1][1]

        if entropy_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i + 1, self.n_endmembers):
                    self.Ws[i][j] = entropy_interaction[i][j - i - 1][0]
                    self.Ws[j][i] = entropy_interaction[i][j - i - 1][1]

        if volume_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i + 1, self.n_endmembers):
                    self.Wv[i][j] = volume_interaction[i][j - i - 1][0]
                    self.Wv[j][i] = volume_interaction[i][j - i - 1][1]

        # initialize ideal solution model
        IdealSolution.__init__(self, endmembers)

    def _non_ideal_function(self, W, molar_fractions):
        n = len(molar_fractions)
        return _non_ideal_interactions_subreg(molar_fractions, n, W)

    def _non_ideal_interactions(self, molar_fractions):
        # equation (6') of Helffrich and Wood, 1989
        Eint = self._non_ideal_function(self.We, molar_fractions)
        Sint = self._non_ideal_function(self.Ws, molar_fractions)
        Vint = self._non_ideal_function(self.Wv, molar_fractions)
        return Eint, Sint, Vint

    def _non_ideal_excess_partial_gibbs(self, pressure, temperature, molar_fractions):
        Eint, Sint, Vint = self._non_ideal_interactions(molar_fractions)
        return Eint - temperature * Sint + pressure * Vint

    def excess_partial_gibbs_free_energies(self, pressure, temperature, molar_fractions):
        ideal_gibbs = IdealSolution._ideal_excess_partial_gibbs(
            self, temperature, molar_fractions)
        non_ideal_gibbs = self._non_ideal_excess_partial_gibbs(
            pressure, temperature, molar_fractions)
        return ideal_gibbs + non_ideal_gibbs

    def excess_partial_entropies(self, pressure, temperature, molar_fractions):
        ideal_entropies = IdealSolution._ideal_excess_partial_entropies(
            self, temperature, molar_fractions)
        non_ideal_entropies = self._non_ideal_function(self.Ws, molar_fractions)
        return ideal_entropies + non_ideal_entropies

    def excess_partial_volumes(self, pressure, temperature, molar_fractions):
        non_ideal_volumes = self._non_ideal_function(self.Wv, molar_fractions)
        return non_ideal_volumes


    def gibbs_hessian(self, pressure, temperature, molar_fractions):
        n = len(molar_fractions)
        ideal_entropy_hessian = IdealSolution._ideal_entropy_hessian(self, temperature, molar_fractions)
        nonideal_gibbs_hessian = _non_ideal_hessian_subreg(molar_fractions, n,
                                                           self.We - temperature*self.Ws +
                                                           pressure*self.Wv)

        return nonideal_gibbs_hessian - temperature*ideal_entropy_hessian

    def entropy_hessian(self, pressure, temperature, molar_fractions):
        n = len(molar_fractions)
        ideal_entropy_hessian = IdealSolution._ideal_entropy_hessian(self, temperature, molar_fractions)
        nonideal_entropy_hessian = _non_ideal_hessian_subreg(molar_fractions, n,
                                                             self.Ws)
        return ideal_entropy_hessian + nonideal_entropy_hessian

    def volume_hessian(self, pressure, temperature, molar_fractions):
        n = len(molar_fractions)
        return _non_ideal_hessian_subreg(molar_fractions, n,
                                         self.Wv)

    def activity_coefficients(self, pressure, temperature, molar_fractions):
        if temperature > 1.e-10:
            return np.exp(self._non_ideal_excess_partial_gibbs(pressure, temperature, molar_fractions) / (constants.gas_constant * temperature))
        else:
            raise Exception("Activity coefficients not defined at 0 K.")

    def activities(self, pressure, temperature, molar_fractions):
        return IdealSolution.activities(self, pressure, temperature, molar_fractions) * self.activity_coefficients(pressure, temperature, molar_fractions)

class AsymmetricRegularSolution_w_magnetism (IdealSolution):
    """
    Solution model implementing the asymmetric regular solution model formulation (Holland and Powell, 2003)
    """

    def __init__( self, endmembers, alphas, magnetic_parameters, enthalpy_interaction, volume_interaction = None, entropy_interaction = None ):

        self.n_endmembers = len(endmembers)

        # Create array of van Laar parameters
        self.alpha=np.array(alphas)

        # Create 2D arrays of interaction parameters
        self.Wh=np.zeros(shape=(self.n_endmembers,self.n_endmembers))
        self.Ws=np.zeros(shape=(self.n_endmembers,self.n_endmembers))
        self.Wv=np.zeros(shape=(self.n_endmembers,self.n_endmembers))

        #setup excess enthalpy interaction matrix
        for i in range(self.n_endmembers):
            for j in range(i+1, self.n_endmembers):
                self.Wh[i][j]=2.*enthalpy_interaction[i][j-i-1]/(self.alpha[i]+self.alpha[j])

        if entropy_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.Ws[i][j]=2.*entropy_interaction[i][j-i-1]/(self.alpha[i]+self.alpha[j])

        if volume_interaction is not None:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.Wv[i][j]=2.*volume_interaction[i][j-i-1]/(self.alpha[i]+self.alpha[j])

        # Magnetism
        self.structural_parameter=np.array(magnetic_parameters[0])
        self.magnetic_moments=np.array(magnetic_parameters[1])
        self.Tcs=np.array(magnetic_parameters[2])
        self.magnetic_moment_excesses=np.array(magnetic_parameters[3])
        self.Tc_excesses=np.array(magnetic_parameters[4])
        self.WTc=np.zeros(shape=(self.n_endmembers,self.n_endmembers))
        self.WB=np.zeros(shape=(self.n_endmembers,self.n_endmembers))
        try:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.WTc[i][j] = 2.*self.Tc_excesses[i][j-i-1]/(self.alpha[i]+self.alpha[j])
        except AttributeError:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.WTc[i][j] = 0.

        try:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.WB[i][j] = 2.*self.magnetic_moment_excesses[i][j-i-1]/(self.alpha[i]+self.alpha[j])
        except AttributeError:
            for i in range(self.n_endmembers):
                for j in range(i+1, self.n_endmembers):
                    self.WB[i][j] = 0.


        #initialize ideal solution model
        IdealSolution.__init__(self, endmembers )

    def _phi( self, molar_fractions):
        phi=np.array([self.alpha[i]*molar_fractions[i] for i in range(self.n_endmembers)])
        phi=np.divide(phi, np.sum(phi))
        return phi

    def _non_ideal_interactions( self, molar_fractions ):
        # -sum(sum(qi.qj.Wij*)
        # equation (2) of Holland and Powell 2003

        phi=self._phi(molar_fractions)

        q=np.zeros(len(molar_fractions))
        Hint=np.zeros(len(molar_fractions))
        Sint=np.zeros(len(molar_fractions))
        Vint=np.zeros(len(molar_fractions))

        for l in range(self.n_endmembers):
            q=np.array([kd(i,l)-phi[i] for i in range(self.n_endmembers)])

            Hint[l]=0.-self.alpha[l]*np.dot(q,np.dot(self.Wh,q))
            Sint[l]=0.-self.alpha[l]*np.dot(q,np.dot(self.Ws,q))
            Vint[l]=0.-self.alpha[l]*np.dot(q,np.dot(self.Wv,q))

        return Hint, Sint, Vint

    def _non_ideal_excess_partial_gibbs( self, pressure, temperature, molar_fractions) :

        Hint, Sint, Vint = self._non_ideal_interactions( molar_fractions )

        magnetic_gibbs = self._magnetic_excess_partial_gibbs(pressure, temperature, molar_fractions)

        return Hint - temperature*Sint + pressure*Vint + magnetic_gibbs

    def _magnetic_params(self, structural_parameter):
        A = (518./1125.) + (11692./15975.)*((1./structural_parameter) - 1.)
        b = (474./497.)*(1./structural_parameter - 1.)
        a = [-79./(140.*structural_parameter), -b/6., -b/135., -b/600., -1./10., -1./315., -1./1500.]
        return A, b, a

    def _magnetic_entropy(self, tau, magnetic_moment, structural_parameter):
        """
        Returns the magnetic contribution to the Gibbs free energy [J/mol]
        Expressions are those used by Chin, Hertzman and Sundman (1987)
        as reported in Sundman in the Journal of Phase Equilibria (1991)
        """

        A, b, a = self._magnetic_params(structural_parameter)
        if tau < 1:
            f=1.+(1./A)*(a[0]/tau + b*(a[1]*np.power(tau, 3.) + a[2]*np.power(tau, 9.) + a[3]*np.power(tau, 15.)))
        else:
            f=(1./A)*(a[4]*np.power(tau,-5) + a[5]*np.power(tau,-15) + a[6]*np.power(tau, -25))

        return -constants.gas_constant*np.log(magnetic_moment + 1.)*f


    def _magnetic_excess_partial_gibbs( self, pressure, temperature, molar_fractions) :
        # This function calculates the partial *excess* gibbs free energies due to magnetic ordering
        # Note that the endmember contributions are subtracted from the excesses
        # i.e. the partial *excess* gibbs free energy at any endmember composition is zero

        phi=self._phi(molar_fractions)

        # Tc, magnetic moment and magnetic Gibbs contribution at P,T,X
        Tc=np.dot(self.Tcs.T, molar_fractions) + np.dot(self.alpha.T,molar_fractions)*np.dot(phi.T,np.dot(self.WTc,phi))

        if Tc > 1.e-12:
            tau=temperature/Tc
            magnetic_moment=np.dot(self.magnetic_moments, molar_fractions) + np.dot(self.alpha.T,molar_fractions)*np.dot(phi.T,np.dot(self.WB,phi))
            Gmag=-temperature*self._magnetic_entropy(tau, magnetic_moment, self.structural_parameter)
            f = Gmag/(constants.gas_constant*temperature*np.log(magnetic_moment + 1.))
            # Partial excesses
            A, b, a = self._magnetic_params(self.structural_parameter)
            dtaudtc=-temperature/(Tc*Tc)
            if tau < 1:
                dfdtau=(1./A)*(-a[0]/(tau*tau) + 3.*a[1]*np.power(tau, 2.) + 9.*a[2]*np.power(tau, 8.) + 15.*a[3]*np.power(tau, 14.))
            else:
                dfdtau=(1./A)*(-5.*a[4]*np.power(tau,-6) - 15.*a[5]*np.power(tau,-16) - 25.*a[6]*np.power(tau, -26))
        else:
            Gmag=dfdtau=dtaudtc=magnetic_moment=f=0.0

        # Calculate the effective partial Tc and magnetic moments at the endmembers
        # (in order to calculate the partial derivatives of Tc and magnetic moment at the composition of interest)
        partial_B=np.zeros(self.n_endmembers)
        partial_Tc=np.zeros(self.n_endmembers)
        endmember_Gmag=np.zeros(self.n_endmembers)
        for l in range(self.n_endmembers):
            if self.Tcs[l]>1.e-12:
                endmember_Gmag[l] = -temperature*self._magnetic_entropy(temperature/self.Tcs[l], self.magnetic_moments[l], self.structural_parameter)
            else:
                endmember_Gmag[l]=0.

            q=np.array([kd(i,l)-phi[i] for i in range(self.n_endmembers)])
            partial_Tc[l]=self.Tcs[l]-self.alpha[l]*np.dot(q,np.dot(self.WTc,q))
            partial_B[l]=self.magnetic_moments[l]-self.alpha[l]*np.dot(q,np.dot(self.WB,q))

        tc_diff = partial_Tc - Tc
        magnetic_moment_diff= partial_B - magnetic_moment

        # Use the chain and product rules on the expression for the magnetic gibbs free energy
        XdGdX=constants.gas_constant*temperature*(magnetic_moment_diff*f/(magnetic_moment + 1.) + dfdtau*dtaudtc*tc_diff*np.log(magnetic_moment + 1.))

        endmember_contributions=np.dot(endmember_Gmag, molar_fractions)

        return Gmag - endmember_contributions + XdGdX

    def excess_magnetic_gibbs_free_energy( self, pressure, temperature, molar_fractions):
        return np.dot(self._magnetic_excess_partial_gibbs(pressure, temperature, molar_fractions),molar_fractions)

    def excess_partial_gibbs_free_energies( self, pressure, temperature, molar_fractions ):
        ideal_gibbs = IdealSolution._ideal_excess_partial_gibbs (self, temperature, molar_fractions)
        non_ideal_gibbs = self._non_ideal_excess_partial_gibbs(pressure, temperature, molar_fractions)
        return ideal_gibbs + non_ideal_gibbs

    def excess_volume ( self, pressure, temperature, molar_fractions ):
        phi=self._phi(molar_fractions)
        V_excess=np.dot(self.alpha.T,molar_fractions)*np.dot(phi.T,np.dot(self.Wv,phi))
        return V_excess

    def excess_entropy( self, pressure, temperature, molar_fractions ):
        phi=self._phi(molar_fractions)
        S_conf=-constants.gas_constant*np.dot(IdealSolution._log_ideal_activities(self, molar_fractions), molar_fractions)
        S_excess=np.dot(self.alpha.T,molar_fractions)*np.dot(phi.T,np.dot(self.Ws,phi))
        S_mag = -np.dot(self._magnetic_excess_partial_gibbs(pressure, temperature, molar_fractions), molar_fractions)/temperature
        return S_conf + S_excess + S_mag

    def excess_enthalpy( self, pressure, temperature, molar_fractions ):
        phi=self._phi(molar_fractions)
        H_excess=np.dot(self.alpha.T,molar_fractions)*np.dot(phi.T,np.dot(self.Wh,phi))
        return H_excess + pressure*self.excess_volume ( pressure, temperature, molar_fractions )


class SymmetricRegularSolution_w_magnetism (AsymmetricRegularSolution_w_magnetism):
    """
    Solution model implementing the symmetric regular solution model
    """
    def __init__( self, endmembers, magnetic_parameters, enthalpy_interaction, volume_interaction = None, entropy_interaction = None ):
        alphas = np.ones( len(endmembers) )
        AsymmetricRegularSolution_w_magnetism.__init__(self, endmembers, alphas, magnetic_parameters, enthalpy_interaction, volume_interaction, entropy_interaction )
