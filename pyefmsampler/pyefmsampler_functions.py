#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 08:59:15 2025

@author: frederik
"""


import numpy as np
import cobra
import tqdm
import matplotlib.pyplot as plt

from collections import Counter
from matplotlib.ticker import MaxNLocator
from scipy.optimize import linprog


###############################################################################
# Helper Functions
###############################################################################




def combine_efms(efm1,efm2,target,model):
    """
    Find specific large blocksets, by combining two EFMs and cancelling a reversible reaction. The corresponding LP is typically significantly less complex than in the general case.
    Parameters
    ----------
    efm1 : np.array
        First EFM
    efm2 : np.array
        Second EFM
    target : int
        Reaction that must be active
    model : FluxCone object
        Fluxcone that has the input vectors as EFMs.

    Returns
    -------
    list
        List containing EFMs that result from combining input EFMs.
    """
    
    rev_supp1 = np.intersect1d(supp(model.rev), supp(efm1))
    rev_supp2 = np.intersect1d(supp(model.rev), supp(efm2))
    
    pos1 = np.intersect1d(np.where(efm1 > 0)[0],rev_supp1)
    pos2 = np.intersect1d(np.where(efm2 > 0)[0],rev_supp1)
    
    neg2 = np.intersect1d(np.where(efm1 < 0)[0],rev_supp1)
    neg2 = np.intersect1d(np.where(efm2 < 0)[0],rev_supp2)
    possible_cancels = np.union1d(np.intersect1d(pos1, neg2),np.intersect1d(pos1,neg2))
    combined_supp = np.union1d(supp(efm1),supp(efm2))
    new_efms = []
    new_supps = []
    if len(possible_cancels) == 0:
        return []
    for cancel_index in possible_cancels:
        blockset = np.union1d(np.setdiff1d(np.arange(model.num_reacs),combined_supp),cancel_index)
        composed_efm =  unsplit_vector(find_efm(model.split_stoich, target, blocked=blockset),model)
        if len(supp(composed_efm>0)) and tuple(supp(composed_efm)) not in new_supps:
            new_efms.append(composed_efm)
            new_supps.append(tuple(supp(composed_efm)))
                
    return new_efms





def supp(vector, tol=1e-8):
    """
    Parameters
    ----------
    vector : np.array or list or tuple
    tol : float
    
    Returns 
    -------
    np.array 
        contains indices of nonzero entries in the input vector
    
    """ 
    return tuple(np.where(abs(np.array(vector)) > tol)[0])


def zero(vector, tol=1e-8):
    """
    Parameters
    ----------
    vector : np.array or list or tuple
    tol : float
    
    Returns 
    -------
    np.array 
        contains indices of zero entries in the input vector
    
    """
    return np.where(abs(np.array(vector)) < tol)[0]


def find_objective_index(cobra_model):
    """
    Searches for the index of the reaction that is optimized in the objective function.


    Parameters
    ----------
    cobra_model : cobra.Model
        
    Returns
    -------
    int
        Index of the reaction that is optimized in the objective function.

    """
    
    import re
    s = str(cobra_model.objective)
    match = re.search(r'\*(.*?)\s*-', s)
    if match:
        result = match.group(1).strip()
    return [rea.id for rea in cobra_model.reactions].index(result)


def unsplit_vector(split_vector, model):
    """
    Rejoins pairs of irreversible reactions that result from splitting a reversible reaction


    Parameters
    ----------
    split_vector : np.array 
        vector containing forward and backward irreversible reactions for each reversible reaction.
    model : FluxCone object
        model providing information which reactions were split to obtain split_vector.

    Returns
    -------
    unsplit : np.array
        vector in the original flux cone, containing reversible reactions.
    """
    
    
    rev_indices = np.where(model.rev)[0]  # Indices of reversible reactions
    original_shape = len(model.rev)
    
    orig = split_vector[:original_shape]  # First part contains original fluxes
    tosub = np.zeros_like(orig)  # To store values to subtract
    splits = split_vector[original_shape:]  # The extra split fluxes
    
    for i, j in enumerate(rev_indices):
        tosub[j] = splits[i]  # Assign the split backward flux
    
    unsplit = orig - tosub  # Reconstruct the original flux vector
    
    return unsplit


def check_essential(S,biomass_index,reaction_index):
    
    """
    Check if reaction_index is essential for biomass_index. Tyoically you want to check whether
    a reaction can be equal to 0 while biomass is still being produces. Thus if a reaction can be
    equal to zero it is not essential and the function returns False.
    Parameters:
    S (np.array2d): Stoichiometri matrix after splitting reactions
    biomass_index (int): Index of the reaction for which the essentiality is checked.
    reaction_index (int): Index of the reaction for which you want to check if it is essential for biomass_index
    
    Returns:
        bool: True if reaction_index is essential for biomass_index, False otherwise
    """
    
    
    # Set up LP to minimize flux through reaction_index, while forcing biomass_index
    A_eq = S
    b_eq = np.zeros(S.shape[0])
    c = np.eye(S.shape[1])[reaction_index]
    bounds = [(1,None) if i==biomass_index else (0,None) for i in range(S.shape[1])]
    result = linprog(c,A_eq = A_eq, b_eq=b_eq,bounds=bounds)
    
    if result.x[reaction_index] > 0:
        return True
    else:
        return False


def find_essential_reactions(S,biomass_index):
    """
    Find all essential reactions for biomass_index
    Parameters
    ----------
    S : np.array - stoichiometric matrix
    biomass_index : int - index for which essential reactions are to be searched
        

    Returns
    -------
    essential_indices : list - list of indices of essential reactions
    """
    rep_efm = find_efm(S, biomass_index)
    essential_indices = []
    for index in tqdm.tqdm(supp(rep_efm)):
        if check_essential(S, biomass_index, index):
            essential_indices.append(index)
    return essential_indices


def find_efm(S, target: int, blocked = [],costs = None):
    """
    Find EFM in a metabolic network, where S is the augmented stoichiometric matrix, where reversible reactions have been
    split into a forward and a backward irreversible reaction. 
    
    Parameters:
    :param S (np.ndarray): Stoichiometric matrix of a metabolic network where reversible reactions have been split
    :param target (int): index of reaction that must be contained in the EFM that is returned
    :param blocked (list): list of reactions that cannot be contained in the EFM that is returned
    :param optimization_direction (np.array): vector that determines which reactios' fluxes are minimized in the EFM that is returned
    
    Returns:
    np.ndarray: EFM containing target reaction and not containing blocked reactions that minimizes c^Tx.
    """
    
    n = S.shape[1]  # Number of variables
    
    # Objective function: minimize sum of all x_i
    if costs is None:
        c = np.ones(n)
    else:
        c = costs
    
    # Equality constraint Sx = 0
    A_eq = S
    b_eq = np.zeros(S.shape[0])
    
    # Bounds: x_i >= 0 for i in I, no bounds for others
    bounds = [(0, 0) if i in blocked else (1, None) if i == target else (0, None) for i in range(n)]

    # Solve the LP using dual simplex method
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ds')
    
    if result.success:
        return result.x
    else:
        raise ValueError("No EFM found.")


def supports_to_binary_matrix(supports, vector_length):
    """
    Convert a list of index lists to a binary matrix of shape (n_samples, vector_length)
    """
    mat = np.zeros((len(supports), vector_length), dtype=int)
    for i, supp in enumerate(supports):
        mat[i, supp] = 1
    return mat



class FluxCone:

    def __init__(self, stoichiometry: np.array, reversibility: np.array):
        """
        This Python function initializes a class instance with stoichiometry and reversibility arrays to
        represent a metabolic network.
        """

        # Stoichiometric matrix
        self.stoich = stoichiometry  # np.array

        # Number of metabolites and reactions
        self.num_metabs, self.num_reacs = np.shape(stoichiometry)  # int

        # {0,1} vector for reversible reactions
        self.rev = reversibility  # np.array

        # {0,1} vector for irreversible reactions
        self.irr = (np.ones(self.num_reacs) - self.rev).astype(int)  # np.array
        
        # stoichiometric matrix after splitting all reversible reactions
        self.split_stoich = np.c_[self.stoich, -self.stoich[:, supp(self.rev)]]


    @classmethod
    def from_sbml(cls, path_to_sbml: str):
        """
        The `from_sbml` function reads an SBML file, extracts the stoichiometric matrix and
        reversibility vector, and initializes a FluxCone object with the extracted parameters.
        """

        # read sbml-file
        sbml_model = cobra.io.read_sbml_model(path_to_sbml)

        # extract stoichiometric matrix
        stoich = cobra.util.array.create_stoichiometric_matrix(sbml_model)

        # extract reversibility vector
        rev = np.array([rea.reversibility for rea in sbml_model.reactions]).astype(int)

        # initialize class object from extracted parameters
        return cls(stoich, rev)
        
    @classmethod
    def from_bigg_id(cls,bigg_id: str):
        """
        The `from_bigg_id` function loads a model from the bigg batabase, extracts the stoichiometric matrix and
        reversibility vector, and initializes a FluxCone object with the extracted parameters.
        """
        
        # load model from bigg database
        bigg_model = cobra.io.load_model(bigg_id)
        
        # extract stoichiometric matrix
        stoich = cobra.util.array.create_stoichiometric_matrix(bigg_model)

        # extract reversibility vector
        rev = np.array([rea.reversibility for rea in bigg_model.reactions]).astype(int)

        # initialize class object from extracted parameters
        return cls(stoich, rev)
    
    def rank_test(self,vector):
        """
        Checks if a given vector is an Elementary Flux Mode (EFM) based on rank
        tests and the support of the vector.
        
        Parameters
        ----------
        vector : np.array

        Returns
        -------
        bool
            True if vector is an EFM of the model, according to the rank test

        """
        # (0,...,0) is not an EFM by defintion
        if len(supp(vector)) == 0:
            return False

        # rank test
        if np.linalg.matrix_rank(self.stoich[:, supp(vector)]) == len(supp(vector)) - 1:
            return True

        return False
    
    
    def degree(self, vector):
        """
        The function calculates the degree of a vector within the flux cone.
        """
        # non-negativity constraints defined by v_irr >= 0
        nonegs = np.eye(self.num_reacs)[np.array(supp(self.irr))]

        # outer description of the flux cone by C = { x | Sx >= 0}
        S = np.r_[self.stoich, nonegs]

        return int(self.num_reacs - np.linalg.matrix_rank(S[zero(np.dot(S, vector))]))
    
    
    
    
    def get_efms_efmtool(self, only_reversible=False,opts = dict({
           "kind": "stoichiometry",
           "arithmetic": "double",
           "zero": "1e-10",
           "compression": "default",
           "log": "console",
           "level": "OFF",
           "maxthreads": "-1",
           "normalize": "max",
           "adjacency-method": "pattern-tree-minzero",
           "rowordering": "MostZerosOrAbsLexMin",})):
        """
        Calculates elementary flux modes using the efmtool library

        Parameters
        ----------
        only_reversible : TYPE, optional
            DESCRIPTION. The default is False. Is set to True, only reversible EFMs will be computed
        opts : TYPE, optional
            DESCRIPTION. Options for efmtool
        Returns
        -------
        np.array : EFMs computed by efmtool as rows
        """
        import efmtool

        # Initiate reaction names and metabolite names from 0 to n resp. m because
        # efmtool needs these lists of strings as input
        # "normalize options:  [max, min, norm2, squared, none]
       

        if only_reversible:
            S = np.r_[self.stoich, np.eye(self.num_reacs)[np.array(supp(self.irr))]]
        else:
            S = self.stoich

        reaction_names = list(np.arange(len(S[0])).astype(str))
        metabolite_names = list(np.arange(len(S)).astype(str))

        efms_cols = efmtool.calculate_efms(S, self.rev, reaction_names, metabolite_names, opts)

        return efms_cols.T
