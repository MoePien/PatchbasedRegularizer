# This code belongs to the paper
#
# M. Piening, F. Altekrüger, J. Hertrich, P. Hagemann, A. Walther and G. Steidl (2023).
# Learning from small data sets: Patch-based regularizers in inverse problems for image reconstruction
# ArXiv preprint:2312.16611
#
# Please cite the paper, if you use the code.
# The functions are from 
# 
# J. Feydy, T. S{\'e}journ{\'e}, F.-X. Vialard, S.-i. Amari, A. Trouve and G. Peyr{\'e} (2019).
# Interpolating between Optimal Transport and MMD using Sinkhorn Divergences.
# The 22nd International Conference on Artificial Intelligence and Statistics, pages 2681--2690.
# (https://github.com/jeanfeydy/geomloss)
#
# Here, a semi-unbalanced case is included. Therefore, some functions are
# modified. More specifically, we modified the function geomloss.sinkhorn_divergence.scaling_parameters
# for the semi-unbalanced case (scaling_parameters_su, ll. 60-74).
# Based on that, we redefine the functions geomloss.sinkhorn_samples.sinkhorn_tensorized
# and geomloss.sinkhorn_samples.sinkhorn_online in ll. 84-231 and ll. 241-317, 
# respectively, where we use the defined function scaling_parameters_su.
# Different parameters for the semi-unbalanced case (rho) are incorporated 
# in the function geomloss.sinkhorn_divergence.sinkhorn_cost, which is 
# modified in ll. 323-402.
# These different parameters need to be used for the iterates in 
# geomloss.sinkhorn_divergence.sinkhorn_loop. The modified function is 
# defined in ll. 409-784.
# Finally, the function geomloss.samples_loss.SamplesLoss is redefined 
# in ll. 817-1246 with the use of the newly defined functions sinkhorn_tensorized_su 
# and sinkhorn_online_su. 
#
# Specific changes of each function are explained above the functions.
# The main difference consists in the change of the unbalancing parameter
# rho from a single parameter to a list of two parameters.

import torch
from torch.nn import Module
from functools import partial
import warnings
from collections.abc import Iterable

from geomloss.kernel_samples import kernel_tensorized, kernel_online, kernel_multiscale
from geomloss.kernel_samples import kernel_tensorized as hausdorff_tensorized
from geomloss.kernel_samples import kernel_online as hausdorff_online
from geomloss.kernel_samples import kernel_multiscale as hausdorff_multiscale
from geomloss.sinkhorn_samples import softmin_tensorized, lse_genred, softmin_online, sinkhorn_multiscale 
from geomloss.sinkhorn_divergence import epsilon_schedule, max_diameter, log_weights, dampening
from geomloss.utils import squared_distances, distances, scal

def divide_if_not_none(nominator, denominator):
    """
    only divides the nominator if the denominator is not None
    """
    if denominator is None:
        return nominator
    else:
        return nominator/denominator

# Modified function, for the original function see geomloss.sinkhorn_divergence.scaling_parameters
# Here, an extension for the semi-unbalanced case is included (ll. 67-72)
def scaling_parameters_su(x, y, p, blur, reach, diameter, scaling):
    r"""Turns high-level arguments into numerical values for the Sinkhorn loop."""
    if diameter is None:
        D = x.shape[-1]
        diameter = max_diameter(x.view(-1, D), y.view(-1, D))

    eps = blur ** p
    if reach is None:
        rho = [None, None]
    elif isinstance(reach, Iterable):
        rho = [None if rho_el is None else rho_el ** p for rho_el in reach]
    else:
        rho = [reach**p, reach**p]
    eps_list = epsilon_schedule(p, diameter, blur, scaling)
    return diameter, eps, eps_list, rho


# Redefinition of geomloss.sinkhorn_samples.sinkhorn_tensorized for the semi-unbalanced case.
# The only difference is the application of scaling_parameters_su
# in l. 200.
cost_routines = {
    1: (lambda x, y: distances(x, y)),
    2: (lambda x, y: squared_distances(x, y) / 2),
}
def sinkhorn_tensorized_su(
    a,
    x,
    b,
    y,
    p=2,
    blur=0.05,
    reach=None,
    diameter=None,
    scaling=0.5,
    cost=None,
    debias=True,
    potentials=False,
    **kwargs,
):
    r"""Vanilla PyTorch implementation of the Sinkhorn divergence.

    Args:
        a ((B, N) Tensor): Weights :math:`\alpha_i` for the first measure,
            with a batch dimension.

        x ((B, N, D) Tensor): Sampling locations :math:`x_i` for the first measure,
            with a batch dimension.

        b ((B, M) Tensor): Weights :math:`\beta_j` for the second measure,
            with a batch dimension.

        y ((B, M, D) Tensor): Sampling locations :math:`y_j` for the second measure,
            with a batch dimension.

        p (int, optional): Exponent of the ground cost function
            :math:`C(x_i,y_j)`, which is equal to
            :math:`\tfrac{1}{p}\|x_i-y_j\|^p` if it is not provided
            explicitly through the `cost` optional argument.
            Defaults to 2.

        blur (float, optional): Target value for the blurring scale
            of the Gibbs kernel
            :math:`K_{i,j} = \exp(-C(x_i,y_j)/\varepsilon) = \exp(-\|x_i-y_j\|^p / p \text{blur}^p).
            In the Sinkhorn algorithm, the temperature :math:`\varepsilon`
            is computed as :math:`\text{blur}^p`.
            Defaults to 0.05.

        reach (float or None (= +infty), optional): Typical scale for the
            maximum displacement between any two points :math:`x_i` and :math:`y_j`
            in the optimal transport model.
            In the unbalanced Sinkhorn divergence,
            the strength :math:`\rho` of the soft marginal constraints
            is computed as :math:`\rho = \text{reach}^p`.
            Defaults to None.

        diameter (float or None, optional): Upper bound on the value
            of the distance :math:`\|x_i-y_j\|` between any two samples.
            This will be used as a first value of the `blur` radius
            in the epsilon-scaling annealing descent.
            Defaults to None: an upper bound will be estimated on the fly.

        scaling (float in (0, 1), optional): Ratio between two successive
            values of the blur radius in the epsilon-scaling annealing descent.
            Defaults to 0.5.

        cost (function, optional): Cost function :math:`C(x_i,y_j)`.
            It should take as input two point clouds `x` and `y`
            with a batch dimension, encoded as `(B, N, D)`, `(B, M, D)`
            torch Tensors and return a `(B, N, M)` torch Tensor.
            Defaults to None: we use a Euclidean cost
            :math:`C(x_i,y_j) = \tfrac{1}{p}\|x_i-y_j\|^p`.

        debias (bool, optional): Should we used the "de-biased" Sinkhorn divergence
            :math:`\text{S}_{\varepsilon, \rho}(\al,\be)` instead
            of the "raw" entropic OT cost
            :math:`\text{OT}_{\varepsilon, \rho}(\al,\be)`?
            This slows down the OT solver but guarantees that our approximation
            of the Wasserstein distance will be positive and definite
            - up to convergence of the Sinkhorn loop.
            For a detailed discussion of the influence of this parameter,
            see e.g. Fig. 3.21 in Jean Feydy's PhD thesis.
            Defaults to True.

        potentials (bool, optional): Should we return the optimal dual potentials
            instead of the cost value?
            Defaults to False.

    Returns:
        (B,) Tensor or pair of (B, N), (B, M) Tensors: if `potentials` is True,
            we return a pair of (B, N), (B, M) Tensors that encode the optimal dual vectors,
            respectively supported by :math:`x_i` and :math:`y_j`.
            Otherwise, we return a (B,) Tensor of values for the Sinkhorn divergence.
    """

    # Retrieve the batch size B, the numbers of samples N, M
    # and the size of the ambient space D:
    B, N, D = x.shape
    _, M, _ = y.shape

    # By default, our cost function :math:`C(x_i,y_j)` is a halved,
    # squared Euclidean distance (p=2) or a simple Euclidean distance (p=1):
    if cost is None:
        cost = cost_routines[p]

    # Compute the relevant cost matrices C(x_i, y_j), C(y_j, x_i), etc.
    # Note that we "detach" the gradients of the "right-hand sides":
    # this is coherent with the way we compute our gradients
    # in the `sinkhorn_loop(...)` routine, in the `sinkhorn_divergence.py` file.
    # Please refer to the comments in this file for more details.
    C_xy = cost(x, y.detach())  # (B,N,M) torch Tensor
    C_yx = cost(y, x.detach())  # (B,M,N) torch Tensor

    # N.B.: The "auto-correlation" matrices C(x_i, x_j) and C(y_i, y_j)
    #       are only used by the "debiased" Sinkhorn algorithm.
    C_xx = cost(x, x.detach()) if debias else None  # (B,N,N) torch Tensor
    C_yy = cost(y, y.detach()) if debias else None  # (B,M,M) torch Tensor

    # Compute the relevant values of the diameter of the configuration,
    # target temperature epsilon, temperature schedule across itereations
    # and strength of the marginal constraints:
    diameter, eps, eps_list, rho = scaling_parameters_su(
        x, y, p, blur, reach, diameter, scaling
    )

    # Use an optimal transport solver to retrieve the dual potentials:
    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
        softmin_tensorized,
        log_weights(a),
        log_weights(b),
        C_xx,
        C_yy,
        C_xy,
        C_yx,
        eps_list,
        rho,
        debias=debias,
    )

    # Optimal transport cost:
    return sinkhorn_cost(
        eps,
        rho,
        a,
        b,
        f_aa,
        g_bb,
        g_ab,
        f_ba,
        batch=True,
        debias=debias,
        potentials=potentials,
    )


# Redefinition of geomloss.sinkhorn_samples.sinkhorn_online for the semi-unbalanced case.
# The only difference is the application of scaling_parameters_su
# in l. 288.
cost_formulas = {
    1: "Norm2(X-Y)",
    2: "(SqDist(X,Y) / IntCst(2))",
}
def sinkhorn_online_su(
    a,
    x,
    b,
    y,
    p=2,
    blur=0.05,
    reach=None,
    diameter=None,
    scaling=0.5,
    cost=None,
    debias=True,
    potentials=False,
    **kwargs,
):

    B, N, D = x.shape
    B, M, _ = y.shape

    if cost is None and B > 1:
        if True:
            # raise ValueError("Not expected in this benchmark!")
            softmin = partial(softmin_online_lazytensor, p=p)
        else:
            my_lse = lse_lazytensor(p, D, batchdims=(B,))
            softmin = partial(softmin_online, log_conv=my_lse)

    else:
        if B > 1:
            raise ValueError(
                "Custom cost functions are not yet supported with batches." ""
            )

        x = x.squeeze(0)  # (1, N, D) -> (N, D)
        y = y.squeeze(0)  # (1, M, D) -> (M, D)

        if cost is None:
            cost = cost_formulas[p]

        my_lse = lse_genred(cost, D, dtype=str(x.dtype)[6:])
        softmin = partial(softmin_online, log_conv=my_lse)

    # The "cost matrices" are implicitly encoded in the point clouds,
    # and re-computed on-the-fly:
    C_xx, C_yy = ((x, x.detach()), (y, y.detach())) if debias else (None, None)
    C_xy, C_yx = ((x, y.detach()), (y, x.detach()))

    diameter, eps, eps_list, rho = scaling_parameters_su(
        x, y, p, blur, reach, diameter, scaling
    )

    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
        softmin,
        log_weights(a),
        log_weights(b),
        C_xx,
        C_yy,
        C_xy,
        C_yx,
        eps_list,
        rho,
        debias=debias,
    )

    return sinkhorn_cost(
        eps,
        rho,
        a,
        b,
        f_aa,
        g_bb,
        g_ab,
        f_ba,
        batch=True,
        debias=debias,
        potentials=potentials,
    )


# Modified function, for the original function see geomloss.sinkhorn_divergence.sinkhorn_cost
# Here, an extension for the semi-unbalanced case is included. Different
# parameters for rho are included, see ll. 375-382 and 399-401.
def sinkhorn_cost(
    eps, rho, a, b, f_aa, g_bb, g_ab, f_ba, batch=False, debias=True, potentials=False
):
    r"""Returns the required information (cost, etc.) from a set of dual potentials.

    Args:
        eps (float): Target (i.e. final) temperature.
        rho (float or None (:math:`+\infty`)): Strength of the marginal constraints.

        a ((..., N) Tensor, nonnegative): Weights for the "source" measure on the points :math:`x_i`.
        b ((..., M) Tensor, nonnegative): Weights for the "target" measure on the points :math:`y_j`.
        f_aa ((..., N) Tensor)): Dual potential for the "a <-> a" problem.
        g_bb ((..., M) Tensor)): Dual potential for the "b <-> b" problem.
        g_ab ((..., M) Tensor)): Dual potential supported by :math:`y_j` for the "a <-> b" problem.
        f_ba ((..., N) Tensor)): Dual potential supported by :math:`x_i`  for the "a <-> a" problem.
        batch (bool, optional): Are we working in batch mode? Defaults to False.
        debias (bool, optional): Are we working with the "debiased" or the "raw" Sinkhorn divergence?
            Defaults to True.
        potentials (bool, optional): Shall we return the dual vectors instead of the cost value?
            Defaults to False.

    Returns:
        Tensor or pair of Tensors: if `potentials` is True, we return a pair
            of (..., N), (..., M) Tensors that encode the optimal dual vectors,
            respectively supported by :math:`x_i` and :math:`y_j`.
            Otherwise, we return a (,) or (B,) Tensor of values for the Sinkhorn divergence.
    """

    if potentials:  # Just return the dual potentials
        if debias:  # See Eq. (3.209) in Jean Feydy's PhD thesis.
            # N.B.: This formula does not make much sense in the unbalanced mode
            #       (i.e. if reach is not None).
            return f_ba - f_aa, g_ab - g_bb
        else:  # See Eq. (3.207) in Jean Feydy's PhD thesis.
            return f_ba, g_ab

    else:  # Actually compute the Sinkhorn divergence
        if (
            debias
        ):  # UNBIASED Sinkhorn divergence, S_eps(a,b) = OT_eps(a,b) - .5*OT_eps(a,a) - .5*OT_eps(b,b)
            if rho == [None, None]:  # Balanced case:
                # See Eq. (3.209) in Jean Feydy's PhD thesis.
                return scal(a, f_ba - f_aa, batch=batch) + scal(
                    b, g_ab - g_bb, batch=batch
                )
            else:
                # Unbalanced case:
                # See Proposition 12 (Dual formulas for the Sinkhorn costs)
                # in "Sinkhorn divergences for unbalanced optimal transport",
                # Sejourne et al., https://arxiv.org/abs/1910.12958.
                return scal(
                    a,
                    UnbalancedWeight(eps, rho[0])(
                        (-divide_if_not_none(f_aa, rho[0])).exp() - (-divide_if_not_none(f_ba, rho[0])).exp()
                    ),
                    batch=batch,
                ) + scal(
                    b,
                    UnbalancedWeight(eps, rho[1])(
                        (-divide_if_not_none(g_bb , rho[1])).exp() - (-divide_if_not_none(g_ab , rho[1])).exp()
                    ),
                    batch=batch,
                )

        else:  # Classic, BIASED entropized Optimal Transport OT_eps(a,b)
            if None in rho:  # Balanced case:
                # See Eq. (3.207) in Jean Feydy's PhD thesis.
                return scal(a, f_ba, batch=batch) + scal(b, g_ab, batch=batch)
            else:
                # Unbalanced case:
                # See Proposition 12 (Dual formulas for the Sinkhorn costs)
                # in "Sinkhorn divergences for unbalanced optimal transport",
                # Sejourne et al., https://arxiv.org/abs/1910.12958.
                # N.B.: Even if this quantity is never used in practice,
                #       we may want to re-check this computation...
                return scal(
                    a, UnbalancedWeight(eps, rho[0])(1 - (-divide_if_not_none(f_ba , rho[0])).exp()), batch=batch
                ) + scal(
                    b, UnbalancedWeight(eps, rho[1])(1 - (-divide_if_not_none(g_ab, rho[1])).exp()), batch=batch
                )


# Modified function, for the original function see geomloss.sinkhorn_divergence.sinkhorn_loop
# Here, an extension for the semi-unbalanced case is included. The difference
# consists in different damping paramters (ll. 597-599) and their use for
# the both iterates (ll. 613-779).
def sinkhorn_loop(
    softmin,
    a_logs,
    b_logs,
    C_xxs,
    C_yys,
    C_xys,
    C_yxs,
    eps_list,
    rho,
    jumps=[],
    kernel_truncation=None,
    truncate=5,
    cost=None,
    extrapolate=None,
    debias=True,
    last_extrapolation=True,
):
    r"""Implements the (possibly multiscale) symmetric Sinkhorn loop,
    with the epsilon-scaling (annealing) heuristic.

    This is the main "core" routine of GeomLoss. It is written to
    solve optimal transport problems efficiently in all the settings
    that are supported by the library: (generalized) point clouds,
    images and volumes.

    This algorithm is described in Section 3.3.3 of Jean Feydy's PhD thesis,
    "Geometric data analysis, beyond convolutions" (Universite Paris-Saclay, 2020)
    (https://www.jeanfeydy.com/geometric_data_analysis.pdf).
    Algorithm 3.5 corresponds to the case where `kernel_truncation` is None,
    while Algorithm 3.6 describes the full multiscale algorithm.

    Args:
        softmin (function): This routine must implement the (soft-)C-transform
            between dual vectors, which is the core computation for
            Auction- and Sinkhorn-like optimal transport solvers.
            If `eps` is a float number, `C_xy` encodes a cost matrix :math:`C(x_i,y_j)`
            and `g` encodes a dual potential :math:`g_j` that is supported by the points
            :math:`y_j`'s, then `softmin(eps, C_xy, g)` must return a dual potential
            `f` for ":math:`f_i`", supported by the :math:`x_i`'s, that is equal to:

            .. math::
                f_i \gets - \varepsilon \log \sum_{j=1}^{\text{M}} \exp
                \big[ g_j - C(x_i, y_j) / \varepsilon \big]~.

            For more detail, see e.g. Section 3.3 and Eq. (3.186) in Jean Feydy's PhD thesis.

        a_logs (list of Tensors): List of log-weights :math:`\log(\alpha_i)`
            for the first input measure at different resolutions.

        b_logs (list of Tensors): List of log-weights :math:`\log(\beta_i)`
            for the second input measure at different resolutions.

        C_xxs (list): List of objects that encode the cost matrices
            :math:`C(x_i, x_j)` between the samples of the first input
            measure at different scales.
            These will be passed to the `softmin` function as second arguments.

        C_yys (list): List of objects that encode the cost matrices
            :math:`C(y_i, y_j)` between the samples of the second input
            measure at different scales.
            These will be passed to the `softmin` function as second arguments.

        C_xys (list): List of objects that encode the cost matrices
            :math:`C(x_i, y_j)` between the samples of the first and second input
            measures at different scales.
            These will be passed to the `softmin` function as second arguments.

        C_yxs (list): List of objects that encode the cost matrices
            :math:`C(y_i, x_j)` between the samples of the second and first input
            measures at different scales.
            These will be passed to the `softmin` function as second arguments.

        eps_list (list of float): List of successive values for the temperature
            :math:`\varepsilon`. The number of iterations in the loop
            is equal to the length of this list.

        rho (float or None): Strength of the marginal constraints for unbalanced OT.
            None stands for :math:`\rho = +\infty`, i.e. balanced OT.

        jumps (list, optional): List of iteration numbers where we "jump"
            from a coarse resolution to a finer one by looking
            one step further in the lists `a_logs`, `b_logs`, `C_xxs`, etc.
            Count starts at iteration 0.
            Defaults to [] - single-scale mode without jumps.

        kernel_truncation (function, optional): Implements the kernel truncation trick.
            Defaults to None.

        truncate (int, optional): Optional argument for `kernel_truncation`.
            Defaults to 5.

        cost (string or function, optional): Optional argument for `kernel_truncation`.
            Defaults to None.

        extrapolate (function, optional): Function.
            If
            `f_ba` is a dual potential that is supported by the :math:`x_i`'s,
            `g_ab` is a dual potential that is supported by the :math:`y_j`'s,
            `eps` is the current value of the temperature :math:`\varepsilon`,
            `damping` is the current value of the damping coefficient for unbalanced OT,
            `C_xy` encodes the cost matrix :math:`C(x_i, y_j)` at the current
            ("coarse") resolution,
            `b_log` denotes the log-weights :math:`\log(\beta_j)`
            that are supported by the :math:`y_j`'s at the coarse resolution,
            and
            `C_xy_fine` encodes the cost matrix :math:`C(x_i, y_j)` at the next
            ("fine") resolution,
            then
            `extrapolate(f_ba, g_ab, eps, damping, C_xy, b_log, C_xy_fine)`
            will be used to compute the new values of the dual potential
            `f_ba` on the point cloud :math:`x_i` at a finer resolution.
            Defaults to None - it is not needed in single-scale mode.

        debias (bool, optional): Should we used the "de-biased" Sinkhorn divergence
            :math:`\text{S}_{\varepsilon, \rho}(\al,\be)` instead
            of the "raw" entropic OT cost
            :math:`\text{OT}_{\varepsilon, \rho}(\al,\be)`?
            This slows down the OT solver but guarantees that our approximation
            of the Wasserstein distance will be positive and definite
            - up to convergence of the Sinkhorn loop.
            For a detailed discussion of the influence of this parameter,
            see e.g. Fig. 3.21 in Jean Feydy's PhD thesis.
            Defaults to True.

        last_extrapolation (bool, optional): Should we perform a last, "full"
            Sinkhorn iteration before returning the dual potentials?
            This allows us to retrieve correct gradients without having
            to backpropagate trough the full Sinkhorn loop.
            Defaults to True.

    Returns:
        4-uple of Tensors: The four optimal dual potentials
            `(f_aa, g_bb, g_ab, f_ba)` that are respectively
            supported by the first, second, second and first input measures
            and associated to the "a <-> a", "b <-> b",
            "a <-> b" and "a <-> b" optimal transport problems.
    """

    # Number of iterations, specified by our epsilon-schedule
    Nits = len(eps_list)

    # The multiscale algorithm may loop over several representations
    # of the input measures.
    # In this routine, the convention is that "myvars" denotes
    # the list of "myvar" across different scales.
    if type(a_logs) is not list:
        # The "single-scale" use case is simply encoded
        # using lists of length 1.

        # Logarithms of the weights:
        a_logs, b_logs = [a_logs], [b_logs]

        # Cost "matrices" C(x_i, y_j) and C(y_i, x_j):
        C_xys, C_yxs = [C_xys], [C_yxs]  # Used for the "a <-> b" problem.

        # Cost "matrices" C(x_i, x_j) and C(y_i, y_j):
        if debias:  # Only used for the "a <-> a" and "b <-> b" problems.
            C_xxs, C_yys = [C_xxs], [C_yys]

    # N.B.: We don't let users backprop through the Sinkhorn iterations
    #       and branch instead on an explicit formula "at convergence"
    #       using some "advanced" PyTorch syntax at the end of the loop.
    #       This acceleration "trick" relies on the "envelope theorem":
    #       it works very well if users are only interested in the gradient
    #       of the Sinkhorn loss, but may not produce correct results
    #       if one attempts to compute order-2 derivatives,
    #       or differentiate "non-standard" quantities that
    #       are defined using the optimal dual potentials.
    #
    #       We may wish to alter this behaviour in the future.
    #       For reference on the question, see Eq. (3.226-227) in
    #       Jean Feydy's PhD thesis and e.g.
    #       "Super-efficiency of automatic differentiation for
    #       functions defined as a minimum", Ablin, Peyré, Moreau (2020)
    #       https://arxiv.org/pdf/2002.03722.pdf.
    torch.autograd.set_grad_enabled(False)

    # Line 1 (in Algorithm 3.6 from Jean Feydy's PhD thesis) ---------------------------

    # We start at the coarsest resolution available:
    k = 0  # Scale index
    eps = eps_list[k]  # First value of the temperature (typically, = diameter**p)

    # Damping factor: equal to 1 for balanced OT,
    # < 1 for unbalanced OT with KL penalty on the marginal constraints.
    # For reference, see Table 1 in "Sinkhorn divergences for unbalanced
    # optimal transport", Sejourne et al., https://arxiv.org/abs/1910.12958.
    damping0 = dampening(eps, rho[0])
    damping1 = dampening(eps, rho[1])

    # Load the measures and cost matrices at the current scale:
    a_log, b_log = a_logs[k], b_logs[k]
    C_xy, C_yx = C_xys[k], C_yxs[k]  # C(x_i, y_j), C(y_i, x_j)
    if debias:  # Info for the "a <-> a" and "b <-> b" problems
        C_xx, C_yy = C_xxs[k], C_yys[k]  # C(x_i, x_j), C(y_j, y_j)

    # Line 2 ---------------------------------------------------------------------------
    # Start with a decent initialization for the dual vectors:
    # N.B.: eps is really large here, so the log-sum-exp behaves as a sum
    #       and the softmin is basically
    #       a convolution with the cost function (i.e. the limit for eps=+infty).
    #       The algorithm was originally written with this convolution
    #       - but in this implementation, we use "softmin" for the sake of simplicity.
    g_ab = damping0 * softmin(eps, C_yx, a_log)  # a -> b
    f_ba = damping1 * softmin(eps, C_xy, b_log)  # b -> a
    if debias:
        f_aa = damping0 * softmin(eps, C_xx, a_log)  # a -> a
        g_bb = damping1 * softmin(eps, C_yy, b_log)  # a -> a

    # Lines 4-5: eps-scaling descent ---------------------------------------------------
    for i, eps in enumerate(eps_list):  # See Fig. 3.25-26 in Jean Feydy's PhD thesis.

        # Line 6: update the damping coefficient ---------------------------------------
        #damping = dampening(eps, rho)  # eps and damping change across iterations
        damping0 = dampening(eps, rho[0])
        damping1 = dampening(eps, rho[1])
        # Line 7: "coordinate ascent" on the dual problems -----------------------------
        # N.B.: As discussed in Section 3.3.3 of Jean Feydy's PhD thesis,
        #       we perform "symmetric" instead of "alternate" updates
        #       of the dual potentials "f" and "g".
        #       To this end, we first create buffers "ft", "gt"
        #       (for "f-tilde", "g-tilde") using the standard
        #       Sinkhorn formulas, and update both dual vectors
        #       simultaneously.
        ft_ba = damping0 * softmin(eps, C_xy, b_log + g_ab / eps)  # b -> a
        gt_ab = damping1 * softmin(eps, C_yx, a_log + f_ba / eps)  # a -> b

        # See Fig. 3.21 in Jean Feydy's PhD thesis to see the importance
        # of debiasing when the target "blur" or "eps**(1/p)" value is larger
        # than the average distance between samples x_i, y_j and their neighbours.
        if debias:
            ft_aa = damping0 * softmin(eps, C_xx, a_log + f_aa / eps)  # a -> a
            gt_bb = damping1 * softmin(eps, C_yy, b_log + g_bb / eps)  # b -> b

        # Symmetrized updates - see Fig. 3.24.b in Jean Feydy's PhD thesis:
        f_ba, g_ab = 0.5 * (f_ba + ft_ba), 0.5 * (g_ab + gt_ab)  # OT(a,b) wrt. a, b
        if debias:
            f_aa, g_bb = 0.5 * (f_aa + ft_aa), 0.5 * (g_bb + gt_bb)  # OT(a,a), OT(b,b)

        # Line 8: jump from a coarse to a finer scale ----------------------------------
        # In multi-scale mode, we work we increasingly detailed representations
        # of the input measures: this type of strategy is known as "multi-scale"
        # in computer graphics, "multi-grid" in numerical analysis,
        # "coarse-to-fine" in signal processing or "divide and conquer"
        # in standard complexity theory (e.g. for the quick-sort algorithm).
        #
        # In the Sinkhorn loop with epsilon-scaling annealing, our
        # representations of the input measures are fine enough to ensure
        # that the typical distance between any two samples x_i, y_j is always smaller
        # than the current value of "blur = eps**(1/p)".
        # As illustrated in Fig. 3.26 of Jean Feydy's PhD thesis, this allows us
        # to reach a satisfying level of precision while speeding up the computation
        # of the Sinkhorn iterations in the first few steps.
        #
        # In practice, different multi-scale representations of the input measures
        # are generated by the "parent" code of this solver and stored in the
        # lists a_logs, b_logs, C_xxs, etc.
        #
        # The switch between different scales is specified by the list of "jump" indices,
        # that is generated in conjunction with the list of temperatures "eps_list".
        #
        # N.B.: In single-scale mode, jumps = []: the code below is never executed
        #       and we retrieve "Algorithm 3.5" from Jean Feydy's PhD thesis.
        if i in jumps:

            if i == len(eps_list) - 1:  # Last iteration: just extrapolate!

                C_xy_fine, C_yx_fine = C_xys[k + 1], C_yxs[k + 1]
                if debias:
                    C_xx_fine, C_yy_fine = C_xxs[k + 1], C_yys[k + 1]

                last_extrapolation = False  # No need to re-extrapolate after the loop
                torch.autograd.set_grad_enabled(True)

            else:  # It's worth investing some time on kernel truncation...
                # The lines below implement the Kernel truncation trick,
                # described in Eq. (3.222-3.224) in Jean Feydy's PhD thesis and in
                # "Stabilized sparse scaling algorithms for entropy regularized transport
                #  problems", Schmitzer (2016-2019), (https://arxiv.org/pdf/1610.06519.pdf).
                #
                # A more principled and "controlled" variant is also described in
                # "Capacity constrained entropic optimal transport, Sinkhorn saturated
                #  domain out-summation and vanishing temperature", Benamou and Martinet
                #  (2020), (https://hal.archives-ouvertes.fr/hal-02563022/).
                #
                # On point clouds, this code relies on KeOps' block-sparse routines.
                # On grids, it is a "dummy" call: we do not perform any "truncation"
                # and rely instead on the separability of the Gaussian convolution kernel.

                # Line 9: a <-> b ------------------------------------------------------
                C_xy_fine, C_yx_fine = kernel_truncation(
                    C_xy,
                    C_yx,
                    C_xys[k + 1],
                    C_yxs[k + 1],
                    f_ba,
                    g_ab,
                    eps,
                    truncate=truncate,
                    cost=cost,
                )

                if debias:
                    # Line 10: a <-> a  ------------------------------------------------
                    C_xx_fine, _ = kernel_truncation(
                        C_xx,
                        C_xx,
                        C_xxs[k + 1],
                        C_xxs[k + 1],
                        f_aa,
                        f_aa,
                        eps,
                        truncate=truncate,
                        cost=cost,
                    )
                    # Line 11: b <-> b -------------------------------------------------
                    C_yy_fine, _ = kernel_truncation(
                        C_yy,
                        C_yy,
                        C_yys[k + 1],
                        C_yys[k + 1],
                        g_bb,
                        g_bb,
                        eps,
                        truncate=truncate,
                        cost=cost,
                    )

            # Line 12: extrapolation step ----------------------------------------------
            # We extra/inter-polate the values of the dual potentials from
            # the "coarse" to the "fine" resolution.
            #
            # On point clouds, we use the expressions of the dual potentials
            # detailed e.g. in Eqs. (3.194-3.195) of Jean Feydy's PhD thesis.
            # On images and volumes, we simply rely on (bi/tri-)linear interpolation.
            #
            # N.B.: the cross-updates below *must* be done in parallel!
            f_ba, g_ab = (
                extrapolate(f_ba, g_ab, eps, damping0, C_xy, b_log, C_xy_fine),
                extrapolate(g_ab, f_ba, eps, damping1, C_yx, a_log, C_yx_fine),
            )

            # Extrapolation for the symmetric problems:
            if debias:
                f_aa = extrapolate(f_aa, f_aa, eps, damping0, C_xx, a_log, C_xx_fine)
                g_bb = extrapolate(g_bb, g_bb, eps, damping1, C_yy, b_log, C_yy_fine)

            # Line 13: update the measure weights and cost "matrices" ------------------
            k = k + 1
            a_log, b_log = a_logs[k], b_logs[k]
            C_xy, C_yx = C_xy_fine, C_yx_fine
            if debias:
                C_xx, C_yy = C_xx_fine, C_yy_fine

    # As a very last step, we perform a final "Sinkhorn" iteration.
    # As detailed above (around "torch.autograd.set_grad_enabled(False)"),
    # this allows us to retrieve correct expressions for the gradient
    # without having to backprop through the whole Sinkhorn loop.
    torch.autograd.set_grad_enabled(True)

    if last_extrapolation:
        # The cross-updates should be done in parallel!
        f_ba, g_ab = (
            damping0 * softmin(eps, C_xy, (b_log + g_ab / eps).detach()),
            damping1 * softmin(eps, C_yx, (a_log + f_ba / eps).detach()),
        )

        if debias:
            f_aa = damping0 * softmin(eps, C_xx, (a_log + f_aa / eps).detach())
            g_bb = damping1 * softmin(eps, C_yy, (b_log + g_bb / eps).detach())

    if debias:
        return f_aa, g_bb, g_ab, f_ba
    else:
        return None, None, g_ab, f_ba

# Redefinition of geomloss.samples_loss.SamplesLoss for the semi-unbalanced case.
# The only difference is the application of the functions defined for the
# semi-unbalanced case. In particular, as defined in the routines, we use
# 'sinkhorn_tensorized_su' and 'sinkhorn_online_su' in ll. 792-793.
routines = {
    "sinkhorn": {
        "tensorized": sinkhorn_tensorized_su,
        "online": sinkhorn_online_su,
        "multiscale": sinkhorn_multiscale,
    },
    "hausdorff": {
        "tensorized": hausdorff_tensorized,
        "online": hausdorff_online,
        "multiscale": hausdorff_multiscale,
    },
    "energy": {
        "tensorized": partial(kernel_tensorized, name="energy"),
        "online": partial(kernel_online, name="energy"),
        "multiscale": partial(kernel_multiscale, name="energy"),
    },
    "gaussian": {
        "tensorized": partial(kernel_tensorized, name="gaussian"),
        "online": partial(kernel_online, name="gaussian"),
        "multiscale": partial(kernel_multiscale, name="gaussian"),
    },
    "laplacian": {
        "tensorized": partial(kernel_tensorized, name="laplacian"),
        "online": partial(kernel_online, name="laplacian"),
        "multiscale": partial(kernel_multiscale, name="laplacian"),
    },
}
class SamplesLoss(Module):
    """Creates a criterion that computes distances between sampled measures on a vector space.

    Warning:
        If **loss** is ``"sinkhorn"`` and **reach** is **None** (balanced Optimal Transport),
        the resulting routine will expect measures whose total masses are equal with each other.

    Parameters:
        loss (string, default = ``"sinkhorn"``): The loss function to compute.
            The supported values are:

              - ``"sinkhorn"``: (Un-biased) Sinkhorn divergence, which interpolates
                between Wasserstein (blur=0) and kernel (blur= :math:`+\infty` ) distances.
              - ``"hausdorff"``: Weighted Hausdorff distance, which interpolates
                between the ICP loss (blur=0) and a kernel distance (blur= :math:`+\infty` ).
              - ``"energy"``: Energy Distance MMD, computed using the kernel
                :math:`k(x,y) = -\|x-y\|_2`.
              - ``"gaussian"``: Gaussian MMD, computed using the kernel
                :math:`k(x,y) = \exp \\big( -\|x-y\|_2^2 \,/\, 2\sigma^2)`
                of standard deviation :math:`\sigma` = **blur**.
              - ``"laplacian"``: Laplacian MMD, computed using the kernel
                :math:`k(x,y) = \exp \\big( -\|x-y\|_2 \,/\, \sigma)`
                of standard deviation :math:`\sigma` = **blur**.

        p (int, default=2): If **loss** is ``"sinkhorn"`` or ``"hausdorff"``,
            specifies the ground cost function between points.
            The supported values are:

              - **p** = 1: :math:`~~C(x,y) ~=~ \|x-y\|_2`.
              - **p** = 2: :math:`~~C(x,y) ~=~ \\tfrac{1}{2}\|x-y\|_2^2`.

        blur (float, default=.05): The finest level of detail that
            should be handled by the loss function - in
            order to prevent overfitting on the samples' locations.

            - If **loss** is ``"gaussian"`` or ``"laplacian"``,
              it is the standard deviation :math:`\sigma` of the convolution kernel.
            - If **loss** is ``"sinkhorn"`` or ``"hausdorff"``,
              it is the typical scale :math:`\sigma` associated
              to the temperature :math:`\\varepsilon = \sigma^p`.
              The default value of .05 is sensible for input
              measures that lie in the unit square/cube.

            Note that the *Energy Distance* is scale-equivariant, and won't
            be affected by this parameter.

        reach (float, default=None= :math:`+\infty` ): If **loss** is ``"sinkhorn"``
            or ``"hausdorff"``,
            specifies the typical scale :math:`\\tau` associated
            to the constraint strength :math:`\\rho = \\tau^p`.

        diameter (float, default=None): A rough indication of the maximum
            distance between points, which is used to tune the :math:`\\varepsilon`-scaling
            descent and provide a default heuristic for clustering **multiscale** schemes.
            If **None**, a conservative estimate will be computed on-the-fly.

        scaling (float, default=.5): If **loss** is ``"sinkhorn"``,
            specifies the ratio between successive values
            of :math:`\sigma=\\varepsilon^{1/p}` in the
            :math:`\\varepsilon`-scaling descent.
            This parameter allows you to specify the trade-off between
            speed (**scaling** < .4) and accuracy (**scaling** > .9).

        truncate (float, default=None= :math:`+\infty`): If **backend**
            is ``"multiscale"``, specifies the effective support of
            a Gaussian/Laplacian kernel as a multiple of its standard deviation.
            If **truncate** is not **None**, kernel truncation
            steps will assume that
            :math:`\\exp(-x/\sigma)` or
            :math:`\\exp(-x^2/2\sigma^2) are zero when
            :math:`\|x\| \,>\, \\text{truncate}\cdot \sigma`.


        cost (function or string, default=None): if **loss** is ``"sinkhorn"``
            or ``"hausdorff"``, specifies the cost function that should
            be used instead of :math:`\\tfrac{1}{p}\|x-y\|^p`:

            - If **backend** is ``"tensorized"``, **cost** should be a
              python function that takes as input a
              (B,N,D) torch Tensor **x**, a (B,M,D) torch Tensor **y**
              and returns a batched Cost matrix as a (B,N,M) Tensor.
            - Otherwise, if **backend** is ``"online"`` or ``"multiscale"``,
              **cost** should be a `KeOps formula <http://www.kernel-operations.io/api/math-operations.html>`_,
              given as a string, with variables ``X`` and ``Y``.
              The default values are ``"Norm2(X-Y)"`` (for **p** = 1) and
              ``"(SqDist(X,Y) / IntCst(2))"`` (for **p** = 2).

        cluster_scale (float, default=None): If **backend** is ``"multiscale"``,
            specifies the coarse scale at which cluster centroids will be computed.
            If **None**, a conservative estimate will be computed from
            **diameter** and the ambient space's dimension,
            making sure that memory overflows won't take place.

        debias (bool, default=True): If **loss** is ``"sinkhorn"``,
            specifies if we should compute the **unbiased**
            Sinkhorn divergence instead of the classic,
            entropy-regularized "SoftAssign" loss.

        potentials (bool, default=False): When this parameter is set to True,
            the :mod:`SamplesLoss` layer returns a pair of optimal dual potentials
            :math:`F` and :math:`G`, sampled on the input measures,
            instead of differentiable scalar value.
            These dual vectors :math:`(F(x_i))` and :math:`(G(y_j))`
            are encoded as Torch tensors, with the same shape
            as the input weights :math:`(\\alpha_i)` and :math:`(\\beta_j)`.

        verbose (bool, default=False): If **backend** is ``"multiscale"``,
            specifies whether information on the clustering and
            :math:`\\varepsilon`-scaling descent should be displayed
            in the standard output.

        backend (string, default = ``"auto"``): The implementation that
            will be used in the background; this choice has a major impact
            on performance. The supported values are:

              - ``"auto"``: Choose automatically, using a simple
                heuristic based on the inputs' shapes.
              - ``"tensorized"``: Relies on a full cost/kernel matrix, computed
                once and for all and stored on the device memory.
                This method is fast, but has a quadratic
                memory footprint and does not scale beyond ~5,000 samples per measure.
              - ``"online"``: Computes cost/kernel values on-the-fly, leveraging
                online map-reduce CUDA routines provided by
                the `pykeops <https://www.kernel-operations.io>`_ library.
              - ``"multiscale"``: Fast implementation that scales to millions
                of samples in dimension 1-2-3, relying on the block-sparse
                reductions provided by the `pykeops <https://www.kernel-operations.io>`_ library.

    """

    def __init__(
        self,
        loss="sinkhorn",
        p=2,
        blur=0.05,
        reach=None,
        diameter=None,
        scaling=0.5,
        truncate=5,
        cost=None,
        kernel=None,
        cluster_scale=None,
        debias=True,
        potentials=False,
        verbose=False,
        backend="auto",
    ):

        super(SamplesLoss, self).__init__()
        self.loss = loss
        self.backend = backend
        self.p = p
        self.blur = blur
        self.reach = reach
        self.truncate = truncate
        self.diameter = diameter
        self.scaling = scaling
        self.cost = cost
        self.kernel = kernel
        self.cluster_scale = cluster_scale
        self.debias = debias
        self.potentials = potentials
        self.verbose = verbose

    def forward(self, *args):
        """Computes the loss between sampled measures.

        Documentation and examples: Soon!
        Until then, please check the tutorials :-)"""

        l_x, α, x, l_y, β, y = self.process_args(*args)
        B, N, M, D, l_x, α, l_y, β = self.check_shapes(l_x, α, x, l_y, β, y)

        backend = (
            self.backend
        )  # Choose the backend -----------------------------------------
        if l_x is not None or l_y is not None:
            if backend in ["auto", "multiscale"]:
                backend = "multiscale"
            else:
                raise ValueError(
                    'Explicit cluster labels are only supported with the "auto" and "multiscale" backends.'
                )

        elif backend == "auto":
            if M * N <= 5000 ** 2:
                backend = (
                    "tensorized"  # Fast backend, with a quadratic memory footprint
                )
            else:
                if (
                    D <= 3
                    and self.loss == "sinkhorn"
                    and M * N > 10000 ** 2
                    and self.p == 2
                ):
                    backend = "multiscale"  # Super scalable algorithm in low dimension
                else:
                    backend = "online"  # Play it safe, without kernel truncation

        # Check compatibility between the batchsize and the backend --------------------------

        if backend in ["multiscale"]:  # multiscale routines work on single measures
            if B == 1:
                α, x, β, y = α.squeeze(0), x.squeeze(0), β.squeeze(0), y.squeeze(0)
            elif B > 1:
                warnings.warn(
                    "The 'multiscale' backend do not support batchsize > 1. "
                    + "Using 'tensorized' instead: beware of memory overflows!"
                )
                backend = "tensorized"

        if B == 0 and backend in [
            "tensorized",
            "online",
        ]:  # tensorized and online routines work on batched tensors
            α, x, β, y = α.unsqueeze(0), x.unsqueeze(0), β.unsqueeze(0), y.unsqueeze(0)
        

        # Run --------------------------------------------------------------------------------
        values = routines[self.loss][backend](
            α,
            x,
            β,
            y,
            p=self.p,
            blur=self.blur,
            reach=self.reach,
            diameter=self.diameter,
            scaling=self.scaling,
            truncate=self.truncate,
            cost=self.cost,
            kernel=self.kernel,
            cluster_scale=self.cluster_scale,
            debias=self.debias,
            potentials=self.potentials,
            labels_x=l_x,
            labels_y=l_y,
            verbose=self.verbose,
        )
        
        # Make sure that the output has the correct shape ------------------------------------
        if (
            self.potentials
        ):  # Return some dual potentials (= test functions) sampled on the input measures
            F, G = values
            return F.view_as(α), G.view_as(β)

        else:  # Return a scalar cost value
            if backend in ["multiscale"]:  # KeOps backends return a single scalar value
                if B == 0:
                    return values  # The user expects a scalar value
                else:
                    return values.view(
                        -1
                    )  # The user expects a "batch list" of distances

            else:  # "tensorized" backend returns a "batch vector" of values
                if B == 0:
                    return values[0]  # The user expects a scalar value
                else:
                    return values  # The user expects a "batch vector" of distances

    def process_args(self, *args):
        if len(args) == 6:
            return args
        if len(args) == 4:
            α, x, β, y = args
            return None, α, x, None, β, y
        elif len(args) == 2:
            x, y = args
            α = self.generate_weights(x)
            β = self.generate_weights(y)
            return None, α, x, None, β, y
        else:
            raise ValueError(
                "A SamplesLoss accepts two (x, y), four (α, x, β, y) or six (l_x, α, x, l_y, β, y)  arguments."
            )

    def generate_weights(self, x):
        if x.dim() == 2:  #
            N = x.shape[0]
            return torch.ones(N).type_as(x) / N
        elif x.dim() == 3:
            B, N, _ = x.shape
            return torch.ones(B, N).type_as(x) / N
        else:
            raise ValueError(
                "Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors."
            )

    def check_shapes(self, l_x, α, x, l_y, β, y):

        if α.dim() != β.dim():
            raise ValueError(
                "Input weights 'α' and 'β' should have the same number of dimensions."
            )
        if x.dim() != y.dim():
            raise ValueError(
                "Input samples 'x' and 'y' should have the same number of dimensions."
            )
        if x.shape[-1] != y.shape[-1]:
            raise ValueError(
                "Input samples 'x' and 'y' should have the same last dimension."
            )

        if (
            x.dim() == 2
        ):  # No batch --------------------------------------------------------------------
            B = 0  # Batchsize
            N, D = x.shape  # Number of "i" samples, dimension of the feature space
            M, _ = y.shape  # Number of "j" samples, dimension of the feature space

            if α.dim() not in [1, 2]:
                raise ValueError(
                    "Without batches, input weights 'α' and 'β' should be encoded as (N,) or (N,1) tensors."
                )
            elif α.dim() == 2:
                if α.shape[1] > 1:
                    raise ValueError(
                        "Without batches, input weights 'α' should be encoded as (N,) or (N,1) tensors."
                    )
                if β.shape[1] > 1:
                    raise ValueError(
                        "Without batches, input weights 'β' should be encoded as (M,) or (M,1) tensors."
                    )
                α, β = α.view(-1), β.view(-1)

            if l_x is not None:
                if l_x.dim() not in [1, 2]:
                    raise ValueError(
                        "Without batches, the vector of labels 'l_x' should be encoded as an (N,) or (N,1) tensor."
                    )
                elif l_x.dim() == 2:
                    if l_x.shape[1] > 1:
                        raise ValueError(
                            "Without batches, the vector of labels 'l_x' should be encoded as (N,) or (N,1) tensors."
                        )
                    l_x = l_x.view(-1)
                if len(l_x) != N:
                    raise ValueError(
                        "The vector of labels 'l_x' should have the same length as the point cloud 'x'."
                    )

            if l_y is not None:
                if l_y.dim() not in [1, 2]:
                    raise ValueError(
                        "Without batches, the vector of labels 'l_y' should be encoded as an (M,) or (M,1) tensor."
                    )
                elif l_y.dim() == 2:
                    if l_y.shape[1] > 1:
                        raise ValueError(
                            "Without batches, the vector of labels 'l_y' should be encoded as (M,) or (M,1) tensors."
                        )
                    l_y = l_y.view(-1)
                if len(l_y) != M:
                    raise ValueError(
                        "The vector of labels 'l_y' should have the same length as the point cloud 'y'."
                    )

            N2, M2 = α.shape[0], β.shape[0]

        elif (
            x.dim() == 3
        ):  # batch computation ---------------------------------------------------------
            (
                B,
                N,
                D,
            ) = x.shape
            # Batchsize, number of "i" samples, dimension of the feature space
            (
                B2,
                M,
                _,
            ) = y.shape
            # Batchsize, number of "j" samples, dimension of the feature space
            if B != B2:
                raise ValueError("Samples 'x' and 'y' should have the same batchsize.")

            if α.dim() not in [2, 3]:
                raise ValueError(
                    "With batches, input weights 'α' and 'β' should be encoded as (B,N) or (B,N,1) tensors."
                )
            elif α.dim() == 3:
                if α.shape[2] > 1:
                    raise ValueError(
                        "With batches, input weights 'α' should be encoded as (B,N) or (B,N,1) tensors."
                    )
                if β.shape[2] > 1:
                    raise ValueError(
                        "With batches, input weights 'β' should be encoded as (B,M) or (B,M,1) tensors."
                    )
                α, β = α.squeeze(-1), β.squeeze(-1)

            if l_x is not None:
                raise NotImplementedError(
                    'The "multiscale" backend has not been implemented with batches.'
                )
            if l_y is not None:
                raise NotImplementedError(
                    'The "multiscale" backend has not been implemented with batches.'
                )

            B2, N2 = α.shape
            B3, M2 = β.shape
            if B != B2:
                raise ValueError(
                    "Samples 'x' and weights 'α' should have the same batchsize."
                )
            if B != B3:
                raise ValueError(
                    "Samples 'y' and weights 'β' should have the same batchsize."
                )

        else:
            raise ValueError(
                "Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors."
            )

        if N != N2:
            raise ValueError(
                "Weights 'α' and samples 'x' should have compatible shapes."
            )
        if M != M2:
            raise ValueError(
                "Weights 'β' and samples 'y' should have compatible shapes."
            )

        return B, N, M, D, l_x, α, l_y, β
