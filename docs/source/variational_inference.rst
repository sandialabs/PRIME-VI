Variational Inference
=====================

We will compare the Bayesian posterior sampled with MCMC with posterior models obtained using VI which recasts approximate 
inference as an optimization problem. In particular, as the exact posterior is intractable, we consider a family of approximating 
densities :math:`\mathcal{F}= \{q(\mathbf{m};\boldsymbol{\phi}) \ \vert \  \boldsymbol{\phi} \in \boldsymbol{\Phi} \subseteq \mathbb{R}^d \}` 
and seek to find a density :math:`q(\mathbf{m};\boldsymbol{\phi}^*)` that minimizes the KL-divergence with respect to the posterior

.. math::
   :nowrap:

    \begin{equation}
        \boldsymbol{\phi}^* = \mathrm{argmin} \hspace{2mm} D_{\mathrm{KL}} \left( q(\mathbf{m};\boldsymbol{\phi}) \lVert p(\mathbf{m} \ \vert \  \mathcal{D} ) \right)
    \end{equation}

This can be re-expressed as minimizing the objective function  :math:`\mathcal{L}(\boldsymbol{\phi})` based on the evidence lower bound (ELBO) :cite:p:`Kingma:2019`

.. math::
   :nowrap:

    \begin{equation}
        \mathcal{L}(\boldsymbol{\phi}) = -\mathbb{H}[q(\mathbf{m};\boldsymbol{\phi})] - \mathbb{E}_{q(\mathbf{m};\boldsymbol{\phi}) } [\log p(\mathcal{D}  \ \vert \  \mathbf{m}) + \log p(\mathbf{m})]
        \label{eq:ELBO}
    \end{equation}

where the first term in Eq.~\eqref{eq:ELBO} is the entropy of the surrogate posterior and the second, data-dependent term is an expectation with 
respect to the surrogate posterior that reflects both the expected data-fit and the prior. Here we take :math:`\mathcal{F}` to be the set of mean-field Gaussian distributions, i.e.,

.. math::
   :nowrap:

    \begin{equation}
        q\left(\mathbf{m};\ \boldsymbol{\phi} \right) = \prod_{i=1}^d q_i\left(\theta_i; \ \mu_i, \sigma_i\right)
        \label{eq:mean_field_Gaussian}
    \end{equation}

where :math:`q_i(\theta_i ; \mu_i,\sigma_i) = \mathcal{N}(\theta_i ; \mu_i,\sigma_i)`, :math:`\boldsymbol{\phi} = (\boldsymbol{\mu},\boldsymbol{\sigma})` 
and we arrive at an optimization problem over :math:`2d` parameters where :math:`d` is the number of parameters defining the Epidemiological model. To 
carry out the above minimization problem, we aim to use a gradient-based iterative scheme as the expectation in Eq.~\eqref{eq:ELBO} cannot 
be evaluated explicitly due to the nonlinearity of the forward model. Furthermore, :math:`\mathcal{L}(\boldsymbol{\phi})` is potentially a 
non-convex objective. Note that the gradient and expectation operators do not commute, i.e., 

.. math::
   :nowrap:

    \begin{equation}
        \nabla_{\boldsymbol{\phi}} \mathbb{E}_{q \left(\mathbf{m};\boldsymbol{\phi} \right) } \left[ \log p \left(\mathcal{D}  \ \vert \  \mathbf{m} \right) p(\mathbf{m}) \right] \neq \mathbb{E}_{q \left(\mathbf{m};\boldsymbol{\phi} \right) } \left[\nabla_{\boldsymbol{\phi}} \log p \left(\mathcal{D}  \ \vert \  \mathbf{m} \right) p \left(\mathbf{m}\right) \right]
    \end{equation}

so some care has to be taken to arrive at a Monte Carlo estimator for the gradient :math:`\nabla_{\boldsymbol{\phi}} \mathcal{L}(\boldsymbol{\phi})`. 
Two widely used approaches are: (a) the Score function estimator, described in \S~\ref{sec:appendix-ELBO-score} (in the Appendix), which 
forms the basis of black box VI and requires only evaluations of the log likelihood, and (b) the reparametrization approach which requires 
gradients of the log likelihood. The score function estimator typically displays much larger variance as seen in Kucukelbir et. al. :cite:p:`Kucukelbir:2017`
where two orders of magnitude more samples were needed to arrive at the same variance as a reparametrization estimator. A similar trend was 
confirmed for the outbreak problem (Fig.~\ref{fig:ELBO_conv}) suggesting that the reparametrization approach would lead to superior scalability. 
Reparametrization proceeds by expressing :math:`\mathbf{m}` as a differentiable transformation :math:`\mathbf{m} = t(\boldsymbol{\epsilon},\boldsymbol{\phi})` 
of a :math:`\boldsymbol{\phi}`-independent random variable :math:`\boldsymbol{\epsilon} \sim q(\boldsymbol{\epsilon})` such 
that :math:`\mathbf{m}(\boldsymbol{\epsilon},\boldsymbol{\phi}) \sim q(\mathbf{m}, \boldsymbol{\phi})`. This allows the gradient to be expressed as

.. math::
   :nowrap:

    \begin{equation}
        \nabla_{\boldsymbol{\phi}} \mathcal{L} \left(\boldsymbol{\phi} \right) = - \nabla_{\boldsymbol{\phi}} \mathbb{H} \left[ q \left(\mathbf{m};\boldsymbol{\phi} \right) \right] - \mathbb{E}_{q\left(\boldsymbol{\epsilon} \right) } \left[\nabla_{\boldsymbol{\phi}}  \log p \left(\mathcal{D}  \ \vert \  \mathbf{m}(\boldsymbol{\epsilon},\boldsymbol{\phi}) \right) + \nabla_{\boldsymbol{\phi}} \log p \left(\mathbf{m} \left(\boldsymbol{\epsilon},\boldsymbol{\phi} \right) \right) \right]
        \label{eq:ELBO-grad}
    \end{equation}

where gradients of the entropy term in Eq.~\eqref{eq:ELBO-grad} are available analytically for the Gaussian surrogate 
posterior and the second term can now be approximated with Monte Carlo given a method to compute the required gradients. 
For many machine learning models, automatic differentiation can be exploited to calculate the gradient of the log-likelihood 
with respect to parameters :math:`\mathbf{m}`. Here, the objective function involves the log of the likelihood (Eq.~\eqref{eq:likelihood}) 
where derivatives of matrix inverses and determinants with respect to parameters are required to compute the gradient. 
Gradients such as these are not available using most automatic differentiation libraries. Instead, matrix calculus and 
quadrature were used to compute the derivatives of the log-likelihood with respect to model predictions :math:`\mathbf{y}^{(p)}_i` and 
to approximate the derivatives of the model predictions with respect to parameters, respectively.

