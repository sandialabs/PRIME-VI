Appendix: mathematical details
==============================

Score gradients
---------------

We briefly review the score estimator, or black-box, approach to estimating the gradient of the ELBO. Recall that the ELBO is given by \ref{eq:ELBO} which, 
for the sake of clarity, can be written in a more generic form

.. math::
   :nowrap:

    \begin{equation}
        \mathcal{L}(\boldsymbol{\phi}) = \mathbb{E}_{q(\boldsymbol{\theta};\boldsymbol{\phi})} \left[ f(\boldsymbol{\theta}) \right] 
    \end{equation}

where $f(\boldsymbol{\theta})$ encapsulates the dependence on the random vector $\boldsymbol{\theta}$. To derive an estimator for the gradient, 
we can carry out the following manipulations

.. math::
   :nowrap:

    \begin{align}
        \nabla_{\boldsymbol{\phi}} \mathcal{L}(\boldsymbol{\phi}) &= \nabla_{\boldsymbol{\phi}} \mathbb{E}_{q(\boldsymbol{\theta};\boldsymbol{\phi})} \left[ f(\boldsymbol{\theta}) \right] \\
        &= \nabla_{\boldsymbol{\phi}} \int q(\boldsymbol{\theta};\boldsymbol{\phi}) f(\boldsymbol{\theta})  d\,\boldsymbol{\theta} \\
        &=  \int \nabla_{\boldsymbol{\phi}} q(\boldsymbol{\theta};\boldsymbol{\phi}) f(\boldsymbol{\theta})  d\,\boldsymbol{\theta} \\
        &= \int q(\boldsymbol{\theta};\boldsymbol{\phi}) \frac{\nabla_{\boldsymbol{\phi}} q(\boldsymbol{\theta};\boldsymbol{\phi})}{q(\boldsymbol{\theta};\boldsymbol{\phi})}f(\boldsymbol{\theta}) d\,\boldsymbol{\theta} \\
        &= \mathbb{E}_{q(\boldsymbol{\theta};\boldsymbol{\phi})} \left[ f(\boldsymbol{\theta}) \nabla_{\boldsymbol{\phi}} \log q(\boldsymbol{\theta};\boldsymbol{\phi}) \right]
    \end{align}
Hence, the gradient can be expressed as an expectation with respect to :math:`q(\boldsymbol{\theta};\boldsymbol{\phi})` where only the 
log of the surrogate posterior needs to be differentiated with respect to the variational parameters :math:`\boldsymbol{\phi}`.

Reparametrization gradients
---------------------------

The likelihood and log likelihood are given by

.. math::
   :nowrap:

    \begin{align}
        p(\mathcal{D} \vert \boldsymbol{\theta}) &= \prod_{i=1}^{N_d} 2\pi^{-N_r / 2} \det(\mathbf{\Sigma}_i)^{-1/2} \exp \left( -\frac{1}{2} (\mathbf{y}_i^{(o)} - \mathbf{y}_i)^T \mathbf{\Sigma}_i^{-1} (\mathbf{y}_i^{(o)} - \mathbf{y}_i) \right) \\
        l(\boldsymbol{\theta}) &= -\frac{N_d N_r 2 \pi}{2} -\frac{1}{2} \sum_{i=1}^{N_d} \log \det(\mathbf{\Sigma}_i) +(\mathbf{y}_i^{(o)} - \mathbf{y}_i)^T \mathbf{\Sigma}_i^{-1} (\mathbf{y}_i^{(o)} - \mathbf{y}_i)
    \end{align}

Using the reparametrization trick, we can write the ELBO \eqref{eq:ELBO} and its gradient in the form

.. math::
   :nowrap:

    \begin{align}
        \mathcal{L}(\boldsymbol{\phi}) &= -\mathbb{H}[q(\boldsymbol{\theta};\boldsymbol{\phi})] - \mathbb{E}_{q(\boldsymbol{\epsilon}) } [ \log p(\mathcal{D} \vert \boldsymbol{\theta}(\boldsymbol{\epsilon},\boldsymbol{\phi})) + \log p(\boldsymbol{\theta}(\boldsymbol{\epsilon},\boldsymbol{\phi}))] \\
        \nabla_{\boldsymbol{\phi}} \mathcal{L}(\boldsymbol{\phi}) &= - \nabla_{\boldsymbol{\phi}} \mathbb{H}[q(\boldsymbol{\theta};\boldsymbol{\phi})] - \mathbb{E}_{q(\boldsymbol{\epsilon}) } [\nabla_{\boldsymbol{\phi}}  \log p(\mathcal{D} \vert \boldsymbol{\theta}(\boldsymbol{\epsilon},\boldsymbol{\phi})) + \nabla_{\boldsymbol{\phi}} \log p(\boldsymbol{\theta}(\boldsymbol{\epsilon},\boldsymbol{\phi}))]
    \end{align}

where :math:`\boldsymbol{\phi} = (\boldsymbol{\mu},\boldsymbol{\rho})`, :math:`\boldsymbol{\theta} = \boldsymbol{\mu} + \boldsymbol{\sigma}(\boldsymbol{\rho}) \odot \boldsymbol{\epsilon}` 
with :math:`\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})`. Here, :math:`\sigma` is a positive transformation of the unconstrained variable :math:`\boldsymbol{\rho}` 
to ensure the variance is constrained to be positive. A Monte Carlo estimator of the gradient can then be written as

.. math::
   :nowrap:

    \begin{align}
        \nabla_{\boldsymbol{\phi}} \mathcal{L}(\boldsymbol{\phi}) &\approx - \nabla_{\boldsymbol{\phi}} \mathbb{H}[q(\boldsymbol{\theta};\boldsymbol{\phi})]  -\frac{1}{N_s}\sum_{i=1}^{N_s} \nabla_{\boldsymbol{\phi}}  \log p(\mathcal{D} \vert \boldsymbol{\theta}(\boldsymbol{\epsilon}_i,\boldsymbol{\phi})) + \nabla_{\boldsymbol{\phi}} \log p(\boldsymbol{\theta}(\boldsymbol{\epsilon}_i,\boldsymbol{\phi})) \\
        &= - \nabla_{\boldsymbol{\phi}} \mathbb{H}[q(\boldsymbol{\theta};\boldsymbol{\phi})]  -\frac{1}{N_s}\sum_{i=1}^{N_s} \left( \nabla_{\boldsymbol{\theta}}  \log p(\mathcal{D} \vert \boldsymbol{\theta}(\boldsymbol{\epsilon}_i,\boldsymbol{\phi}))  + \nabla_{\boldsymbol{\theta}} \log p(\boldsymbol{\theta}(\boldsymbol{\epsilon}_i,\boldsymbol{\phi})) \right) \odot \nabla_{\boldsymbol{\phi}} \boldsymbol{\theta}(\boldsymbol{\epsilon}_i,\boldsymbol{\phi})
    \end{align}

where the last line is given by the chain rule and the fact that :math:`\boldsymbol{\theta}` is defined by an element-wise transformation of :math:`\boldsymbol{\phi}`. Observe that

.. math::
   :nowrap:

    \begin{align}
        \nabla_{\boldsymbol{\mu}}  \boldsymbol{\theta} &= \mathbf{1} \\
        \nabla_{\boldsymbol{\rho}} \boldsymbol{\theta} &= \nabla_{\boldsymbol{\rho}} \boldsymbol{\sigma}(\boldsymbol{\rho}) \odot \boldsymbol{\epsilon}_i
        \\
        \nabla_{\boldsymbol{\mu}} \mathbb{H}[q(\boldsymbol{\theta};\boldsymbol{\phi})] &= \mathbf{0} \\
        \nabla_{\boldsymbol{\rho}} \mathbb{H}[q(\boldsymbol{\theta};\boldsymbol{\phi})] &= \frac{1}{\boldsymbol{\sigma}(\boldsymbol{\rho})} \odot \nabla_{\boldsymbol{\rho}} \boldsymbol{\sigma}(\boldsymbol{\rho})
    \end{align}

so that it remains to compute the gradients of the log-likelihood and prior. 