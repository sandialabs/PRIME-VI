Bayesian Spatial Model
======================
The noisy model predictions are defined as

.. math::
   :nowrap:

    \begin{equation} 
        \mathbf{y}^{(o)}_i = \mathbf{y}^{(p)}_i + \boldsymbol{\epsilon}_i = \mathcal{M}(t_i; \mathbf{m}) + \boldsymbol{\epsilon}_i, \hspace{3mm} \boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma}_i).
        \label{eq:obs_eqn}
    \end{equation}

Here :math:`\mathbf{m} = \mathrm{vec} \left( \mathbf{m}_r \right)`, where :math:`\mathbf{m}_r^T = (t_0^r, N^r, k^r, \theta^r)`,  are the region specific 
model parameters  for :math:`r = 1,\ldots,R`. To account for spatial correlations and heteroscedastic noise seen in case-counts, the noise is 
assumed to be composed of two terms :math:`\boldsymbol{\epsilon}_i = \boldsymbol{\epsilon}_{i,1} + \boldsymbol{\epsilon}_{i,2}` where the first 
is given by a Gaussian Markov Random Field (GMRF) model while the second represents temporally-varying, independent Gaussian noise. 
Letting :math:`\mathcal{D}` and :math:`\Theta` represent the data and parameters, respectively, the likelihood then takes the form

.. math::
   :label: likelihood

    \begin{equation}
        p(\mathcal{D}  \ \vert \  \Theta)=\prod_{i=1}^{N_d}\frac{1}{(2\pi)^{N_r/2}
        \mathrm{det}\mathbf{\Sigma}_i^{1/2}}\exp\left(-\frac{1}{2}(\mathbf{y}^{(o)}_i-\mathbf{y}^{(p)}_i)
        \mathbf{\Sigma}_i^{-1}(\mathbf{y}^{(o)}_i-\mathbf{y}^{(p)}_i)^T\right)
    \end{equation}

where :math:`\mathbf{\Sigma}_i` is given by

.. math::
   :label: noise-model

    \begin{equation}\label{eq:noise-model}
        \mathbf{\Sigma}_i = \tau_{\Phi}
        P^{-1} + \mathrm{diag}\left(\sigma_a+\sigma_m \mathbf{y}^{(p)}_i\right)^2.
    \end{equation}

The first term :math:`P = \left[D-\lambda_{\Phi}\mathbf{W}\right]` in :eq:`noise-model` forms the precision matrix of a GMRF component
of the noise where the strength of correlations induced by adjacent regions is governed by :math:`\lambda_{\Phi}`. The relative topology of 
regions is encoded by :math:`\mathbf{W}`, the county adjacency matrix, defined as

.. math::
   :nowrap:

    \begin{equation}
        w_{kk}=0\,\textrm{and}\,
        w_{kl}=\begin{cases}
        1 & \textrm{if regions $k$ and $l$ are adjacent}\\
        0 & \textrm{otherwise}
        \end{cases}
    \end{equation}

Here, :math:`D = \mathrm{diag}\{g_1,g_2,\ldots,g_{R}\}` where :math:`g_i` is the number of regions adjacent to region :math:`i`.

The second term captures prediction-dependent, uncorrelated noise with additive and multiplicative components governed by :math:`\sigma_a` and :math:`\sigma_b`, respectively. 
The the relative contribution between the correlated GMRF noise and the uncorrelated noise is controlled by :math:`\tau_{\Phi}`.

The set of parameters :math:`\mathbf{m}` defining the likelihood in :eq:`likelihood` is given by 

.. math::
   :nowrap:

    \begin{equation}
        \mathbf{m} = \mathrm{vec}\left(\theta_i\right) = \mathrm{vec}\left(  
        \begin{bmatrix}
        \mathbf{m}_1 & \cdots & \mathbf{m}_{R} & \boldsymbol{\eta}
        \end{bmatrix}
        \right)
    \end{equation}

where :math:`\boldsymbol{\eta} =(\tau_{\Phi}, \lambda_{\Phi}, \sigma_a, \sigma_m)` are the global noise parameters. Inference consists of forming the 
posterior distribution :math:`p\left(\mathbf{m} \ \vert \  \mathcal{D}  \right) = p\left(\mathcal{D}  \ \vert \ \mathbf{m} \right) p\left( \mathbf{m} \right) / p \left(\mathcal{D}  \right)` 
over uncertain parameters :math:`\mathbf{m}`. As the posterior is intractable, we instead look to approximate it using VI. Hence, the following sections 
describe how VI is  formulated carried out to approximate the posterior :math:`p(\mathbf{m} \ \vert \ \mathcal{D} )` for the outbreak model as well as how the 
prior :math:`p(\mathbf{m})` is defined to regularize the inverse problem.