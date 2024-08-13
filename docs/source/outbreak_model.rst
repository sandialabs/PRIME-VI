Outbreak Model
==============
We propose an epidemiological model to forecast infection rates across adjacent geographical regions and use these forecasts to detect emergent outbreaks. 
The model is an extension of previous work by Safta :cite:p:`Safta:2021` and Blonigan :cite:p:`Blonigan:2021` for epidemic forecasts over a single region wave to multi-region 
outbreak detection. In this section we will first describe the single region model and then present statistical approaches to estimate the model parameters over adjacent geographical regions.

The epidemiological model is defined by a spatially varying infection rate model and an incubation model given by

.. math::
   :nowrap:

    \begin{eqnarray}
        f_{inf}(t;t_0^r,k^r,\theta^r) &=& (\theta^r)^{-k}(t-t_0^r)^{k-1}\exp(-(t-t_0^r)/\theta^r)\big/\Gamma(k^r) \\
        F_{inc}(t;\mu,\sigma) &=& \frac{1}{2}\mathrm{erfc}\left(-\frac{\log t-\mu}{\sigma\sqrt{2}}\right)
    \end{eqnarray}

where the infection rate :math:`f_{inf}` is a Gamma distribution with shape and scale parameters :math:`k^r` and :math:`\theta^r`, respectively. 
The parameter :math:`t_0^r` represents the start of the outbreak and will be inferred along with the infection rate paramters. Note that :math:`1 \leq r \leq R` indexes the spatial 
region. The number of people that turn symptomatic over the time interval :math:`[t_{i-1},t_i]` is given by

.. math::
   :nowrap:

    \begin{eqnarray}
        y_r(i;t_0^r, N^r, k^r,\theta^r)  &=&N_r\int_{t_0^r}^{t_i} f_{inf}(\tau-t_0^r;k^r,\theta^r)
        (F_{inc}(t_i-\tau;\mu,\sigma)-(F_{inc}(t_{i-1}-\tau;\mu,\sigma))d\,\tau
        \label{eq:model-pred}
    \end{eqnarray}

so that :math:`\mathbf{y}(i)= [y_1(i) \cdots y_{R}(i)]^T = \mathbf{y}_i` and :math:`i` represents the time-dependence of the predictions.
Here, :math:`N^r` is the fourth and final region-dependent parameter and represents the total number of people infected during the entire 
epidemic wave in spatial region :math:`r` normalized by the population of region :math:`r`.