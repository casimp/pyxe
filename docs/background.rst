Background
==========

The following section provides a broad and basic overview of the key information and concepts used in the construction of the pyXe package. These details are by no means comprehensive. A comprehensive introduction to the topic can be found in the Synchrotron X-ray Diffraction (P. J. Withers) chapter of Practical Residual Stress Measurements by G. S. Shajer [1].

The package was originally developed for use with the I12:JEEP beamline and their EDXD detector. The following section details some information regarding their setup.

I12 and EDXD
------------

*Beamline I12-JEEP (Joint Engineering, Environmental, and Processing) is a high energy X-ray beamline for imaging, diffraction and scattering, which operates at energies of 53-150 keV.*

The I12-JEEP beamline is located at the Diamond Light Source (DLS) in Oxfordshire, UK. Specifications for the beamline have be detailed in the Journal of Synchrontron Radiation [2]. Much of this information is replicated on the DLS's website:

http://www.diamond.ac.uk/Beamlines/Engineering-and-Environment/I12.html

The I12-JEEP beamline can be accessed through two separate experimental hutches. Experimental Hutch 2 (EH2), which is the larger of the hutches, contains the energy dispersive X-ray detector (EDXD). The layout of the detector can be seen below:

.. figure:: EDXD.png
    :figwidth: 400px
    :width: 500px
    :alt: EDXD setup

*The EDXD system. (a) Geometry of the detector, detector slits and sample slits showing the semi-annular arrangement of 23 independent Ge crystals [2].*

..

The detector is comprised of 23 elements spaced in steps of 8.2°, covering an azimuthal range from 0 to 180°. An additional, unused, detector is available in the case that one detector should fail. The data array that is output contains reference to this detector but it is ignored during the analysis. Detailed information about the EDXD setup can be found in the previously noted journal article and on the DLS website:

http://www.diamond.ac.uk/Beamlines/Engineering-and-Environment/I12/detectors/EDXD.html


Strain Calculation
------------------

Strain is calculated from each specified peak individually (i.e. this is not a Reitveld type refinement) although the strain from many individual peaks may be calculated and stored. Strain is calculated against the unstrained inter-planar spacing, :math:`d_0`, such that:

.. math::
    \epsilon = \frac{d_n - d_0}{d_0}

or, more specifically, in terms of the scattering vector, q:

.. math::
    \epsilon = \frac{q_0}{q_n - q_0}.

The unstrained lattice parameter (:math:`d_0`) much either be explicitly given or specified via a NeXus file containing EDXD measurements from an unstrained source.
A consideration of the methods by which to extract unstrained lattice parameters can be found in work by Withers et al [3].


Principal and Shear Strains
---------------------------

The detector, and therefore angle, specific strain values can be further utilised to calculate the principal strains and the angle that the strain element is rotated relative to the principal axis. Knowing this it is then possible to calculate the shear strain. Further to this, these parameters allow for the extraction of strain at any defined angle. This can be more accurate than the equivalent detector specific strain due to the additional information that is effectively leveraged in the calculations.

.. figure:: example_fitmohrs.png
    :figwidth: 900px
    :width: 1000px
    :alt: Strain fit

*(a) An example of the fit made through the strain array corresponding to the 23-element detector array. (b) The corresponding Mohr's circle highlighting both the principal strain and the strain and shear strain at 0° and 90°.*

Stress calculations
-------------------

In a 3D strain state, the normal stresses can be calculated according to the following equation:

.. math:: \sigma_{xx} = \frac{E}{(1 + \mu)(1 - 2\mu)} \left[(1 - \mu)\epsilon_{xx} + \mu(\epsilon_{yy} + \epsilon_{zz})\right].


The EDXD system captures the peak shifts and therefore the strain in 2D (nominally in x and y). The peak shifts and strain information in the orientation along the beam are not computed. Stress cannot be calculated unless additional information is available. One situation in which it is possible to calculate stress is under a plane strain criterion. In this scenario material along one axis (in this case along the beam direction) is under constraint and the strain can be approximated to zero. Ignoring poisson ratio effects, the full strain tensor collapses down to the 2D in-plane state such that:

.. math::  \epsilon_{ij} =
  \begin{pmatrix}  \epsilon_{xx} & \epsilon_{xy} & \epsilon_{xz} \\
  \epsilon_{yx} & \epsilon_{yy} & \epsilon_{yz} \\
  \epsilon_{zx} & \epsilon_{zy} & \epsilon_{zz}
  \end{pmatrix} =
  \begin{pmatrix}  \epsilon_{xx} & \epsilon_{xy}\\
  \epsilon_{yx} & \epsilon_{yy}
  \end{pmatrix}.

This then allows for the convenient calculation of stress:

.. math:: \sigma_{xx} = \frac{E}{(1 + \mu)(1 - 2\mu)} \left[(1 - \mu)\epsilon_{xx} + \mu(\epsilon_{yy})\right]


References
----------
1. Withers, P. (2013). Synchrotron X-ray Diffraction. In - Practical Residual Stress Measurement Methods (pp. 163–194).

2. Drakopoulos, M., Connolley, T., Reinhard, C., Atwood, R., Magdysyuk, O., Vo, N., … Wanelik, K. (2015). I12: the Joint Engineering , Environment and Processing ( JEEP ) beamline at Diamond Light Source. Journal of Synchrotron Radiation, (2015), 828–838. http://doi.org/10.1107/S1600577515003513

3. Withers, P. J., Preuss, M., Steuwer, a., & Pang, J. W. L. (2007). Methods for obtaining the strain-free lattice parameter when using diffraction to determine residual stress. Journal of Applied Crystallography, 40(5), 891–904. http://doi.org/10.1107/S0021889807030269
