1.0 Strain Calculations
=======================

Strain is calculated from each specified peak individually (i.e. this is not a Reitveld type refinement) although the strain from many individual peaks may be calculated and stored.
Strain is calculated against the unstrained inter-planar spacing, :math:`d_0`, such that:

.. math::
    \epsilon = \frac{d_n - d_0}{d_0}

or, more specifically, in terms of the scattering vector, q:

.. math::
    \epsilon = \frac{q_0}{q_n - q_0}.

The unstrained lattice parameter (:math:`d_0`) much either be explicity given or specified via a NeXus file containing EDXD measurements from an unstrained source.
A consideration of the methods by which to extract unstrained lattice parameters can be found in work by Withers et al. (REF).


1.1 In-plane strain state
~~~~~~~~~~~~~~~~~~~~~~~~~

The detector, and therefore angle, specific strain values can be further utilised to fit and extract a full description of the in-plane strain state.
This is beneficial due to the additional information that is then available, notably the principal in-plane strains and shear strain. 
Further to this, these parameters allow for the extraction of strain at any defined angle. 
This can be more accurate than the equivalent detector specific strain due to the additional information that is effectively leveraged in the calculations.

1.2 Plane-strain stress calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a system in which material along one axis is under constraint and the strain can be approximated to zero, the full strain trensor collapses down to the 2D in-plane state.
This then allows for the convenient calculation of stress. It must be emphasised that this is only a valid calculation in material under plane strain conditions.