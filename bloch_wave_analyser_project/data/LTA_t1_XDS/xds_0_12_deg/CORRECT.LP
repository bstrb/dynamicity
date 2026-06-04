
 ***** CORRECT ***** (VERSION Jan 19, 2025  BUILT=20251103)  13-May-2026

 INPUT PARAMETER VALUES
 ----------------------
 SPACE_GROUP_NUMBER=  221 as used in the INTEGRATE step
 UNIT_CELL_CONSTANTS=    11.909    11.909    11.909  90.000  90.000  90.000 as used by INTEGRATE
 FRIEDEL'S_LAW=FALSE
 PROFILE_FITTING= TRUE 
 OVERLOAD=    660000     MINPK=  75.00000    WFAC1=  10.0
 INCLUDE_RESOLUTION_RANGE=    20.000     0.600
 NAME_TEMPLATE_OF_DATA_FRAMES=../image/?????.img SMV
 DATA_RANGE=       1     757
 ROTATION_AXIS=  0.147330  0.989086 -0.001486
 OSCILLATION_RANGE=  0.120000
 STARTING_ANGLE=   -45.000     STARTING_FRAME=       1
 X-RAY_WAVELENGTH=  0.019690
 INCIDENT_BEAM_DIRECTION=  0.000014  0.000008  1.000000
 FRACTION_OF_POLARIZATION=   0.500
 POLARIZATION_PLANE_NORMAL=  0.000000  0.000000  0.000000
 AIR=   0.00010
 DETECTOR=UNKNOWN
 SILICON=  0.402235 SENSOR_THICKNESS=  0.000000
 NUMBER OF DETECTOR SEGMENTS   1
 NX=   512  NY=   512    QX=  0.055000  QY=  0.055000
 ORGX=    243.89  ORGY=    249.30
 DETECTOR_DISTANCE=   444.191
 DIRECTION_OF_DETECTOR_X-AXIS=   1.00000   0.00000   0.00000
 DIRECTION_OF_DETECTOR_Y-AXIS=   0.00000   1.00000   0.00000
 BEAM_DIVERGENCE_E.S.D.=   0.014
 REFLECTING_RANGE_E.S.D.=   0.052
 MINIMUM_ZETA=     0.050
 MAXIMUM_ERROR_OF_SPOT_POSITION=   3.0
 MAXIMUM_ERROR_OF_SPINDLE_POSITION=   2.0
 MINIMUM_I/SIGMA=   3.0
 REFLECTIONS/CORRECTION_FACTOR=      50
 SNRC=  0.10 I/sigma for use in scaling factor determination
 STRICT_ABSORPTION_CORRECTION=FALSE 
 CORRECTIONS= DECAY MODULATION ABSORPTION                      
 TEST_RESOLUTION_RANGE=    10.000     5.000     MIN_RFL_Rmeas=    50     MAX_FAC_Rmeas=    2.00
 REFERENCE_DATA_SET=                                                  
 FIT_B-FACTOR_TO_REFERENCE_DATA_SET=FALSE 

   15210 REFLECTIONS ON FILE "INTEGRATE.HKL"
       0 CORRUPTED REFLECTION RECORDS (IGNORED)
       0 MATCHING REFLECTIONS SPECIFIED IN FILE "FILTER.HKL"
       0 REFLECTIONS INCOMPLETE OR OUTSIDE IMAGE RANGE       1 ...     757
       0 OVERLOADED REFLECTIONS (IGNORED)
       0 REFLECTIONS OUTSIDE ACCEPTED RESOLUTION RANGES
                OR TOO CLOSE TO ROTATION AXIS (IGNORED)
   15210 REFLECTIONS ACCEPTED



 ******************************************************************************
                      AUTOMATIC SPACE GROUP ASSIGNMENT
 ******************************************************************************

 XDS adopts the following approach.
 (1) it looks for possible symmetries of the crystal lattice
 (2) it computes a redundancy independent R-factor for all enantiomorphous
     point groups compatible with the observed lattice symmetry.
 (3) it selects the group which explains the intensity data at an acceptable,
     redundancy-independent R-factor (Rmeas, Rrim) using a minimum number of
     unique reflections.

 This approach does not test for the presence of screw axes. Consequently,
 orthorhombic cell axes will be specified in increasing length (following
 conventions), despite the possibility that different assignments for the
 cell axes could become necessary for space groups P222(1) and P2(1)2(1)2
 containing one or two screw axes, respectively.

 The user can always override the automatic decisions by specifying the
 correct space group number and unit cell constants in XDS.INP and repeating
 the CORRECT step of XDS. This provides a simple way to rename orthorhombic
 cell constants if screw axes are present.

 In addition, the user has the option to specify in XDS.INP
 (a) a reference data set or
 (b) a reindexing transformation or
 (c) the three basis vectors (if known from processing a previous data set
 taken at the same crystal orientation in a multi-wavelength experiment).
 These features of XDS are useful for resolving the issue of alternative
 settings of polar or rhombohedral cells (like P4, P6, R3).

 =====> Specifications provided by the user in XDS.INP:
 SPACE_GROUP_NUMBER=  221 (0: unknown symmetry and unit cell)
 UNIT_CELL_CONSTANTS=    11.91    11.91    11.91  90.000  90.000  90.000
 UNIT_CELL_A-AXIS=     7.403    -4.953     7.905 (0 0 0 : unknown)
 UNIT_CELL_B-AXIS=     7.412     9.251    -1.146 (0 0 0 : unknown)
 UNIT_CELL_C-AXIS=    -5.664     5.633     8.833 (0 0 0 : unknown)
 REIDX=  0  0  0  0  0  0  0  0  0  0  0  0 (all 0 : not specified)
 TEST_RESOLUTION_RANGE=   10.00    5.00 (Angstrom) for space group determination
 NUMBER OF ACCEPTED REFLECTIONS FOR CALCULATING Rmeas               22
 NUMBER OF ACCEPTED UNIQUE REFLECTIONS FROM REFERENCE DATA SET       0


 *********** DETERMINATION OF LATTICE CHARACTER AND BRAVAIS LATTICE ***********

 The CHARACTER OF A LATTICE is defined by the metrical parameters of its
 reduced cell as described in the INTERNATIONAL TABLES FOR CRYSTALLOGRAPHY
 Volume A, p. 746 (KLUWER ACADEMIC PUBLISHERS, DORDRECHT/BOSTON/LONDON, 1989).
 Note that more than one lattice character may have the same BRAVAIS LATTICE.

 A lattice character is marked "*" to indicate a lattice consistent with the
 observed locations of the diffraction spots. These marked lattices must have
 low values for the QUALITY OF FIT and their implicated UNIT CELL CONSTANTS
 should not violate the ideal values by more than
 MAX_CELL_AXIS_ERROR=  0.03
 MAX_CELL_ANGLE_ERROR=           2.0 (Degrees)

 The REINDEXING TRANSFORMATION REIDX() consists of 12 integers that relate
 the original indices H,K,L from file INTEGRATE.HKL to the indices H',K',L'
 with respect to the new cell.
    H'=(REIDX(1)*H+REIDX( 2)*K+REIDX( 3)*L)/IDXV+REIDX( 4)
    K'=(REIDX(5)*H+REIDX( 6)*K+REIDX( 7)*L)/IDXV+REIDX( 8)
    L'=(REIDX(9)*H+REIDX(10)*K+REIDX(11)*L)/IDXV+REIDX(12)
 The value of the integer IDXV depends on the lattice type used for
 specifying reflections on file INTEGRATE.HKL; IDXV is
 1 for a primitive,
 2 for a face or body centred,
 3 for a rhombohedral,
 4 for a lattice centred on all faces.
 IDXV is set by XDS and cannot be input by the user.

  LATTICE-  BRAVAIS-   QUALITY  UNIT CELL CONSTANTS (ANGSTROEM & DEGREES)    REINDEXING TRANSFORMATION
 CHARACTER  LATTICE     OF FIT      a      b      c   alpha  beta gamma

 *  31        aP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    0  1  0  0  1  0  0  0  0  0 -1  0
 *  44        aP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    0  1  0  0 -1  0  0  0  0  0  1  0
 *  33        mP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    0  1  0  0 -1  0  0  0  0  0  1  0
 *  34        mP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    0 -1  0  0  0  0 -1  0  1  0  0  0
 *  32        oP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    0  1  0  0 -1  0  0  0  0  0  1  0
 *  35        mP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    1  0  0  0  0 -1  0  0  0  0 -1  0
 *  20        mC          0.0      16.8   16.8   11.9  90.0  90.0  90.0    1  0 -1  0  1  0  1  0  0 -1  0  0
 *  25        mC          0.0      16.8   16.8   11.9  90.0  90.0  90.0   -1  0  1  0  1  0  1  0  0  1  0  0
 *  10        mC          0.0      16.8   16.8   11.9  90.0  90.0  90.0    1  1  0  0 -1  1  0  0  0  0  1  0
 *  13        oC          0.0      16.8   16.8   11.9  90.0  90.0  90.0   -1  1  0  0 -1 -1  0  0  0  0  1  0
 *  14        mC          0.0      16.8   16.8   11.9  90.0  90.0  90.0   -1  1  0  0 -1 -1  0  0  0  0  1  0
 *  11        tP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    0  1  0  0 -1  0  0  0  0  0  1  0
 *  21        tP          0.0      11.9   11.9   11.9  90.0  90.0  90.0   -1  0  0  0  0  0  1  0  0  1  0  0
 *  23        oC          0.0      16.8   16.8   11.9  90.0  90.0  90.0   -1  0  1  0  1  0  1  0  0  1  0  0
 *   2        hR          0.0      16.8   16.8   20.6  90.0  90.0 120.0   -1  1  0  0  0 -1 -1  0 -1 -1  1  0
 *   3        cP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    0  1  0  0 -1  0  0  0  0  0  1  0
 *   4        hR          0.0      16.8   16.8   20.6  90.0  90.0 120.0    1  1  0  0  0 -1  1  0  1 -1 -1  0
     5        cI        166.7      16.8   16.8   16.8  60.0  60.0  60.0    0  1  1  0 -1  1  0  0 -1  0  1  0
    28        mC        250.0      11.9   26.6   11.9  90.0  90.0  63.4    0 -1  0  0  0 -1 -2  0  1  0  0  0
    29        mC        250.0      11.9   26.6   11.9  90.0  90.0  63.4    0  1  0  0 -2  1  0  0  0  0  1  0
    36        oC        250.0      11.9   26.6   11.9  90.0  90.0 116.6    0  1  0  0  0 -1 -2  0 -1  0  0  0
    38        oC        250.0      11.9   26.6   11.9  90.0  90.0 116.6    0 -1  0  0 -2  1  0  0  0  0 -1  0
    37        mC        250.0      26.6   11.9   11.9  90.0  90.0  63.4    0  1  2  0  0  1  0  0 -1  0  0  0
    39        mC        250.0      26.6   11.9   11.9  90.0  90.0  63.4    2 -1  0  0  0 -1  0  0  0  0 -1  0
    30        mC        250.0      11.9   26.6   11.9  90.0  90.0  63.4    0 -1  0  0  0 -1 -2  0  1  0  0  0
    12        hP        250.0      11.9   11.9   11.9  90.0  90.0  90.0    0  1  0  0 -1  0  0  0  0  0  1  0
    40        oC        250.0      11.9   26.6   11.9  90.0  90.0 116.6    0  1  0  0  0 -1 -2  0 -1  0  0  0
    41        mC        250.0      26.6   11.9   11.9  90.0  90.0  63.4    0  1  2  0  0  1  0  0 -1  0  0  0
     1        cF        250.0      20.6   20.6   20.6 109.5 109.5 109.5   -1  1 -1  0  1  1  1  0  1 -1 -1  0
    22        hP        250.0      11.9   11.9   11.9  90.0  90.0  90.0   -1  0  0  0  0  0  1  0  0  1  0  0
    27        mC        500.0      26.6   11.9   16.8  90.0 129.2  63.4    2 -1  0  0  0 -1  0  0 -1  0 -1  0
    19        oI        500.0      11.9   16.8   20.6  90.0  54.7  90.0    0 -1  0  0 -1  0 -1  0  1 -1 -1  0
    15        tI        500.0      11.9   11.9   29.2  65.9  65.9  90.0    0  1  0  0 -1  0  0  0 -1  1  2  0
    42        oI        500.0      11.9   11.9   29.2 114.1 114.1  90.0   -1  0  0  0  0  1  0  0  1 -1 -2  0
    26        oF        625.0      11.9   26.6   26.6  78.5 116.6 116.6    0  1  0  0  2 -1  0  0  0 -1 -2  0
    18        tI        625.0      16.8   20.6   11.9  54.7  90.0  90.0   -1  0 -1  0 -1  1  1  0  0  1  0  0
     9        hR        750.0      11.9   16.8   39.5  90.0 107.5 135.0    0  1  0  0  1 -1  0  0 -1 -1 -3  0
     6        tI        999.0      16.8   16.8   16.8  60.0  60.0  60.0   -1  0  1  0  0  1  1  0 -1  1  0  0
     7        tI        999.0      16.8   16.8   16.8  60.0  60.0  60.0    0  1  1  0 -1  1  0  0 -1  0  1  0
     8        oI        999.0      16.8   16.8   16.8  60.0  60.0  60.0    1 -1  0  0  0 -1 -1  0  1  0 -1  0
    16        oF        999.0      16.8   16.8   29.2  90.0 125.3  90.0    1 -1  0  0  1  1  0  0 -1  1  2  0
    17        mC        999.0      16.8   16.8   16.8  60.0 120.0  90.0    1  1  0  0  1 -1  0  0  0 -1 -1  0
    24        hR        999.0      29.2   16.8   11.9  90.0  65.9 106.8   -2  1  1  0  1  0  1  0  0  1  0  0
    43        mI        999.0      16.8   29.2   11.9 114.1 135.0  54.7   -1  1  0  0 -1  1  2  0  1  0  0  0

 ****** LATTICE SYMMETRY IMPLICATED BY SPACE GROUP SYMMETRY ******

BRAVAIS-            POSSIBLE SPACE-GROUPS FOR PROTEIN CRYSTALS
  TYPE                     [SPACE GROUP NUMBER,SYMBOL]
  aP      [1,P1]
  mP      [3,P2] [4,P2(1)]
 mC,mI    [5,C2]
  oP      [16,P222] [17,P222(1)] [18,P2(1)2(1)2] [19,P2(1)2(1)2(1)]
  oC      [21,C222] [20,C222(1)]
  oF      [22,F222]
  oI      [23,I222] [24,I2(1)2(1)2(1)]
  tP      [75,P4] [76,P4(1)] [77,P4(2)] [78,P4(3)] [89,P422] [90,P42(1)2]
          [91,P4(1)22] [92,P4(1)2(1)2] [93,P4(2)22] [94,P4(2)2(1)2]
          [95,P4(3)22] [96,P4(3)2(1)2]
  tI      [79,I4] [80,I4(1)] [97,I422] [98,I4(1)22]
  hP      [143,P3] [144,P3(1)] [145,P3(2)] [149,P312] [150,P321] [151,P3(1)12]
          [152,P3(1)21] [153,P3(2)12] [154,P3(2)21] [168,P6] [169,P6(1)]
          [170,P6(5)] [171,P6(2)] [172,P6(4)] [173,P6(3)] [177,P622]
          [178,P6(1)22] [179,P6(5)22] [180,P6(2)22] [181,P6(4)22] [182,P6(3)22]
  hR      [146,R3] [155,R32]
  cP      [195,P23] [198,P2(1)3] [207,P432] [208,P4(2)32] [212,P4(3)32]
          [213,P4(1)32]
  cF      [196,F23] [209,F432] [210,F4(1)32]
  cI      [197,I23] [199,I2(1)3] [211,I432] [214,I4(1)32]


 Several solutions exist for the given space group that explain the data. The
 selected solution (marked "*") correlates best with the given reference data set.
 The independent settings of the cell can be obtained by application of the
 REINDEXING TRANSFORMATION to the original indices H,K,L from file INTEGRATE.HKL

 CORRELATION = correlation factor with the reference data set
 NPAIR       = number of unique reflection pairs correlated 
 Rmeas=Rrim  = redundancy independent R-factor (intensities)
               For definition see:
               Rmeas : Diederichs & Karplus,
                       Nature Struct. Biol. 4, 269-275 (1997);
               Rrim  : Weiss & Hilgenfeld,
                       J.Appl.Cryst. 30,203-205 (1997).
 COMPARED    = number of reflections used for calculating Rmeas
 ESD         = Agreement with given unit cell geometry

 CORRELATION  NPAIR  Rmeas  COMPARED  ESD       REINDEX TRANSFORMATION

  *   0.00        0   10.1      18    0.00  1  0  0  0  0  1  0  0  0  0  1  0

 ************ SELECTED SPACE GROUP AND UNIT CELL FOR THIS DATA SET ************

 SPACE_GROUP_NUMBER=  221
 UNIT_CELL_CONSTANTS=    11.91    11.91    11.91  90.000  90.000  90.000
 UNIT_CELL_A-AXIS=     7.407    -4.950     7.903
 UNIT_CELL_B-AXIS=     7.407     9.255    -1.145
 UNIT_CELL_C-AXIS=    -5.666     5.627     8.835
 REFLECTIONS H,K,L FROM "INTEGRATE.HKL" WILL BE REINDEXED BY
 H' = (  1*H +  0*K +  0*L)/ 1 +  0
 K' = (  0*H +  1*K +  0*L)/ 1 +  0
 L' = (  0*H +  0*K +  1*L)/ 1 +  0

 ***** 48 EQUIVALENT POSITIONS IN SPACE GROUP #221 *****

    If x',y',z' is an equivalent position to x,y,z, then
        x'=x*ML(1)+y*ML( 2)+z*ML( 3)+ML( 4)/12.0
        y'=x*ML(5)+y*ML( 6)+z*ML( 7)+ML( 8)/12.0
        z'=x*ML(9)+y*ML(10)+z*ML(11)+ML(12)/12.0

    #    1  2  3  4    5  6  7  8    9 10 11 12 
    1    1  0  0  0    0  1  0  0    0  0  1  0
    2    0  0  1  0    1  0  0  0    0  1  0  0
    3    0  1  0  0    0  0  1  0    1  0  0  0
    4    1  0  0  0    0 -1  0  0    0  0 -1  0
    5    0  0  1  0   -1  0  0  0    0 -1  0  0
    6    0  1  0  0    0  0 -1  0   -1  0  0  0
    7   -1  0  0  0    0  1  0  0    0  0 -1  0
    8    0  0 -1  0    1  0  0  0    0 -1  0  0
    9    0 -1  0  0    0  0  1  0   -1  0  0  0
   10   -1  0  0  0    0 -1  0  0    0  0  1  0
   11    0  0 -1  0   -1  0  0  0    0  1  0  0
   12    0 -1  0  0    0  0 -1  0    1  0  0  0
   13    1  0  0  0    0  0  1  0    0  1  0  0
   14    0  1  0  0    1  0  0  0    0  0  1  0
   15    0  0  1  0    0  1  0  0    1  0  0  0
   16    1  0  0  0    0  0 -1  0    0 -1  0  0
   17    0  1  0  0   -1  0  0  0    0  0 -1  0
   18    0  0  1  0    0 -1  0  0   -1  0  0  0
   19   -1  0  0  0    0  0  1  0    0 -1  0  0
   20    0 -1  0  0    1  0  0  0    0  0 -1  0
   21    0  0 -1  0    0  1  0  0   -1  0  0  0
   22   -1  0  0  0    0  0 -1  0    0  1  0  0
   23    0 -1  0  0   -1  0  0  0    0  0  1  0
   24    0  0 -1  0    0 -1  0  0    1  0  0  0
   25   -1  0  0  0    0 -1  0  0    0  0 -1  0
   26    0  0 -1  0   -1  0  0  0    0 -1  0  0
   27    0 -1  0  0    0  0 -1  0   -1  0  0  0
   28   -1  0  0  0    0  1  0  0    0  0  1  0
   29    0  0 -1  0    1  0  0  0    0  1  0  0
   30    0 -1  0  0    0  0  1  0    1  0  0  0
   31    1  0  0  0    0 -1  0  0    0  0  1  0
   32    0  0  1  0   -1  0  0  0    0  1  0  0
   33    0  1  0  0    0  0 -1  0    1  0  0  0
   34    1  0  0  0    0  1  0  0    0  0 -1  0
   35    0  0  1  0    1  0  0  0    0 -1  0  0
   36    0  1  0  0    0  0  1  0   -1  0  0  0
   37   -1  0  0  0    0  0 -1  0    0 -1  0  0
   38    0 -1  0  0   -1  0  0  0    0  0 -1  0
   39    0  0 -1  0    0 -1  0  0   -1  0  0  0
   40   -1  0  0  0    0  0  1  0    0  1  0  0
   41    0 -1  0  0    1  0  0  0    0  0  1  0
   42    0  0 -1  0    0  1  0  0    1  0  0  0
   43    1  0  0  0    0  0 -1  0    0  1  0  0
   44    0  1  0  0   -1  0  0  0    0  0  1  0
   45    0  0  1  0    0 -1  0  0    1  0  0  0
   46    1  0  0  0    0  0  1  0    0 -1  0  0
   47    0  1  0  0    1  0  0  0    0  0 -1  0
   48    0  0  1  0    0  1  0  0   -1  0  0  0

 ******************************************************************************
      MEAN DISCREPANCIES BETWEEN OBSERVED AND CALCULATED SPOT LOCATIONS
 ******************************************************************************

 The discrepancies in X- and Y-coordinates of the spots are depicted in the
 two images DX-CORRECTIONS.cbf and DY-CORRECTIONS.cbf for inspection with
 the XDS-Viewer.



 ******************************************************************************
  REFINEMENT OF DIFFRACTION PARAMETERS USING ALL IMAGES
 ******************************************************************************


 REFINED VALUES OF DIFFRACTION PARAMETERS DERIVED FROM     14083 INDEXED SPOTS
 REFINED PARAMETERS:   AXIS ORIENTATION POSITION
 STANDARD DEVIATION OF SPOT    POSITION (PIXELS)     1.14
 STANDARD DEVIATION OF SPINDLE POSITION (DEGREES)    0.14
 SPACE GROUP NUMBER    221
 UNIT CELL PARAMETERS     11.909    11.909    11.909  90.000  90.000  90.000
 E.S.D. OF CELL PARAMETERS -1.0E+00-1.0E+00-1.0E+00-1.0E+00-1.0E+00-1.0E+00
 REC. CELL PARAMETERS   0.083968  0.083968  0.083968  90.000  90.000  90.000
 COORDINATES OF UNIT CELL A-AXIS     7.405    -4.951     7.905
 COORDINATES OF UNIT CELL B-AXIS     7.410     9.252    -1.146
 COORDINATES OF UNIT CELL C-AXIS    -5.665     5.631     8.833
 CRYSTAL MOSAICITY (DEGREES)     0.052
 LAB COORDINATES OF ROTATION AXIS  0.153832  0.988096 -0.001386
 DIRECT BEAM COORDINATES (REC. ANGSTROEM)   0.000000  0.000000 50.787209
 DETECTOR COORDINATES (PIXELS) OF DIRECT BEAM     243.98    249.38
 DETECTOR ORIGIN (PIXELS) AT                      243.98    249.38
 CRYSTAL TO DETECTOR DISTANCE (mm)       444.50
 LAB COORDINATES OF DETECTOR X-AXIS  1.000000  0.000000  0.000000
 LAB COORDINATES OF DETECTOR Y-AXIS  0.000000  1.000000  0.000000


 THE DATA COLLECTION STATISTICS REPORTED BELOW ASSUMES:
 SPACE_GROUP_NUMBER=  221
 UNIT_CELL_CONSTANTS=    11.91    11.91    11.91  90.000  90.000  90.000


 ******************************************************************************
        MEAN INTENSITY AS FUNCTION OF SPINDLE POSITION WITHIN DATA IMAGE
 ******************************************************************************

 This statistics could serve as a diagnostic tool for detecting
 shutter problems (suggested by Kay Diederichs).
 Data are corrected for this effect if PATCH_SHUTTER_PROBLEM=TRUE in XDS.INP.
 ===> Selected : PATCH_SHUTTER_PROBLEM=FALSE

 INTERVAL = Angular interval in units of the oscillation range
            covered by a data image.
 NUMBER   = Number of reflections in each interval
 INTENSITY= Mean intensity of the reflections in each interval
 FACTOR   = Correction factor applied to intensities;
            INTENSITY(overall)/INTENSITY(interval)

   INTERVAL     NUMBER      INTENSITY      FACTOR
 -0.05  0.05      1455         95.031       1.088
  0.05  0.15      1549        107.376       0.963
  0.15  0.25      1538        101.960       1.014
  0.25  0.35      1460        103.708       0.997
  0.35  0.45      1498        103.402       1.000
  0.45  0.55      1563        110.892       0.932
  0.55  0.65      1548        107.362       0.963
  0.65  0.75      1539         98.618       1.048
  0.75  0.85      1498        100.613       1.028
  0.85  0.95      1562        104.303       0.991

 -0.05  0.95     15210        103.396       1.000


 ******************************************************************************
           CORRECTION FACTORS AS FUNCTION OF IMAGE NUMBER & RESOLUTION
 ******************************************************************************

 RECIPROCAL CORRECTION FACTORS FOR INPUT DATA SETS MERGED TO
 OUTPUT FILE: XDS_ASCII.HKL                                     

 THE CALCULATIONS ASSUME         FRIEDEL'S_LAW= TRUE
 TOTAL NUMBER OF CORRECTION FACTORS DEFINED      289
 DEGREES OF FREEDOM OF CHI^2 FIT             14297.0
 CHI^2-VALUE OF FIT OF CORRECTION FACTORS      0.867
 NUMBER OF CYCLES CARRIED OUT                      3

 CORRECTION FACTORS for visual inspection by XDS-Viewer DECAY.cbf           
 XMIN=     0.3 XMAX=   756.6 NXBIN=   17
 YMIN= 0.00705 YMAX= 2.77092 NYBIN=   17
 NUMBER OF REFLECTIONS USED FOR DETERMINING CORRECTION FACTORS      15174


 ******************************************************************************
  CORRECTION FACTORS AS FUNCTION OF X (fast) & Y(slow) IN THE DETECTOR PLANE
 ******************************************************************************

 RECIPROCAL CORRECTION FACTORS FOR INPUT DATA SETS MERGED TO
 OUTPUT FILE: XDS_ASCII.HKL                                     

 THE CALCULATIONS ASSUME         FRIEDEL'S_LAW= TRUE
 TOTAL NUMBER OF CORRECTION FACTORS DEFINED      289
 DEGREES OF FREEDOM OF CHI^2 FIT             14292.0
 CHI^2-VALUE OF FIT OF CORRECTION FACTORS      0.865
 NUMBER OF CYCLES CARRIED OUT                      3

 CORRECTION FACTORS for visual inspection by XDS-Viewer MODPIX.cbf          
 XMIN=     3.6 XMAX=   508.1 NXBIN=   17
 YMIN=     3.6 YMAX=   508.1 NYBIN=   17
 NUMBER OF REFLECTIONS USED FOR DETERMINING CORRECTION FACTORS      15174


 ******************************************************************************
   CORRECTION FACTORS AS FUNCTION OF IMAGE NUMBER & DETECTOR SURFACE POSITION
 ******************************************************************************

 RECIPROCAL CORRECTION FACTORS FOR INPUT DATA SETS MERGED TO
 OUTPUT FILE: XDS_ASCII.HKL                                     

 THE CALCULATIONS ASSUME         FRIEDEL'S_LAW= TRUE
 TOTAL NUMBER OF CORRECTION FACTORS DEFINED      221
 DEGREES OF FREEDOM OF CHI^2 FIT             14297.0
 CHI^2-VALUE OF FIT OF CORRECTION FACTORS      0.864
 NUMBER OF CYCLES CARRIED OUT                      3

 CORRECTION FACTORS for visual inspection by XDS-Viewer ABSORP.cbf          
 XMIN=     0.3 XMAX=   756.6 NXBIN=   17
 DETECTOR_SURFACE_POSITION=     256     256
 DETECTOR_SURFACE_POSITION=     342     342
 DETECTOR_SURFACE_POSITION=     170     342
 DETECTOR_SURFACE_POSITION=     170     170
 DETECTOR_SURFACE_POSITION=     342     170
 DETECTOR_SURFACE_POSITION=     450     336
 DETECTOR_SURFACE_POSITION=     336     450
 DETECTOR_SURFACE_POSITION=     176     450
 DETECTOR_SURFACE_POSITION=      62     336
 DETECTOR_SURFACE_POSITION=      62     176
 DETECTOR_SURFACE_POSITION=     176      62
 DETECTOR_SURFACE_POSITION=     336      62
 DETECTOR_SURFACE_POSITION=     450     176
 NUMBER OF REFLECTIONS USED FOR DETERMINING CORRECTION FACTORS      15174


 ******************************************************************************
    CORRECTION PARAMETERS FOR THE STANDARD ERROR OF REFLECTION INTENSITIES
 ******************************************************************************

 The variance v0(I) of the intensity I obtained from counting statistics is
 replaced by v(I)=a*(v0(I)+b*I^2). The model parameters a, b are chosen to
 minimize the discrepancies between v(I) and the variance estimated from
 sample statistics of symmetry related reflections. This model implicates
 an asymptotic limit ISa=1/SQRT(a*b) for the highest I/Sigma(I) that the
 experimental setup can produce (Diederichs (2010) Acta Cryst D66, 733-740).

     a        b          ISa
 1.388E+02  1.037E-03    2.64


 ******************************************************************************
      STANDARD ERROR OF REFLECTION INTENSITIES AS FUNCTION OF RESOLUTION
      FOR DATA SET  XDS_ASCII.HKL                                     
 ******************************************************************************

  I/Sigma  = mean intensity/Sigma of a reflection in shell
  Chi^2    = goodness of fit between sample variances of
             symmetry-related intensities and their errors
             (Chi^2 = 1 for perfect agreement)
  R-FACTOR
  observed = (SUM(ABS(I(h,i)-I(h))))/(SUM(I(h,i)))
  expected = expected R-FACTOR derived from Sigma(I)

   NUMBER  = number of reflections in resolution shell
             used for calculation of R-FACTOR
  ACCEPTED = number of accepted reflections
  REJECTED = number of rejected reflections (MISFITS),
             recognized by comparison with symmetry-related
             reflections.

 RESOLUTION RANGE  I/Sigma  Chi^2  R-FACTOR  R-FACTOR  NUMBER ACCEPTED REJECTED
                                   observed  expected

   11.909   4.799     2.63   0.51     12.12     27.38      32      32       0
    4.799   3.540     2.63   0.47     20.77     28.96      45      45       0
    3.540   2.934     2.63   0.32     16.91     28.93      39      39       0
    2.934   2.561     2.58   0.67     27.29     29.27      83      83       0
    2.561   2.301     2.50   0.67     22.04     28.77      71      71       0
    2.301   2.107     2.51   1.22     32.13     29.13      79      79       0
    2.107   1.955     2.47   0.85     32.67     29.21      94      94       0
    1.955   1.832     2.50   1.47     37.44     29.81     118     118       0
    1.832   1.729     2.42   1.87     39.72     30.06      80      80       0
    1.729   1.642     2.45   0.76     20.68     29.20     102     102       0
    1.642   1.567     2.53   1.17     26.21     29.62     129     129       0
    1.567   1.502     2.58   0.94     21.84     29.85     129     129       0
    1.502   1.444     2.44   1.59     30.77     29.29     130     130       0
    1.444   1.392     2.54   0.78     21.78     29.65     119     119       0
    1.392   1.345     2.51   1.11     24.27     30.02     159     159       0
    1.345   1.303     2.50   0.73     21.48     29.43     117     117       0
    1.303   1.265     2.44   1.03     33.46     30.44     124     124       0
    1.265   1.229     2.54   1.00     34.29     30.19     166     166       0
    1.229   1.197     2.50   1.10     26.01     29.78     166     166       0
    1.197   1.167     2.43   0.72     21.95     29.61     164     164       0
    1.167   1.139     2.48   1.00     25.04     29.79     172     172       0
    1.139   1.113     2.33   1.21     31.98     30.34     163     163       0
    1.113   1.089     1.96   1.42     46.85     32.90     148     148       0
    1.089   1.066     2.10   1.80     57.35     32.84     142     142       0
    1.066   1.045     2.10   1.45     51.71     33.39     224     224       0
    1.045   1.025     2.26   1.51     45.27     31.51     214     214       0
    1.025   1.006     2.11   1.15     28.94     30.84     170     170       0
    1.006   0.988     2.09   1.42     51.42     32.16     134     134       0
    0.988   0.971     2.15   1.20     39.76     31.97     278     278       0
    0.971   0.954     2.18   1.24     43.12     32.81     199     199       0
    0.954   0.939     2.09   1.15     33.02     31.46      95      95       0
    0.939   0.924     1.90   1.44     55.83     35.21     313     313       0
    0.924   0.910     1.90   1.16     44.47     33.42     190     190       0
    0.910   0.897     2.06   1.28     49.07     34.84     179     179       0
    0.897   0.884     2.06   1.14     43.15     33.67     221     221       0
    0.884   0.872     2.05   0.99     38.18     34.12     260     260       0
    0.872   0.860     2.01   0.99     41.73     34.94     135     135       0
    0.860   0.848     1.88   1.00     42.25     34.04     261     261       0
    0.848   0.838     1.84   1.08     48.41     37.40     225     225       0
    0.838   0.827     1.88   1.01     47.30     37.61     238     238       0
    0.827   0.817     2.11   0.88     31.05     32.63     244     244       0
    0.817   0.807     1.75   0.97     46.57     37.75     185     185       0
    0.807   0.798     1.99   1.05     42.12     35.55     268     268       0
    0.798   0.789     1.97   1.00     43.69     35.62     198     198       0
    0.789   0.780     2.05   0.76     31.99     33.51     284     284       0
    0.780   0.772     1.76   0.95     48.38     39.43     283     283       0
    0.772   0.763     1.95   0.84     26.86     33.26     193     193       0
    0.763   0.755     1.56   0.73     45.56     43.84     237     237       0
    0.755   0.748     1.75   0.88     41.44     36.65     241     241       0
    0.748   0.740     1.38   0.84     43.32     42.06     239     239       0
    0.740   0.733     1.77   0.81     38.97     38.22     277     277       0
    0.733   0.726     1.82   0.76     31.95     34.89     340     340       0
    0.726   0.719     1.56   0.74     44.66     43.51     238     238       0
    0.719   0.712     1.48   0.61     41.75     40.39     223     223       0
    0.712   0.706     1.71   0.80     32.16     36.38     226     226       0
    0.706   0.700     1.58   0.78     41.59     41.18     230     230       0
    0.700   0.693     1.48   0.71     39.37     40.88     369     369       0
    0.693   0.687     1.52   0.62     41.90     41.25     312     312       0
    0.687   0.682     1.33   0.73     53.37     50.03     223     223       0
    0.682   0.676     1.66   0.80     39.95     39.51     330     330       0
    0.676   0.670     1.37   0.66     44.23     47.48     298     298       0
    0.670   0.665     1.46   0.77     48.07     42.93     146     146       0
    0.665   0.660     1.36   0.67     43.77     45.26     338     338       0
    0.660   0.654     1.26   0.55     47.59     53.14     353     353       0
    0.654   0.649     1.05   0.49     54.07     64.85     175     175       0
    0.649   0.645     1.12   0.55     53.31     57.85     359     359       0
    0.645   0.640     1.27   0.48     41.05     51.52     229     229       0
    0.640   0.635     1.20   0.56     47.29     51.00     232     232       0
    0.635   0.630     1.23   0.49     41.82     52.12     260     260       0
    0.630   0.626     1.04   0.42     43.69     59.19     260     260       0
    0.626   0.621     1.07   0.51     48.78     54.08     225     225       0
    0.621   0.617     1.13   0.50     47.90     56.63     262     262       0
    0.617   0.613     1.05   0.39     36.97     54.55     293     293       0
    0.613   0.609     1.17   0.39     33.19     50.01     230     230       0
    0.609   0.605     1.01   0.37     47.67     66.04     183     183       0
    0.605   0.601     0.89   0.34     45.84     69.27     117     120       0
   --------------------------------------------------------------------------
   11.909   0.601     1.77   0.87     30.87     31.94   15207   15210       0



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR VARIOUS SUBSETS OF INCLUDED DATA IMAGES
 ******************************************************************************

 Data set statistics is reported below several times, each with a different
 upper limit on the number of images included. This provides the user with
 the information for deciding which data images should be excluded from the
 final data set because of radiation damage or other defects. If the user
 decides for a subset of "good" images that differs from the specification
 as given by the input parameter (see DATA_RANGE= in XDS.INP) the CORRECT-
 step of XDS must be repeated with the new parameter values for DATA_RANGE=.


 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1      85
 ******************************************************************************


          COMPLETENESS AND QUALITY OF DATA SET
          ------------------------------------

 R-FACTOR
 observed = (SUM(ABS(I(h,i)-I(h))))/(SUM(I(h,i)))
 expected = expected R-FACTOR derived from Sigma(I)

 COMPARED = number of reflections used for calculating R-FACTOR
 I/SIGMA  = mean of intensity/Sigma(I) of unique reflections
            (after merging symmetry-related observations)
 Sigma(I) = standard deviation of reflection intensity I
            estimated from sample statistics

 R-meas   = redundancy independent R-factor (intensities)
            Diederichs & Karplus (1997), Nature Struct. Biol. 4, 269-275.

 CC(1/2)  = percentage of correlation between intensities from
            random half-datasets. Correlation significant at
            the 0.1% level is marked by an asterisk.
            Karplus & Diederichs (2012), Science 336, 1030-33
 Anomal   = percentage of correlation between random half-sets
  Corr      of anomalous intensity differences. Correlation
            significant at the 0.1% level is marked.
 SigAno   = mean anomalous difference in units of its estimated
            standard deviation (|F(+)-F(-)|/Sigma). F(+), F(-)
            are structure factor estimates obtained from the
            merged intensity observations in each parity class.
  Nano    = Number of unique reflections used to calculate
            Anomal_Corr & SigAno. At least two observations
            for each (+ and -) parity are required.

       NOTE:      Friedel pairs are treated as different reflections.

 SUBSET OF INTENSITY DATA WITH SIGNAL/NOISE >= -3.0 AS FUNCTION OF RESOLUTION
 RESOLUTION     NUMBER OF REFLECTIONS    COMPLETENESS R-FACTOR  R-FACTOR COMPARED I/SIGMA   R-meas  CC(1/2)  Anomal  SigAno   Nano
   LIMIT     OBSERVED  UNIQUE  POSSIBLE     OF DATA   observed  expected                                      Corr

     1.78          68      23        51       45.1%      31.6%     26.5%       65    3.98     36.8%    85.8*     0    0.000       0
     1.27         120      37        69       53.6%      31.0%     26.2%      114    3.94     36.4%    89.5*     0    0.000       0
     1.04         162      49        83       59.0%      37.4%     27.6%      156    3.58     44.1%    75.1*     0    0.000       0
     0.90         173      52        93       55.9%      50.7%     30.2%      166    3.33     60.7%    34.5      0    0.000       0
     0.81         212      64        99       64.6%      36.2%     32.2%      198    3.16     41.5%    77.3*     0    0.000       0
     0.74         224      66       113       58.4%      31.7%     33.8%      213    2.83     36.6%    95.3*     0    0.000       0
     0.68         252      76       115       66.1%      36.5%     37.7%      235    2.55     42.3%    90.4*     0    0.000       0
     0.64         276      81       126       64.3%      45.2%     46.2%      261    2.00     52.6%    65.1*     0    0.000       0
     0.60         218      81       132       61.4%      39.2%     54.1%      188    1.48     47.3%    84.0*     0    0.000       0
    total        1705     529       881       60.0%      34.5%     28.8%     1596    2.74     40.4%    90.4*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES      1705
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                         1705
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    529



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1     169
 ******************************************************************************


          COMPLETENESS AND QUALITY OF DATA SET
          ------------------------------------

 R-FACTOR
 observed = (SUM(ABS(I(h,i)-I(h))))/(SUM(I(h,i)))
 expected = expected R-FACTOR derived from Sigma(I)

 COMPARED = number of reflections used for calculating R-FACTOR
 I/SIGMA  = mean of intensity/Sigma(I) of unique reflections
            (after merging symmetry-related observations)
 Sigma(I) = standard deviation of reflection intensity I
            estimated from sample statistics

 R-meas   = redundancy independent R-factor (intensities)
            Diederichs & Karplus (1997), Nature Struct. Biol. 4, 269-275.

 CC(1/2)  = percentage of correlation between intensities from
            random half-datasets. Correlation significant at
            the 0.1% level is marked by an asterisk.
            Karplus & Diederichs (2012), Science 336, 1030-33
 Anomal   = percentage of correlation between random half-sets
  Corr      of anomalous intensity differences. Correlation
            significant at the 0.1% level is marked.
 SigAno   = mean anomalous difference in units of its estimated
            standard deviation (|F(+)-F(-)|/Sigma). F(+), F(-)
            are structure factor estimates obtained from the
            merged intensity observations in each parity class.
  Nano    = Number of unique reflections used to calculate
            Anomal_Corr & SigAno. At least two observations
            for each (+ and -) parity are required.

       NOTE:      Friedel pairs are treated as different reflections.

 SUBSET OF INTENSITY DATA WITH SIGNAL/NOISE >= -3.0 AS FUNCTION OF RESOLUTION
 RESOLUTION     NUMBER OF REFLECTIONS    COMPLETENESS R-FACTOR  R-FACTOR COMPARED I/SIGMA   R-meas  CC(1/2)  Anomal  SigAno   Nano
   LIMIT     OBSERVED  UNIQUE  POSSIBLE     OF DATA   observed  expected                                      Corr

     1.78         132      37        51       72.5%      28.3%     27.1%      126    4.34     31.9%    93.0*     0    0.000       0
     1.27         235      55        69       79.7%      24.1%     27.4%      233    4.56     27.1%    91.4*     0    0.000       0
     1.04         325      70        83       84.3%      29.3%     27.7%      323    4.24     32.8%    95.2*     0    0.000       0
     0.90         362      80        93       86.0%      42.7%     30.3%      356    3.87     48.0%    81.0*     0    0.000       0
     0.81         409      85        99       85.9%      34.0%     32.6%      403    3.81     37.7%    85.3*     0    0.000       0
     0.74         470      97       113       85.8%      30.9%     34.3%      462    3.46     34.6%    94.0*     0    0.000       0
     0.68         494     102       115       88.7%      33.9%     37.4%      488    3.09     37.8%    93.2*     0    0.000       0
     0.64         534     109       126       86.5%      42.3%     46.2%      525    2.48     46.9%    75.7*     0    0.000       0
     0.60         434     108       132       81.8%      39.0%     56.8%      412    1.79     44.4%    88.6*     0    0.000       0
    total        3395     743       881       84.3%      29.9%     29.7%     3328    3.31     33.6%    94.8*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES      3395
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                         3395
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    743



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1     253
 ******************************************************************************


          COMPLETENESS AND QUALITY OF DATA SET
          ------------------------------------

 R-FACTOR
 observed = (SUM(ABS(I(h,i)-I(h))))/(SUM(I(h,i)))
 expected = expected R-FACTOR derived from Sigma(I)

 COMPARED = number of reflections used for calculating R-FACTOR
 I/SIGMA  = mean of intensity/Sigma(I) of unique reflections
            (after merging symmetry-related observations)
 Sigma(I) = standard deviation of reflection intensity I
            estimated from sample statistics

 R-meas   = redundancy independent R-factor (intensities)
            Diederichs & Karplus (1997), Nature Struct. Biol. 4, 269-275.

 CC(1/2)  = percentage of correlation between intensities from
            random half-datasets. Correlation significant at
            the 0.1% level is marked by an asterisk.
            Karplus & Diederichs (2012), Science 336, 1030-33
 Anomal   = percentage of correlation between random half-sets
  Corr      of anomalous intensity differences. Correlation
            significant at the 0.1% level is marked.
 SigAno   = mean anomalous difference in units of its estimated
            standard deviation (|F(+)-F(-)|/Sigma). F(+), F(-)
            are structure factor estimates obtained from the
            merged intensity observations in each parity class.
  Nano    = Number of unique reflections used to calculate
            Anomal_Corr & SigAno. At least two observations
            for each (+ and -) parity are required.

       NOTE:      Friedel pairs are treated as different reflections.

 SUBSET OF INTENSITY DATA WITH SIGNAL/NOISE >= -3.0 AS FUNCTION OF RESOLUTION
 RESOLUTION     NUMBER OF REFLECTIONS    COMPLETENESS R-FACTOR  R-FACTOR COMPARED I/SIGMA   R-meas  CC(1/2)  Anomal  SigAno   Nano
   LIMIT     OBSERVED  UNIQUE  POSSIBLE     OF DATA   observed  expected                                      Corr

     1.78         193      42        51       82.4%      26.8%     27.6%      188    4.95     29.6%    91.0*     0    0.000       0
     1.27         359      61        69       88.4%      24.1%     28.2%      358    5.40     26.4%    93.0*     0    0.000       0
     1.04         481      78        83       94.0%      27.8%     28.5%      481    4.99     30.5%    95.7*     0    0.000       0
     0.90         553      88        93       94.6%      41.7%     30.9%      550    4.44     45.5%    94.4*     0    0.000       0
     0.81         612      92        99       92.9%      34.9%     33.4%      612    4.43     37.8%    86.3*     0    0.000       0
     0.74         707     106       113       93.8%      33.2%     35.5%      706    4.03     36.0%    96.2*     0    0.000       0
     0.68         732     109       115       94.8%      32.0%     37.3%      730    3.65     34.9%    97.3*     0    0.000       0
     0.64         788     121       126       96.0%      42.3%     46.3%      782    2.89     45.7%    83.8*     0    0.000       0
     0.60         654     125       132       94.7%      37.3%     55.7%      638    2.11     41.2%    88.2*     0    0.000       0
    total        5079     822       881       93.3%      29.4%     30.4%     5045    3.85     32.2%    94.9*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES      5079
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                         5079
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    822



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1     337
 ******************************************************************************


          COMPLETENESS AND QUALITY OF DATA SET
          ------------------------------------

 R-FACTOR
 observed = (SUM(ABS(I(h,i)-I(h))))/(SUM(I(h,i)))
 expected = expected R-FACTOR derived from Sigma(I)

 COMPARED = number of reflections used for calculating R-FACTOR
 I/SIGMA  = mean of intensity/Sigma(I) of unique reflections
            (after merging symmetry-related observations)
 Sigma(I) = standard deviation of reflection intensity I
            estimated from sample statistics

 R-meas   = redundancy independent R-factor (intensities)
            Diederichs & Karplus (1997), Nature Struct. Biol. 4, 269-275.

 CC(1/2)  = percentage of correlation between intensities from
            random half-datasets. Correlation significant at
            the 0.1% level is marked by an asterisk.
            Karplus & Diederichs (2012), Science 336, 1030-33
 Anomal   = percentage of correlation between random half-sets
  Corr      of anomalous intensity differences. Correlation
            significant at the 0.1% level is marked.
 SigAno   = mean anomalous difference in units of its estimated
            standard deviation (|F(+)-F(-)|/Sigma). F(+), F(-)
            are structure factor estimates obtained from the
            merged intensity observations in each parity class.
  Nano    = Number of unique reflections used to calculate
            Anomal_Corr & SigAno. At least two observations
            for each (+ and -) parity are required.

       NOTE:      Friedel pairs are treated as different reflections.

 SUBSET OF INTENSITY DATA WITH SIGNAL/NOISE >= -3.0 AS FUNCTION OF RESOLUTION
 RESOLUTION     NUMBER OF REFLECTIONS    COMPLETENESS R-FACTOR  R-FACTOR COMPARED I/SIGMA   R-meas  CC(1/2)  Anomal  SigAno   Nano
   LIMIT     OBSERVED  UNIQUE  POSSIBLE     OF DATA   observed  expected                                      Corr

     1.78         264      51        51      100.0%      24.4%     27.5%      262    5.28     26.7%    93.1*     0    0.000       0
     1.27         470      69        69      100.0%      25.9%     28.3%      468    5.81     28.1%    95.4*     0    0.000       0
     1.04         635      83        83      100.0%      29.7%     29.0%      634    5.64     31.9%    95.4*     0    0.000       0
     0.90         739      93        93      100.0%      42.4%     31.2%      739    5.00     45.6%    97.1*     0    0.000       0
     0.81         815      99        99      100.0%      37.6%     33.4%      815    4.89     40.4%    97.4*     0    0.000       0
     0.74         935     113       113      100.0%      36.5%     35.5%      935    4.51     39.0%    98.2*     0    0.000       0
     0.68         986     115       115      100.0%      38.0%     38.3%      986    4.13     41.0%    87.7*     0    0.000       0
     0.64        1053     126       126      100.0%      43.8%     46.4%     1053    3.34     46.9%    84.5*     0    0.000       0
     0.60         859     130       132       98.5%      39.7%     55.1%      853    2.42     43.3%    92.4*     0    0.000       0
    total        6756     879       881       99.8%      30.1%     30.6%     6745    4.33     32.6%    96.0*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES      6756
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                         6756
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    879



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1     421
 ******************************************************************************


          COMPLETENESS AND QUALITY OF DATA SET
          ------------------------------------

 R-FACTOR
 observed = (SUM(ABS(I(h,i)-I(h))))/(SUM(I(h,i)))
 expected = expected R-FACTOR derived from Sigma(I)

 COMPARED = number of reflections used for calculating R-FACTOR
 I/SIGMA  = mean of intensity/Sigma(I) of unique reflections
            (after merging symmetry-related observations)
 Sigma(I) = standard deviation of reflection intensity I
            estimated from sample statistics

 R-meas   = redundancy independent R-factor (intensities)
            Diederichs & Karplus (1997), Nature Struct. Biol. 4, 269-275.

 CC(1/2)  = percentage of correlation between intensities from
            random half-datasets. Correlation significant at
            the 0.1% level is marked by an asterisk.
            Karplus & Diederichs (2012), Science 336, 1030-33
 Anomal   = percentage of correlation between random half-sets
  Corr      of anomalous intensity differences. Correlation
            significant at the 0.1% level is marked.
 SigAno   = mean anomalous difference in units of its estimated
            standard deviation (|F(+)-F(-)|/Sigma). F(+), F(-)
            are structure factor estimates obtained from the
            merged intensity observations in each parity class.
  Nano    = Number of unique reflections used to calculate
            Anomal_Corr & SigAno. At least two observations
            for each (+ and -) parity are required.

       NOTE:      Friedel pairs are treated as different reflections.

 SUBSET OF INTENSITY DATA WITH SIGNAL/NOISE >= -3.0 AS FUNCTION OF RESOLUTION
 RESOLUTION     NUMBER OF REFLECTIONS    COMPLETENESS R-FACTOR  R-FACTOR COMPARED I/SIGMA   R-meas  CC(1/2)  Anomal  SigAno   Nano
   LIMIT     OBSERVED  UNIQUE  POSSIBLE     OF DATA   observed  expected                                      Corr

     1.78         326      51        51      100.0%      24.9%     27.9%      326    5.88     27.1%    91.4*     0    0.000       0
     1.27         598      69        69      100.0%      25.3%     28.8%      598    6.62     26.9%    98.1*     0    0.000       0
     1.04         790      83        83      100.0%      31.8%     29.5%      790    6.25     33.7%    95.5*     0    0.000       0
     0.90         930      93        93      100.0%      42.5%     31.7%      930    5.56     45.1%    97.5*     0    0.000       0
     0.81        1022      99        99      100.0%      41.1%     33.9%     1022    5.46     43.6%    80.8*     0    0.000       0
     0.74        1171     113       113      100.0%      37.2%     35.9%     1171    5.10     39.2%    98.6*     0    0.000       0
     0.68        1232     115       115      100.0%      40.1%     38.9%     1232    4.58     42.4%    84.0*     0    0.000       0
     0.64        1318     126       126      100.0%      45.3%     46.7%     1318    3.75     47.8%    91.3*     0    0.000       0
     0.60        1064     130       132       98.5%      40.4%     54.2%     1060    2.76     43.5%    89.1*     0    0.000       0
    total        8451     879       881       99.8%      30.9%     31.0%     8447    4.86     33.0%    96.1*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES      8451
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                         8451
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    879



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1     505
 ******************************************************************************


          COMPLETENESS AND QUALITY OF DATA SET
          ------------------------------------

 R-FACTOR
 observed = (SUM(ABS(I(h,i)-I(h))))/(SUM(I(h,i)))
 expected = expected R-FACTOR derived from Sigma(I)

 COMPARED = number of reflections used for calculating R-FACTOR
 I/SIGMA  = mean of intensity/Sigma(I) of unique reflections
            (after merging symmetry-related observations)
 Sigma(I) = standard deviation of reflection intensity I
            estimated from sample statistics

 R-meas   = redundancy independent R-factor (intensities)
            Diederichs & Karplus (1997), Nature Struct. Biol. 4, 269-275.

 CC(1/2)  = percentage of correlation between intensities from
            random half-datasets. Correlation significant at
            the 0.1% level is marked by an asterisk.
            Karplus & Diederichs (2012), Science 336, 1030-33
 Anomal   = percentage of correlation between random half-sets
  Corr      of anomalous intensity differences. Correlation
            significant at the 0.1% level is marked.
 SigAno   = mean anomalous difference in units of its estimated
            standard deviation (|F(+)-F(-)|/Sigma). F(+), F(-)
            are structure factor estimates obtained from the
            merged intensity observations in each parity class.
  Nano    = Number of unique reflections used to calculate
            Anomal_Corr & SigAno. At least two observations
            for each (+ and -) parity are required.

       NOTE:      Friedel pairs are treated as different reflections.

 SUBSET OF INTENSITY DATA WITH SIGNAL/NOISE >= -3.0 AS FUNCTION OF RESOLUTION
 RESOLUTION     NUMBER OF REFLECTIONS    COMPLETENESS R-FACTOR  R-FACTOR COMPARED I/SIGMA   R-meas  CC(1/2)  Anomal  SigAno   Nano
   LIMIT     OBSERVED  UNIQUE  POSSIBLE     OF DATA   observed  expected                                      Corr

     1.78         388      51        51      100.0%      24.7%     28.3%      388    6.37     26.6%    84.4*     0    0.000       0
     1.27         713      69        69      100.0%      25.2%     29.1%      713    7.17     26.6%    98.2*     0    0.000       0
     1.04         955      83        83      100.0%      30.6%     29.8%      955    6.90     32.1%    96.9*     0    0.000       0
     0.90        1116      93        93      100.0%      41.9%     32.0%     1116    6.04     44.0%    97.5*     0    0.000       0
     0.81        1230      99        99      100.0%      40.5%     34.1%     1230    6.02     42.5%    94.1*     0    0.000       0
     0.74        1412     113       113      100.0%      36.7%     36.0%     1412    5.62     38.4%    98.3*     0    0.000       0
     0.68        1483     115       115      100.0%      38.8%     38.5%     1483    5.07     40.6%    97.4*     0    0.000       0
     0.64        1577     126       126      100.0%      45.2%     46.9%     1577    4.11     47.3%    92.4*     0    0.000       0
     0.60        1288     130       132       98.5%      39.7%     54.0%     1284    3.07     42.2%    96.2*     0    0.000       0
    total       10162     879       881       99.8%      30.5%     31.3%    10158    5.33     32.3%    93.6*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES     10162
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                        10162
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    879



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1     589
 ******************************************************************************


          COMPLETENESS AND QUALITY OF DATA SET
          ------------------------------------

 R-FACTOR
 observed = (SUM(ABS(I(h,i)-I(h))))/(SUM(I(h,i)))
 expected = expected R-FACTOR derived from Sigma(I)

 COMPARED = number of reflections used for calculating R-FACTOR
 I/SIGMA  = mean of intensity/Sigma(I) of unique reflections
            (after merging symmetry-related observations)
 Sigma(I) = standard deviation of reflection intensity I
            estimated from sample statistics

 R-meas   = redundancy independent R-factor (intensities)
            Diederichs & Karplus (1997), Nature Struct. Biol. 4, 269-275.

 CC(1/2)  = percentage of correlation between intensities from
            random half-datasets. Correlation significant at
            the 0.1% level is marked by an asterisk.
            Karplus & Diederichs (2012), Science 336, 1030-33
 Anomal   = percentage of correlation between random half-sets
  Corr      of anomalous intensity differences. Correlation
            significant at the 0.1% level is marked.
 SigAno   = mean anomalous difference in units of its estimated
            standard deviation (|F(+)-F(-)|/Sigma). F(+), F(-)
            are structure factor estimates obtained from the
            merged intensity observations in each parity class.
  Nano    = Number of unique reflections used to calculate
            Anomal_Corr & SigAno. At least two observations
            for each (+ and -) parity are required.

       NOTE:      Friedel pairs are treated as different reflections.

 SUBSET OF INTENSITY DATA WITH SIGNAL/NOISE >= -3.0 AS FUNCTION OF RESOLUTION
 RESOLUTION     NUMBER OF REFLECTIONS    COMPLETENESS R-FACTOR  R-FACTOR COMPARED I/SIGMA   R-meas  CC(1/2)  Anomal  SigAno   Nano
   LIMIT     OBSERVED  UNIQUE  POSSIBLE     OF DATA   observed  expected                                      Corr

     1.78         451      51        51      100.0%      24.4%     28.5%      451    6.85     26.1%    84.0*     0    0.000       0
     1.27         832      69        69      100.0%      24.8%     29.3%      832    7.72     25.9%    98.3*     0    0.000       0
     1.04        1120      83        83      100.0%      30.8%     30.1%     1120    7.44     32.1%    97.3*     0    0.000       0
     0.90        1301      93        93      100.0%      41.6%     32.2%     1301    6.53     43.4%    97.7*     0    0.000       0
     0.81        1440      99        99      100.0%      40.4%     34.2%     1440    6.52     42.0%    93.0*     0    0.000       0
     0.74        1651     113       113      100.0%      36.9%     36.3%     1651    6.06     38.4%    98.2*     0    0.000       0
     0.68        1720     115       115      100.0%      37.9%     38.8%     1720    5.48     39.4%    97.1*     0    0.000       0
     0.64        1831     126       126      100.0%      44.5%     47.0%     1831    4.42     46.2%    95.1*     0    0.000       0
     0.60        1494     130       132       98.5%      40.4%     54.7%     1491    3.30     42.5%    91.3*     0    0.000       0
    total       11840     879       881       99.8%      30.3%     31.5%    11837    5.75     31.8%    93.7*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES     11840
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                        11840
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    879



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1     673
 ******************************************************************************


          COMPLETENESS AND QUALITY OF DATA SET
          ------------------------------------

 R-FACTOR
 observed = (SUM(ABS(I(h,i)-I(h))))/(SUM(I(h,i)))
 expected = expected R-FACTOR derived from Sigma(I)

 COMPARED = number of reflections used for calculating R-FACTOR
 I/SIGMA  = mean of intensity/Sigma(I) of unique reflections
            (after merging symmetry-related observations)
 Sigma(I) = standard deviation of reflection intensity I
            estimated from sample statistics

 R-meas   = redundancy independent R-factor (intensities)
            Diederichs & Karplus (1997), Nature Struct. Biol. 4, 269-275.

 CC(1/2)  = percentage of correlation between intensities from
            random half-datasets. Correlation significant at
            the 0.1% level is marked by an asterisk.
            Karplus & Diederichs (2012), Science 336, 1030-33
 Anomal   = percentage of correlation between random half-sets
  Corr      of anomalous intensity differences. Correlation
            significant at the 0.1% level is marked.
 SigAno   = mean anomalous difference in units of its estimated
            standard deviation (|F(+)-F(-)|/Sigma). F(+), F(-)
            are structure factor estimates obtained from the
            merged intensity observations in each parity class.
  Nano    = Number of unique reflections used to calculate
            Anomal_Corr & SigAno. At least two observations
            for each (+ and -) parity are required.

       NOTE:      Friedel pairs are treated as different reflections.

 SUBSET OF INTENSITY DATA WITH SIGNAL/NOISE >= -3.0 AS FUNCTION OF RESOLUTION
 RESOLUTION     NUMBER OF REFLECTIONS    COMPLETENESS R-FACTOR  R-FACTOR COMPARED I/SIGMA   R-meas  CC(1/2)  Anomal  SigAno   Nano
   LIMIT     OBSERVED  UNIQUE  POSSIBLE     OF DATA   observed  expected                                      Corr

     1.78         523      51        51      100.0%      24.5%     28.8%      523    7.30     25.9%    91.5*     0    0.000       0
     1.27         944      69        69      100.0%      25.7%     29.5%      944    8.18     26.7%    99.2*     0    0.000       0
     1.04        1274      83        83      100.0%      30.7%     30.2%     1274    7.91     31.8%    97.8*     0    0.000       0
     0.90        1480      93        93      100.0%      41.5%     32.3%     1480    6.97     43.0%    98.0*     0    0.000       0
     0.81        1639      99        99      100.0%      40.5%     34.4%     1639    6.95     42.0%    81.7*     0    0.000       0
     0.74        1884     113       113      100.0%      37.5%     36.4%     1884    6.45     38.7%    98.9*     0    0.000       0
     0.68        1967     115       115      100.0%      38.1%     39.0%     1967    5.81     39.4%    97.3*     0    0.000       0
     0.64        2108     126       126      100.0%      44.3%     47.0%     2108    4.75     45.8%    92.6*     0    0.000       0
     0.60        1717     131       132       99.2%      41.3%     55.4%     1713    3.50     43.3%    92.8*     0    0.000       0
    total       13536     880       881       99.9%      30.6%     31.7%    13532    6.12     31.8%    96.6*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES     13536
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                        13536
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    880



 ******************************************************************************
  STATISTICS OF SAVED DATA SET "XDS_ASCII.HKL" (DATA_RANGE=       1     757)
 FILE TYPE:         XDS_ASCII      MERGE=FALSE          FRIEDEL'S_LAW=FALSE
 ******************************************************************************

   REFLECTIONS OF TYPE H,0,0  0,K,0  0,0,L OR EXPECTED TO BE ABSENT (*)
   --------------------------------------------------------------------

   H    K    L  RESOLUTION  INTENSITY     SIGMA    INTENSITY/SIGMA  #OBSERVED

    1    0    0    11.909   0.4316E+03  0.1163E+03         3.71           2 
    2    0    0     5.955   0.9939E+03  0.1936E+03         5.13           4 
    3    0    0     3.970   0.4476E+03  0.8581E+02         5.22           4 
    4    0    0     2.977   0.7114E+03  0.1469E+03         4.84           4 
    5    0    0     2.382   0.1595E+04  0.3211E+03         4.97           4 
    6    0    0     1.985   0.7182E+03  0.1382E+03         5.20           4 
    7    0    0     1.701   0.1990E+03  0.4353E+02         4.57           4 
    8    0    0     1.489   0.1713E+04  0.3276E+03         5.23           4 
    9    0    0     1.323   0.5409E+03  0.1043E+03         5.19           4 
   10    0    0     1.191   0.1763E+04  0.3357E+03         5.25           4 
   11    0    0     1.083   0.1433E+03  0.2888E+02         4.96           4 
   12    0    0     0.992   0.3024E+03  0.5920E+02         5.11           4 
   13    0    0     0.916   0.6252E+02  0.1352E+02         4.62           4 
   14    0    0     0.851   0.2058E+02  0.6332E+01         3.25           4 
   15    0    0     0.794   0.6725E+02  0.1448E+02         4.64           4 
   16    0    0     0.744   0.4135E+03  0.8079E+02         5.12           4 
   17    0    0     0.701   0.1047E+03  0.2174E+02         4.82           4 
   18    0    0     0.662   0.1690E+02  0.5336E+01         3.17           4 
   19    0    0     0.627   0.1256E+01  0.2352E+01         0.53           4 

          COMPLETENESS AND QUALITY OF DATA SET
          ------------------------------------

 R-FACTOR
 observed = (SUM(ABS(I(h,i)-I(h))))/(SUM(I(h,i)))
 expected = expected R-FACTOR derived from Sigma(I)

 COMPARED = number of reflections used for calculating R-FACTOR
 I/SIGMA  = mean of intensity/Sigma(I) of unique reflections
            (after merging symmetry-related observations)
 Sigma(I) = standard deviation of reflection intensity I
            estimated from sample statistics

 R-meas   = redundancy independent R-factor (intensities)
            Diederichs & Karplus (1997), Nature Struct. Biol. 4, 269-275.

 CC(1/2)  = percentage of correlation between intensities from
            random half-datasets. Correlation significant at
            the 0.1% level is marked by an asterisk.
            Karplus & Diederichs (2012), Science 336, 1030-33
 Anomal   = percentage of correlation between random half-sets
  Corr      of anomalous intensity differences. Correlation
            significant at the 0.1% level is marked.
 SigAno   = mean anomalous difference in units of its estimated
            standard deviation (|F(+)-F(-)|/Sigma). F(+), F(-)
            are structure factor estimates obtained from the
            merged intensity observations in each parity class.
  Nano    = Number of unique reflections used to calculate
            Anomal_Corr & SigAno. At least two observations
            for each (+ and -) parity are required.

       NOTE:      Friedel pairs are treated as different reflections.

 SUBSET OF INTENSITY DATA WITH SIGNAL/NOISE >= -3.0 AS FUNCTION OF RESOLUTION
 RESOLUTION     NUMBER OF REFLECTIONS    COMPLETENESS R-FACTOR  R-FACTOR COMPARED I/SIGMA   R-meas  CC(1/2)  Anomal  SigAno   Nano
   LIMIT     OBSERVED  UNIQUE  POSSIBLE     OF DATA   observed  expected                                      Corr

     1.78         587      51        51      100.0%      24.6%     29.0%      587    7.76     25.9%    91.5*     0    0.000       0
     1.27        1063      69        69      100.0%      25.8%     29.7%     1063    8.69     26.8%    98.2*     0    0.000       0
     1.04        1432      83        83      100.0%      31.1%     30.3%     1432    8.38     32.2%    97.5*     0    0.000       0
     0.90        1672      93        93      100.0%      41.8%     32.5%     1672    7.44     43.1%    96.4*     0    0.000       0
     0.81        1836      99        99      100.0%      40.8%     34.5%     1836    7.33     42.1%    94.8*     0    0.000       0
     0.74        2120     113       113      100.0%      38.4%     36.6%     2120    6.81     39.6%    97.4*     0    0.000       0
     0.68        2207     115       115      100.0%      38.1%     39.2%     2207    6.13     39.3%    98.8*     0    0.000       0
     0.64        2356     126       126      100.0%      45.1%     47.4%     2356    5.00     46.4%    97.0*     0    0.000       0
     0.60        1937     131       132       99.2%      42.3%     55.9%     1934    3.69     44.0%    92.0*     0    0.000       0
    total       15210     880       881       99.9%      30.9%     31.9%    15207    6.47     32.1%    96.6*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES     15210
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                        15210
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    880



 ******************************************************************************
    WILSON STATISTICS OF DATA SET  "XDS_ASCII.HKL"
 ******************************************************************************

 Data is divided into resolution shells and a straight line 
 A - 2*B*SS is fitted to log<I>, where
   RES    = mean resolution (Angstrom) in shell
   SS     = mean of (sin(THETA)/LAMBDA)**2 in shell
   <I>    = mean reflection intensity in shell
   BO     = (A - log<I>)/(2*SS)
    #     = number of reflections in resolution shell

   WILSON LINE (using all data) : A=   5.968 B=   2.875 CORRELATION=  0.98
      #      RES      SS        <I>       log(<I>)       BO
      22     3.512   0.020  5.0602E+02       6.227      -6.4
      27     2.085   0.058  2.5087E+02       5.525       3.8
      29     1.638   0.093  2.6733E+02       5.588       2.0
      35     1.389   0.130  2.0023E+02       5.299       2.6
      37     1.225   0.167  1.9819E+02       5.289       2.0
      36     1.115   0.201  1.3174E+02       4.881       2.7
      42     1.028   0.236  7.5319E+01       4.322       3.5
      48     0.955   0.274  5.5769E+01       4.021       3.5
      46     0.895   0.312  4.4737E+01       3.801       3.5
      48     0.847   0.349  3.2796E+01       3.490       3.6
      48     0.807   0.384  4.2396E+01       3.747       2.9
      52     0.772   0.419  3.6785E+01       3.605       2.8
      58     0.739   0.457  3.2556E+01       3.483       2.7
      47     0.712   0.493  2.5145E+01       3.225       2.8
      64     0.688   0.528  2.1876E+01       3.085       2.7
      59     0.664   0.567  1.5623E+01       2.749       2.8
      56     0.644   0.602  1.0898E+01       2.389       3.0
      64     0.626   0.638  1.0840E+01       2.383       2.8
      62     0.609   0.675  1.0054E+01       2.308       2.7


 HIGHER ORDER MOMENTS OF WILSON DISTRIBUTION OF  CENTRIC DATA
    AS COMPARED WITH THEORETICAL VALUES. (EXPECTED: 1.00)    
      #      RES        <I**2>/      <I**3>/      <I**4>/  
                         3<I>**2     15<I>**3    105<I>**4 

      22     3.512        0.526        0.204        0.062
      27     2.085        0.653        0.372        0.186
      29     1.638        0.634        0.339        0.156
      35     1.389        0.817        0.697        0.524
      37     1.225        0.521        0.210        0.072
      36     1.115        0.732        0.469        0.258
      42     1.028        0.679        0.475        0.321
      48     0.955        0.539        0.237        0.089
      46     0.895        0.565        0.292        0.139
      48     0.847        0.493        0.195        0.066
      48     0.807        0.556        0.254        0.098
      52     0.772        0.706        0.513        0.345
      58     0.739        0.871        0.958        1.037
      47     0.712        0.615        0.339        0.165
      64     0.688        0.547        0.238        0.087
      59     0.664        0.577        0.310        0.154
      56     0.644        0.538        0.242        0.094
      64     0.626        0.527        0.242        0.101
      62     0.609        0.715        0.572        0.436
     880   overall        0.620        0.382        0.241


 HIGHER ORDER MOMENTS OF WILSON DISTRIBUTION OF ACENTRIC DATA
    AS COMPARED WITH THEORETICAL VALUES. (EXPECTED: 1.00)    
      #      RES        <I**2>/      <I**3>/      <I**4>/  
                         2<I>**2      6<I>**3     24<I>**4 

       0   overall        0.000        0.000        0.000

   ======= CUMULATIVE INTENSITY DISTRIBUTION =======
 DEFINITIONS:
   <I>    = mean reflection intensity
 Na(Z)exp = expected number of acentric reflections with I <= Z*<I>
 Na(Z)obs = observed number of acentric reflections with I <= Z*<I>
 Nc(Z)exp = expected number of  centric reflections with I <= Z*<I>
 Nc(Z)obs = observed number of  centric reflections with I <= Z*<I>



 Nc(Z)obs/Nc(Z)exp versus resolution and Z (0.1-1.0)
      #      RES     0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9   1.0

      22     3.512  0.18  0.39  0.55  0.58  0.61  0.65  0.69  0.72  0.90  0.93
      27     2.085  0.45  0.54  0.53  0.70  0.78  0.73  0.68  0.71  0.79  0.81
      29     1.638  0.28  0.50  0.58  0.66  0.73  0.80  0.81  0.82  0.89  0.91
      35     1.389  0.12  0.17  0.34  0.72  0.93  0.92  0.91  0.95  0.96  0.96
      37     1.225  0.00  0.47  0.45  0.40  0.62  0.67  0.77  0.77  0.78  0.87
      36     1.115  0.11  0.32  0.87  0.94  0.85  0.94  1.07  1.02  1.06  1.02
      42     1.028  0.00  0.21  0.40  0.50  0.69  0.85  0.92  0.95  0.98  1.01
      48     0.955  0.00  0.06  0.30  0.53  0.64  0.74  0.73  0.80  0.92  0.95
      46     0.895  0.09  0.31  0.31  0.46  0.46  0.58  0.66  0.80  0.86  0.92
      48     0.847  0.08  0.06  0.15  0.22  0.36  0.56  0.73  0.86  0.95  0.98
      48     0.807  0.00  0.06  0.30  0.66  0.68  0.67  0.77  0.76  0.89  0.98
      52     0.772  0.00  0.11  0.37  0.53  0.66  0.75  0.84  0.89  1.02  1.04
      58     0.739  0.00  0.25  0.46  0.62  0.83  0.92  0.98  0.99  1.00  1.04
      47     0.712  0.09  0.12  0.26  0.49  0.69  0.83  0.86  0.98  1.00  0.97
      64     0.688  0.06  0.23  0.38  0.53  0.60  0.70  0.84  0.85  0.83  0.89
      59     0.664  0.00  0.10  0.16  0.25  0.55  0.69  0.91  1.00  0.95  0.97
      56     0.644  0.00  0.10  0.13  0.34  0.62  0.76  0.84  0.85  0.90  0.94
      64     0.626  0.06  0.18  0.19  0.33  0.42  0.45  0.73  0.85  0.98  1.01
      62     0.609  0.13  0.19  0.27  0.55  0.65  0.75  0.95  0.90  0.86  0.90
     880   overall  0.07  0.20  0.34  0.50  0.64  0.73  0.83  0.87  0.93  0.96


 Na(Z)obs/Na(Z)exp versus resolution and Z (0.1-1.0)
      #      RES     0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9   1.0

       0   overall  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00

 NUMBER OF UNIQUE ALIEN REFLECTIONS WITH A Z-SCORE ABOVE LIMIT         0
 (ALIENS ABOVE LIMIT (REJECT_ALIEN=      20.0) ARE MARKED INVALID)

 NUMBER OF REFLECTION RECORDS ON OUTPUT FILE "XDS_ASCII.HKL"       15210
 NUMBER OF ACCEPTED OBSERVATIONS (INCLUDING SYSTEMATIC ABSENCES)   15210
 NUMBER OF REJECTED MISFITS & ALIENS (marked by -1*SIGMA(IOBS))        0


 cpu time used                  0.3 sec
 elapsed wall-clock time        0.5 sec
