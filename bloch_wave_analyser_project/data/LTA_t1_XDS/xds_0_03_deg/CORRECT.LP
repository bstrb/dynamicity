
 ***** CORRECT ***** (VERSION Jan 19, 2025  BUILT=20251103)  22-Apr-2026

 INPUT PARAMETER VALUES
 ----------------------
 SPACE_GROUP_NUMBER=  221 as used in the INTEGRATE step
 UNIT_CELL_CONSTANTS=    11.909    11.909    11.909  90.000  90.000  90.000 as used by INTEGRATE
 FRIEDEL'S_LAW=FALSE
 PROFILE_FITTING= TRUE 
 OVERLOAD=    660000     MINPK=  75.00000    WFAC1=  10.0
 INCLUDE_RESOLUTION_RANGE=    20.000     0.600
 NAME_TEMPLATE_OF_DATA_FRAMES=../../../LTA_t1/image_16/??????.tiff TIFF
 DATA_RANGE=       1    3032
 ROTATION_AXIS=  0.151350  0.988479 -0.001324
 OSCILLATION_RANGE=  0.030000
 STARTING_ANGLE=   -45.000     STARTING_FRAME=       1
 X-RAY_WAVELENGTH=  0.019690
 INCIDENT_BEAM_DIRECTION=  0.000019  0.000014  1.000000
 FRACTION_OF_POLARIZATION=   0.500
 POLARIZATION_PLANE_NORMAL=  0.000000  0.000000  0.000000
 AIR=   0.00010
 DETECTOR=UNKNOWN
 SILICON=  0.402235 SENSOR_THICKNESS=  0.000000
 NUMBER OF DETECTOR SEGMENTS   1
 NX=   512  NY=   512    QX=  0.055000  QY=  0.055000
 ORGX=    243.91  ORGY=    249.32
 DETECTOR_DISTANCE=   443.865
 DIRECTION_OF_DETECTOR_X-AXIS=   1.00000   0.00000   0.00000
 DIRECTION_OF_DETECTOR_Y-AXIS=   0.00000   1.00000   0.00000
 BEAM_DIVERGENCE_E.S.D.=   0.015
 REFLECTING_RANGE_E.S.D.=   0.038
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

   15197 REFLECTIONS ON FILE "INTEGRATE.HKL"
       0 CORRUPTED REFLECTION RECORDS (IGNORED)
       0 MATCHING REFLECTIONS SPECIFIED IN FILE "FILTER.HKL"
       0 REFLECTIONS INCOMPLETE OR OUTSIDE IMAGE RANGE       1 ...    3032
       0 OVERLOADED REFLECTIONS (IGNORED)
       0 REFLECTIONS OUTSIDE ACCEPTED RESOLUTION RANGES
                OR TOO CLOSE TO ROTATION AXIS (IGNORED)
   15197 REFLECTIONS ACCEPTED



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

 *  31        aP          0.0      11.9   11.9   11.9  90.0  90.0  90.0   -1  0  0  0  0  0 -1  0  0 -1  0  0
 *  44        aP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    0  0 -1  0 -1  0  0  0  0  1  0  0
 *  35        mP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    1  0  0  0  0  0  1  0  0 -1  0  0
 *  33        mP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    1  0  0  0  0  0  1  0  0 -1  0  0
 *  10        mC          0.0      16.8   16.8   11.9  90.0  90.0  90.0   -1  0 -1  0 -1  0  1  0  0  1  0  0
 *  14        mC          0.0      16.8   16.8   11.9  90.0  90.0  90.0   -1  0 -1  0 -1  0  1  0  0  1  0  0
 *  11        tP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    0  0 -1  0 -1  0  0  0  0  1  0  0
 *  13        oC          0.0      16.8   16.8   11.9  90.0  90.0  90.0   -1  0 -1  0 -1  0  1  0  0  1  0  0
 *  32        oP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    0  0 -1  0 -1  0  0  0  0  1  0  0
 *  34        mP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    0  0  1  0  0 -1  0  0  1  0  0  0
 *  20        mC          0.0      16.8   16.8   11.9  90.0  90.0  90.0    1  1  0  0  1 -1  0  0  0  0 -1  0
 *  23        oC          0.0      16.8   16.8   11.9  90.0  90.0  90.0   -1  1  0  0  1  1  0  0  0  0 -1  0
 *  25        mC          0.0      16.8   16.8   11.9  90.0  90.0  90.0   -1  1  0  0  1  1  0  0  0  0 -1  0
 *   2        hR          0.0      16.8   16.8   20.6  90.0  90.0 120.0   -1  0  1  0  0  1 -1  0 -1 -1 -1  0
 *   4        hR          0.0      16.8   16.8   20.6  90.0  90.0 120.0    1  0 -1  0  0  1  1  0  1 -1  1  0
 *  21        tP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    1  0  0  0  0  0  1  0  0 -1  0  0
 *   3        cP          0.0      11.9   11.9   11.9  90.0  90.0  90.0    0  0 -1  0 -1  0  0  0  0  1  0  0
     5        cI        166.7      16.8   16.8   16.8  60.0  60.0  60.0    0  1 -1  0 -1  0 -1  0 -1  1  0  0
    41        mC        250.0      26.6   11.9   11.9  90.0  90.0  63.4    1 -2  0  0  1  0  0  0  0  0  1  0
    37        mC        250.0      26.6   11.9   11.9  90.0  90.0  63.4    1 -2  0  0  1  0  0  0  0  0  1  0
    28        mC        250.0      11.9   26.6   11.9  90.0  90.0  63.4    1  0  0  0  1 -2  0  0  0  0 -1  0
    30        mC        250.0      11.9   26.6   11.9  90.0  90.0  63.4    1  0  0  0  1 -2  0  0  0  0 -1  0
    40        oC        250.0      11.9   26.6   11.9  90.0  90.0 116.6    1  0  0  0 -1  2  0  0  0  0  1  0
    36        oC        250.0      11.9   26.6   11.9  90.0  90.0 116.6    1  0  0  0 -1  2  0  0  0  0  1  0
    29        mC        250.0      11.9   26.6   11.9  90.0  90.0  63.4   -1  0  0  0 -1  0  2  0  0  1  0  0
    39        mC        250.0      26.6   11.9   11.9  90.0  90.0  63.4    2  0  1  0  0  0  1  0  0 -1  0  0
    12        hP        250.0      11.9   11.9   11.9  90.0  90.0  90.0    0  0 -1  0 -1  0  0  0  0  1  0  0
    22        hP        250.0      11.9   11.9   11.9  90.0  90.0  90.0   -1  0  0  0  0  1  0  0  0  0 -1  0
    38        oC        250.0      11.9   26.6   11.9  90.0  90.0 116.6    1  0  0  0 -1  2  0  0  0  0  1  0
     1        cF        250.0      20.6   20.6   20.6 109.5 109.5 109.5   -1  1  1  0  1 -1  1  0  1  1 -1  0
    15        tI        500.0      11.9   11.9   29.2  65.9  65.9  90.0    0  0 -1  0 -1  0  0  0 -1  2 -1  0
    27        mC        500.0      26.6   11.9   16.8  90.0 129.2  63.4    1  0 -2  0  1  0  0  0  0 -1  1  0
    42        oI        500.0      11.9   11.9   29.2 114.1 114.1  90.0    0  0  1  0  1  0  0  0 -1  2 -1  0
    19        oI        500.0      11.9   16.8   20.6  90.0  54.7  90.0    1  0  0  0  0 -1  1  0  1 -1 -1  0
    26        oF        625.0      11.9   26.6   26.6  78.5 116.6 116.6   -1  0  0  0  1  0 -2  0  1 -2  0  0
    18        tI        625.0      16.8   20.6   11.9  54.7  90.0  90.0    0 -1  1  0 -1  1  1  0 -1  0  0  0
     9        hR        750.0      11.9   16.8   39.5  90.0 107.5 135.0   -1  0  0  0  1  0 -1  0  1 -3  1  0
     6        tI        999.0      16.8   16.8   16.8  60.0  60.0  60.0    0 -1  1  0  1 -1  0  0  1  0  1  0
     7        tI        999.0      16.8   16.8   16.8  60.0  60.0  60.0    1 -1  0  0  1  0  1  0  0 -1  1  0
     8        oI        999.0      16.8   16.8   16.8  60.0  60.0  60.0   -1  0 -1  0 -1  1  0  0  0  1 -1  0
    16        oF        999.0      16.8   16.8   29.2  90.0 125.3  90.0   -1  0 -1  0  1  0 -1  0  1 -2  1  0
    17        mC        999.0      16.8   16.8   16.8  60.0 120.0  90.0    1  0 -1  0 -1  0 -1  0 -1  1  0  0
    24        hR        999.0      29.2   16.8   11.9  90.0  65.9 106.8    1 -1  2  0  0 -1 -1  0  1  0  0  0
    43        mI        999.0      16.8   29.2   11.9 114.1 135.0  54.7   -1  0 -1  0 -1 -2 -1  0  0  0  1  0

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

  *   0.00        0   11.6      18    0.00  1  0  0  0  0  1  0  0  0  0  1  0

 ************ SELECTED SPACE GROUP AND UNIT CELL FOR THIS DATA SET ************

 SPACE_GROUP_NUMBER=  221
 UNIT_CELL_CONSTANTS=    11.91    11.91    11.91  90.000  90.000  90.000
 UNIT_CELL_A-AXIS=     7.410    -4.952     7.899
 UNIT_CELL_B-AXIS=     7.408     9.254    -1.148
 UNIT_CELL_C-AXIS=    -5.661     5.628     8.838
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


 REFINED VALUES OF DIFFRACTION PARAMETERS DERIVED FROM     13362 INDEXED SPOTS
 REFINED PARAMETERS:   AXIS ORIENTATION POSITION
 STANDARD DEVIATION OF SPOT    POSITION (PIXELS)     1.11
 STANDARD DEVIATION OF SPINDLE POSITION (DEGREES)    0.14
 SPACE GROUP NUMBER    221
 UNIT CELL PARAMETERS     11.909    11.909    11.909  90.000  90.000  90.000
 E.S.D. OF CELL PARAMETERS -1.0E+00-1.0E+00-1.0E+00-1.0E+00-1.0E+00-1.0E+00
 REC. CELL PARAMETERS   0.083968  0.083968  0.083968  90.000  90.000  90.000
 COORDINATES OF UNIT CELL A-AXIS     7.408    -4.952     7.901
 COORDINATES OF UNIT CELL B-AXIS     7.410     9.252    -1.149
 COORDINATES OF UNIT CELL C-AXIS    -5.660     5.631     8.837
 CRYSTAL MOSAICITY (DEGREES)     0.038
 LAB COORDINATES OF ROTATION AXIS  0.153991  0.988071 -0.001344
 DIRECT BEAM COORDINATES (REC. ANGSTROEM)   0.000000  0.000000 50.787209
 DETECTOR COORDINATES (PIXELS) OF DIRECT BEAM     243.98    249.38
 DETECTOR ORIGIN (PIXELS) AT                      243.98    249.38
 CRYSTAL TO DETECTOR DISTANCE (mm)       444.48
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
 -0.05  0.05      1480        194.264       1.056
  0.05  0.15      1467        196.870       1.042
  0.15  0.25      1489        200.449       1.023
  0.25  0.35      1529        208.608       0.983
  0.35  0.45      1560        225.391       0.910
  0.45  0.55      1549        210.846       0.973
  0.55  0.65      1494        195.442       1.049
  0.65  0.75      1534        212.211       0.966
  0.75  0.85      1593        215.871       0.950
  0.85  0.95      1502        188.281       1.089

 -0.05  0.95     15197        205.051       1.000


 ******************************************************************************
           CORRECTION FACTORS AS FUNCTION OF IMAGE NUMBER & RESOLUTION
 ******************************************************************************

 RECIPROCAL CORRECTION FACTORS FOR INPUT DATA SETS MERGED TO
 OUTPUT FILE: XDS_ASCII.HKL                                     

 THE CALCULATIONS ASSUME         FRIEDEL'S_LAW= TRUE
 TOTAL NUMBER OF CORRECTION FACTORS DEFINED      289
 DEGREES OF FREEDOM OF CHI^2 FIT             14304.0
 CHI^2-VALUE OF FIT OF CORRECTION FACTORS      0.883
 NUMBER OF CYCLES CARRIED OUT                      3

 CORRECTION FACTORS for visual inspection by XDS-Viewer DECAY.cbf           
 XMIN=     0.7 XMAX=  3027.0 NXBIN=   17
 YMIN= 0.00705 YMAX= 2.77092 NYBIN=   17
 NUMBER OF REFLECTIONS USED FOR DETERMINING CORRECTION FACTORS      15179


 ******************************************************************************
  CORRECTION FACTORS AS FUNCTION OF X (fast) & Y(slow) IN THE DETECTOR PLANE
 ******************************************************************************

 RECIPROCAL CORRECTION FACTORS FOR INPUT DATA SETS MERGED TO
 OUTPUT FILE: XDS_ASCII.HKL                                     

 THE CALCULATIONS ASSUME         FRIEDEL'S_LAW= TRUE
 TOTAL NUMBER OF CORRECTION FACTORS DEFINED      289
 DEGREES OF FREEDOM OF CHI^2 FIT             14299.3
 CHI^2-VALUE OF FIT OF CORRECTION FACTORS      0.881
 NUMBER OF CYCLES CARRIED OUT                      3

 CORRECTION FACTORS for visual inspection by XDS-Viewer MODPIX.cbf          
 XMIN=     3.5 XMAX=   508.1 NXBIN=   17
 YMIN=     3.5 YMAX=   508.1 NYBIN=   17
 NUMBER OF REFLECTIONS USED FOR DETERMINING CORRECTION FACTORS      15179


 ******************************************************************************
   CORRECTION FACTORS AS FUNCTION OF IMAGE NUMBER & DETECTOR SURFACE POSITION
 ******************************************************************************

 RECIPROCAL CORRECTION FACTORS FOR INPUT DATA SETS MERGED TO
 OUTPUT FILE: XDS_ASCII.HKL                                     

 THE CALCULATIONS ASSUME         FRIEDEL'S_LAW= TRUE
 TOTAL NUMBER OF CORRECTION FACTORS DEFINED      221
 DEGREES OF FREEDOM OF CHI^2 FIT             14304.0
 CHI^2-VALUE OF FIT OF CORRECTION FACTORS      0.880
 NUMBER OF CYCLES CARRIED OUT                      3

 CORRECTION FACTORS for visual inspection by XDS-Viewer ABSORP.cbf          
 XMIN=     0.7 XMAX=  3027.0 NXBIN=   17
 DETECTOR_SURFACE_POSITION=     256     256
 DETECTOR_SURFACE_POSITION=     342     342
 DETECTOR_SURFACE_POSITION=     170     342
 DETECTOR_SURFACE_POSITION=     170     170
 DETECTOR_SURFACE_POSITION=     342     170
 DETECTOR_SURFACE_POSITION=     450     336
 DETECTOR_SURFACE_POSITION=     336     450
 DETECTOR_SURFACE_POSITION=     175     450
 DETECTOR_SURFACE_POSITION=      62     336
 DETECTOR_SURFACE_POSITION=      62     175
 DETECTOR_SURFACE_POSITION=     175      62
 DETECTOR_SURFACE_POSITION=     336      62
 DETECTOR_SURFACE_POSITION=     450     175
 NUMBER OF REFLECTIONS USED FOR DETERMINING CORRECTION FACTORS      15179


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
 1.786E+02  8.916E-04    2.51


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

   11.909   4.773     2.50   0.53     17.65     29.13      31      32       0
    4.773   3.519     2.50   0.56     23.93     30.51      45      45       0
    3.519   2.916     2.50   0.34     20.03     30.40      39      39       0
    2.916   2.544     2.45   1.34     32.00     30.79      84      84       0
    2.544   2.286     2.40   0.66     22.45     29.88      88      88       0
    2.286   2.094     2.37   1.36     45.90     31.37      68      68       0
    2.094   1.943     2.34   0.71     24.59     30.71      88      88       0
    1.943   1.820     2.38   1.39     38.52     31.35     118     118       0
    1.820   1.718     2.31   1.74     38.46     31.24      83      83       0
    1.718   1.632     2.36   0.62     19.54     31.02     134     134       0
    1.632   1.557     2.38   1.37     35.14     31.22     107     107       0
    1.557   1.492     2.46   0.88     23.30     31.40     117     117       0
    1.492   1.434     2.31   1.64     32.88     30.80     130     130       0
    1.434   1.383     2.41   0.93     25.20     31.41     180     180       0
    1.383   1.337     2.38   1.03     22.92     31.43      97      97       0
    1.337   1.295     2.37   1.02     26.26     31.11     139     139       0
    1.295   1.256     2.33   1.00     33.03     31.87     169     169       0
    1.256   1.221     2.38   1.26     42.59     32.00     150     150       0
    1.221   1.189     2.41   0.87     21.45     30.76     141     141       0
    1.189   1.159     2.34   0.91     30.45     31.93     193     193       0
    1.159   1.132     2.37   0.88     25.12     31.35     185     185       0
    1.132   1.106     2.17   1.28     32.62     31.75     118     118       0
    1.106   1.082     1.87   1.67     65.46     35.30     187     187       0
    1.082   1.059     2.00   1.57     56.40     35.36     226     226       0
    1.059   1.038     2.12   1.68     50.40     33.27     167     167       0
    1.038   1.018     2.10   1.36     41.79     32.84     150     150       0
    1.018   0.999     2.08   1.19     37.94     32.73     219     219       0
    0.999   0.981     2.01   1.41     41.43     32.61     184     184       0
    0.981   0.964     2.07   1.09     42.51     35.01     191     191       0
    0.964   0.948     1.99   1.29     41.44     33.94     203     203       0
    0.948   0.933     1.97   1.43     51.01     34.80     218     218       0
    0.933   0.918     1.78   1.32     55.18     37.11     168     168       0
    0.918   0.904     1.89   1.18     46.12     34.97     262     262       0
    0.904   0.891     1.98   1.24     52.15     35.79     161     161       0
    0.891   0.878     2.06   0.95     36.31     34.52     225     225       0
    0.878   0.866     1.89   1.04     46.32     36.64     301     301       0
    0.866   0.854     2.02   1.08     42.45     33.91     163     163       0
    0.854   0.843     1.66   1.11     56.13     40.13     192     192       0
    0.843   0.832     1.83   1.05     48.62     38.11     221     221       0
    0.832   0.822     1.89   0.89     36.33     35.27     345     345       0
    0.822   0.812     1.90   0.93     38.52     35.95     157     157       0
    0.812   0.802     1.83   1.17     49.96     37.07     192     192       0
    0.802   0.793     1.88   1.03     45.72     37.77     264     264       0
    0.793   0.784     1.95   0.83     37.54     35.73     302     302       0
    0.784   0.775     1.74   0.77     35.99     37.41     252     252       0
    0.775   0.766     1.78   1.02     49.24     38.90     188     188       0
    0.766   0.758     1.68   0.79     31.60     36.84     312     312       0
    0.758   0.750     1.72   0.84     42.44     37.94     269     269       0
    0.750   0.743     1.37   0.88     43.32     41.84     211     211       0
    0.743   0.735     1.60   0.84     43.49     40.92     275     275       0
    0.735   0.728     1.80   0.81     41.85     36.97     245     245       0
    0.728   0.721     1.64   0.74     30.00     37.47     261     261       0
    0.721   0.714     1.49   0.71     45.02     41.53     342     342       0
    0.714   0.708     1.66   0.81     33.71     37.52     228     228       0
    0.708   0.701     1.46   0.75     45.19     45.46     176     176       0
    0.701   0.695     1.53   0.78     41.17     39.91     339     339       0
    0.695   0.689     1.41   0.65     44.40     41.65     272     272       0
    0.689   0.683     1.46   0.72     47.97     47.03     255     255       0
    0.683   0.677     1.51   0.80     43.46     41.62     377     377       0
    0.677   0.671     1.47   0.67     42.11     44.54     272     272       0
    0.671   0.666     1.36   0.77     51.93     45.20     204     204       0
    0.666   0.661     1.35   0.66     44.41     45.57     350     350       0
    0.661   0.655     1.25   0.58     50.48     52.73     323     323       0
    0.655   0.650     1.11   0.53     53.54     59.89     185     185       0
    0.650   0.645     1.06   0.58     59.05     57.10     230     230       0
    0.645   0.640     1.28   0.54     44.55     51.66     334     334       0
    0.640   0.636     1.14   0.55     51.00     52.21     276     276       0
    0.636   0.631     1.22   0.51     43.98     51.74     260     260       0
    0.631   0.626     1.01   0.42     45.44     59.30     178     178       0
    0.626   0.622     1.09   0.51     49.12     54.09     307     307       0
    0.622   0.617     1.13   0.52     49.38     55.42     261     261       0
    0.617   0.613     1.05   0.42     38.25     53.59     294     294       0
    0.613   0.609     1.18   0.42     34.92     49.46     230     230       0
    0.609   0.605     1.01   0.36     46.87     65.69     175     175       0
    0.605   0.601     0.91   0.34     46.72     68.76     112     113       0
   --------------------------------------------------------------------------
   11.909   0.601     1.72   0.88     32.79     33.29   15195   15197       0



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
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1     338
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

     1.78          68      23        51       45.1%      33.9%     27.9%       65    3.77     39.7%    66.6      0    0.000       0
     1.27         119      37        69       53.6%      32.9%     27.6%      113    3.70     38.6%    86.2*     0    0.000       0
     1.04         162      49        83       59.0%      37.9%     29.0%      156    3.38     44.5%    84.0*     0    0.000       0
     0.90         173      52        93       55.9%      52.4%     31.6%      166    3.17     62.6%    24.8      0    0.000       0
     0.81         211      64        99       64.6%      38.2%     33.5%      196    3.01     43.8%    65.5*     0    0.000       0
     0.74         224      66       113       58.4%      34.6%     34.9%      213    2.72     40.5%    87.4*     0    0.000       0
     0.68         249      75       115       65.2%      38.1%     38.1%      234    2.46     44.4%    88.9*     0    0.000       0
     0.64         275      81       126       64.3%      46.6%     45.8%      259    1.97     54.1%    77.7*     0    0.000       0
     0.60         211      77       132       58.3%      39.5%     52.7%      185    1.52     47.7%    74.3*     0    0.000       0
    total        1692     524       881       59.5%      36.3%     30.0%     1587    2.65     42.5%    81.9*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES      1692
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                         1692
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    524



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1     675
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

     1.78         133      37        51       72.5%      29.4%     28.5%      128    4.15     33.3%    87.1*     0    0.000       0
     1.27         235      55        69       79.7%      25.4%     28.9%      233    4.31     28.6%    91.2*     0    0.000       0
     1.04         323      70        83       84.3%      29.8%     29.0%      320    3.99     33.4%    95.4*     0    0.000       0
     0.90         362      80        93       86.0%      44.3%     31.7%      356    3.67     49.8%    80.2*     0    0.000       0
     0.81         411      85        99       85.9%      35.2%     33.9%      405    3.67     39.0%    85.4*     0    0.000       0
     0.74         470      97       113       85.8%      32.8%     35.4%      462    3.33     36.9%    92.1*     0    0.000       0
     0.68         490     101       115       87.8%      34.7%     38.0%      485    3.00     38.7%    95.9*     0    0.000       0
     0.64         533     109       126       86.5%      44.0%     45.9%      524    2.43     48.7%    73.1*     0    0.000       0
     0.60         424     105       132       79.5%      39.5%     55.1%      404    1.81     45.1%    85.8*     0    0.000       0
    total        3381     739       881       83.9%      31.0%     30.9%     3317    3.19     34.9%    92.6*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES      3381
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                         3381
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    739



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1    1011
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

     1.78         194      42        51       82.4%      28.3%     29.1%      189    4.72     31.4%    86.3*     0    0.000       0
     1.27         359      61        69       88.4%      25.2%     29.7%      358    5.09     27.6%    92.8*     0    0.000       0
     1.04         479      78        83       94.0%      28.7%     29.8%      479    4.69     31.5%    94.5*     0    0.000       0
     0.90         554      88        93       94.6%      43.3%     32.3%      551    4.21     47.4%    97.0*     0    0.000       0
     0.81         615      93        99       93.9%      36.5%     34.7%      614    4.19     39.7%    84.0*     0    0.000       0
     0.74         707     106       113       93.8%      35.3%     36.6%      706    3.86     38.5%    92.4*     0    0.000       0
     0.68         728     109       115       94.8%      33.1%     38.0%      726    3.51     36.1%    96.4*     0    0.000       0
     0.64         790     121       126       96.0%      44.1%     46.2%      784    2.83     47.7%    90.2*     0    0.000       0
     0.60         641     121       132       91.7%      38.2%     54.3%      628    2.13     42.3%    89.0*     0    0.000       0
    total        5067     819       881       93.0%      30.7%     31.7%     5035    3.70     33.7%    92.9*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES      5067
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                         5067
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    819



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1    1348
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

     1.78         264      51        51      100.0%      26.0%     28.9%      261    5.00     28.5%    92.3*     0    0.000       0
     1.27         470      69        69      100.0%      27.0%     29.8%      468    5.48     29.4%    92.3*     0    0.000       0
     1.04         636      83        83      100.0%      31.1%     30.5%      635    5.30     33.4%    95.9*     0    0.000       0
     0.90         739      93        93      100.0%      43.8%     32.6%      739    4.73     47.3%    94.4*     0    0.000       0
     0.81         818      99        99      100.0%      39.3%     34.7%      818    4.66     42.2%    97.3*     0    0.000       0
     0.74         937     113       113      100.0%      38.2%     36.6%      937    4.32     41.0%    94.6*     0    0.000       0
     0.68         983     115       115      100.0%      39.3%     39.2%      983    3.97     42.4%    63.4*     0    0.000       0
     0.64        1053     126       126      100.0%      45.3%     46.5%     1053    3.25     48.5%    85.0*     0    0.000       0
     0.60         848     127       132       96.2%      40.9%     54.2%      845    2.41     44.6%    93.4*     0    0.000       0
    total        6748     876       881       99.4%      31.6%     31.9%     6739    4.15     34.2%    95.2*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES      6748
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                         6748
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    876



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1    1685
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

     1.78         326      51        51      100.0%      25.5%     29.4%      325    5.58     27.7%    90.4*     0    0.000       0
     1.27         598      69        69      100.0%      26.5%     30.3%      598    6.23     28.2%    98.4*     0    0.000       0
     1.04         791      83        83      100.0%      33.5%     31.0%      791    5.88     35.5%    97.1*     0    0.000       0
     0.90         929      93        93      100.0%      44.0%     33.1%      929    5.25     46.7%    96.2*     0    0.000       0
     0.81        1026      99        99      100.0%      43.0%     35.3%     1026    5.19     45.6%    90.7*     0    0.000       0
     0.74        1175     113       113      100.0%      39.5%     37.1%     1175    4.87     41.8%    96.0*     0    0.000       0
     0.68        1231     115       115      100.0%      41.6%     39.7%     1231    4.40     43.9%    82.8*     0    0.000       0
     0.64        1318     126       126      100.0%      46.9%     46.7%     1318    3.64     49.4%    91.2*     0    0.000       0
     0.60        1051     127       132       96.2%      41.4%     53.4%     1050    2.74     44.5%    93.0*     0    0.000       0
    total        8445     876       881       99.4%      32.2%     32.3%     8443    4.64     34.3%    96.1*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES      8445
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                         8445
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    876



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1    2022
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

     1.78         389      51        51      100.0%      25.0%     29.7%      388    6.04     26.9%    90.5*     0    0.000       0
     1.27         714      69        69      100.0%      26.2%     30.6%      714    6.75     27.6%    97.6*     0    0.000       0
     1.04         956      83        83      100.0%      32.2%     31.3%      956    6.49     33.8%    93.8*     0    0.000       0
     0.90        1117      93        93      100.0%      43.6%     33.4%     1117    5.69     45.8%    97.5*     0    0.000       0
     0.81        1234      99        99      100.0%      42.4%     35.4%     1234    5.72     44.5%    90.4*     0    0.000       0
     0.74        1415     113       113      100.0%      38.7%     37.1%     1415    5.37     40.5%    96.2*     0    0.000       0
     0.68        1481     115       115      100.0%      40.7%     39.4%     1481    4.86     42.5%    95.2*     0    0.000       0
     0.64        1574     126       126      100.0%      46.8%     46.8%     1574    3.98     49.0%    91.1*     0    0.000       0
     0.60        1276     127       132       96.2%      41.0%     53.2%     1275    3.05     43.5%    93.7*     0    0.000       0
    total       10156     876       881       99.4%      31.7%     32.6%    10154    5.10     33.5%    95.6*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES     10156
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                        10156
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    876



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1    2358
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

     1.78         454      51        51      100.0%      25.2%     30.0%      453    6.51     26.8%    92.3*     0    0.000       0
     1.27         833      69        69      100.0%      25.8%     30.9%      833    7.27     27.0%    97.9*     0    0.000       0
     1.04        1120      83        83      100.0%      32.4%     31.6%     1120    6.99     33.8%    97.6*     0    0.000       0
     0.90        1299      93        93      100.0%      43.5%     33.6%     1299    6.14     45.3%    96.6*     0    0.000       0
     0.81        1442      99        99      100.0%      42.4%     35.6%     1442    6.18     44.2%    79.8*     0    0.000       0
     0.74        1651     113       113      100.0%      38.9%     37.4%     1651    5.78     40.4%    98.0*     0    0.000       0
     0.68        1722     115       115      100.0%      39.8%     39.6%     1722    5.24     41.4%    95.8*     0    0.000       0
     0.64        1834     126       126      100.0%      46.3%     46.8%     1834    4.29     48.0%    93.8*     0    0.000       0
     0.60        1488     128       132       97.0%      41.6%     53.8%     1487    3.29     43.8%    96.2*     0    0.000       0
    total       11843     877       881       99.5%      31.6%     32.9%    11841    5.49     33.2%    96.5*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES     11843
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                        11843
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    877



 ******************************************************************************
  SUMMARY OF DATA SET STATISTICS FOR IMAGE   DATA_RANGE=       1    2695
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

     1.78         525      51        51      100.0%      25.0%     30.3%      524    6.93     26.4%    90.6*     0    0.000       0
     1.27         946      69        69      100.0%      26.8%     31.0%      946    7.70     27.9%    98.6*     0    0.000       0
     1.04        1276      83        83      100.0%      32.2%     31.7%     1276    7.43     33.5%    97.0*     0    0.000       0
     0.90        1480      93        93      100.0%      43.2%     33.7%     1480    6.55     44.8%    97.6*     0    0.000       0
     0.81        1647      99        99      100.0%      42.5%     35.7%     1647    6.61     44.0%    92.7*     0    0.000       0
     0.74        1885     113       113      100.0%      39.4%     37.6%     1885    6.15     40.7%    98.7*     0    0.000       0
     0.68        1968     115       115      100.0%      40.0%     39.8%     1968    5.57     41.3%    92.2*     0    0.000       0
     0.64        2112     126       126      100.0%      46.1%     46.8%     2112    4.61     47.6%    97.0*     0    0.000       0
     0.60        1704     128       132       97.0%      42.7%     54.5%     1703    3.50     44.7%    94.8*     0    0.000       0
    total       13543     877       881       99.5%      31.8%     33.1%    13541    5.85     33.1%    95.8*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES     13543
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                        13543
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    877



 ******************************************************************************
  STATISTICS OF SAVED DATA SET "XDS_ASCII.HKL" (DATA_RANGE=       1    3032)
 FILE TYPE:         XDS_ASCII      MERGE=FALSE          FRIEDEL'S_LAW=FALSE
 ******************************************************************************

   REFLECTIONS OF TYPE H,0,0  0,K,0  0,0,L OR EXPECTED TO BE ABSENT (*)
   --------------------------------------------------------------------

   H    K    L  RESOLUTION  INTENSITY     SIGMA    INTENSITY/SIGMA  #OBSERVED

    1    0    0    11.909   0.1088E+04  0.4344E+03         2.50           1 
    2    0    0     5.955   0.2338E+04  0.5042E+03         4.64           4 
    3    0    0     3.970   0.9911E+03  0.2089E+03         4.74           4 
    4    0    0     2.977   0.2130E+04  0.4410E+03         4.83           4 
    5    0    0     2.382   0.4178E+04  0.8590E+03         4.86           4 
    6    0    0     1.985   0.1581E+04  0.3188E+03         4.96           4 
    7    0    0     1.701   0.3880E+03  0.8971E+02         4.33           4 
    8    0    0     1.489   0.3405E+04  0.6871E+03         4.96           4 
    9    0    0     1.323   0.1141E+04  0.2309E+03         4.94           4 
   10    0    0     1.191   0.3600E+04  0.7211E+03         4.99           4 
   11    0    0     1.083   0.2719E+03  0.5814E+02         4.68           4 
   12    0    0     0.992   0.5988E+03  0.1236E+03         4.85           4 
   13    0    0     0.916   0.1121E+03  0.2568E+02         4.36           4 
   14    0    0     0.851   0.3751E+02  0.1216E+02         3.08           4 
   15    0    0     0.794   0.1277E+03  0.2907E+02         4.39           4 
   16    0    0     0.744   0.8439E+03  0.1730E+03         4.88           4 
   17    0    0     0.701   0.2030E+03  0.4418E+02         4.59           4 
   18    0    0     0.662   0.3385E+02  0.1100E+02         3.08           4 
   19    0    0     0.627   0.2699E+01  0.4916E+01         0.55           4 

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

     1.78         588      51        51      100.0%      27.3%     30.5%      587    7.30     28.7%    85.5*     0    0.000       0
     1.27        1064      69        69      100.0%      27.4%     31.2%     1064    8.13     28.4%    98.5*     0    0.000       0
     1.04        1433      83        83      100.0%      32.6%     31.9%     1433    7.86     33.7%    97.2*     0    0.000       0
     0.90        1669      93        93      100.0%      43.4%     33.9%     1669    6.99     44.8%    98.7*     0    0.000       0
     0.81        1839      99        99      100.0%      42.7%     35.9%     1839    6.95     44.1%    94.8*     0    0.000       0
     0.74        2117     113       113      100.0%      40.2%     37.8%     2117    6.48     41.4%    97.8*     0    0.000       0
     0.68        2207     115       115      100.0%      40.1%     40.1%     2207    5.87     41.3%    97.3*     0    0.000       0
     0.64        2358     126       126      100.0%      46.9%     47.3%     2358    4.84     48.2%    96.5*     0    0.000       0
     0.60        1922     128       132       97.0%      43.7%     55.1%     1921    3.68     45.5%    92.1*     0    0.000       0
    total       15197     877       881       99.5%      32.8%     33.3%    15195    6.17     34.0%    94.6*     0    0.000       0


 NUMBER OF REFLECTIONS IN SELECTED SUBSET OF IMAGES     15197
 NUMBER OF REJECTED MISFITS                                 0
 NUMBER OF SYSTEMATIC ABSENT REFLECTIONS                    0
 NUMBER OF ACCEPTED OBSERVATIONS                        15197
 NUMBER OF UNIQUE ACCEPTED REFLECTIONS                    877



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

   WILSON LINE (using all data) : A=   6.654 B=   2.885 CORRELATION=  0.98
      #      RES      SS        <I>       log(<I>)       BO
      22     3.512   0.020  9.8119E+02       6.889      -5.8
      27     2.085   0.058  5.1119E+02       6.237       3.6
      29     1.638   0.093  5.3241E+02       6.277       2.0
      35     1.389   0.130  3.9569E+02       5.981       2.6
      37     1.225   0.167  3.9214E+02       5.972       2.1
      36     1.115   0.201  2.6156E+02       5.567       2.7
      42     1.028   0.236  1.4861E+02       5.001       3.5
      48     0.955   0.274  1.1014E+02       4.702       3.6
      46     0.895   0.312  8.7767E+01       4.475       3.5
      48     0.847   0.349  6.4787E+01       4.171       3.6
      48     0.807   0.384  8.4011E+01       4.431       2.9
      52     0.772   0.419  7.2661E+01       4.286       2.8
      58     0.739   0.457  6.4340E+01       4.164       2.7
      47     0.712   0.493  4.9391E+01       3.900       2.8
      64     0.688   0.528  4.3116E+01       3.764       2.7
      59     0.664   0.567  3.0509E+01       3.418       2.9
      56     0.644   0.602  2.1321E+01       3.060       3.0
      64     0.626   0.638  2.0994E+01       3.044       2.8
      59     0.609   0.674  2.0162E+01       3.004       2.7


 HIGHER ORDER MOMENTS OF WILSON DISTRIBUTION OF  CENTRIC DATA
    AS COMPARED WITH THEORETICAL VALUES. (EXPECTED: 1.00)    
      #      RES        <I**2>/      <I**3>/      <I**4>/  
                         3<I>**2     15<I>**3    105<I>**4 

      22     3.512        0.527        0.207        0.065
      27     2.085        0.643        0.353        0.169
      29     1.638        0.639        0.345        0.159
      35     1.389        0.826        0.715        0.543
      37     1.225        0.522        0.211        0.072
      36     1.115        0.756        0.507        0.295
      42     1.028        0.693        0.498        0.346
      48     0.955        0.548        0.247        0.095
      46     0.895        0.572        0.300        0.145
      48     0.847        0.501        0.203        0.071
      48     0.807        0.558        0.255        0.098
      52     0.772        0.721        0.542        0.377
      58     0.739        0.893        1.013        1.126
      47     0.712        0.620        0.346        0.170
      64     0.688        0.553        0.244        0.090
      59     0.664        0.588        0.323        0.164
      56     0.644        0.540        0.243        0.094
      64     0.626        0.531        0.249        0.108
      59     0.609        0.713        0.565        0.424
     877   overall        0.627        0.393        0.254


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

      22     3.512  0.18  0.39  0.33  0.48  0.61  0.65  0.76  0.72  0.90  0.93
      27     2.085  0.45  0.54  0.62  0.70  0.78  0.73  0.68  0.82  0.79  0.81
      29     1.638  0.28  0.50  0.58  0.73  0.73  0.86  0.81  0.82  0.89  0.91
      35     1.389  0.12  0.17  0.34  0.72  0.93  0.97  0.96  0.95  0.96  0.96
      37     1.225  0.00  0.47  0.45  0.40  0.57  0.63  0.77  0.77  0.78  0.87
      36     1.115  0.11  0.48  0.87  0.94  0.91  0.94  1.07  1.02  1.06  1.02
      42     1.028  0.00  0.21  0.40  0.50  0.69  0.89  0.92  0.91  0.98  1.01
      48     0.955  0.00  0.06  0.35  0.57  0.76  0.74  0.73  0.80  0.92  0.95
      46     0.895  0.09  0.31  0.37  0.46  0.46  0.58  0.66  0.83  0.86  0.89
      48     0.847  0.08  0.06  0.15  0.26  0.44  0.56  0.73  0.86  0.92  0.98
      48     0.807  0.00  0.12  0.30  0.66  0.68  0.67  0.80  0.76  0.92  0.98
      52     0.772  0.00  0.11  0.42  0.57  0.63  0.72  0.87  0.86  1.02  1.04
      58     0.739  0.00  0.25  0.50  0.66  0.83  0.95  0.98  0.96  1.00  1.04
      47     0.712  0.09  0.12  0.26  0.45  0.69  0.83  0.89  0.98  1.00  0.97
      64     0.688  0.06  0.27  0.38  0.53  0.63  0.67  0.84  0.85  0.86  0.89
      59     0.664  0.00  0.10  0.16  0.29  0.62  0.69  0.94  1.00  0.95  0.97
      56     0.644  0.00  0.10  0.13  0.38  0.65  0.83  0.84  0.85  0.90  0.94
      64     0.626  0.06  0.18  0.19  0.30  0.42  0.47  0.71  0.92  0.98  1.01
      59     0.609  0.00  0.10  0.28  0.57  0.65  0.85  0.94  0.92  0.88  0.97
     877   overall  0.06  0.21  0.35  0.52  0.65  0.74  0.84  0.88  0.93  0.96


 Na(Z)obs/Na(Z)exp versus resolution and Z (0.1-1.0)
      #      RES     0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9   1.0

       0   overall  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00

 NUMBER OF UNIQUE ALIEN REFLECTIONS WITH A Z-SCORE ABOVE LIMIT         0
 (ALIENS ABOVE LIMIT (REJECT_ALIEN=      20.0) ARE MARKED INVALID)

 NUMBER OF REFLECTION RECORDS ON OUTPUT FILE "XDS_ASCII.HKL"       15197
 NUMBER OF ACCEPTED OBSERVATIONS (INCLUDING SYSTEMATIC ABSENCES)   15197
 NUMBER OF REJECTED MISFITS & ALIENS (marked by -1*SIGMA(IOBS))        0


 cpu time used                  0.4 sec
 elapsed wall-clock time        0.3 sec
