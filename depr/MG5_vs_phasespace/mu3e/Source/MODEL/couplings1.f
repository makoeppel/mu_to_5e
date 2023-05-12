ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP1()

      IMPLICIT NONE
      INCLUDE 'model_functions.inc'

      DOUBLE PRECISION PI, ZERO
      PARAMETER  (PI=3.141592653589793D0)
      PARAMETER  (ZERO=0D0)
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      GC_3 = -(MDL_EE*MDL_COMPLEXI)
      GC_50 = -(MDL_CW*MDL_EE*MDL_COMPLEXI)/(2.000000D+00*MDL_SW)
      GC_53 = (MDL_CW*MDL_EE*MDL_COMPLEXI)/MDL_SW
      GC_59 = (MDL_EE*MDL_COMPLEXI*MDL_SW)/(2.000000D+00*MDL_CW)
      GC_62 = (MDL_CW*MDL_EE*MDL_COMPLEXI)/(2.000000D+00*MDL_SW)
     $ +(MDL_EE*MDL_COMPLEXI*MDL_SW)/(2.000000D+00*MDL_CW)
      GC_72 = (MDL_EE__EXP__2*MDL_COMPLEXI*MDL_VEV)/(2.000000D+00
     $ *MDL_SW__EXP__2)
      GC_89 = -((MDL_COMPLEXI*MDL_YE)/MDL_SQRT__2)
      GC_93 = -((MDL_COMPLEXI*MDL_YM)/MDL_SQRT__2)
      GC_100 = (MDL_EE*MDL_COMPLEXI*MDL_CONJG__CKM1X1)/(MDL_SW
     $ *MDL_SQRT__2)
      END
