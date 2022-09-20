from typing import Callable, Dict, List

MSADirType = str
FamiliesType = List[str]
RateMatrixPathType = str
PhylogenyEstimatorReturnType = Dict[str, str]
PhylogenyEstimatorType = Callable[
    [
        MSADirType,
        FamiliesType,
        RateMatrixPathType,
    ],
    PhylogenyEstimatorReturnType,
]
