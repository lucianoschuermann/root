from .square_root import SquareRootModel
from .square_root_extended import SquareRootExtendedModel
from .kissell_istar import KissellIStarModel
from .jpmorgan_spread import JPMorganSpreadModel
from .jpmorgan_no_spread import JPMorganNoSpreadModel
from .bloomberg import BloombergModel

ALL_MODELS = [
    SquareRootModel(),
    SquareRootExtendedModel(),
    KissellIStarModel(),
    JPMorganSpreadModel(),
    JPMorganNoSpreadModel(),
    BloombergModel(),
]

MODEL_LOOKUP = {m.name: m for m in ALL_MODELS}