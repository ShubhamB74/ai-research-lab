from ..taxonomy import Trap
from .leakage import LeakageGenerator
from .multiple_testing import MultipleTestingGenerator
from .survivorship import SurvivorshipGenerator

REGISTRY = {
    Trap.LEAKAGE: LeakageGenerator(),
    Trap.SURVIVORSHIP: SurvivorshipGenerator(),
    Trap.MULTIPLE_TESTING: MultipleTestingGenerator(),
}
