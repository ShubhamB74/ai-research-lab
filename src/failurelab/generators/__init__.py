from ..taxonomy import Trap
from .leakage import LeakageGenerator

# Only implemented families are registered. Add survivorship / multiple_testing in Phase 2.
REGISTRY = {
    Trap.LEAKAGE: LeakageGenerator(),
}
